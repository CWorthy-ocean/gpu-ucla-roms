#include "mem_attr.inc"
module prsgrd_mod
  ! Shchepetkin density-Jacobian PGF (prsgrd_mod.F90). Frozen v1 is
  ! SPLIT_EOS, no NHMG, no isoneutral. P is a column kernel; XI/ETA
  ! faces are (i,j) launches per k. Halo is before/after, not inside.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use halo_mod
  use arch_mod, only: sync_arch
  use time_state
  implicit none
  private
  public :: prsgrd
  public :: dump_pgf_col

  ! dR/dZ (density and depth differences) alias col_fc/col_cf.
  real(rp), WORK_ATTR allocatable :: FC(:,:), rx(:,:), dZx(:,:), dRx(:,:)

contains

  subroutine ensure_scratch()
    integer(ip) :: is, ie, js, je
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    if (allocated(FC)) then
      if (lbound(FC, 1) == is .and. ubound(FC, 1) == ie .and. &
          lbound(FC, 2) == js .and. ubound(FC, 2) == je) return
      deallocate(FC, rx, dZx, dRx)
    endif
    allocate(FC(is:ie, js:je), rx(is:ie, js:je))
    allocate(dZx(is:ie, js:je), dRx(is:ie, js:je))
    FC = 0.0_rp; rx = 0.0_rp; dZx = 0.0_rp; dRx = 0.0_rp
  end subroutine ensure_scratch

  subroutine prsgrd()
    integer(ip) :: i, j, k, nx, ny, nz
    integer(ip) :: i_fc_lo, i_fc_hi, j_fc_lo, j_fc_hi
    integer(ip) :: i_ru_lo, i_ru_hi, j_rv_lo, j_rv_hi
    real(rp) :: HalfGRho, dpth, cff, dzk
    integer(ip) :: ii, im_lo, im_hi
    real(rp) :: za_m, za_0, za_p, ra_m, ra_0, ra_p, zc_m, zc_0, rc_m, rc_0
    logical :: do_tides, edge_w, edge_e, edge_s, edge_n

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    HalfGRho = 0.5_rp * g / rho0
    do_tides = pot_tides
    edge_w = decomp%edge_w
    edge_e = decomp%edge_e
    edge_s = decomp%edge_s
    edge_n = decomp%edge_n

    call hg_begin()
    call hg_add(F_ZR)
    call hg_add(F_ZW)
    call hg_add(F_HZ)
    call hg_add(F_RHO1)
    call hg_add(F_QP1)
    call hg_end()
    call ensure_scratch()

    ! Zero ru, rv on the device (not a host memset of managed memory).
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = decomp%js, decomp%je
        do i = decomp%is, decomp%ie
          ru(i, j, k) = 0.0_rp
          rv(i, j, k) = 0.0_rp
        enddo
      enddo
    enddo

    ! CPU column kernel is j=0:ny, i=0:nx. Do not build P on the
    ! north/east OBC rho rows (j=ny+1 / i=nx+1).
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny
      do i = 0, nx
        do k = 1, nz - 1
          col_cf(i, j, k) = z_r(i, j, k + 1) - z_r(i, j, k)
          col_fc(i, j, k) = adiabatic_diff(rho1(i, j, k + 1), rho1(i, j, k), &
                                       qp1(i, j, k + 1), qp1(i, j, k), &
                                       z_r(i, j, k + 1), z_r(i, j, k))
        enddo
        col_fc(i, j, nz) = col_fc(i, j, nz - 1)
        col_fc(i, j, 0)  = col_fc(i, j, 1)
        col_cf(i, j, nz) = col_cf(i, j, nz - 1)
        col_cf(i, j, 0)  = col_cf(i, j, 1)
        do k = nz, 1, -1
          ! Keep the averaged col_cf in a register: nvfortran reused the
          ! pre-store col_cf(i,j,k) load in the correction term below.
          cff = 2.0_rp * col_cf(i, j, k) * col_cf(i, j, k - 1)
          dzk = cff / max(col_cf(i, j, k) + col_cf(i, j, k - 1), 1.0e-30_rp)
          col_cf(i, j, k) = dzk
          dpth = -z_r(i, j, k)
          col_fc(i, j, k) = harmonic_avg(col_fc(i, j, k), col_fc(i, j, k - 1)) &
                        - qp1(i, j, k) * dzk * (1.0_rp - 2.0_rp * qp2 * dpth)
          rho(i, j, k) = split_eos_insitu(rho1(i, j, k), qp1(i, j, k), z_r(i, j, k))
        enddo
        P(i, j, nz) = surface_P(z_w(i, j, nz), z_r(i, j, nz), z_r(i, j, nz - 1), &
                                rho(i, j, nz), rho(i, j, nz - 1))
        if (do_tides) P(i, j, nz) = P(i, j, nz) - g * ptide(i, j)
        do k = nz - 1, 1, -1
          P(i, j, k) = P(i, j, k + 1) + HalfGRho * dens_jacobian( &
                       rho(i, j, k + 1), rho(i, j, k), &
                       z_r(i, j, k + 1), z_r(i, j, k), &
                       col_fc(i, j, k + 1), col_fc(i, j, k), col_cf(i, j, k + 1), col_cf(i, j, k))
        enddo
      enddo
    enddo
    call dump_pgf_h('after pgf P')

    if (edge_w) then
      i_fc_lo = 2
    else
      i_fc_lo = 0
    endif
    if (edge_e) then
      i_fc_hi = nx
    else
      i_fc_hi = nx + 2
    endif
    if (edge_s) then
      ! Match CPU SOUTHERN_EDGE: compute FC/rx from jstrV=2, then copy
      ! the south-of-first-interior face from that value (FC(1)=FC(2)).
      j_fc_lo = 2
    else
      j_fc_lo = 0
    endif
    if (edge_n) then
      ! Match CPU NORTHERN_EDGE: stop FC/rx at last interior V-face
      ! (j=ny) and copy the OBC face (FC(ny+1)=FC(ny)).
      j_fc_hi = ny
    else
      j_fc_hi = ny + 2
    endif
    i_ru_lo = merge(2, 1, edge_w)
    i_ru_hi = merge(nx, nx + 1, edge_e)
    j_rv_lo = merge(2, 1, edge_s)
    j_rv_hi = merge(ny, ny + 1, edge_n)

    if (probe_on) then
      ! Probe steps: per-level path (dump_pgf_v reads the 2-D work arrays).
      do k = nz, 1, -1
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = 1, ny
          do i = i_fc_lo, i_fc_hi
            FC(i, j) = apply_mask(z_r(i, j, k) - z_r(i - 1, j, k), umask(i, j))
            rx(i, j) = apply_mask(adiabatic_diff(rho1(i, j, k), rho1(i - 1, j, k), &
                                                 qp1(i, j, k), qp1(i - 1, j, k), &
                                                 z_r(i, j, k), z_r(i - 1, j, k)), umask(i, j))
          enddo
        enddo
        if (edge_w) then
#ifdef USE_CUDA
          !$cuf kernel do(1) <<<*,*>>>
#endif
          do j = 1, ny
            FC(i_fc_lo - 1, j) = FC(i_fc_lo, j)
            rx(i_fc_lo - 1, j) = rx(i_fc_lo, j)
          enddo
        endif
        if (edge_e) then
#ifdef USE_CUDA
          !$cuf kernel do(1) <<<*,*>>>
#endif
          do j = 1, ny
            FC(i_fc_hi + 1, j) = FC(i_fc_hi, j)
            rx(i_fc_hi + 1, j) = rx(i_fc_hi, j)
          enddo
        endif
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = 1, ny
          do i = i_ru_lo - 1, i_ru_hi
            dZx(i, j) = harmonic_avg(FC(i, j), FC(i + 1, j))
            dRx(i, j) = harmonic_avg(rx(i, j), rx(i + 1, j))
            dRx(i, j) = dRx(i, j) - qp1(i, j, k) * dZx(i, j) * (1.0_rp + 2.0_rp * qp2 * z_r(i, j, k))
          enddo
        enddo
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = 1, ny
          do i = i_ru_lo, i_ru_hi
            ! CPU prsgrd writes ru unmasked; umask is applied later in set_HUV1.
            ! Do not write the east OBC face i=nx+1 (CPU iend=nx on EASTERN_EDGE).
            ru(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j) * ( &
                          P(i - 1, j, k) - P(i, j, k) - HalfGRho * dens_jacobian( &
                          rho(i, j, k), rho(i - 1, j, k), &
                          z_r(i, j, k), z_r(i - 1, j, k), &
                          dRx(i, j), dRx(i - 1, j), dZx(i, j), dZx(i - 1, j)))
          enddo
        enddo
      enddo

      do k = nz, 1, -1
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do i = 1, nx
          do j = j_fc_lo, j_fc_hi
            FC(i, j) = apply_mask(z_r(i, j, k) - z_r(i, j - 1, k), vmask(i, j))
            rx(i, j) = apply_mask(adiabatic_diff(rho1(i, j, k), rho1(i, j - 1, k), &
                                                 qp1(i, j, k), qp1(i, j - 1, k), &
                                                 z_r(i, j, k), z_r(i, j - 1, k)), vmask(i, j))
          enddo
        enddo
        if (edge_s) then
#ifdef USE_CUDA
          !$cuf kernel do(1) <<<*,*>>>
#endif
          do i = 1, nx
            FC(i, j_fc_lo - 1) = FC(i, j_fc_lo)
            rx(i, j_fc_lo - 1) = rx(i, j_fc_lo)
          enddo
        endif
        if (edge_n) then
#ifdef USE_CUDA
          !$cuf kernel do(1) <<<*,*>>>
#endif
          do i = 1, nx
            FC(i, j_fc_hi + 1) = FC(i, j_fc_hi)
            rx(i, j_fc_hi + 1) = rx(i, j_fc_hi)
          enddo
        endif
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = j_rv_lo - 1, j_rv_hi
          do i = 1, nx
            dZx(i, j) = harmonic_avg(FC(i, j), FC(i, j + 1))
            dRx(i, j) = harmonic_avg(rx(i, j), rx(i, j + 1))
            dRx(i, j) = dRx(i, j) - qp1(i, j, k) * dZx(i, j) * (1.0_rp + 2.0_rp * qp2 * z_r(i, j, k))
          enddo
        enddo
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = j_rv_lo, j_rv_hi
          do i = 1, nx
            ! CPU prsgrd writes rv unmasked; vmask is applied later in set_HUV1.
            rv(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j) * ( &
                          P(i, j - 1, k) - P(i, j, k) - HalfGRho * dens_jacobian( &
                          rho(i, j, k), rho(i, j - 1, k), &
                          z_r(i, j, k), z_r(i, j - 1, k), &
                          dRx(i, j), dRx(i, j - 1), dZx(i, j), dZx(i, j - 1)))
          enddo
        enddo
        call dump_pgf_v('after pgf rv', k)
      enddo
    else
      ! One kernel per direction: the face differences (former wrk3a/wrk3b,
      ! with the edge copies folded into the face selection), the harmonic
      ! averages (former wrk3c/wrk3d) and the update, all per point with the
      ! same expressions: bit-identical to the multi-kernel version.
      ! Scalar call arguments parenthesized: see rho_eos (fixed single-block launch otherwise).
      im_lo = i_fc_lo; im_hi = i_fc_hi
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = 1, ny
          do i = i_ru_lo, i_ru_hi
            ! faces i-1, i, i+1 (clamped to the computed range at the open edges)
            ii = max(i - 1, im_lo)
            za_m = apply_mask(z_r(ii, j, k) - z_r(ii - 1, j, k), umask(ii, j))
            ra_m = apply_mask(adiabatic_diff(rho1(ii, j, k), rho1(ii - 1, j, k), qp1(ii, j, k), qp1(ii - 1, j, k), &
                                             z_r(ii, j, k), z_r(ii - 1, j, k)), umask(ii, j))
            za_0 = apply_mask(z_r(i, j, k) - z_r(i - 1, j, k), umask(i, j))
            ra_0 = apply_mask(adiabatic_diff(rho1(i, j, k), rho1(i - 1, j, k), qp1(i, j, k), qp1(i - 1, j, k), &
                                             z_r(i, j, k), z_r(i - 1, j, k)), umask(i, j))
            ii = min(i + 1, im_hi)
            za_p = apply_mask(z_r(ii, j, k) - z_r(ii - 1, j, k), umask(ii, j))
            ra_p = apply_mask(adiabatic_diff(rho1(ii, j, k), rho1(ii - 1, j, k), qp1(ii, j, k), qp1(ii - 1, j, k), &
                                             z_r(ii, j, k), z_r(ii - 1, j, k)), umask(ii, j))
            zc_m = harmonic_avg((za_m), (za_0))
            rc_m = harmonic_avg((ra_m), (ra_0))
            rc_m = rc_m - qp1(i - 1, j, k) * zc_m * (1.0_rp + 2.0_rp * qp2 * z_r(i - 1, j, k))
            zc_0 = harmonic_avg((za_0), (za_p))
            rc_0 = harmonic_avg((ra_0), (ra_p))
            rc_0 = rc_0 - qp1(i, j, k) * zc_0 * (1.0_rp + 2.0_rp * qp2 * z_r(i, j, k))
            ru(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i - 1, j, k)) * dn_u(i, j) * ( &
                          P(i - 1, j, k) - P(i, j, k) - HalfGRho * dens_jacobian( &
                          rho(i, j, k), rho(i - 1, j, k), &
                          z_r(i, j, k), z_r(i - 1, j, k), &
                          (rc_0), (rc_m), (zc_0), (zc_m)))
          enddo
        enddo
      enddo
      im_lo = j_fc_lo; im_hi = j_fc_hi
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 1, nz
        do j = j_rv_lo, j_rv_hi
          do i = 1, nx
            ii = max(j - 1, im_lo)
            za_m = apply_mask(z_r(i, ii, k) - z_r(i, ii - 1, k), vmask(i, ii))
            ra_m = apply_mask(adiabatic_diff(rho1(i, ii, k), rho1(i, ii - 1, k), qp1(i, ii, k), qp1(i, ii - 1, k), &
                                             z_r(i, ii, k), z_r(i, ii - 1, k)), vmask(i, ii))
            za_0 = apply_mask(z_r(i, j, k) - z_r(i, j - 1, k), vmask(i, j))
            ra_0 = apply_mask(adiabatic_diff(rho1(i, j, k), rho1(i, j - 1, k), qp1(i, j, k), qp1(i, j - 1, k), &
                                             z_r(i, j, k), z_r(i, j - 1, k)), vmask(i, j))
            ii = min(j + 1, im_hi)
            za_p = apply_mask(z_r(i, ii, k) - z_r(i, ii - 1, k), vmask(i, ii))
            ra_p = apply_mask(adiabatic_diff(rho1(i, ii, k), rho1(i, ii - 1, k), qp1(i, ii, k), qp1(i, ii - 1, k), &
                                             z_r(i, ii, k), z_r(i, ii - 1, k)), vmask(i, ii))
            zc_m = harmonic_avg((za_m), (za_0))
            rc_m = harmonic_avg((ra_m), (ra_0))
            rc_m = rc_m - qp1(i, j - 1, k) * zc_m * (1.0_rp + 2.0_rp * qp2 * z_r(i, j - 1, k))
            zc_0 = harmonic_avg((za_0), (za_p))
            rc_0 = harmonic_avg((ra_0), (ra_p))
            rc_0 = rc_0 - qp1(i, j, k) * zc_0 * (1.0_rp + 2.0_rp * qp2 * z_r(i, j, k))
            rv(i, j, k) = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j) * ( &
                          P(i, j - 1, k) - P(i, j, k) - HalfGRho * dens_jacobian( &
                          rho(i, j, k), rho(i, j - 1, k), &
                          z_r(i, j, k), z_r(i, j - 1, k), &
                          (rc_0), (rc_m), (zc_0), (zc_m)))
          enddo
        enddo
      enddo
    endif

    call hg_begin()
    call hg_add(F_RU)
    call hg_add(F_RV)
    call hg_end()
  end subroutine prsgrd

  subroutine dump_pgf_col(tag)
    ! Column geometry / EOS at P1 i=318, j=431:433.
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, jj, kk, nx, ny
    integer(ip) :: js(3), ks(4)
    real(rp) :: pt, zt
    if (.not. probe_on) return
    call sync_arch()
    nx = decomp%nx
    ny = decomp%ny
    i = iprb_pt(1)
    if (i < 1 .or. i > nx) return
    js = (/ ny - 1, ny, ny + 1 /)
    ks = (/ 1, 50, 99, 100 /)
    zt = zeta(i, ny, knew)
    do jj = 1, 3
      j = js(jj)
      if (j < decomp%js .or. j > decomp%je) cycle
      pt = 0.0_rp
      if (allocated(ptide)) pt = ptide(i, j)
      do kk = 1, 4
        k = ks(kk)
        if (k < 1 .or. k > decomp%nz) cycle
        write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,10es26.17)') 'diag: pgfC  i=', i, &
             ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, &
             '  zr,Hz,zw,rho1,qp1,T,S,zetaN,rmask,ptide=', &
             z_r(i, j, k), Hz(i, j, k), z_w(i, j, k), rho1(i, j, k), qp1(i, j, k), &
             t(i, j, k, nrhs, itemp), t(i, j, k, nrhs, isalt), zt, rmask(i, j), pt
      enddo
    enddo
    flush(6)
  end subroutine dump_pgf_col

  subroutine dump_pgf_h(tag)
    character(len=*), intent(in) :: tag
    integer(ip) :: i, j, k, jj, kk, ny
    integer(ip) :: js(3), ks(4)
    if (.not. probe_on) return
    call sync_arch()
    ny = decomp%ny
    i = iprb_pt(1)
    if (i < 1 .or. i > decomp%nx) return
    js = (/ ny - 1, ny, ny + 1 /)
    ks = (/ 1, 50, 99, 100 /)
    do jj = 1, 3
      j = js(jj)
      if (j < decomp%js .or. j > decomp%je) cycle
      do kk = 1, 4
        k = ks(kk)
        if (k < 1 .or. k > decomp%nz) cycle
        write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,6es26.17)') 'diag: pgfH  i=', i, &
             ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, &
             '  dR,dZ,rho,P,zr,Hz=', col_fc(i, j, k), col_cf(i, j, k), rho(i, j, k), &
             P(i, j, k), z_r(i, j, k), Hz(i, j, k)
      enddo
    enddo
    flush(6)
  end subroutine dump_pgf_h

  subroutine dump_pgf_v(tag, k)
    character(len=*), intent(in) :: tag
    integer(ip), intent(in) :: k
    integer(ip) :: i, j, jj, ny
    integer(ip) :: js(2)
    real(rp) :: dP, jac, rv_raw, HalfGRho
    if (.not. probe_on) return
    if (.not. (k == 1 .or. k == 50 .or. k == 99 .or. k == 100)) return
    call sync_arch()
    ny = decomp%ny
    i = iprb_pt(1)
    if (i < 1 .or. i > decomp%nx) return
    HalfGRho = 0.5_rp * g / rho0
    js = (/ ny - 1, ny /)
    do jj = 1, 2
      j = js(jj)
      if (j < decomp%js + 1 .or. j > decomp%je) cycle
      dP = P(i, j - 1, k) - P(i, j, k)
      jac = dens_jacobian(rho(i, j, k), rho(i, j - 1, k), z_r(i, j, k), z_r(i, j - 1, k), &
           dRx(i, j), dRx(i, j - 1), dZx(i, j), dZx(i, j - 1))
      rv_raw = 0.5_rp * (Hz(i, j, k) + Hz(i, j - 1, k)) * dm_v(i, j) * (dP - HalfGRho * jac)
      write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,16es26.17)') 'diag: pgfV  i=', i, &
           ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, &
           '  Hz,HzS,dmv,vmask,P,PS,rho,rhoS,zr,zrS,dP,jac,rv,rv_raw,rmask,rmaskS=', &
           Hz(i, j, k), Hz(i, j - 1, k), dm_v(i, j), vmask(i, j), &
           P(i, j, k), P(i, j - 1, k), rho(i, j, k), rho(i, j - 1, k), &
           z_r(i, j, k), z_r(i, j - 1, k), dP, jac, rv(i, j, k), rv_raw, &
           rmask(i, j), rmask(i, j - 1)
      write(*,'(a,i6,a,i6,a,a,a,i8,a,i4,a,8es26.17)') 'diag: pgfVd i=', i, &
           ' j=', j, '  ', trim(tag), '  iic=', iic, '  k=', k, &
           '  dZx,dZxS,dRx,dRxS,FC,FCN,rx,rxN=', &
           dZx(i, j), dZx(i, j - 1), dRx(i, j), dRx(i, j - 1), &
           FC(i, j), FC(i, j + 1), rx(i, j), rx(i, j + 1)
    enddo
    flush(6)
  end subroutine dump_pgf_v

end module prsgrd_mod
