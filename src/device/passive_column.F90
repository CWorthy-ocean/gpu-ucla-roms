#include "mem_attr.inc"
module passive_column_mod
  ! Vertical advection + implicit vertical diffusion of the passive
  ! tracers, fused into one column kernel (see passive_column_point).
  ! Separate from passive_mod so that the explicit kernel launch is not
  ! inlined into its caller.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  private
  integer(ip), parameter :: NTB = 2   ! tracers per thread in passive_column_point
  public :: passive_column_run

contains

  ! ------------------------------------------- column factorization
  subroutine passive_factor(dtau, use_fwd)
    ! Thomas factors of the implicit vertical-diffusion matrix for every
    ! column (col_fc = 1/diag, col_cf = upper factor, col_wc = fc +
    ! max(wc,0)), the salinity diffusivity; same arithmetic as
    ! implicit_tracer_columns. Shared by all passive tracers.
    real(rp), intent(in) :: dtau
    logical, intent(in) :: use_fwd
    integer(ip) :: i, j, k, nx, ny, nz, iAkt
    real(rp) :: cff, dc0, hz_k, hz_kp, fck, wck, cfk, fcm, wcm, cfm
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    iAkt = min(2_ip, int(size(Akt, 4), ip))
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        dc0 = dtau * pm(i, j) * pn(i, j)
        if (use_fwd) then
          hz_k = Hz_fwd(i, j, 1); hz_kp = Hz_fwd(i, j, 2)
        else
          hz_k = Hz(i, j, 1); hz_kp = Hz(i, j, 2)
        endif
        fck = 2.0_rp * dtau * Akt(i, j, 1, iAkt) / (hz_kp + hz_k)
        wck = dc0 * Wi(i, j, 1)
        cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp))
        cfk = cff * (fck - min(wck, 0.0_rp))
        col_fc(i, j, 1) = cff
        col_cf(i, j, 1) = cfk
        col_wc(i, j, 1) = fck + max(wck, 0.0_rp)
        do k = 2, nz - 1
          fcm = fck; wcm = wck; cfm = cfk
          if (use_fwd) then
            hz_k = Hz_fwd(i, j, k); hz_kp = Hz_fwd(i, j, k + 1)
          else
            hz_k = Hz(i, j, k); hz_kp = Hz(i, j, k + 1)
          endif
          fck = 2.0_rp * dtau * Akt(i, j, k, iAkt) / (hz_k + hz_kp)
          wck = dc0 * Wi(i, j, k)
          cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp) + &
                          fcm - min(wcm, 0.0_rp) - &
                          cfm * (fcm + max(wcm, 0.0_rp)))
          cfk = cff * (fck - min(wck, 0.0_rp))
          col_fc(i, j, k) = cff
          col_cf(i, j, k) = cfk
          col_wc(i, j, k) = fck + max(wck, 0.0_rp)
        enddo
        if (use_fwd) then
          hz_k = Hz_fwd(i, j, nz)
        else
          hz_k = Hz(i, j, nz)
        endif
        ! top: 1 / (hz + fc(nz-1) - min(wc(nz-1),0) - cf(nz-1)*(fc+max(wc,0))(nz-1))
        col_fc(i, j, nz) = 1.0_rp / (hz_k + fck - min(wck, 0.0_rp) - cfk * (fck + max(wck, 0.0_rp)))
        col_wc(i, j, nz) = fck + max(wck, 0.0_rp)
      enddo
    enddo
  end subroutine passive_factor

  ! ------------------------- vertical advection + implicit solve, fused
  DEVICE_ATTR
  subroutine passive_column_point(i, j, itb, ts, nn, ctr, dtp, is, ie, js, je, nz, ntr, nslot, nts, isl, mir, &
                                  tp_, Hz_, We_, pm_, pn_, rmask_, stflx_, ghat_, fc_, cf_, wc_)
    ! One (tracer block, column): the vertical flux divergence (advect's
    ! vert_column_fused arithmetic, sliding window over k) feeds the Thomas
    ! forward sweep level by level, with the intermediate dc kept in place
    ! in tp(nn); then the back substitution. Same arithmetic and operation
    ! order as the former passive_vertical + passive_implicit pair, so the
    ! FP64 build is bit-identical to it; the column factors come from
    ! passive_factor and are read once per tracer block. Arrays arrive as
    ! explicit-shape arguments (the pattern that works in device code with
    ! nvfortran 25.9; module arrays of other modules do not).
    ! ctr = 1: centred (predictor), 0: upwind-weighted (corrector).
    ! mir = 1: tracer 1 receives salinity's surface flux and the KPP
    ! nonlocal term between the vertical advection and the solve.
    integer(ip), intent(in) :: i, j, itb, ts, nn, ctr, is, ie, js, je, nz, ntr, nslot, nts, isl, mir
    real(rp), intent(in) :: dtp
    real(pp), intent(inout) :: tp_(is:ie, js:je, nz, nslot, ntr)
    real(rp), intent(in) :: Hz_(is:ie, js:je, nz), We_(is:ie, js:je, 0:nz)
    real(rp), intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp), intent(in) :: stflx_(is:ie, js:je, nts), ghat_(is:ie, js:je, 0:nz)
    real(rp), intent(in) :: fc_(is:ie, js:je, 0:nz), cf_(is:ie, js:je, 0:nz), wc_(is:ie, js:je, 0:nz)
    integer(ip) :: k, it, itr
    real(rp) :: dtpp, rm, w, sfx, cffk, cffm, tt
    real(rp) :: Hz_km1, Hz_k, Hz_kp1, Hz_kp2, fck, wcm, cfk
    real(rp) :: fz_lo(NTB), fz_k, tplus, tminus, rhs, dck(NTB)
    real(rp) :: t_km1(NTB), t_k(NTB), t_kp1(NTB), t_kp2(NTB)
    logical :: at_bottom, at_surface
    dtpp = dtp * pm_(i, j) * pn_(i, j)
    rm = rmask_(i, j)
    sfx = 0.0_rp
    if (mir == 1 .and. itb == 1) sfx = stflx_(i, j, isl)
    Hz_k = Hz_(i, j, 1)
    Hz_kp1 = Hz_(i, j, 2)
    Hz_km1 = Hz_k
    Hz_kp2 = Hz_kp1
    w = 0.0_rp
    do it = 1, NTB
      itr = (itb - 1) * NTB + it
      if (itr <= ntr) then
        t_k(it) = real(tp_(i, j, 1, ts, itr), rp)
        t_kp1(it) = real(tp_(i, j, 2, ts, itr), rp)
      else
        t_k(it) = 0.0_rp; t_kp1(it) = 0.0_rp
      endif
      t_km1(it) = t_k(it)
      t_kp2(it) = t_kp1(it)
      fz_lo(it) = 0.0_rp
      dck(it) = 0.0_rp
    enddo
    wcm = 0.0_rp
    ! ---- bottom to surface: vertical fluxes and forward elimination
    do k = 1, nz
      at_bottom = (k == 1)
      at_surface = (k == nz - 1)
      if (k < nz) then
        if (at_surface) then
          Hz_kp2 = Hz_kp1
        else
          Hz_kp2 = Hz_(i, j, k + 2)
        endif
        w = We_(i, j, k)
      endif
      fck = fc_(i, j, k)
      do it = 1, NTB
        itr = (itb - 1) * NTB + it
        if (itr > ntr) cycle
        if (k < nz) then
          if (at_surface) then
            t_kp2(it) = t_kp1(it)
          else
            t_kp2(it) = real(tp_(i, j, k + 2, ts, itr), rp)
          endif
          tplus = thickness_tplus(t_km1(it), t_k(it), t_kp1(it), Hz_km1, Hz_k, Hz_kp1, at_bottom)
          tminus = thickness_tminus(t_km1(it), t_k(it), t_kp1(it), t_kp2(it), Hz_km1, Hz_k, Hz_kp1, Hz_kp2, &
                                    at_bottom, at_surface)
          if (ctr == 1) then
            fz_k = 0.5_rp * (tplus + tminus) * w
          else
            fz_k = tplus * max(w, 0.0_rp) + tminus * min(w, 0.0_rp)
          endif
          fz_k = apply_mask(fz_k, rm)
        else
          fz_k = 0.0_rp
        endif
        rhs = real(real(tp_(i, j, k, nn, itr), rp) - dtpp * (fz_k - fz_lo(it)), pp)
        fz_lo(it) = fz_k
        t_km1(it) = t_k(it); t_k(it) = t_kp1(it); t_kp1(it) = t_kp2(it)
        if (mir == 1 .and. itr == 1) then
          ! salinity forcing of the mirror tracer, in the original order
          if (k == nz) rhs = real(rhs + dtp * sfx, pp)
          if (k < nz) then
            cffk = -dtp * ghat_(i, j, k) * sfx
            rhs = real(rhs + cffk, pp)
          endif
          if (k > 1) then
            cffm = -dtp * ghat_(i, j, k - 1) * sfx
            rhs = real(rhs - cffm, pp)
          endif
        endif
        if (k == 1) then
          dck(it) = fck * rhs
          tp_(i, j, k, nn, itr) = real(dck(it), pp)
        else if (k < nz) then
          dck(it) = fck * (rhs + dck(it) * wcm)
          tp_(i, j, k, nn, itr) = real(dck(it), pp)
        else
          tt = apply_mask((rhs + dck(it) * wcm) * fck, rm)
          tp_(i, j, k, nn, itr) = real(tt, pp)
          dck(it) = tt
        endif
      enddo
      if (k < nz) then
        wcm = wc_(i, j, k)
        Hz_km1 = Hz_k; Hz_k = Hz_kp1; Hz_kp1 = Hz_kp2
      endif
    enddo
    ! ---- surface to bottom: back substitution (dck holds the level above)
    do k = nz - 1, 1, -1
      cfk = cf_(i, j, k)
      do it = 1, NTB
        itr = (itb - 1) * NTB + it
        if (itr > ntr) cycle
        tt = apply_mask(real(tp_(i, j, k, nn, itr), rp) + cfk * dck(it), rm)
        tp_(i, j, k, nn, itr) = real(tt, pp)
        dck(it) = tt
      enddo
    enddo
  end subroutine passive_column_point

#ifdef USE_CUDA
  attributes(global) subroutine passive_column_kernel(ts, nn, ctr, dtp, is, ie, js, je, nx, ny, nz, ntr, nslot, nts, &
                                                      isl, nblk, mir, tp_, Hz_, We_, pm_, pn_, rmask_, stflx_, ghat_, &
                                                      fc_, cf_, wc_)
    integer(ip), value :: ts, nn, ctr, is, ie, js, je, nx, ny, nz, ntr, nslot, nts, isl, nblk, mir
    real(rp), value :: dtp
    real(pp), device, intent(inout) :: tp_(is:ie, js:je, nz, nslot, ntr)
    real(rp), device, intent(in) :: Hz_(is:ie, js:je, nz), We_(is:ie, js:je, 0:nz)
    real(rp), device, intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp), device, intent(in) :: stflx_(is:ie, js:je, nts), ghat_(is:ie, js:je, 0:nz)
    real(rp), device, intent(in) :: fc_(is:ie, js:je, 0:nz), cf_(is:ie, js:je, 0:nz), wc_(is:ie, js:je, 0:nz)
    integer(ip) :: i, j, itb
    i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
    itb = blockIdx%z
    if (i > nx .or. j > ny .or. itb > nblk) return
    call passive_column_point(i, j, itb, ts, nn, ctr, dtp, is, ie, js, je, nz, ntr, nslot, nts, isl, mir, &
                              tp_, Hz_, We_, pm_, pn_, rmask_, stflx_, ghat_, fc_, cf_, wc_)
  end subroutine passive_column_kernel
#endif

  subroutine passive_column_run(tp, ntr, ts, nn, dtp, ctr, use_fwd, mirror)
    ! Factorization, then one thread per (tracer block of NTB, column).
    ! Lives in its own module so the kernel launch is never inlined into
    ! the caller (nvfortran 25.9 segfaulted in cuLaunchKernel when it was).
    integer(ip), intent(in) :: ntr, ts, nn, ctr
    real(pp), WORK_ATTR intent(inout) :: tp(decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz, 3, ntr)
    real(rp), intent(in) :: dtp
    logical, intent(in) :: use_fwd, mirror
    integer(ip) :: nx, ny, nz, nblk, mir, is, ie, js, je, nts, isl
#ifdef USE_CUDA
    type(dim3) :: grid, block
#else
    integer(ip) :: i, j, itb
#endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nts = int(size(stflx, 3), ip); isl = int(isalt, ip)
    nblk = (ntr + NTB - 1) / NTB
    mir = merge(1_ip, 0_ip, mirror)
    call passive_factor(dtp, use_fwd)
#ifdef USE_CUDA
    block = dim3(32, 4, 1)
    grid = dim3((nx + 31) / 32, (ny + 3) / 4, nblk)
    call passive_column_kernel<<<grid, block>>>(ts, nn, ctr, dtp, is, ie, js, je, nx, ny, nz, ntr, 3_ip, nts, &
         isl, nblk, mir, tp, Hz, We, pm, pn, rmask, stflx, ghat, col_fc, col_cf, col_wc)
#else
    do itb = 1, nblk
      do j = 1, ny
        do i = 1, nx
          call passive_column_point(i, j, itb, ts, nn, ctr, dtp, is, ie, js, je, nz, ntr, 3_ip, nts, isl, mir, &
                                    tp, Hz, We, pm, pn, rmask, stflx, ghat, col_fc, col_cf, col_wc)
        enddo
      enddo
    enddo
#endif
  end subroutine passive_column_run


end module passive_column_mod
