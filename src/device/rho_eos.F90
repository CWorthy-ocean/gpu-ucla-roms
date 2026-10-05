module rho_eos_mod
  ! Jackett-McDougall SPLIT_EOS + DUKO_2001 + VAR_RHO_2D.
  ! Writes rho1, qp1, and the 2-D density weights rhoA, rhoS.
  ! Interior is one downward column sweep (EOS, N^2 from the level above,
  ! rhoA/rhoS sums); halo is before/after, not inside.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use halo_mod
  use arch_mod, only: sync_arch
  implicit none
  private
  public :: rho_eos

contains

  subroutine rho_eos(tidx)
    integer(ip), intent(in) :: tidx
    integer(ip) :: i, j, k, nx, ny, nz, tid
    integer(ip) :: i0, i1, j0, j1
    real(rp) :: Tt, Ts, r1, q1, dpth, cff, cff1, cffg, rm
    real(rp) :: r1u, q1u, zru, r1m, q1m, zrk, bv, rS, rA
    logical :: has_salt, edge_w, edge_e, edge_s, edge_n

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    tid = tidx
    has_salt = (size(t, 5) >= isalt)
    edge_w = decomp%edge_w
    edge_e = decomp%edge_e
    edge_s = decomp%edge_s
    edge_n = decomp%edge_n
    ! CPU rho_eos uses istrR:iendR / jstrR:jendR (physical-boundary
    ! ghost row included, MPI halo left to exchange).
    i0 = 1; i1 = nx
    j0 = 1; j1 = ny
    if (edge_w) i0 = 0
    if (edge_e) i1 = nx + 1
    if (edge_s) j0 = 0
    if (edge_n) j1 = ny + 1

    call halo_fill_3d_t(t, STAGGER_RHO, tid, itemp)
    if (has_salt) call halo_fill_3d_t(t, STAGGER_RHO, tid, isalt)
    call halo_fill_3d(Hz, STAGGER_RHO)
    call halo_fill_3d(z_r, STAGGER_RHO)
    call halo_fill_3d_w(z_w, STAGGER_RHO)
    write(*,'(a)') 'step: rho_eos prehalo'
    flush(6)
    call sync_arch()

    cffg = g / rho0
    cff1 = 1.0_rp / rho0
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = j0, j1
      do i = i0, i1
        rm = rmask(i, j)
        ! Level nz: EOS, start of the vertical sums.
        Tt = t(i, j, nz, tid, itemp)
        if (has_salt) then
          Ts = t(i, j, nz, tid, isalt)
        else
          Ts = 35.0_rp
        endif
        r1 = jackett_rho1(Tt, Ts)
        q1 = jackett_qp1(Tt, Ts, r1)
        r1u = apply_mask(r1, rm)
        q1u = apply_mask(q1, rm)
        zru = z_r(i, j, nz)
        rho1(i, j, nz) = r1u
        qp1(i, j, nz) = q1u
        rho(i, j, nz) = apply_mask(split_eos_insitu(r1u, q1u, zru), rm)
        dpth = -zru
        cff = Hz(i, j, nz) * (r1u + q1u * dpth * (1.0_rp - qp2 * dpth))
        rS = 0.5_rp * cff * Hz(i, j, nz)
        rA = cff
        bv = 0.0_rp
        ! Levels nz-1 .. 1: EOS, N^2 at the w-level above, sums.
        do k = nz - 1, 1, -1
          Tt = t(i, j, k, tid, itemp)
          if (has_salt) then
            Ts = t(i, j, k, tid, isalt)
          else
            Ts = 35.0_rp
          endif
          r1 = jackett_rho1(Tt, Ts)
          q1 = jackett_qp1(Tt, Ts, r1)
          r1m = apply_mask(r1, rm)
          q1m = apply_mask(q1, rm)
          zrk = z_r(i, j, k)
          rho1(i, j, k) = r1m
          qp1(i, j, k) = q1m
          rho(i, j, k) = apply_mask(split_eos_insitu(r1m, q1m, zrk), rm)
          bv = apply_mask(-cffg * adiabatic_diff(r1u, r1m, q1u, q1m, zru, zrk) / &
               max(zru - zrk, 1.0e-30_rp), rm)
          bvf(i, j, k) = bv
          if (k == nz - 1) bvf(i, j, nz) = bv
          dpth = -zrk
          cff = Hz(i, j, k) * (r1m + q1m * dpth * (1.0_rp - qp2 * dpth))
          rS = rS + Hz(i, j, k) * (rA + 0.5_rp * cff)
          rA = rA + cff
          r1u = r1m; q1u = q1m; zru = zrk
        enddo
        bvf(i, j, 0) = bv
        cff = 1.0_rp / max(z_w(i, j, nz) - z_w(i, j, 0), 1.0e-30_rp)
        rhoA(i, j) = apply_mask(cff * cff1 * rA, rm)
        rhoS(i, j) = apply_mask(2.0_rp * cff * cff * cff1 * rS, rm)
      enddo
    enddo
    call sync_arch()
    write(*,'(a)') 'step: rho_eos kernel'
    flush(6)

    call dump_eos(tid)

    call halo_fill_3d(rho1, STAGGER_RHO)
    call halo_fill_3d(qp1, STAGGER_RHO)
    call halo_fill_3d(rho, STAGGER_RHO)
    call halo_fill_3d_w(bvf, STAGGER_RHO)
    call halo_fill_2d(rhoA, STAGGER_RHO)
    call halo_fill_2d(rhoS, STAGGER_RHO)
  end subroutine rho_eos

  subroutine dump_eos(tind)
    ! Wet rho (455,56): T/S/rho1/qp1/zr/bvf feeding Riraw in LMD.
    use time_state
    integer(ip), intent(in) :: tind
    integer(ip), parameter :: i = 455_ip, j = 56_ip
    integer(ip) :: k, nz
    real(rp) :: dpth, adr, dzr, Tt, Ts
    if (iic - ntstart < kpp_probe_from .or. iic - ntstart > kpp_probe_to) return
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    call sync_arch()
    nz = decomp%nz
    do k = nz - 1, 1, -1
      Tt = t(i, j, k, tind, itemp)
      Ts = t(i, j, k, tind, isalt)
      dpth = -0.5_rp * (z_r(i, j, k + 1) + z_r(i, j, k))
      adr = adiabatic_diff(rho1(i, j, k + 1), rho1(i, j, k), &
           qp1(i, j, k + 1), qp1(i, j, k), z_r(i, j, k + 1), z_r(i, j, k))
      dzr = z_r(i, j, k + 1) - z_r(i, j, k)
      write(*,'(a,i6,i2,i4,7es26.17)') &
           'diag: eos P1 iic,tind,k,T,S,rho1,qp1,zr,bvf=', &
           iic, tind, k, Tt, Ts, rho1(i, j, k), qp1(i, j, k), z_r(i, j, k), bvf(i, j, k)
      write(*,'(a,i6,i2,i4,8es26.17)') &
           'diag: eos d P1 iic,tind,k,Thi,Shi,rho1hi,qp1hi,zr1,dpth,adr,dzr=', &
           iic, tind, k, t(i, j, k + 1, tind, itemp), t(i, j, k + 1, tind, isalt), &
           rho1(i, j, k + 1), qp1(i, j, k + 1), z_r(i, j, k + 1), dpth, adr, dzr
    enddo
    flush(6)
  end subroutine dump_eos

end module rho_eos_mod
