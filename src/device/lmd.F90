#include "mem_attr.inc"
module lmd_mod
  ! LMD94 interior Ri mixing + KPP / BKPP / nonlocal.
  ! MiniPac: LMD_MIXING + LMD_KPP + LMD_BKPP + LMD_NONLOCAL + LMD_RIMIX
  ! + SMOOTH_RIG + INT_AT_RHO_POINTS + SMOOTH_HBL. No DDMIX / CONVEC /
  ! MERGE_OVERLAP / LIMIT_UNSTABLE_ONLY. Jwt=1. Cartesian.
  ! Column-local loops launch on the GPU. SMOOTH_RIG still halo-syncs
  ! per k; the KPP column search is a CUF do(2) kernel.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use time_state
  use halo_mod
  use arch_mod, only: sync_arch
  implicit none
  private
  public :: lmd_vmix
  public :: lmd_swr_frac

  ! Kv/Kt/Ks/Rig alias col_wc/wrk3d/wrk3e/col_dc (dead after lmd_vmix).
  real(rp), WORK_ATTR allocatable :: hbl(:,:), bbl(:,:), ustar(:,:), Bo(:,:), Bosol(:,:)

contains

  subroutine ensure_scratch()
    integer(ip) :: is, ie, js, je
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    call ensure_wrk3()
    if (allocated(hbl)) then
      if (lbound(hbl, 1) == is .and. ubound(hbl, 1) == ie .and. &
          lbound(hbl, 2) == js .and. ubound(hbl, 2) == je) return
      deallocate(hbl, bbl, ustar, Bo, Bosol)
    endif
    allocate(hbl(is:ie, js:je), bbl(is:ie, js:je), ustar(is:ie, js:je))
    allocate(Bo(is:ie, js:je), Bosol(is:ie, js:je))
    hbl = 0.0_rp; bbl = 0.0_rp; ustar = 0.0_rp; Bo = 0.0_rp; Bosol = 0.0_rp
  end subroutine ensure_scratch

  subroutine lmd_swr_frac()
    integer(ip) :: i, j, k, nx, ny, nz
    real(rp) :: attn1, attn2, swdk1, swdk2

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    attn1 = -1.0_rp / jwt1_mu1
    attn2 = -1.0_rp / jwt1_mu2

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        swdk1 = jwt1_r1
        swdk2 = 1.0_rp - jwt1_r1
        swr_frac(i, j, nz) = 1.0_rp
        do k = nz, 1, -1
          swdk1 = swr_band_atten(swdk1, attn1, Hz(i, j, k))
          swdk2 = swr_band_atten(swdk2, attn2, Hz(i, j, k))
          swr_frac(i, j, k - 1) = apply_mask(swdk1 + swdk2, rmask(i, j))
        enddo
      enddo
    enddo
    call sync_arch()
    call halo_fill_3d_w(swr_frac, STAGGER_RHO)
  end subroutine lmd_swr_frac

  subroutine lmd_vmix(tind)
    integer(ip), intent(in) :: tind

    call halo_fill_3d_s(u, STAGGER_U, tind)
    call halo_fill_3d_s(v, STAGGER_V, tind)
    if (tind /= nstp) then
      call halo_fill_3d_s(u, STAGGER_U, nstp)
      call halo_fill_3d_s(v, STAGGER_V, nstp)
    endif
    call halo_fill_3d_t(t, STAGGER_RHO, nstp, itemp)
    if (size(t, 5) >= isalt) call halo_fill_3d_t(t, STAGGER_RHO, nstp, isalt)
    call halo_fill_3d(z_r, STAGGER_RHO)
    call halo_fill_3d_w(z_w, STAGGER_RHO)
    call halo_fill_3d(Hz, STAGGER_RHO)
    call halo_fill_3d_w(bvf, STAGGER_RHO)
    call halo_fill_2d(sustr, STAGGER_U)
    call halo_fill_2d(svstr, STAGGER_V)
    call halo_fill_2d_st(stflx, STAGGER_RHO, itemp)
    if (size(stflx, 3) >= isalt) call halo_fill_2d_st(stflx, STAGGER_RHO, isalt)
    call halo_fill_2d(srflx, STAGGER_RHO)
    call halo_fill_2d(f, STAGGER_RHO)
    call halo_fill_2d(hbls, STAGGER_RHO)
    call halo_fill_2d(hbbl, STAGGER_RHO)

    call ensure_scratch()
    call lmd_swr_frac()
    call zero_mix_coefs()
    call lmd_interior(tind)
    call lmd_kpp(tind)
  end subroutine lmd_vmix

  subroutine lmd_interior(tind)
    integer(ip), intent(in) :: tind
    integer(ip) :: i, j, k, nx, ny, nz, tid
    real(rp) :: cff, dudz, dvdz, nu, mult
    real(rp) :: akv_loc, akt_t, akt_s

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    tid = tind
    akv_loc = akv_bak
    akt_t = akt_bak(itemp)
    akt_s = akt_bak(isalt)

    ! SMOOTH_RIG: halo + isotropic smooth between levels.
    ! Levels are independent: each stage runs for all levels at once.
    call ensure_wrk3()
    call lmd_rig_3d(tid)
    call pad_phys_edges_3s(col_dc, 1, nz - 1)
    ! One 3-D exchange (levels 0:nz) instead of nz-1 per-level exchanges;
    ! levels 1:nz-1 receive the same values, levels 0 and nz are not read.
    call halo_fill_3d_w(col_dc, STAGGER_RHO)
    call smooth_iso_3d_s(col_dc, 1, nz - 1)

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        do k = 1, nz - 1
          nu = lmd_nu_sx(col_dc(i, j, k))
          col_wc(i, j, k) = lmd_nuwm + lmd_nu0m * nu
          wrk3d(i, j, k) = lmd_nuws + lmd_nu0s * nu
          wrk3e(i, j, k) = wrk3d(i, j, k)
          mult = lmd_bottom_mult(z_w(i, j, k) - z_w(i, j, 0))
          col_wc(i, j, k) = col_wc(i, j, k) * mult
          wrk3d(i, j, k) = wrk3d(i, j, k) * mult
          wrk3e(i, j, k) = wrk3e(i, j, k) * mult
        enddo
        col_wc(i, j, nz) = col_wc(i, j, nz - 1) + akv_loc
        wrk3d(i, j, nz) = wrk3d(i, j, nz - 1) + akt_t
        wrk3e(i, j, nz) = wrk3e(i, j, nz - 1) + akt_s
        col_wc(i, j, 0)  = col_wc(i, j, 1) + akv_loc
        wrk3d(i, j, 0)  = wrk3d(i, j, 1) + akt_t
        wrk3e(i, j, 0)  = wrk3e(i, j, 1) + akt_s
        ! Sequential in k, matching lmd_vmix_tile (uses already-smoothed k-1).
        do k = 1, nz - 1
          col_wc(i, j, k) = 0.5_rp * col_wc(i, j, k) + 0.25_rp * col_wc(i, j, k - 1) + &
                        0.25_rp * col_wc(i, j, k + 1) + akv_loc
          wrk3d(i, j, k) = 0.5_rp * wrk3d(i, j, k) + 0.25_rp * wrk3d(i, j, k - 1) + &
                        0.25_rp * wrk3d(i, j, k + 1) + akt_t
          wrk3e(i, j, k) = 0.5_rp * wrk3e(i, j, k) + 0.25_rp * wrk3e(i, j, k - 1) + &
                        0.25_rp * wrk3e(i, j, k + 1) + akt_s
        enddo
      enddo
    enddo
    call sync_arch()
    call dump_ri(tid)
  end subroutine lmd_interior

  subroutine lmd_kpp(tind)
    integer(ip), intent(in) :: tind
    integer(ip) :: i, j, k, nx, ny, nz, tid, n_stp, first, kbls, kbbl
    logical :: bulk
    real(rp) :: Cg, Vtc, alpha, beta, Bfsfc, zscale, ws, wm, Kern, Vtsq
    real(rp) :: ssgm, z_bl, cff, cff1, cff_up, cff_dn, amp, wmb, wsb, sgmb, swdk

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    tid = tind
    n_stp = nstp
    first = merge(1_ip, 0_ip, is_first_time_step())
    Cg = lmd_Cg()
    Vtc = lmd_Vtc()
    bulk = have_bulk

    ! --- per-column surface forcing, friction velocity, previous BL depths
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        alpha = jackett_alpha(t(i, j, nz, n_stp, itemp), t(i, j, nz, n_stp, isalt))
        beta  = jackett_beta(t(i, j, nz, n_stp, itemp), t(i, j, nz, n_stp, isalt))
        Bo(i, j) = g * (alpha * (stflx(i, j, itemp) - srflx(i, j)) - &
                        beta * stflx(i, j, isalt))
        Bosol(i, j) = g * alpha * srflx(i, j)
        if (bulk) then
          ustar(i, j) = sqrt(sqrt(sustr_r(i, j)**2 + svstr_r(i, j)**2))
        else
          ustar(i, j) = sqrt(sqrt(lmd_ustar_w * ( &
               sustr(i, j)**2 + sustr(i + 1, j)**2 + sustr(i, j) * sustr(i + 1, j) + &
               svstr(i, j)**2 + svstr(i, j + 1)**2 + svstr(i, j) * svstr(i, j + 1))))
        endif
        hbl(i, j) = hbls(i, j)
        bbl(i, j) = hbbl(i, j)
      enddo
    enddo

    ! --- per-level unresolved-shear term Vtsq (independent levels)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          swdk = sqrt(max(0.0_rp, swr_frac(i, j, k) * swr_frac(i, j, k - 1)))
          zscale = z_w(i, j, nz) - z_r(i, j, k)
          Bfsfc = Bo(i, j) + Bosol(i, j) * (1.0_rp - swdk)
          ws = lmd_ws(zscale, Bfsfc, ustar(i, j), hbl(i, j), rmask(i, j))
          wrk3f(i, j, k) = lmd_Vtsq_fix * Vtc * ws * sqrt(max(1.0e-5_rp, bvf(i, j, k - 1)))
        enddo
      enddo
    enddo

    ! --- column: bulk Richardson integral (recurrence) and the BL searches
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        kbls = 0
        col_cf(i, j, nz) = 0.0_rp
        col_cf(i, j, 0) = 0.0_rp
        col_fc(i, j, nz) = 0.0_rp
        do k = nz - 1, 1, -1
          cff_up = (z_w(i, j, nz) - z_w(i, j, k))**2
          cff_dn = (z_w(i, j, k) - z_w(i, j, 0))**2
          Kern = cff_up * cff_dn / max( &
               (cff_up + (lmd_epssfc * hbl(i, j))**2) * &
               (cff_dn + (lmd_epssfc * bbl(i, j))**2), 1.0e-30_rp)
          col_fc(i, j, k) = col_fc(i, j, k + 1) + Kern * ( &
               0.5_rp * ((u(i, j, k + 1, tid) + u(i + 1, j, k + 1, tid) - &
                          u(i, j, k, tid) - u(i + 1, j, k, tid))**2 + &
                         (v(i, j, k + 1, tid) + v(i, j + 1, k + 1, tid) - &
                          v(i, j, k, tid) - v(i, j + 1, k, tid))**2) / &
               max(Hz(i, j, k) + Hz(i, j, k + 1), 1.0e-30_rp) - &
               0.5_rp * (Hz(i, j, k) + Hz(i, j, k + 1)) * ( &
               lmd_Ri_inv * bvf(i, j, k) + lmd_C_Ek * f(i, j) * f(i, j)))
        enddo
        z_bl = z_w(i, j, 0) + 0.25_rp * Hz(i, j, 1)
        cff_up = (z_w(i, j, nz) - z_bl)**2
        cff_dn = (z_bl - z_w(i, j, 0))**2
        Kern = cff_up * cff_dn / max( &
             (cff_up + (lmd_epssfc * hbl(i, j))**2) * &
             (cff_dn + (lmd_epssfc * bbl(i, j))**2), 1.0e-30_rp)
        col_fc(i, j, 0) = col_fc(i, j, 1) + Kern * ( &
             0.5_rp * ((u(i, j, 1, tid) + u(i + 1, j, 1, tid))**2 + &
                       (v(i, j, 1, tid) + v(i, j + 1, 1, tid))**2) / &
             max(Hz(i, j, 1), 1.0e-30_rp) - &
             0.5_rp * Hz(i, j, 1) * (lmd_Ri_inv * bvf(i, j, 1) + &
             lmd_C_Ek * f(i, j) * f(i, j)))

        do k = nz, 1, -1
          col_cf(i, j, k) = col_fc(i, j, k) + wrk3f(i, j, k)
          if (kbls == 0 .and. col_cf(i, j, k) < 0.0_rp) kbls = k
        enddo
        if (kbls > 0) then
          k = kbls
          if (k == nz) then
            hbl(i, j) = z_w(i, j, nz) - z_r(i, j, nz)
          else
            hbl(i, j) = z_w(i, j, nz) - (z_r(i, j, k) * col_cf(i, j, k + 1) - &
                 z_r(i, j, k + 1) * col_cf(i, j, k)) / max(col_cf(i, j, k + 1) - col_cf(i, j, k), 1.0e-30_rp)
          endif
        else
          hbl(i, j) = z_w(i, j, nz) - z_w(i, j, 0)
        endif
        hbl(i, j) = apply_mask(hbl(i, j), rmask(i, j))

        kbbl = 0
        col_cf(i, j, 0) = 0.0_rp
        do k = 1, nz
          col_cf(i, j, k) = col_fc(i, j, k) - col_fc(i, j, 0)
          if (kbbl == 0 .and. col_cf(i, j, k) > 0.0_rp) kbbl = k
        enddo
        if (kbbl > 0) then
          k = kbbl
          if (k == 1) then
            bbl(i, j) = z_r(i, j, 1) - z_w(i, j, 0)
          else
            bbl(i, j) = (z_r(i, j, k - 1) * col_cf(i, j, k) - z_r(i, j, k) * col_cf(i, j, k - 1)) / &
                 max(col_cf(i, j, k) - col_cf(i, j, k - 1), 1.0e-30_rp) - z_w(i, j, 0)
          endif
        else
          bbl(i, j) = z_w(i, j, nz) - z_w(i, j, 0)
        endif
        bbl(i, j) = apply_mask(bbl(i, j), rmask(i, j))
      enddo
    enddo

    call dump_kpp(1, tid)
    call pad_phys_edges(hbl)
    call halo_fill_2d(hbl, STAGGER_RHO)
    call smooth_iso_2d(hbl)
    call pad_phys_edges(bbl)
    call halo_fill_2d(bbl, STAGGER_RHO)
    call smooth_iso_2d(bbl)
    call dump_kpp(2, tid)

    ! --- per column: time-averaged depths, surface forcing at the BL base
    !     (wrk2a), bottom friction velocity (wrk2b), stored depths.
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (first == 0) then
          hbl(i, j) = 0.5_rp * (hbl(i, j) + hbls(i, j))
          bbl(i, j) = 0.5_rp * (bbl(i, j) + hbbl(i, j))
        endif

        kbls = nz
        do k = nz - 1, 1, -1
          if (z_w(i, j, k) > z_w(i, j, nz) - hbl(i, j)) kbls = k
        enddo
        k = kbls
        z_bl = z_w(i, j, nz) - hbl(i, j)
        if (swr_frac(i, j, k - 1) > 0.0_rp) then
          Bfsfc = Bo(i, j) + Bosol(i, j) * (1.0_rp - swr_frac(i, j, k - 1) * &
               swr_frac(i, j, k) * (z_w(i, j, k) - z_w(i, j, k - 1)) / max( &
               swr_frac(i, j, k) * (z_w(i, j, k) - z_bl) + &
               swr_frac(i, j, k - 1) * (z_bl - z_w(i, j, k - 1)), 1.0e-30_rp))
        else
          Bfsfc = Bo(i, j) + Bosol(i, j)
        endif
        wrk2a(i, j) = Bfsfc

        wmb = vonkar * vonkar * sqrt(lmd_ustar_w * ( &
             u(i, j, 1, n_stp)**2 + u(i + 1, j, 1, n_stp)**2 + &
             u(i, j, 1, n_stp) * u(i + 1, j, 1, n_stp) + &
             v(i, j, 1, n_stp)**2 + v(i, j + 1, 1, n_stp)**2 + &
             v(i, j, 1, n_stp) * v(i, j + 1, 1, n_stp))) / &
             log(1.0_rp + 0.5_rp * Hz(i, j, 1) / max(zob, 1.0e-30_rp))
        wrk2b(i, j) = wmb
        hbls(i, j) = hbl(i, j)
        hbbl(i, j) = bbl(i, j)
      enddo
    enddo

    ! --- per level: surface BL profile, bottom BL profile, Akv/Akt
    !     (same per-level order as the former column loops: surface, then
    !     bottom, then the masked copy).
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 0, nz
      do j = 1, ny
        do i = 1, nx
          Bfsfc = wrk2a(i, j)
          zscale = z_w(i, j, nz) - z_w(i, j, k)
          wm = lmd_wm(zscale, Bfsfc, ustar(i, j), hbl(i, j), rmask(i, j))
          ws = lmd_ws(zscale, Bfsfc, ustar(i, j), hbl(i, j), rmask(i, j))
          ssgm = (z_w(i, j, nz) - z_w(i, j, k)) / max(hbl(i, j), lmd_eps)
          if (ssgm < 1.0_rp) then
            if (ssgm < 0.07_rp) then
              cff = 0.5_rp * (ssgm - 0.07_rp)**2 / 0.07_rp
            else
              cff = 0.0_rp
            endif
            cff = cff + ssgm * (1.0_rp - ssgm)**2
            amp = ssgm**2
            col_wc(i, j, k) = sqrt((amp * col_wc(i, j, k))**2 + (wm * hbl(i, j) * cff)**2)
            wrk3d(i, j, k) = sqrt((amp * wrk3d(i, j, k))**2 + (ws * hbl(i, j) * cff)**2)
            wrk3e(i, j, k) = sqrt((amp * wrk3e(i, j, k))**2 + (ws * hbl(i, j) * cff)**2)
            if (Bfsfc < 0.0_rp) then
              ghat(i, j, k) = -Cg * ssgm * (1.0_rp - ssgm)**2
            else
              ghat(i, j, k) = 0.0_rp
            endif
          else
            ghat(i, j, k) = 0.0_rp
          endif

          wmb = wrk2b(i, j)
          wsb = wmb
          sgmb = (z_w(i, j, k) - z_w(i, j, 0) + zob) / max(bbl(i, j) + zob, 1.0e-30_rp)
          if (sgmb < 1.0_rp) then
            cff1 = sgmb * (1.0_rp - sgmb)**2
            col_wc(i, j, k) = sqrt(col_wc(i, j, k)**2 + (wmb * bbl(i, j) * cff1)**2)
            wrk3d(i, j, k) = sqrt(wrk3d(i, j, k)**2 + (wsb * bbl(i, j) * cff1)**2)
            wrk3e(i, j, k) = sqrt(wrk3e(i, j, k)**2 + (wsb * bbl(i, j) * cff1)**2)
          endif

          if (rmask(i, j) > 0.5_rp) then
            Akv(i, j, k) = col_wc(i, j, k)
            Akt(i, j, k, itemp) = wrk3d(i, j, k)
            Akt(i, j, k, isalt) = wrk3e(i, j, k)
          else
            Akv(i, j, k) = 0.0_rp
            Akt(i, j, k, itemp) = 0.0_rp
            Akt(i, j, k, isalt) = 0.0_rp
          endif
        enddo
      enddo
    enddo
    call dump_kpp(3, tid)

    call pad_phys_edges(hbls)
    call pad_phys_edges(hbbl)
    call halo_fill_2d(hbls, STAGGER_RHO)
    call halo_fill_2d(hbbl, STAGGER_RHO)
    call halo_fill_3d_w(Akv, STAGGER_RHO)
    call halo_fill_3d_w_s(Akt, STAGGER_RHO, itemp)
    call halo_fill_3d_w_s(Akt, STAGGER_RHO, isalt)
    call halo_fill_3d_w(ghat, STAGGER_RHO)
  end subroutine lmd_kpp

  subroutine dump_kpp(stage, tind)
    ! West rho neighbor (455,56) of U-face (456,56). Labeled P1 so the
    ! CPU kpp probe (nkpp_probe=1) matches.
    integer(ip), intent(in) :: stage, tind
    integer(ip), parameter :: i = 455_ip, j = 56_ip
    integer(ip) :: k, nz, kbls
    real(rp) :: z_bl, Bfsfc
    if (iic - ntstart < kpp_probe_from .or. iic - ntstart > kpp_probe_to) return
    if (i < 1 .or. i > decomp%nx .or. j < 1 .or. j > decomp%ny) return
    call sync_arch()
    nz = decomp%nz
    select case (stage)
    case (1)
      write(*,'(a,i6,i2,8es26.17)') 'diag: kpp in P1 iic,tind,ustar,Bo,Bosol,hbls,hbl,sustr_r,svstr_r,f=', &
           iic, tind, ustar(i, j), Bo(i, j), Bosol(i, j), hbls(i, j), hbl(i, j), &
           sustr_r(i, j), svstr_r(i, j), f(i, j)
      write(*,'(a,i6,i2,8es26.17)') 'diag: kpp uv P1 iic,tind,uW,uE,vS,vN,Tnz,Snz,stflxT,srflx=', &
           iic, tind, u(i, j, nz, tind), u(i + 1, j, nz, tind), &
           v(i, j, nz, tind), v(i, j + 1, nz, tind), &
           t(i, j, nz, nstp, itemp), t(i, j, nz, nstp, isalt), &
           stflx(i, j, itemp), srflx(i, j)
      do k = nz, 1, -1
        write(*,'(a,i6,i2,i4,5es26.17)') 'diag: kppk P1 iic,tind,k,FC,bvf,swr,T,zr=', &
             iic, tind, k, col_fc(i, j, k), bvf(i, j, k), &
             swr_frac(i, j, k), t(i, j, k, nstp, itemp), z_r(i, j, k)
      enddo
    case (2)
      write(*,'(a,i6,i2,es26.17)') 'diag: kpp sm P1 iic,tind,hbl=', iic, tind, hbl(i, j)
    case default
      kbls = nz
      do k = nz - 1, 1, -1
        if (z_w(i, j, k) > z_w(i, j, nz) - hbl(i, j)) kbls = k
      enddo
      k = kbls
      z_bl = z_w(i, j, nz) - hbl(i, j)
      if (swr_frac(i, j, k - 1) > 0.0_rp) then
        Bfsfc = Bo(i, j) + Bosol(i, j) * (1.0_rp - swr_frac(i, j, k - 1) * &
             swr_frac(i, j, k) * (z_w(i, j, k) - z_w(i, j, k - 1)) / &
             (swr_frac(i, j, k) * (z_w(i, j, k) - z_bl) + &
              swr_frac(i, j, k - 1) * (z_bl - z_w(i, j, k - 1))))
      else
        Bfsfc = Bo(i, j) + Bosol(i, j)
      endif
      write(*,'(a,i6,i2,i4,6es26.17)') 'diag: kpp out P1 iic,tind,kbls,hbl,Bfsfc,Akt99,Akt97,Akv99,ghat99=', &
           iic, tind, kbls, hbl(i, j), Bfsfc, Akt(i, j, nz - 1, itemp), &
           Akt(i, j, nz - 3, itemp), Akv(i, j, nz - 1), ghat(i, j, nz - 1)
      do k = nz, 1, -1
        write(*,'(a,i6,i2,i4,3es26.17)') 'diag: kppak P1 iic,tind,k,Akt,Akv,ssgm=', &
             iic, tind, k, Akt(i, j, k, itemp), Akv(i, j, k), &
             (z_w(i, j, nz) - z_w(i, j, k)) / max(hbl(i, j), lmd_eps)
      enddo
    end select
    flush(6)
  end subroutine dump_kpp

  subroutine dump_ri(tind)
    ! Interior Ri mixing at (455,56) — AkvW feeder for U-face (456,56) —
    ! and its 3x3 SMOOTH_RIG stencil.
    integer(ip), intent(in) :: tind
    integer(ip), parameter :: i0 = 455_ip, j0 = 56_ip
    integer(ip) :: i, j, k, nz, di, dj, ii, jj
    real(rp) :: cff, dudz, dvdz, sh2, riraw, nu
    if (iic - ntstart < kpp_probe_from .or. iic - ntstart > kpp_probe_to) return
    if (i0 < 1 .or. i0 > decomp%nx .or. j0 < 1 .or. j0 > decomp%ny) return
    call sync_arch()
    nz = decomp%nz
    i = i0
    j = j0
    do k = nz - 1, 1, -1
      cff = 0.5_rp / max(z_r(i, j, k + 1) - z_r(i, j, k), 1.0e-30_rp)
      dudz = cff * (u(i, j, k + 1, tind) - u(i, j, k, tind) + &
                    u(i + 1, j, k + 1, tind) - u(i + 1, j, k, tind))
      dvdz = cff * (v(i, j, k + 1, tind) - v(i, j, k, tind) + &
                    v(i, j + 1, k + 1, tind) - v(i, j + 1, k, tind))
      sh2 = dudz * dudz + dvdz * dvdz
      riraw = lmd_gradient_ri(bvf(i, j, k), dudz, dvdz)
      write(*,'(a,i6,i2,i4,5es26.17)') &
           'diag: ri sh P1 iic,tind,k,dudz,dvdz,sh2,bvf,Riraw=', &
           iic, tind, k, dudz, dvdz, sh2, bvf(i, j, k), riraw
      nu = lmd_nu_sx(col_dc(i, j, k))
      write(*,'(a,i6,i2,i4,4es26.17)') &
           'diag: ri sm P1 iic,tind,k,Rig,nu,Kv,Kt=', &
           iic, tind, k, col_dc(i, j, k), nu, col_wc(i, j, k), wrk3d(i, j, k)
      do dj = -1_ip, 1_ip
        do di = -1_ip, 1_ip
          ii = i + di
          jj = j + dj
          if (ii < 1 .or. ii > decomp%nx .or. jj < 1 .or. jj > decomp%ny) cycle
          cff = 0.5_rp / max(z_r(ii, jj, k + 1) - z_r(ii, jj, k), 1.0e-30_rp)
          dudz = cff * (u(ii, jj, k + 1, tind) - u(ii, jj, k, tind) + &
                        u(ii + 1, jj, k + 1, tind) - u(ii + 1, jj, k, tind))
          dvdz = cff * (v(ii, jj, k + 1, tind) - v(ii, jj, k, tind) + &
                        v(ii, jj + 1, k + 1, tind) - v(ii, jj + 1, k, tind))
          riraw = lmd_gradient_ri(bvf(ii, jj, k), dudz, dvdz)
          write(*,'(a,i6,i2,i4,i3,i3,5es26.17)') &
               'diag: ri nb P1 iic,tind,k,di,dj,dudz,dvdz,bvf,Riraw,Rigsm=', &
               iic, tind, k, di, dj, dudz, dvdz, bvf(ii, jj, k), riraw, col_dc(ii, jj, k)
          write(*,'(a,i6,i2,i4,i3,i3,3es26.17)') &
               'diag: ri msk P1 iic,tind,k,di,dj,rmask,umask,vmask=', &
               iic, tind, k, di, dj, rmask(ii, jj), umask(ii, jj), vmask(ii, jj)
        enddo
      enddo
      ! Same column: west U-face (455,56) and east U-face (456,56).
      cff = 0.5_rp / max(z_r(i, j, k + 1) - z_r(i, j, k), 1.0e-30_rp)
      dudz = cff * (u(i, j, k + 1, tind) - u(i, j, k, tind) + &
                    u(i + 1, j, k + 1, tind) - u(i + 1, j, k, tind))
      dvdz = cff * (v(i, j, k + 1, tind) - v(i, j, k, tind) + &
                    v(i, j + 1, k + 1, tind) - v(i, j + 1, k, tind))
      write(*,'(a,i6,i2,i4,8es26.17)') &
           'diag: ri u P1 iic,tind,k,zr,zr1,cff,uWlo,uWhi,uElo,uEhi,dudz=', &
           iic, tind, k, z_r(i, j, k), z_r(i, j, k + 1), cff, &
           u(i, j, k, tind), u(i, j, k + 1, tind), &
           u(i + 1, j, k, tind), u(i + 1, j, k + 1, tind), dudz
      write(*,'(a,i6,i2,i4,5es26.17)') &
           'diag: ri v P1 iic,tind,k,vSlo,vShi,vNlo,vNhi,dvdz=', &
           iic, tind, k, &
           v(i, j, k, tind), v(i, j, k + 1, tind), &
           v(i, j + 1, k, tind), v(i, j + 1, k + 1, tind), dvdz
    enddo
    flush(6)
  end subroutine dump_ri

  subroutine pad_phys_edges(a)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip) :: i, j, nx, ny
    logical :: edge_w, edge_e, edge_s, edge_n

    nx = decomp%nx
    ny = decomp%ny
    edge_w = decomp%edge_w
    edge_e = decomp%edge_e
    edge_s = decomp%edge_s
    edge_n = decomp%edge_n
    if (edge_w) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 0, ny + 1
        a(0, j) = a(1, j)
      enddo
    endif
    if (edge_e) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 0, ny + 1
        a(nx + 1, j) = a(nx, j)
      enddo
    endif
    if (edge_s) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = 0, nx + 1
        a(i, 0) = a(i, 1)
      enddo
    endif
    if (edge_n) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = 0, nx + 1
        a(i, ny + 1) = a(i, ny)
      enddo
    endif
    ! Corners on the device (one thread): no host stores to managed a.
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<1,1>>>
#endif
    do i = 1, 1
      if (edge_w .and. edge_s) a(0, 0) = a(1, 1)
      if (edge_w .and. edge_n) a(0, ny + 1) = a(1, ny)
      if (edge_e .and. edge_s) a(nx + 1, 0) = a(nx, 1)
      if (edge_e .and. edge_n) a(nx + 1, ny + 1) = a(nx, ny)
    enddo
  end subroutine pad_phys_edges

  subroutine smooth_iso_2d(fld)
    ! Isotropic 1/12 + 3/16 Laplacian (SMOOTH_RIG / SMOOTH_HBL).
    real(rp), WORK_ATTR intent(inout) :: fld(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip) :: i, j, nx, ny

    nx = decomp%nx
    ny = decomp%ny
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny + 1
      do i = 1, nx + 1
        wrk2a(i, j) = (fld(i, j) - fld(i - 1, j)) * umask(i, j)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 0, nx + 1
        wrk2c(i, j) = (fld(i, j) - fld(i, j - 1)) * vmask(i, j)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        wrk2b(i, j) = wrk2c(i, j) + lmd_smooth_c * (wrk2a(i + 1, j) + wrk2a(i, j - 1) - &
             wrk2a(i, j) - wrk2a(i + 1, j - 1))
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        wrk2a(i, j) = wrk2a(i, j) + lmd_smooth_c * (wrk2c(i, j + 1) + wrk2c(i - 1, j) - &
             wrk2c(i, j) - wrk2c(i - 1, j + 1))
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        fld(i, j) = apply_mask(fld(i, j) + lmd_smooth_c1 * (wrk2a(i + 1, j) - wrk2a(i, j) + &
             wrk2b(i, j + 1) - wrk2b(i, j)), rmask(i, j))
      enddo
    enddo
  end subroutine smooth_iso_2d

  subroutine pad_phys_edges_s(a, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    integer(ip), intent(in) :: slot
    integer(ip) :: i, j, nx, ny, n
    logical :: edge_w, edge_e, edge_s, edge_n
    nx = decomp%nx
    ny = decomp%ny
    n = slot
    edge_w = decomp%edge_w
    edge_e = decomp%edge_e
    edge_s = decomp%edge_s
    edge_n = decomp%edge_n
    if (edge_w) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 0, ny + 1
        a(0, j, n) = a(1, j, n)
      enddo
    endif
    if (edge_e) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do j = 0, ny + 1
        a(nx + 1, j, n) = a(nx, j, n)
      enddo
    endif
    if (edge_s) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = 0, nx + 1
        a(i, 0, n) = a(i, 1, n)
      enddo
    endif
    if (edge_n) then
#ifdef USE_CUDA
      !$cuf kernel do(1) <<<*,*>>>
#endif
      do i = 0, nx + 1
        a(i, ny + 1, n) = a(i, ny, n)
      enddo
    endif
    ! Corners on the device too (one thread): no host stores to managed a.
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<1,1>>>
#endif
    do i = 1, 1
      if (edge_w .and. edge_s) a(0, 0, n) = a(1, 1, n)
      if (edge_w .and. edge_n) a(0, ny + 1, n) = a(1, ny, n)
      if (edge_e .and. edge_s) a(nx + 1, 0, n) = a(nx, 1, n)
      if (edge_e .and. edge_n) a(nx + 1, ny + 1, n) = a(nx, ny, n)
    enddo
  end subroutine pad_phys_edges_s

  subroutine smooth_iso_2d_s(fld, slot)
    real(rp), WORK_ATTR intent(inout) :: fld(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    integer(ip), intent(in) :: slot
    integer(ip) :: i, j, nx, ny, n
    nx = decomp%nx
    ny = decomp%ny
    n = slot
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 0, ny + 1
      do i = 1, nx + 1
        wrk2a(i, j) = (fld(i, j, n) - fld(i - 1, j, n)) * umask(i, j)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 0, nx + 1
        wrk2c(i, j) = (fld(i, j, n) - fld(i, j - 1, n)) * vmask(i, j)
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny + 1
      do i = 1, nx
        wrk2b(i, j) = wrk2c(i, j) + lmd_smooth_c * (wrk2a(i + 1, j) + wrk2a(i, j - 1) - &
             wrk2a(i, j) - wrk2a(i + 1, j - 1))
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx + 1
        wrk2a(i, j) = wrk2a(i, j) + lmd_smooth_c * (wrk2c(i, j + 1) + wrk2c(i - 1, j) - &
             wrk2c(i, j) - wrk2c(i - 1, j + 1))
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        ! Match CPU SMOOTH_RIG: no final rmask. Land col_dc stays finite and
        ! enters wet-column cross terms. (SMOOTH_HBL uses smooth_iso_2d,
        ! which still applies rmask after the smooth.)
        fld(i, j, n) = fld(i, j, n) + lmd_smooth_c1 * (wrk2a(i + 1, j) - wrk2a(i, j) + &
             wrk2b(i, j + 1) - wrk2b(i, j))
      enddo
    enddo
  end subroutine smooth_iso_2d_s


  subroutine zero_mix_coefs()
    ! col_wc, wrk3d, wrk3e, col_dc = 0 as one device kernel (a host assignment would
    ! migrate four 3-D managed arrays to the CPU every call).
    integer(ip) :: i, j, k
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 0, decomp%nz
      do j = decomp%js, decomp%je
        do i = decomp%is, decomp%ie
          col_wc(i, j, k) = 0.0_rp
          wrk3d(i, j, k) = 0.0_rp
          wrk3e(i, j, k) = 0.0_rp
          col_dc(i, j, k) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine zero_mix_coefs

  subroutine lmd_rig_3d(tid)
    ! Gradient Richardson number at all interior w-levels (one launch).
    integer(ip), intent(in) :: tid
    integer(ip) :: i, j, k, nx, ny, nz, td
    real(rp) :: cff, dudz, dvdz
    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    td = tid
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz - 1
      do j = 1, ny
        do i = 1, nx
          cff = 0.5_rp / max(z_r(i, j, k + 1) - z_r(i, j, k), 1.0e-30_rp)
          dudz = cff * (u(i, j, k + 1, td) - u(i, j, k, td) + &
                        u(i + 1, j, k + 1, td) - u(i + 1, j, k, td))
          dvdz = cff * (v(i, j, k + 1, td) - v(i, j, k, td) + &
                        v(i, j + 1, k + 1, td) - v(i, j + 1, k, td))
          col_dc(i, j, k) = lmd_gradient_ri(bvf(i, j, k), dudz, dvdz)
        enddo
      enddo
    enddo
  end subroutine lmd_rig_3d

  subroutine pad_phys_edges_3s(a, k0, k1)
    ! pad_phys_edges_s for levels k0:k1, same kernel order per level.
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    integer(ip), intent(in) :: k0, k1
    integer(ip) :: i, j, nx, ny, n
    logical :: edge_w, edge_e, edge_s, edge_n
    nx = decomp%nx
    ny = decomp%ny
    edge_w = decomp%edge_w
    edge_e = decomp%edge_e
    edge_s = decomp%edge_s
    edge_n = decomp%edge_n
    if (edge_w) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do n = k0, k1
        do j = 0, ny + 1
          a(0, j, n) = a(1, j, n)
        enddo
      enddo
    endif
    if (edge_e) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do n = k0, k1
        do j = 0, ny + 1
          a(nx + 1, j, n) = a(nx, j, n)
        enddo
      enddo
    endif
    if (edge_s) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do n = k0, k1
        do i = 0, nx + 1
          a(i, 0, n) = a(i, 1, n)
        enddo
      enddo
    endif
    if (edge_n) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do n = k0, k1
        do i = 0, nx + 1
          a(i, ny + 1, n) = a(i, ny, n)
        enddo
      enddo
    endif
    ! Corners on the device, one thread per level.
#ifdef USE_CUDA
    !$cuf kernel do(1) <<<*,*>>>
#endif
    do n = k0, k1
      if (edge_w .and. edge_s) a(0, 0, n) = a(1, 1, n)
      if (edge_w .and. edge_n) a(0, ny + 1, n) = a(1, ny, n)
      if (edge_e .and. edge_s) a(nx + 1, 0, n) = a(nx, 1, n)
      if (edge_e .and. edge_n) a(nx + 1, ny + 1, n) = a(nx, ny, n)
    enddo
  end subroutine pad_phys_edges_3s

  subroutine smooth_iso_3d_s(fld, k0, k1)
    ! smooth_iso_2d_s for levels k0:k1 on 3-D work arrays.
    real(rp), WORK_ATTR intent(inout) :: fld(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    integer(ip), intent(in) :: k0, k1
    integer(ip) :: i, j, nx, ny, n
    nx = decomp%nx
    ny = decomp%ny
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do n = k0, k1
      do j = 0, ny + 1
        do i = 1, nx + 1
          wrk3a(i, j, n) = (fld(i, j, n) - fld(i - 1, j, n)) * umask(i, j)
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do n = k0, k1
      do j = 1, ny + 1
        do i = 0, nx + 1
          wrk3c(i, j, n) = (fld(i, j, n) - fld(i, j - 1, n)) * vmask(i, j)
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do n = k0, k1
      do j = 1, ny + 1
        do i = 1, nx
          wrk3b(i, j, n) = wrk3c(i, j, n) + lmd_smooth_c * (wrk3a(i + 1, j, n) + wrk3a(i, j - 1, n) - &
               wrk3a(i, j, n) - wrk3a(i + 1, j - 1, n))
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do n = k0, k1
      do j = 1, ny
        do i = 1, nx + 1
          wrk3a(i, j, n) = wrk3a(i, j, n) + lmd_smooth_c * (wrk3c(i, j + 1, n) + wrk3c(i - 1, j, n) - &
               wrk3c(i, j, n) - wrk3c(i - 1, j + 1, n))
        enddo
      enddo
    enddo
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do n = k0, k1
      do j = 1, ny
        do i = 1, nx
          ! Match CPU SMOOTH_RIG: no final rmask. Land col_dc stays finite and
          ! enters wet-column cross terms. (SMOOTH_HBL uses smooth_iso_2d,
          ! which still applies rmask after the smooth.)
          fld(i, j, n) = fld(i, j, n) + lmd_smooth_c1 * (wrk3a(i + 1, j, n) - wrk3a(i, j, n) + &
               wrk3b(i, j + 1, n) - wrk3b(i, j, n))
        enddo
      enddo
    enddo
  end subroutine smooth_iso_3d_s

end module lmd_mod
