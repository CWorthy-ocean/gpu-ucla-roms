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
  ! swr_frac depends only on Hz, rmask and the dry map, none of which change
  ! between the predictor and corrector lmd_vmix calls of a step: compute it
  ! once per step (iic of the last computation).
  logical, save :: swr_done = .false.
  integer(ip), save :: swr_iic = 0

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
    integer(ip) :: i, j, k, nx, ny, nz, lim
    real(rp) :: attn1, attn2, swdk1, swdk2

    if (swr_done .and. swr_iic == iic) return
    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    attn1 = -1.0_rp / jwt1_mu1
    attn2 = -1.0_rp / jwt1_mu2
    lim = dry_lim(G_LMD)

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
        swdk1 = jwt1_r1
        swdk2 = 1.0_rp - jwt1_r1
        swr_frac(i, j, nz) = 1.0_rp
        do k = nz, 1, -1
          ! Once a band is exactly zero, swr_band_atten returns +0 for every
          ! deeper level (0 * exp(xi) with xi <= 0, or its xi <= -20 branch):
          ! skip the exp.
          if (swdk1 /= 0.0_rp) swdk1 = swr_band_atten(swdk1, attn1, Hz(i, j, k))
          if (swdk2 /= 0.0_rp) swdk2 = swr_band_atten(swdk2, attn2, Hz(i, j, k))
          swr_frac(i, j, k - 1) = apply_mask(swdk1 + swdk2, rmask(i, j))
        enddo
      enddo
    enddo
    ! No halo fill: every reader of swr_frac (lmd_kpp, step3d_t, the step
    ! probes) reads interior columns only, so the halo values were dead.
    swr_done = .true.
    swr_iic = iic
  end subroutine lmd_swr_frac

  subroutine lmd_vmix(tind)
    integer(ip), intent(in) :: tind
    logical :: old

    ! Only the surface forcing needs its halo here (2026-10-08 audit): u, v
    ! (tind and nstp), T(nstp), the geometry, bvf, f, hbls and hbbl are
    ! current (exchanged after their last write, or static).
    call hg_begin()
    call hg_add(F_SUSTR)
    call hg_add(F_SVSTR)
    call hg_add(F_STFLX, itr=itemp)
    if (size(stflx, 3) >= isalt) call hg_add(F_STFLX, itr=isalt)
    call hg_add(F_SRFLX)
    call hg_end()

    call ensure_scratch()
    call lmd_swr_frac()
    ! Probe steps keep the former path (zeroing, 3-D smoothing, interior
    ! kernel, per-level KPP kernel): dump_ri reads the smoothed col_dc and
    ! col_wc/wrk3d. Otherwise the smoothed Rig, the interior coefficients
    ! and the KPP profiles are computed in one column kernel (lmd_kpp).
    ! The gradient Richardson number comes from the KPP column search.
    old = probe_on .or. (iic - ntstart >= kpp_probe_from .and. iic - ntstart <= kpp_probe_to)
    if (old) then
      call zero_mix_coefs()
    else
      call lmd_zero_deep_frame()
    endif
    call lmd_kpp(tind, old)
  end subroutine lmd_vmix

  subroutine lmd_interior(tind)
    integer(ip), intent(in) :: tind
    integer(ip) :: i, j, k, nx, ny, nz, tid, lim
    real(rp) :: cff, dudz, dvdz, nu, mult
    real(rp) :: akv_loc, akt_t, akt_s

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    tid = tind
    akv_loc = akv_bak
    akt_t = akt_bak(itemp)
    akt_s = akt_bak(isalt)

    ! Probe-step path. SMOOTH_RIG: isotropic smooth between levels of the
    ! exchanged col_dc (lmd_kpp); each stage runs for all levels at once.
    call smooth_iso_3d_s(col_dc, 1, nz - 1)
    lim = dry_lim(G_LMD)

#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
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

  subroutine lmd_kpp(tind, old)
    integer(ip), intent(in) :: tind
    logical, intent(in) :: old
    integer(ip) :: i, j, k, n, nx, ny, nz, tid, n_stp, first, kbls, kbbl, lim, keep, hw, reuse
    logical :: bulk
    real(rp) :: Cg, Vtc, alpha, beta, Bfsfc, zscale, ws, wm, Kern, Vtsq
    real(rp) :: ssgm, z_bl, cff, cff1, cff_up, cff_dn, amp, wmb, wsb, sgmb, swdk
    real(rp) :: zwn, zw0, zwk, hb, bb, bo_ij, bos_ij, us_ij, rm_ij, eh, eb
    real(rp) :: fck, fc0, cf_up, cf_hi, cf_lo, hbn, bbn
    real(rp) :: akv_loc, akt_t, akt_s, nu, mult, rig, gh
    real(rp) :: a1_00, a1_10, a1_0m, a1_1m, a1_0p, a1_1p
    real(rp) :: c_00, c_0p, c_m0, c_mp, c_10, c_1p, b_00, b_0p, a_00, a_10
    real(rp) :: cv_n, ct_n, cv_c, ct_c, pv, pt, ps, nv, nt, ns, kv, kt, ks
    real(rp) :: uw0, ue0, vs0, vn0, hz0, uw1, ue1, vs1, vn1, hz1, sw0, sw1, bv0, bv1
    real(rp) :: zc, zc_prev, bf_prev, wm_prev, ws_prev, zr0, zr1, cr, dudz, dvdz

    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    tid = tind
    n_stp = nstp
    first = merge(1_ip, 0_ip, is_first_time_step())
    Cg = lmd_Cg()
    Vtc = lmd_Vtc()
    bulk = have_bulk
    lim = dry_lim(G_LMD)
    akv_loc = akv_bak
    akt_t = akt_bak(itemp)
    akt_s = akt_bak(isalt)

    ! --- per-column surface forcing, friction velocity, previous BL depths
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
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

    ! --- column: bulk Richardson integral (recurrence) and the BL searches.
    !     The unresolved-shear term Vtsq (former wrk3f kernel) is evaluated
    !     in the downward sweep, only down to the first hit of the surface
    !     search; the bottom search stops at its first hit. col_fc is kept
    !     (bottom search, dump_kpp); col_cf lives in registers. The sweep
    !     also stores the gradient Richardson number of each interior
    !     w-level in col_dc (former lmd_rig_3d kernel, same expression; zero
    !     on skipped columns), from the u, v, bvf it loads anyway.
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,(32,2,1)>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then
          if (colact(i, j) == 0) then
            do k = 1, nz - 1
              col_dc(i, j, k) = 0.0_rp
            enddo
            cycle
          end if
        end if
        zwn = z_w(i, j, nz)
        zw0 = z_w(i, j, 0)
        hb = hbl(i, j)
        bb = bbl(i, j)
        bo_ij = Bo(i, j)
        bos_ij = Bosol(i, j)
        us_ij = ustar(i, j)
        rm_ij = rmask(i, j)
        eh = (lmd_epssfc * hb)**2
        eb = (lmd_epssfc * bb)**2
        kbls = 0
        fck = 0.0_rp
        cf_up = 0.0_rp
        cf_hi = 0.0_rp
        cf_lo = 0.0_rp
        hw = 0
        zc_prev = 0.0_rp
        bf_prev = 0.0_rp
        ws_prev = 0.0_rp
        ! level k+1 values carried from the previous iteration
        uw1 = u(i, j, nz, tid); ue1 = u(i + 1, j, nz, tid)
        vs1 = v(i, j, nz, tid); vn1 = v(i, j + 1, nz, tid)
        hz1 = Hz(i, j, nz)
        sw1 = swr_frac(i, j, nz)
        bv1 = bvf(i, j, nz)
        zr1 = 0.0_rp
        do k = nz, 1, -1
          bv0 = bvf(i, j, k - 1)
          zr0 = z_r(i, j, k)
          if (k < nz) then
            uw0 = u(i, j, k, tid); ue0 = u(i + 1, j, k, tid)
            vs0 = v(i, j, k, tid); vn0 = v(i, j + 1, k, tid)
            hz0 = Hz(i, j, k)
            cr = 0.5_rp / max(zr1 - zr0, 1.0e-30_rp)
            dudz = cr * (uw1 - uw0 + &
                         ue1 - ue0)
            dvdz = cr * (vs1 - vs0 + &
                         vn1 - vn0)
            col_dc(i, j, k) = lmd_gradient_ri((bv1), (dudz), (dvdz))
            zwk = z_w(i, j, k)
            cff_up = (zwn - zwk)**2
            cff_dn = (zwk - zw0)**2
            Kern = cff_up * cff_dn / max( &
                 (cff_up + eh) * &
                 (cff_dn + eb), 1.0e-30_rp)
            fck = fck + Kern * ( &
                 0.5_rp * ((uw1 + ue1 - &
                            uw0 - ue0)**2 + &
                           (vs1 + vn1 - &
                            vs0 - vn0)**2) / &
                 max(hz0 + hz1, 1.0e-30_rp) - &
                 0.5_rp * (hz0 + hz1) * ( &
                 lmd_Ri_inv * bv1 + lmd_C_Ek * f(i, j) * f(i, j)))
            uw1 = uw0; ue1 = ue0; vs1 = vs0; vn1 = vn0; hz1 = hz0
          endif
          col_fc(i, j, k) = fck
          if (kbls == 0) then
            sw0 = swr_frac(i, j, k - 1)
            swdk = sqrt(max(0.0_rp, sw1 * sw0))
            zscale = zwn - zr0
            Bfsfc = bo_ij + bos_ij * (1.0_rp - swdk)
            ! lmd_ws uses zscale only through min(zscale, hbl*epssfc): reuse
            ! the last value when the clamped depth and Bfsfc repeat.
            zc = min(zscale, hb * lmd_epssfc)
            reuse = 0
            if (hw /= 0) then
              if (zc == zc_prev .and. Bfsfc == bf_prev) reuse = 1
            endif
            if (reuse /= 0) then
              ws = ws_prev
            else
              ws = lmd_ws((zscale), (Bfsfc), (us_ij), (hb), (rm_ij))
              hw = 1
              zc_prev = zc
              bf_prev = Bfsfc
              ws_prev = ws
            endif
            Vtsq = lmd_Vtsq_fix * Vtc * ws * sqrt(max(1.0e-5_rp, bv0))
            cff = fck + Vtsq
            if (cff < 0.0_rp) then
              kbls = k
              cf_hi = cf_up
              cf_lo = cff
            endif
            cf_up = cff
            sw1 = sw0
          endif
          bv1 = bv0
          zr1 = zr0
        enddo
        z_bl = zw0 + 0.25_rp * Hz(i, j, 1)
        cff_up = (zwn - z_bl)**2
        cff_dn = (z_bl - zw0)**2
        Kern = cff_up * cff_dn / max( &
             (cff_up + eh) * &
             (cff_dn + eb), 1.0e-30_rp)
        fc0 = fck + Kern * ( &
             0.5_rp * ((u(i, j, 1, tid) + u(i + 1, j, 1, tid))**2 + &
                       (v(i, j, 1, tid) + v(i, j + 1, 1, tid))**2) / &
             max(Hz(i, j, 1), 1.0e-30_rp) - &
             0.5_rp * Hz(i, j, 1) * (lmd_Ri_inv * bvf(i, j, 1) + &
             lmd_C_Ek * f(i, j) * f(i, j)))
        col_fc(i, j, 0) = fc0

        if (kbls > 0) then
          k = kbls
          if (k == nz) then
            hbn = zwn - z_r(i, j, nz)
          else
            hbn = zwn - (z_r(i, j, k) * cf_hi - &
                 z_r(i, j, k + 1) * cf_lo) / max(cf_hi - cf_lo, 1.0e-30_rp)
          endif
        else
          hbn = zwn - zw0
        endif
        hbl(i, j) = apply_mask(hbn, rm_ij)

        kbbl = 0
        cf_lo = 0.0_rp
        cf_hi = 0.0_rp
        do k = 1, nz
          cf_hi = col_fc(i, j, k) - fc0
          if (cf_hi > 0.0_rp) then
            kbbl = k
            exit
          endif
          cf_lo = cf_hi
        enddo
        if (kbbl > 0) then
          k = kbbl
          if (k == 1) then
            bbn = z_r(i, j, 1) - zw0
          else
            bbn = (z_r(i, j, k - 1) * cf_hi - z_r(i, j, k) * cf_lo) / &
                 max(cf_hi - cf_lo, 1.0e-30_rp) - zw0
          endif
        else
          bbn = zwn - zw0
        endif
        bbl(i, j) = apply_mask(bbn, rm_ij)
      enddo
    enddo

    call dump_kpp(1, tid)

    ! SMOOTH_RIG input: pad and exchange the Richardson number (one 3-D
    ! exchange of levels 0:nz instead of nz-1 per-level exchanges; levels 0
    ! and nz are not read). Probe steps then run the former smoothing and
    ! interior kernels (lmd_interior, dump_ri).
    call pad_phys_edges_3s(col_dc, 1, nz - 1)
    call halo_fill_3d_w(col_dc, STAGGER_RHO)
    if (old) call lmd_interior(tid)

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
        if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
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

    if (old) then
      ! --- per level: surface BL profile, bottom BL profile, Akv/Akt
      !     (same per-level order as the former column loops: surface, then
      !     bottom, then the masked copy).
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do k = 0, nz
        do j = 1, ny
          do i = 1, nx
            if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
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
    else
    ! --- one thread per column, levels 0..nz in order: SMOOTH_RIG of the
    !     exchanged col_dc (the five smooth_iso_3d_s stages for this point,
    !     same operands and order), the LMD94 interior coefficients with
    !     their sequential vertical smoothing (former lmd_interior; the
    !     unsmoothed values run one level ahead), then the surface and
    !     bottom BL profiles and Akv/Akt/ghat of the former per-level
    !     kernel. wm/ws are only formed inside the BL (ssgm < 1), where they
    !     are used. Kv/Kt/Ks stay in registers; they are stored (col_wc,
    !     wrk3d, wrk3e) only on deep-land columns, where the former kernels
    !     left them for later readers (zero when those columns are skipped).
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (lim > 0) then
          if (colact(i, j) == 0) then
            do k = 0, nz
              col_wc(i, j, k) = 0.0_rp
              wrk3d(i, j, k) = 0.0_rp
              wrk3e(i, j, k) = 0.0_rp
            enddo
            cycle
          end if
        end if
        keep = 1 - colact(i, j)
        zwn = z_w(i, j, nz)
        zw0 = z_w(i, j, 0)
        hb = hbl(i, j)
        bb = bbl(i, j)
        Bfsfc = wrk2a(i, j)
        wmb = wrk2b(i, j)
        us_ij = ustar(i, j)
        rm_ij = rmask(i, j)
        cv_n = 0.0_rp; ct_n = 0.0_rp
        cv_c = 0.0_rp; ct_c = 0.0_rp
        hw = 0
        zc_prev = 0.0_rp
        wm_prev = 0.0_rp; ws_prev = 0.0_rp
        pv = 0.0_rp; pt = 0.0_rp; ps = 0.0_rp
        do k = 0, nz
          ! unsmoothed interior coefficients at level k + 1 (1..nz-1)
          if (k + 1 <= nz - 1) then
            n = k + 1
            a1_00 = (col_dc(i, j, n) - col_dc(i - 1, j, n)) * umask(i, j)
            a1_10 = (col_dc(i + 1, j, n) - col_dc(i, j, n)) * umask(i + 1, j)
            a1_0m = (col_dc(i, j - 1, n) - col_dc(i - 1, j - 1, n)) * umask(i, j - 1)
            a1_1m = (col_dc(i + 1, j - 1, n) - col_dc(i, j - 1, n)) * umask(i + 1, j - 1)
            a1_0p = (col_dc(i, j + 1, n) - col_dc(i - 1, j + 1, n)) * umask(i, j + 1)
            a1_1p = (col_dc(i + 1, j + 1, n) - col_dc(i, j + 1, n)) * umask(i + 1, j + 1)
            c_00 = (col_dc(i, j, n) - col_dc(i, j - 1, n)) * vmask(i, j)
            c_0p = (col_dc(i, j + 1, n) - col_dc(i, j, n)) * vmask(i, j + 1)
            c_m0 = (col_dc(i - 1, j, n) - col_dc(i - 1, j - 1, n)) * vmask(i - 1, j)
            c_mp = (col_dc(i - 1, j + 1, n) - col_dc(i - 1, j, n)) * vmask(i - 1, j + 1)
            c_10 = (col_dc(i + 1, j, n) - col_dc(i + 1, j - 1, n)) * vmask(i + 1, j)
            c_1p = (col_dc(i + 1, j + 1, n) - col_dc(i + 1, j, n)) * vmask(i + 1, j + 1)
            b_00 = c_00 + lmd_smooth_c * (a1_10 + a1_0m - &
                 a1_00 - a1_1m)
            b_0p = c_0p + lmd_smooth_c * (a1_1p + a1_00 - &
                 a1_0p - a1_10)
            a_00 = a1_00 + lmd_smooth_c * (c_0p + c_m0 - &
                 c_00 - c_mp)
            a_10 = a1_10 + lmd_smooth_c * (c_1p + c_00 - &
                 c_10 - c_0p)
            rig = col_dc(i, j, n) + lmd_smooth_c1 * (a_10 - a_00 + &
                 b_0p - b_00)
            nu = lmd_nu_sx((rig))
            cv_n = lmd_nuwm + lmd_nu0m * nu
            ct_n = lmd_nuws + lmd_nu0s * nu
            mult = lmd_bottom_mult(z_w(i, j, n) - zw0)
            cv_n = cv_n * mult
            ct_n = ct_n * mult
          endif
          if (k == 0) then
            kv = cv_n + akv_loc
            kt = ct_n + akt_t
            ks = ct_n + akt_s
          else if (k < nz) then
            if (k < nz - 1) then
              nv = cv_n; nt = ct_n; ns = ct_n
            else
              nv = cv_c + akv_loc; nt = ct_c + akt_t; ns = ct_c + akt_s
            endif
            kv = 0.5_rp * cv_c + 0.25_rp * pv + &
                 0.25_rp * nv + akv_loc
            kt = 0.5_rp * ct_c + 0.25_rp * pt + &
                 0.25_rp * nt + akt_t
            ks = 0.5_rp * ct_c + 0.25_rp * ps + &
                 0.25_rp * ns + akt_s
          else
            kv = cv_c + akv_loc
            kt = ct_c + akt_t
            ks = ct_c + akt_s
          endif
          pv = kv; pt = kt; ps = ks
          cv_c = cv_n; ct_c = ct_n

          ! surface BL profile
          zwk = z_w(i, j, k)
          zscale = zwn - zwk
          ssgm = (zwn - zwk) / max(hb, lmd_eps)
          if (ssgm < 1.0_rp) then
            ! lmd_wm/lmd_ws use zscale only through min(zscale, hbl*epssfc);
            ! Bfsfc, ustar, hbl and rmask are column constants. Below
            ! epssfc*hbl the clamped depth repeats: reuse the last values.
            zc = min(zscale, hb * lmd_epssfc)
            reuse = 0
            if (hw /= 0) then
              if (zc == zc_prev) reuse = 1
            endif
            if (reuse /= 0) then
              wm = wm_prev
              ws = ws_prev
            else
              wm = lmd_wm((zscale), (Bfsfc), (us_ij), (hb), (rm_ij))
              ws = lmd_ws((zscale), (Bfsfc), (us_ij), (hb), (rm_ij))
              hw = 1
              zc_prev = zc
              wm_prev = wm
              ws_prev = ws
            endif
            if (ssgm < 0.07_rp) then
              cff = 0.5_rp * (ssgm - 0.07_rp)**2 / 0.07_rp
            else
              cff = 0.0_rp
            endif
            cff = cff + ssgm * (1.0_rp - ssgm)**2
            amp = ssgm**2
            kv = sqrt((amp * kv)**2 + (wm * hb * cff)**2)
            kt = sqrt((amp * kt)**2 + (ws * hb * cff)**2)
            ks = sqrt((amp * ks)**2 + (ws * hb * cff)**2)
            if (Bfsfc < 0.0_rp) then
              gh = -Cg * ssgm * (1.0_rp - ssgm)**2
            else
              gh = 0.0_rp
            endif
          else
            gh = 0.0_rp
          endif
          ghat(i, j, k) = gh

          ! bottom BL profile
          wsb = wmb
          sgmb = (zwk - zw0 + zob) / max(bb + zob, 1.0e-30_rp)
          if (sgmb < 1.0_rp) then
            cff1 = sgmb * (1.0_rp - sgmb)**2
            kv = sqrt(kv**2 + (wmb * bb * cff1)**2)
            kt = sqrt(kt**2 + (wsb * bb * cff1)**2)
            ks = sqrt(ks**2 + (wsb * bb * cff1)**2)
          endif

          if (keep /= 0) then
            col_wc(i, j, k) = kv
            wrk3d(i, j, k) = kt
            wrk3e(i, j, k) = ks
          endif
          if (rm_ij > 0.5_rp) then
            Akv(i, j, k) = kv
            Akt(i, j, k, itemp) = kt
            Akt(i, j, k, isalt) = ks
          else
            Akv(i, j, k) = 0.0_rp
            Akt(i, j, k, itemp) = 0.0_rp
            Akt(i, j, k, isalt) = 0.0_rp
          endif
        enddo
      enddo
    enddo
    endif
    call dump_kpp(3, tid)

    call pad_phys_edges(hbls)
    call pad_phys_edges(hbbl)
    call hg_begin()
    call hg_add(F_HBLS)
    call hg_add(F_HBBL)
    call hg_add(F_AKV)
    call hg_add(F_AKT, itr=itemp)
    call hg_add(F_AKT, itr=isalt)
    call hg_add(F_GHAT)
    call hg_end()
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

  subroutine lmd_zero_deep_frame()
    ! The former full zeroing of col_wc, wrk3d, wrk3e and col_dc
    ! (zero_mix_coefs, kept for probe steps) is dead apart from the deep-
    ! land (colact = 0) columns: visc3d skips those and reads wrk3d there
    ! as lmd_vmix left it. Interior deep columns are written by the KPP
    ! column kernel; this launch keeps the zero on deep frame (halo)
    ! columns, which exist only for a dry radius below halo + 1. Every other
    ! element is written before it is read:
    ! - col_dc, levels 1..nz-1: interior by the KPP column kernel (zero on
    !   skipped columns), the one-cell ring by pad_phys_edges_3s and the exchange
    !   (the east-west round includes rows 0 / ny+1 at physical south /
    !   north edges, the north-south round columns -1..nx+2), from fresh
    !   values; the smoothing reads only the interior and that ring. Levels
    !   0 and nz and the rest of the frame are not read; after lmd_vmix the
    !   column solvers (step3d_uv, uv_rhs) write col_dc before reading it.
    ! - col_wc, wrk3d, wrk3e: not read in lmd_vmix; afterwards visc3d
    !   writes wrk3d wherever it reads it except on skipped (deep) columns,
    !   and the other users write col_wc first; wrk3e has no other user.
    integer(ip) :: i, j, k, nx, ny, nz, is, ie, js, je
    nx = decomp%nx
    ny = decomp%ny
    nz = decomp%nz
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        if (i >= 1 .and. i <= nx .and. j >= 1 .and. j <= ny) cycle
        if (colact(i, j) /= 0) cycle
        do k = 0, nz
          col_wc(i, j, k) = 0.0_rp
          wrk3d(i, j, k) = 0.0_rp
          wrk3e(i, j, k) = 0.0_rp
        enddo
      enddo
    enddo
  end subroutine lmd_zero_deep_frame


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
    integer(ip) :: i, j, nx, ny, n, lim
    nx = decomp%nx
    ny = decomp%ny
    lim = dry_lim(G_LMD)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do n = k0, k1
      do j = 0, ny + 1
        do i = 1, nx + 1
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
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
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
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
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
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
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
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
          if (lim > 0) then; if (colact(i, j) == 0) cycle; end if
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
