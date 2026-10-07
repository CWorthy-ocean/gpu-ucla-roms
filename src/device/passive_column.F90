#include "mem_attr.inc"
module passive_column_mod
  ! Passive-tracer transport kernels for the tracer-innermost layout
  ! tp(ntr, is:ie, js:je, nz, 3): one thread per (tracer, cell) for the
  ! horizontal steps and the lateral diffusion, one thread per (tracer,
  ! column) for the vertical advection + implicit solve. Consecutive
  ! threads hold consecutive tracers of a cell, so every tracer load of a
  ! warp is one contiguous line. The arithmetic per element is that of the
  ! former per-cell kernels (pc4/pup3 faces, vert_column_fused, the Thomas
  ! solve), so the FP64 build is bit-identical to them.
  ! Separate from passive_mod so the explicit kernel launches are never
  ! inlined into their callers (nvfortran 25.9 segfaulted when they were).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use gpu_state, only: nwet, wet_i, wet_j
  use formulas
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  private
  public :: passive_horiz_run, passive_t3dmix_run, passive_column_run

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

  ! --------------------------------------- horizontal face divergences
  DEVICE_ATTR
  subroutine c4_div(itr, i, j, k, ts, ntr, is, ie, js, je, nz, nslot, t0, wedge, eedge, sedge, nedge, &
                    um0, um1, um2, um3, vm0, vm1, vm2, vm3, fu1, fu2, fv1, fv2, tp_, hdiv)
    ! fxe - fxw + fen - fes with the C4 faces of pc4_fx/pc4_fe (same expressions).
    integer(ip), intent(in) :: itr, i, j, k, ts, ntr, is, ie, js, je, nz, nslot
    real(rp), intent(in) :: t0, um0, um1, um2, um3, vm0, vm1, vm2, vm3, fu1, fu2, fv1, fv2
    logical, intent(in) :: wedge, eedge, sedge, nedge
    real(pp), intent(in) :: tp_(ntr, is:ie, js:je, nz, nslot)
    real(rp), intent(out) :: hdiv
    real(rp) :: tm2, tm1, tp1, tp2, dcw, dlw, drw, dre, fxw, fxe, fes, fen
    tm1 = real(tp_(itr, i - 1, j, k, ts), rp)
    tp1 = real(tp_(itr, i + 1, j, k, ts), rp)
    dcw = apply_mask(t0 - tm1, um1)
    if (wedge) then
      dlw = dcw
    else
      tm2 = real(tp_(itr, i - 2, j, k, ts), rp)
      dlw = apply_mask(tm1 - tm2, um0)
    endif
    drw = apply_mask(tp1 - t0, um2)
    fxw = apply_mask(0.5_rp * (t0 + tm1 - (drw - dlw) / 6.0_rp) * fu1, um1)
    if (eedge) then
      dre = drw
    else
      tp2 = real(tp_(itr, i + 2, j, k, ts), rp)
      dre = apply_mask(tp2 - tp1, um3)
    endif
    fxe = apply_mask(0.5_rp * (tp1 + t0 - (dre - dcw) / 6.0_rp) * fu2, um2)
    tm1 = real(tp_(itr, i, j - 1, k, ts), rp)
    tp1 = real(tp_(itr, i, j + 1, k, ts), rp)
    dcw = apply_mask(t0 - tm1, vm1)
    if (sedge) then
      dlw = dcw
    else
      tm2 = real(tp_(itr, i, j - 2, k, ts), rp)
      dlw = apply_mask(tm1 - tm2, vm0)
    endif
    drw = apply_mask(tp1 - t0, vm2)
    fes = apply_mask(0.5_rp * (t0 + tm1 - (drw - dlw) / 6.0_rp) * fv1, vm1)
    if (nedge) then
      dre = drw
    else
      tp2 = real(tp_(itr, i, j + 2, k, ts), rp)
      dre = apply_mask(tp2 - tp1, vm3)
    endif
    fen = apply_mask(0.5_rp * (tp1 + t0 - (dre - dcw) / 6.0_rp) * fv2, vm2)
    hdiv = fxe - fxw + fen - fes
  end subroutine c4_div

  DEVICE_ATTR
  subroutine up3_div(itr, i, j, k, ts, ntr, is, ie, js, je, nz, nslot, t0, wedge, eedge, sedge, nedge, &
                     um0, um1, um2, um3, vm0, vm1, vm2, vm3, rmw, rmc, rme, rms, rmn, fu1, fu2, fv1, fv2, tp_, hdiv)
    ! fxe - fxw + fen - fes with the UP3 faces of pup3_fx/pup3_fe (same expressions).
    integer(ip), intent(in) :: itr, i, j, k, ts, ntr, is, ie, js, je, nz, nslot
    real(rp), intent(in) :: t0, um0, um1, um2, um3, vm0, vm1, vm2, vm3, rmw, rmc, rme, rms, rmn, fu1, fu2, fv1, fv2
    logical, intent(in) :: wedge, eedge, sedge, nedge
    real(pp), intent(in) :: tp_(ntr, is:ie, js:je, nz, nslot)
    real(rp), intent(out) :: hdiv
    real(rp) :: tm2, tm1, tp1, tp2, dcw, dlw, drw, dre, fxw, fxe, fes, fen, face_t
    tm1 = real(tp_(itr, i - 1, j, k, ts), rp)
    tp1 = real(tp_(itr, i + 1, j, k, ts), rp)
    dcw = apply_mask(t0 - tm1, um1)
    if (wedge) then
      dlw = dcw
    else
      tm2 = real(tp_(itr, i - 2, j, k, ts), rp)
      dlw = apply_mask(tm1 - tm2, um0)
    endif
    drw = apply_mask(tp1 - t0, um2)
    if (rmw > 0.5_rp .and. rmc > 0.5_rp) then
      face_t = 0.5_rp * (tm1 + t0)
    elseif (rmw > 0.5_rp) then
      face_t = tm1
    elseif (rmc > 0.5_rp) then
      face_t = t0
    else
      face_t = 0.0_rp
    endif
    fxw = apply_mask(face_t * fu1 - coef_up3 * ( &
          (dcw - dlw) * um0 * um1 * max(fu1, 0.0_rp) + (drw - dcw) * um1 * um2 * min(fu1, 0.0_rp)), um1)
    if (eedge) then
      dre = drw
    else
      tp2 = real(tp_(itr, i + 2, j, k, ts), rp)
      dre = apply_mask(tp2 - tp1, um3)
    endif
    if (rmc > 0.5_rp .and. rme > 0.5_rp) then
      face_t = 0.5_rp * (t0 + tp1)
    elseif (rmc > 0.5_rp) then
      face_t = t0
    elseif (rme > 0.5_rp) then
      face_t = tp1
    else
      face_t = 0.0_rp
    endif
    fxe = apply_mask(face_t * fu2 - coef_up3 * ( &
          (drw - dcw) * um1 * um2 * max(fu2, 0.0_rp) + (dre - drw) * um2 * um3 * min(fu2, 0.0_rp)), um2)
    tm1 = real(tp_(itr, i, j - 1, k, ts), rp)
    tp1 = real(tp_(itr, i, j + 1, k, ts), rp)
    dcw = apply_mask(t0 - tm1, vm1)
    if (sedge) then
      dlw = dcw
    else
      tm2 = real(tp_(itr, i, j - 2, k, ts), rp)
      dlw = apply_mask(tm1 - tm2, vm0)
    endif
    drw = apply_mask(tp1 - t0, vm2)
    if (rms > 0.5_rp .and. rmc > 0.5_rp) then
      face_t = 0.5_rp * (tm1 + t0)
    elseif (rms > 0.5_rp) then
      face_t = tm1
    elseif (rmc > 0.5_rp) then
      face_t = t0
    else
      face_t = 0.0_rp
    endif
    fes = apply_mask(face_t * fv1 - coef_up3 * ( &
          (dcw - dlw) * vm0 * vm1 * max(fv1, 0.0_rp) + (drw - dcw) * vm1 * vm2 * min(fv1, 0.0_rp)), vm1)
    if (nedge) then
      dre = drw
    else
      tp2 = real(tp_(itr, i, j + 2, k, ts), rp)
      dre = apply_mask(tp2 - tp1, vm3)
    endif
    if (rmc > 0.5_rp .and. rmn > 0.5_rp) then
      face_t = 0.5_rp * (t0 + tp1)
    elseif (rmc > 0.5_rp) then
      face_t = t0
    elseif (rmn > 0.5_rp) then
      face_t = tp1
    else
      face_t = 0.0_rp
    endif
    fen = apply_mask(face_t * fv2 - coef_up3 * ( &
          (drw - dcw) * vm1 * vm2 * max(fv2, 0.0_rp) + (dre - drw) * vm2 * vm3 * min(fv2, 0.0_rp)), vm2)
    hdiv = fxe - fxw + fen - fes
  end subroutine up3_div

  ! ----------------------------------- horizontal step, one thread per (tracer, cell)
  DEVICE_ATTR
  subroutine horiz_point(itr, i, j, k, mode, ts, ns, nn, nb, cs, cb, dtp, ntr, nx, ny, is, ie, js, je, nz, nslot, &
                         west, east, south, north, tp_, Hz_, Hzb_, pm_, pn_, FlxU_, FlxV_, umask_, vmask_, rmask_)
    ! mode 1: LF-AM3 predictor with C4 faces from slot ts, old field in slot
    ! ns, saved Hz-weighted into slot nb, result into slot nn.
    ! mode 2: corrector, UP3 faces from slot ts applied to slot nn.
    integer(ip), intent(in) :: itr, i, j, k, mode, ts, ns, nn, nb, ntr, nx, ny, is, ie, js, je, nz, nslot
    integer(ip), intent(in) :: west, east, south, north
    real(rp), intent(in) :: cs, cb, dtp
    real(pp), intent(inout) :: tp_(ntr, is:ie, js:je, nz, nslot)
    real(rp), intent(in) :: Hz_(is:ie, js:je, nz), Hzb_(is:ie, js:je, nz), FlxU_(is:ie, js:je, nz), FlxV_(is:ie, js:je, nz)
    real(rp), intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp) :: dtpp, t0, hdiv, told, hb, hzc
    real(rp) :: um0, um1, um2, um3, vm0, vm1, vm2, vm3, fu1, fu2, fv1, fv2
    logical :: wedge, eedge, sedge, nedge
    dtpp = dtp * pm_(i, j) * pn_(i, j)
    um0 = umask_(i - 1, j); um1 = umask_(i, j); um2 = umask_(i + 1, j); um3 = umask_(i + 2, j)
    vm0 = vmask_(i, j - 1); vm1 = vmask_(i, j); vm2 = vmask_(i, j + 1); vm3 = vmask_(i, j + 2)
    fu1 = FlxU_(i, j, k); fu2 = FlxU_(i + 1, j, k)
    fv1 = FlxV_(i, j, k); fv2 = FlxV_(i, j + 1, k)
    wedge = (west == 1 .and. i == 1)
    eedge = (east == 1 .and. i == nx)
    sedge = (south == 1 .and. j == 1)
    nedge = (north == 1 .and. j == ny)
    t0 = real(tp_(itr, i, j, k, ts), rp)
    if (mode == 1) then
      call c4_div(itr, i, j, k, ts, ntr, is, ie, js, je, nz, nslot, t0, wedge, eedge, sedge, nedge, &
                  um0, um1, um2, um3, vm0, vm1, vm2, vm3, fu1, fu2, fv1, fv2, tp_, hdiv)
      hb = Hzb_(i, j, k); hzc = Hz_(i, j, k)
      told = real(tp_(itr, i, j, k, ns), rp)
      tp_(itr, i, j, k, nn) = real(hb * (cs * told + cb * real(tp_(itr, i, j, k, nb), rp)) - dtpp * hdiv, pp)
      tp_(itr, i, j, k, nb) = real(hzc * told, pp)
    else
      call up3_div(itr, i, j, k, ts, ntr, is, ie, js, je, nz, nslot, t0, wedge, eedge, sedge, nedge, &
                   um0, um1, um2, um3, vm0, vm1, vm2, vm3, rmask_(i - 1, j), rmask_(i, j), rmask_(i + 1, j), &
                   rmask_(i, j - 1), rmask_(i, j + 1), fu1, fu2, fv1, fv2, tp_, hdiv)
      tp_(itr, i, j, k, nn) = real(real(tp_(itr, i, j, k, nn), rp) - dtpp * hdiv, pp)
    endif
  end subroutine horiz_point

  ! ------------------------------ lateral diffusion, one thread per (tracer, cell)
  DEVICE_ATTR
  subroutine t3dmix_point(itr, i, j, k, nr, nn, dtl, isl, ntr, is, ie, js, je, nz, nslot, nd, &
                          tp_, Hz_, pm_, pn_, pmon_u_, pnom_v_, umask_, vmask_, diff2_)
    ! Harmonic diffusion along s-surfaces with the salinity diffusivity,
    ! faces computed inline (same expressions as the former passive_t3dmix).
    integer(ip), intent(in) :: itr, i, j, k, nr, nn, isl, ntr, is, ie, js, je, nz, nslot, nd
    real(rp), intent(in) :: dtl
    real(pp), intent(inout) :: tp_(ntr, is:ie, js:je, nz, nslot)
    real(rp), intent(in) :: Hz_(is:ie, js:je, nz), pm_(is:ie, js:je), pn_(is:ie, js:je)
    real(rp), intent(in) :: pmon_u_(is:ie, js:je), pnom_v_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je)
    real(rp), intent(in) :: diff2_(is:ie, js:je, nd)
    real(rp) :: fxw, fxe, fes, fen
    fxw = apply_mask(0.25_rp * (diff2_(i, j, isl) + diff2_(i - 1, j, isl)) * &
         pmon_u_(i, j) * (Hz_(i, j, k) + Hz_(i - 1, j, k)) * &
         (real(tp_(itr, i, j, k, nr), rp) - real(tp_(itr, i - 1, j, k, nr), rp)), umask_(i, j))
    fxe = apply_mask(0.25_rp * (diff2_(i + 1, j, isl) + diff2_(i, j, isl)) * &
         pmon_u_(i + 1, j) * (Hz_(i + 1, j, k) + Hz_(i, j, k)) * &
         (real(tp_(itr, i + 1, j, k, nr), rp) - real(tp_(itr, i, j, k, nr), rp)), umask_(i + 1, j))
    fes = apply_mask(0.25_rp * (diff2_(i, j, isl) + diff2_(i, j - 1, isl)) * &
         pnom_v_(i, j) * (Hz_(i, j, k) + Hz_(i, j - 1, k)) * &
         (real(tp_(itr, i, j, k, nr), rp) - real(tp_(itr, i, j - 1, k, nr), rp)), vmask_(i, j))
    fen = apply_mask(0.25_rp * (diff2_(i, j + 1, isl) + diff2_(i, j, isl)) * &
         pnom_v_(i, j + 1) * (Hz_(i, j + 1, k) + Hz_(i, j, k)) * &
         (real(tp_(itr, i, j + 1, k, nr), rp) - real(tp_(itr, i, j, k, nr), rp)), vmask_(i, j + 1))
    tp_(itr, i, j, k, nn) = real(real(tp_(itr, i, j, k, nn), rp) + dtl * pm_(i, j) * pn_(i, j) * &
         (fxe - fxw + fen - fes) / max(Hz_(i, j, k), 1.0e-30_rp), pp)
  end subroutine t3dmix_point

  ! ------------------- vertical advection + implicit solve, one thread per (tracer, column)
  DEVICE_ATTR
  subroutine column_point(itr, i, j, ts, nn, ctr, dtp, ntr, is, ie, js, je, nz, nslot, nts, isl, mir, &
                          tp_, Hz_, We_, pm_, pn_, rmask_, stflx_, ghat_, fc_, cf_, wc_)
    ! The vertical flux divergence (vert_column_fused arithmetic, sliding
    ! window over k) feeds the Thomas forward sweep level by level, with
    ! the intermediate dc kept in place in tp(nn); then the back
    ! substitution. ctr = 1: centred (predictor), 0: upwind-weighted
    ! (corrector). mir = 1: tracer 1 receives salinity's surface flux and
    ! the KPP nonlocal term between the vertical advection and the solve.
    integer(ip), intent(in) :: itr, i, j, ts, nn, ctr, ntr, is, ie, js, je, nz, nslot, nts, isl, mir
    real(rp), intent(in) :: dtp
    real(pp), intent(inout) :: tp_(ntr, is:ie, js:je, nz, nslot)
    real(rp), intent(in) :: Hz_(is:ie, js:je, nz), We_(is:ie, js:je, 0:nz)
    real(rp), intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp), intent(in) :: stflx_(is:ie, js:je, nts), ghat_(is:ie, js:je, 0:nz)
    real(rp), intent(in) :: fc_(is:ie, js:je, 0:nz), cf_(is:ie, js:je, 0:nz), wc_(is:ie, js:je, 0:nz)
    integer(ip) :: k
    real(rp) :: dtpp, rm, w, sfx, cffk, cffm, tt
    real(rp) :: Hz_km1, Hz_k, Hz_kp1, Hz_kp2, fck, wcm, cfk
    real(rp) :: fz_lo, fz_k, tplus, tminus, rhs, dck
    real(rp) :: t_km1, t_k, t_kp1, t_kp2
    logical :: at_bottom, at_surface
    dtpp = dtp * pm_(i, j) * pn_(i, j)
    rm = rmask_(i, j)
    sfx = 0.0_rp
    if (mir == 1 .and. itr == 1) sfx = stflx_(i, j, isl)
    Hz_k = Hz_(i, j, 1)
    Hz_kp1 = Hz_(i, j, 2)
    Hz_km1 = Hz_k
    Hz_kp2 = Hz_kp1
    w = 0.0_rp
    t_k = real(tp_(itr, i, j, 1, ts), rp)
    t_kp1 = real(tp_(itr, i, j, 2, ts), rp)
    t_km1 = t_k
    t_kp2 = t_kp1
    fz_lo = 0.0_rp
    dck = 0.0_rp
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
      if (k < nz) then
        if (at_surface) then
          t_kp2 = t_kp1
        else
          t_kp2 = real(tp_(itr, i, j, k + 2, ts), rp)
        endif
        tplus = thickness_tplus(t_km1, t_k, t_kp1, Hz_km1, Hz_k, Hz_kp1, at_bottom)
        tminus = thickness_tminus(t_km1, t_k, t_kp1, t_kp2, Hz_km1, Hz_k, Hz_kp1, Hz_kp2, at_bottom, at_surface)
        if (ctr == 1) then
          fz_k = 0.5_rp * (tplus + tminus) * w
        else
          fz_k = tplus * max(w, 0.0_rp) + tminus * min(w, 0.0_rp)
        endif
        fz_k = apply_mask(fz_k, rm)
      else
        fz_k = 0.0_rp
      endif
      rhs = real(real(tp_(itr, i, j, k, nn), rp) - dtpp * (fz_k - fz_lo), pp)
      fz_lo = fz_k
      t_km1 = t_k; t_k = t_kp1; t_kp1 = t_kp2
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
        dck = fck * rhs
        tp_(itr, i, j, k, nn) = real(dck, pp)
      else if (k < nz) then
        dck = fck * (rhs + dck * wcm)
        tp_(itr, i, j, k, nn) = real(dck, pp)
      else
        tt = apply_mask((rhs + dck * wcm) * fck, rm)
        tp_(itr, i, j, k, nn) = real(tt, pp)
        dck = tt
      endif
      if (k < nz) then
        wcm = wc_(i, j, k)
        Hz_km1 = Hz_k; Hz_k = Hz_kp1; Hz_kp1 = Hz_kp2
      endif
    enddo
    ! ---- surface to bottom: back substitution (dck holds the level above)
    do k = nz - 1, 1, -1
      cfk = cf_(i, j, k)
      tt = apply_mask(real(tp_(itr, i, j, k, nn), rp) + cfk * dck, rm)
      tp_(itr, i, j, k, nn) = real(tt, pp)
      dck = tt
    enddo
  end subroutine column_point

#ifdef USE_CUDA
  attributes(global) subroutine horiz_kernel(nel, nwet, wet_i, wet_j, mode, ts, ns, nn, nb, cs, cb, dtp, ntr, nx, ny, nz, is, ie, js, je, nslot, &
                                             west, east, south, north, tp_, Hz_, Hzb_, pm_, pn_, FlxU_, FlxV_, umask_, vmask_, rmask_)
    integer(ip), value :: nel, mode, ts, ns, nn, nb, ntr, nx, ny, nz, is, ie, js, je, nslot, west, east, south, north
    integer(ip), value :: nwet
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    integer(ip) :: cc
    real(rp), value :: cs, cb, dtp
    real(pp), device, intent(inout) :: tp_(ntr, is:ie, js:je, nz, nslot)
    real(rp), device, intent(in) :: Hz_(is:ie, js:je, nz), Hzb_(is:ie, js:je, nz), FlxU_(is:ie, js:je, nz), FlxV_(is:ie, js:je, nz)
    real(rp), device, intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je), rmask_(is:ie, js:je)
    integer(ip) :: tid, c, itr, i, j, k
    tid = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (tid > nel) return
    itr = mod(tid - 1, ntr) + 1; c = (tid - 1) / ntr
    cc = mod(c, nwet) + 1; k = c / nwet + 1
    i = wet_i(cc); j = wet_j(cc)
    call horiz_point(itr, i, j, k, mode, ts, ns, nn, nb, cs, cb, dtp, ntr, nx, ny, is, ie, js, je, nz, nslot, &
                     west, east, south, north, tp_, Hz_, Hzb_, pm_, pn_, FlxU_, FlxV_, umask_, vmask_, rmask_)
  end subroutine horiz_kernel

  attributes(global) subroutine t3dmix_kernel(nel, nwet, wet_i, wet_j, nr, nn, dtl, isl, ntr, nx, ny, nz, is, ie, js, je, nslot, nd, &
                                              tp_, Hz_, pm_, pn_, pmon_u_, pnom_v_, umask_, vmask_, diff2_)
    integer(ip), value :: nel, nr, nn, isl, ntr, nx, ny, nz, is, ie, js, je, nslot, nd
    integer(ip), value :: nwet
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    integer(ip) :: cc
    real(rp), value :: dtl
    real(pp), device, intent(inout) :: tp_(ntr, is:ie, js:je, nz, nslot)
    real(rp), device, intent(in) :: Hz_(is:ie, js:je, nz), pm_(is:ie, js:je), pn_(is:ie, js:je)
    real(rp), device, intent(in) :: pmon_u_(is:ie, js:je), pnom_v_(is:ie, js:je), umask_(is:ie, js:je), vmask_(is:ie, js:je)
    real(rp), device, intent(in) :: diff2_(is:ie, js:je, nd)
    integer(ip) :: tid, c, itr, i, j, k
    tid = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (tid > nel) return
    itr = mod(tid - 1, ntr) + 1; c = (tid - 1) / ntr
    cc = mod(c, nwet) + 1; k = c / nwet + 1
    i = wet_i(cc); j = wet_j(cc)
    call t3dmix_point(itr, i, j, k, nr, nn, dtl, isl, ntr, is, ie, js, je, nz, nslot, nd, &
                      tp_, Hz_, pm_, pn_, pmon_u_, pnom_v_, umask_, vmask_, diff2_)
  end subroutine t3dmix_kernel

  attributes(global) subroutine column_kernel(nel, nwet, wet_i, wet_j, ts, nn, ctr, dtp, ntr, nx, ny, nz, is, ie, js, je, nslot, nts, isl, mir, &
                                              tp_, Hz_, We_, pm_, pn_, rmask_, stflx_, ghat_, fc_, cf_, wc_)
    integer(ip), value :: nel, ts, nn, ctr, ntr, nx, ny, nz, is, ie, js, je, nslot, nts, isl, mir
    integer(ip), value :: nwet
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    integer(ip) :: cc
    real(rp), value :: dtp
    real(pp), device, intent(inout) :: tp_(ntr, is:ie, js:je, nz, nslot)
    real(rp), device, intent(in) :: Hz_(is:ie, js:je, nz), We_(is:ie, js:je, 0:nz)
    real(rp), device, intent(in) :: pm_(is:ie, js:je), pn_(is:ie, js:je), rmask_(is:ie, js:je)
    real(rp), device, intent(in) :: stflx_(is:ie, js:je, nts), ghat_(is:ie, js:je, 0:nz)
    real(rp), device, intent(in) :: fc_(is:ie, js:je, 0:nz), cf_(is:ie, js:je, 0:nz), wc_(is:ie, js:je, 0:nz)
    integer(ip) :: tid, c, itr, i, j
    tid = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (tid > nel) return
    itr = mod(tid - 1, ntr) + 1; c = (tid - 1) / ntr
    i = wet_i(c + 1); j = wet_j(c + 1)
    call column_point(itr, i, j, ts, nn, ctr, dtp, ntr, is, ie, js, je, nz, nslot, nts, isl, mir, &
                      tp_, Hz_, We_, pm_, pn_, rmask_, stflx_, ghat_, fc_, cf_, wc_)
  end subroutine column_kernel
#endif

  ! ------------------------------------------------------------ launchers
  subroutine passive_horiz_run(tp, ntr, mode, ts, ns, nn, nb, cs, cb, dtp)
    integer(ip), intent(in) :: ntr, mode, ts, ns, nn, nb
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz, 3)
    real(rp), intent(in) :: cs, cb, dtp
    integer(ip) :: nx, ny, nz, is, ie, js, je, nel, west, east, south, north
#ifndef USE_CUDA
    integer(ip) :: itr, i, j, k
#endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    west  = merge(1_ip, 0_ip, decomp%edge_w)
    east  = merge(1_ip, 0_ip, decomp%edge_e)
    south = merge(1_ip, 0_ip, decomp%edge_s)
    north = merge(1_ip, 0_ip, decomp%edge_n)
    nel = ntr * nwet * nz
#ifdef USE_CUDA
    call horiz_kernel<<<(nel + 127) / 128, 128>>>(nel, nwet, wet_i, wet_j, mode, ts, ns, nn, nb, cs, cb, dtp, ntr, nx, ny, nz, is, ie, js, je, 3_ip, &
         west, east, south, north, tp, Hz, Hz_bak, pm, pn, FlxU, FlxV, umask, vmask, rmask)
#else
    do k = 1, nz; do j = 1, ny; do i = 1, nx; do itr = 1, ntr
      call horiz_point(itr, i, j, k, mode, ts, ns, nn, nb, cs, cb, dtp, ntr, nx, ny, is, ie, js, je, nz, 3_ip, &
                       west, east, south, north, tp, Hz, Hz_bak, pm, pn, FlxU, FlxV, umask, vmask, rmask)
    enddo; enddo; enddo; enddo
#endif
  end subroutine passive_horiz_run

  subroutine passive_t3dmix_run(tp, ntr, nr, nn, dtl)
    integer(ip), intent(in) :: ntr, nr, nn
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz, 3)
    real(rp), intent(in) :: dtl
    integer(ip) :: nx, ny, nz, is, ie, js, je, nel, isl, nd
#ifndef USE_CUDA
    integer(ip) :: itr, i, j, k
#endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nd = int(size(diff2, 3), ip); isl = min(2_ip, nd)
    nel = ntr * nwet * nz
#ifdef USE_CUDA
    call t3dmix_kernel<<<(nel + 127) / 128, 128>>>(nel, nwet, wet_i, wet_j, nr, nn, dtl, isl, ntr, nx, ny, nz, is, ie, js, je, 3_ip, nd, &
         tp, Hz, pm, pn, pmon_u, pnom_v, umask, vmask, diff2)
#else
    do k = 1, nz; do j = 1, ny; do i = 1, nx; do itr = 1, ntr
      call t3dmix_point(itr, i, j, k, nr, nn, dtl, isl, ntr, is, ie, js, je, nz, 3_ip, nd, &
                        tp, Hz, pm, pn, pmon_u, pnom_v, umask, vmask, diff2)
    enddo; enddo; enddo; enddo
#endif
  end subroutine passive_t3dmix_run

  subroutine passive_column_run(tp, ntr, ts, nn, dtp, ctr, use_fwd, mirror)
    ! Factorization, then one thread per (tracer, column).
    integer(ip), intent(in) :: ntr, ts, nn, ctr
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz, 3)
    real(rp), intent(in) :: dtp
    logical, intent(in) :: use_fwd, mirror
    integer(ip) :: nx, ny, nz, mir, is, ie, js, je, nts, isl, nel
#ifndef USE_CUDA
    integer(ip) :: itr, i, j
#endif
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    nts = int(size(stflx, 3), ip); isl = int(isalt, ip)
    mir = merge(1_ip, 0_ip, mirror)
    nel = ntr * nwet
    call passive_factor(dtp, use_fwd)
#ifdef USE_CUDA
    call column_kernel<<<(nel + 127) / 128, 128>>>(nel, nwet, wet_i, wet_j, ts, nn, ctr, dtp, ntr, nx, ny, nz, is, ie, js, je, 3_ip, nts, isl, mir, &
         tp, Hz, We, pm, pn, rmask, stflx, ghat, col_fc, col_cf, col_wc)
#else
    do j = 1, ny; do i = 1, nx; do itr = 1, ntr
      call column_point(itr, i, j, ts, nn, ctr, dtp, ntr, is, ie, js, je, nz, 3_ip, nts, isl, mir, &
                        tp, Hz, We, pm, pn, rmask, stflx, ghat, col_fc, col_cf, col_wc)
    enddo; enddo; enddo
#endif
  end subroutine passive_column_run

end module passive_column_mod
