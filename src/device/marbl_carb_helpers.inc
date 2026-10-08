! Carbonate-system device helpers shared by marbl_sflux_mod and
! marbl_interior_mod (#included in the contains part of both modules,
! so each module has its own copy and the kernels can inline them;
! a call across modules was not inlined). Needs T0_Kelvin, salt_min,
! xacc, max_bracket_grow_it and maxit from marbl_sflux_mod.

  ! marbl_co2calc_mod: apply_pressure_correction factor
  DEVICE_ATTR
  subroutine pcorr(temp, invRtk, press_bar, dv0, dv1, dv2, ka0, ka1, ka2, kfac)
    ! subroutine with by-reference dummies: nvfortran 25.9 ICEs when a
    ! device subroutine calls a device function with value arguments
    real(rp), intent(in) :: temp, invRtk, press_bar, dv0, dv1, dv2, ka0, ka1, ka2
    real(rp), intent(out) :: kfac
    real(rp) :: deltaV, Kappa, lnKfac
    real(rp), parameter :: p5 = 0.5_rp, p001 = 0.001_rp
    deltaV = dv0 + (dv1 + dv2 * temp) * temp
    Kappa  = (ka0 + (ka1 + ka2 * temp) * temp) * p001
    lnKfac = (-deltaV + p5 * Kappa * press_bar) * press_bar * invRtk
    kfac = exp(lnKfac)
  end subroutine pcorr

  ! marbl_co2calc_mod: comp_co2calc_coeffs
  DEVICE_ATTR
  subroutine co2_coeffs(temp, salt, press_bar, lpress, k0, k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft)
    ! All dummies by reference: nvfortran 25.9 device subroutines that mix
    ! 'value' inputs with by-reference outputs received undefined inputs.
    ! lpress /= 0 applies MARBL's pressure correction (interior path);
    ! the surface path passes lpress = 0.
    real(rp), intent(in) :: temp, salt, press_bar
    integer(ip), intent(in) :: lpress
    real(rp), intent(out) :: k0, k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft
    real(rp) :: salt_lim, tk, tk100, tk1002, invtk, dlogtk, is, is2, sqrtis, sqrts, s2, scl, invRtk
    real(rp) :: arg, log_1_m_1p005em3_s, log_1_p_tot_sulfate_div_ks, kfac
    real(rp), parameter :: c1 = 1.0_rp, c10 = 10.0_rp, c1000 = 1000.0_rp

    salt_lim = max(salt, salt_min)
    tk       = T0_Kelvin + temp
    tk100    = tk * 1e-2_rp
    tk1002   = tk100 * tk100
    invtk    = c1 / tk
    dlogtk   = log(tk)
    invRtk   = (c1 / 83.1451_rp) * invtk
    is       = 19.924_rp * salt_lim / (c1000 - 1.005_rp * salt_lim)
    is2      = is * is
    sqrtis   = sqrt(is)
    sqrts    = sqrt(salt_lim)
    s2       = salt_lim * salt_lim
    scl      = salt_lim / 1.80655_rp
    arg = c1 - 0.001005_rp * salt_lim
    log_1_m_1p005em3_s = log(arg)

    arg = -162.8301_rp + 218.2968_rp / tk100 + &
         90.9241_rp * (dlogtk + log(1e-2_rp)) - 1.47696_rp * tk1002 + &
         salt_lim * (.025695_rp - .025225_rp * tk100 + 0.0049867_rp * tk1002)
    ff = exp(arg)

    arg = 93.4517_rp / tk100 - 60.2409_rp + 23.3585_rp * (dlogtk + log(1e-2_rp)) + &
         salt_lim * (.023517_rp - 0.023656_rp * tk100 + 0.0047036_rp * tk1002)
    k0 = exp(arg)

    arg = 3633.86_rp * invtk - 61.2172_rp + &
          9.67770_rp * dlogtk - 0.011555_rp * salt_lim + &
          0.0001152_rp * s2
    arg = -log(c10) * arg
    k1 = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -25.5_rp, 0.1271_rp, 0.0_rp, -3.08_rp, 0.0877_rp, 0.0_rp, kfac)
      k1 = k1 * kfac
    endif

    arg = 471.78_rp * invtk + 25.9290_rp - &
          3.16967_rp * dlogtk - 0.01781_rp * salt_lim + 0.0001122_rp * s2
    arg = -log(c10) * arg
    k2 = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -15.82_rp, -0.0219_rp, 0.0_rp, 1.13_rp, -0.1475_rp, 0.0_rp, kfac)
      k2 = k2 * kfac
    endif

    arg = (-8966.90_rp - 2890.53_rp * sqrts - &
         77.942_rp * salt_lim + 1.728_rp * salt_lim * sqrts - &
         0.0996_rp * s2) * invtk + &
         (148.0248_rp + 137.1942_rp * sqrts + 1.62142_rp * salt_lim) + &
         (-24.4344_rp - 25.085_rp * sqrts - 0.2474_rp * salt_lim) * dlogtk + &
         0.053105_rp * sqrts * tk
    kb = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -29.48_rp, 0.1622_rp, -0.002608_rp, -2.84_rp, 0.0_rp, 0.0_rp, kfac)
      kb = kb * kfac
    endif

    arg = -4576.752_rp * invtk + 115.525_rp - &
         18.453_rp * dlogtk + &
         (-106.736_rp * invtk + 0.69171_rp) * sqrts + &
         (-0.65643_rp * invtk - 0.01844_rp) * salt_lim
    k1p = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -14.51_rp, 0.1211_rp, -0.000321_rp, -2.67_rp, 0.0427_rp, 0.0_rp, kfac)
      k1p = k1p * kfac
    endif

    arg = -8814.715_rp * invtk + 172.0883_rp - &
         27.927_rp * dlogtk + &
         (-160.340_rp * invtk + 1.3566_rp) * sqrts + &
         (0.37335_rp * invtk - 0.05778_rp) * salt_lim
    k2p = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -23.12_rp, 0.1758_rp, -0.002647_rp, -5.15_rp, 0.09_rp, 0.0_rp, kfac)
      k2p = k2p * kfac
    endif

    arg = -3070.75_rp * invtk - 18.141_rp + &
         (17.27039_rp * invtk + 2.81197_rp) * sqrts + &
         (-44.99486_rp * invtk - 0.09984_rp) * salt_lim
    k3p = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -26.57_rp, 0.202_rp, -0.003042_rp, -4.08_rp, 0.0714_rp, 0.0_rp, kfac)
      k3p = k3p * kfac
    endif

    arg = -8904.2_rp * invtk + 117.385_rp - &
         19.334_rp * dlogtk + &
         (-458.79_rp * invtk + 3.5913_rp) * sqrtis + &
         (188.74_rp * invtk - 1.5998_rp) * is + &
         (-12.1652_rp * invtk + 0.07871_rp) * is2 + &
         log_1_m_1p005em3_s
    ksi = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -29.48_rp, 0.1622_rp, -0.002608_rp, -2.84_rp, 0.0_rp, 0.0_rp, kfac)
      ksi = ksi * kfac
    endif

    arg = -13847.26_rp * invtk + 148.9652_rp - 23.6521_rp * dlogtk + &
         (118.67_rp * invtk - 5.977_rp + 1.0495_rp * dlogtk) * sqrts - &
         0.01615_rp * salt_lim
    kw = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -20.02_rp, 0.1119_rp, -0.001409_rp, -5.13_rp, 0.0794_rp, 0.0_rp, kfac)
      kw = kw * kfac
    endif

    arg = -4276.1_rp * invtk + 141.328_rp - 23.093_rp * dlogtk + &
         (-13856.0_rp * invtk + 324.57_rp - 47.986_rp * dlogtk) * sqrtis + &
         (35474.0_rp * invtk - 771.54_rp + 114.723_rp * dlogtk) * is - &
         2698.0_rp * invtk * is * sqrtis + &
         1776.0_rp * invtk * is2 + &
         log_1_m_1p005em3_s
    ks  = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -18.03_rp, 0.0466_rp, 0.000316_rp, -4.53_rp, 0.09_rp, 0.0_rp, kfac)
      ks = ks * kfac
    endif

    arg = c1 + (0.1400_rp / 96.062_rp) * (scl) / ks
    log_1_p_tot_sulfate_div_ks = log(arg)
    arg = 1590.2_rp * invtk - 12.641_rp + 1.525_rp * sqrtis + &
         log_1_m_1p005em3_s + log_1_p_tot_sulfate_div_ks
    kf  = exp(arg)
    if (lpress /= 0) then
      call pcorr(temp, invRtk, press_bar, -9.78_rp, -0.009_rp, -0.000942_rp, -3.91_rp, 0.054_rp, 0.0_rp, kfac)
      kf = kf * kfac
    endif

    bt = 0.000232_rp / 10.811_rp * scl
    st = 0.14_rp / 96.062_rp * scl
    ft = 0.000067_rp / 18.9984_rp * scl
  end subroutine co2_coeffs

  ! marbl_co2calc_mod: total_alkalinity (one element): TA(x) - ta and d/dx
  DEVICE_ATTR
  subroutine ta_fn(x, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit, fn, df)
    real(rp), intent(in) :: x, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit
    real(rp), intent(out) :: fn, df
    real(rp) :: x1, x1_r, x2, x2_r, x3, k12, k12p, k123p
    real(rp) :: a, a_r, a2_r, da, b, b_r, b2_r, db, c_tmp, c_r
    real(rp) :: kb_p_x1_r, ksi_p_x1_r, c1_p_c_ks_x1_r_r, c1_p_kf_x1_r_r
    real(rp), parameter :: c1 = 1.0_rp, c2 = 2.0_rp, c3 = 3.0_rp

    x1 = x
    x1_r = c1 / x1
    x2 = x1 * x1
    x2_r = x1_r * x1_r
    x3 = x2 * x1
    k12 = k1 * k2
    k12p = k1p * k2p
    k123p = k12p * k3p
    a = x3 + k1p * x2 + k12p * x1 + k123p
    a_r = c1 / a
    a2_r = a_r * a_r
    da = c3 * x2 + c2 * k1p * x1 + k12p
    b = x2 + k1 * x1 + k12
    b_r = c1 / b
    b2_r = b_r * b_r
    db = c2 * x1 + k1
    c_tmp = c1 + st / ks
    c_r = c1 / c_tmp
    kb_p_x1_r = c1 / (kb + x1)
    ksi_p_x1_r = c1 / (ksi + x1)
    c1_p_c_ks_x1_r_r = c1 / (c1 + c_tmp * ks * x1_r)
    c1_p_kf_x1_r_r = c1 / (c1 + kf * x1_r)

    fn = k1 * dic * x1 * b_r &
         + c2 * dic * k12 * b_r &
         + bt * kb * kb_p_x1_r &
         + kw * x1_r &
         + pt * k12p * x1 * a_r &
         + c2 * pt * k123p * a_r &
         + sit * ksi * ksi_p_x1_r &
         - x1 * c_r &
         - st * c1_p_c_ks_x1_r_r &
         - ft * c1_p_kf_x1_r_r &
         - pt * x3 * a_r &
         - ta

    df = k1 * dic * (b - x1 * db) * b2_r &
         - c2 * dic * k12 * db * b2_r &
         - bt * kb * kb_p_x1_r * kb_p_x1_r &
         - kw * x2_r &
         + (pt * k12p * (a - x1 * da)) * a2_r &
         - c2 * pt * k123p * da * a2_r &
         - sit * ksi * ksi_p_x1_r * ksi_p_x1_r &
         - c1 * c_r &
         - st * c1_p_c_ks_x1_r_r * c1_p_c_ks_x1_r_r * (c_tmp * ks * x2_r) &
         - ft * c1_p_kf_x1_r_r * c1_p_kf_x1_r_r * kf * x2_r &
         - pt * x2 * (c3 * a - x1 * da) * a2_r
  end subroutine ta_fn

  ! marbl_co2calc_mod: drtsafe (one element): bracket growth, then safeguarded Newton
  DEVICE_ATTR
  subroutine drtsafe1(k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit, &
                      x1_in, x2_in, rsoln, rierr)
    real(rp), intent(in) :: k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit
    real(rp), intent(in) :: x1_in, x2_in
    real(rp), intent(out) :: rsoln
    integer(ip), intent(out) :: rierr
    real(rp) :: x1, x2, xlo, xhi, flo, fhi, f, df, dxold, dx, temp_loc, soln, dfl, dfh
    integer :: it
    logical :: leave_bracket, dx_decrease
    real(rp), parameter :: c0 = 0.0_rp, p5 = 0.5_rp, c2 = 2.0_rp

    rierr = 0
    rsoln = c0
    x1 = x1_in; x2 = x2_in
    it = 0
    do
      call ta_fn(x1, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit, flo, dfl)
      call ta_fn(x2, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit, fhi, dfh)
      if (.not. ((flo > c0 .and. fhi > c0) .or. (flo < c0 .and. fhi < c0))) exit
      it = it + 1
      if (it > max_bracket_grow_it) then
        rierr = 1
        return
      endif
      dx = sqrt(x2 / x1)
      x2 = x2 * dx
      x1 = x1 / dx
    enddo

    if (flo < c0) then
      xlo = x1
      xhi = x2
    else
      xlo = x2
      xhi = x1
      temp_loc = flo
      flo = fhi
      fhi = temp_loc
    endif
    soln = p5 * (xlo + xhi)
    dxold = abs(xlo - xhi)
    dx = dxold
    call ta_fn(soln, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit, f, df)

    do it = 1, maxit
      leave_bracket = ((soln - xhi) * df - f) * ((soln - xlo) * df - f) >= c0
      dx_decrease = abs(c2 * f) <= abs(dxold * df)
      if (leave_bracket .or. .not. dx_decrease) then
        dxold = dx
        dx = p5 * (xhi - xlo)
        soln = xlo + dx
        if (xlo == soln) then
          rsoln = soln; return
        endif
      else
        dxold = dx
        dx = -f / df
        temp_loc = soln
        soln = soln + dx
        if (temp_loc == soln) then
          rsoln = soln; return
        endif
      endif
      if (abs(dx) < xacc) then
        rsoln = soln; return
      endif
      call ta_fn(soln, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit, f, df)
      if (f < c0) then
        xlo = soln
        flo = f
      else
        xhi = soln
        fhi = f
      endif
    enddo
    rsoln = soln
    rierr = 2
  end subroutine drtsafe1
