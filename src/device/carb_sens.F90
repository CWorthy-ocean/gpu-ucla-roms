#include "mem_attr.inc"
module carb_sens_mod
  ! Surface carbonate sensitivities beta = dDIC/dCO2 and eta = dDIC/dALK on
  ! the device: a transliteration of the CPU's carbonate_sensitivity.F90
  ! (compute_surface_beta_eta), the same arithmetic per cell. That module
  ! uses MARBL-consistent constants (Lueker et al. 2000 K1/K2, total
  ! scale), a safeguarded Newton-Raphson pH solve (rtsafe) bracketed in pH
  ! 2-13 with a relative convergence test and a residual check, and the
  ! Humphreys et al. (2018) isocapnic quotient. A cell that is land, below
  ! the salinity/ALK/DIC floors, or whose solve fails gets beta = eta = 0.
  ! Used by the cdr_gas_exch output (cdr_gas_out); the inputs are the
  ! ALT_CO2 carbon system. The helpers take every argument by reference
  ! (see the toolchain notes in MARBL_GPU_PORT.md).
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: cs_surface

  real(rp), parameter :: rho_sw = 1025.0_rp
  real(rp), parameter :: t0_kelvin = 273.15_rp
  real(rp), parameter :: xacc = 1.0e-10_rp
  integer, parameter :: maxit = 100
  real(rp), parameter :: ph_lo = 2.0_rp, ph_hi = 13.0_rp
  real(rp), parameter :: resid_tol = 1.0e-8_rp
  real(rp), parameter :: salt_min = 0.1_rp
  real(rp), parameter :: dic_min = salt_min / 35.0_rp * 1944.0_rp
  real(rp), parameter :: alk_min = salt_min / 35.0_rp * 2225.0_rp

contains

  DEVICE_ATTR
  subroutine cs_constants(t_c, s, k0, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft)
    ! equilibrium_constants (k_carbonic_opt = 10)
    real(rp), intent(in) :: t_c, s
    real(rp), intent(out) :: k0, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft
    real(rp) :: tk, tk100, tk1002, invtk, dlogtk
    real(rp) :: is, is2, sqrtis, sqrts, s15, s2, scl
    real(rp) :: pk1, pk2
    tk = t0_kelvin + t_c
    tk100 = tk * 1.0e-2_rp
    tk1002 = tk100 * tk100
    invtk = 1.0_rp / tk
    dlogtk = log(tk)
    is = 19.924_rp * s / (1000.0_rp - 1.005_rp * s)
    is2 = is * is
    sqrtis = sqrt(is)
    sqrts = sqrt(s)
    s15 = s**1.5_rp
    s2 = s * s
    scl = s / 1.80655_rp
    k0 = exp(93.4517_rp / tk100 - 60.2409_rp + 23.3585_rp * log(tk100) &
         + s * (0.023517_rp - 0.023656_rp * tk100 + 0.0047036_rp * tk1002))
    pk1 = 3633.86_rp * invtk - 61.2172_rp + 9.67770_rp * dlogtk &
          - 0.011555_rp * s + 0.0001152_rp * s2
    pk2 = 471.78_rp * invtk + 25.9290_rp - 3.16967_rp * dlogtk &
          - 0.01781_rp * s + 0.0001122_rp * s2
    k1 = 10.0_rp**(-pk1)
    k2 = 10.0_rp**(-pk2)
    kb = exp((-8966.90_rp - 2890.53_rp * sqrts - 77.942_rp * s &
         + 1.728_rp * s15 - 0.0996_rp * s2) * invtk &
         + (148.0248_rp + 137.1942_rp * sqrts + 1.62142_rp * s) &
         + (-24.4344_rp - 25.085_rp * sqrts - 0.2474_rp * s) * dlogtk &
         + 0.053105_rp * sqrts * tk)
    k1p = exp(-4576.752_rp * invtk + 115.525_rp - 18.453_rp * dlogtk &
          + (-106.736_rp * invtk + 0.69171_rp) * sqrts &
          + (-0.65643_rp * invtk - 0.01844_rp) * s)
    k2p = exp(-8814.715_rp * invtk + 172.0883_rp - 27.927_rp * dlogtk &
          + (-160.340_rp * invtk + 1.3566_rp) * sqrts &
          + (0.37335_rp * invtk - 0.05778_rp) * s)
    k3p = exp(-3070.75_rp * invtk - 18.141_rp &
          + (17.27039_rp * invtk + 2.81197_rp) * sqrts &
          + (-44.99486_rp * invtk - 0.09984_rp) * s)
    ksi = exp(-8904.2_rp * invtk + 117.385_rp - 19.334_rp * dlogtk &
          + (-458.79_rp * invtk + 3.5913_rp) * sqrtis &
          + (188.74_rp * invtk - 1.5998_rp) * is &
          + (-12.1652_rp * invtk + 0.07871_rp) * is2 &
          + log(1.0_rp - 0.001005_rp * s))
    kw = exp(-13847.26_rp * invtk + 148.9652_rp - 23.6521_rp * dlogtk &
         + (118.67_rp * invtk - 5.977_rp + 1.0495_rp * dlogtk) * sqrts &
         - 0.01615_rp * s)
    ks = exp(-4276.1_rp * invtk + 141.328_rp - 23.093_rp * dlogtk &
         + (-13856.0_rp * invtk + 324.57_rp - 47.986_rp * dlogtk) * sqrtis &
         + (35474.0_rp * invtk - 771.54_rp + 114.723_rp * dlogtk) * is &
         - 2698.0_rp * invtk * is**1.5_rp + 1776.0_rp * invtk * is2 &
         + log(1.0_rp - 0.001005_rp * s))
    kf = exp(1590.2_rp * invtk - 12.641_rp + 1.525_rp * sqrtis &
         + log(1.0_rp - 0.001005_rp * s) &
         + log(1.0_rp + (0.1400_rp / 96.062_rp) * scl / ks))
    bt = 0.000232_rp * scl / 10.811_rp
    st = 0.14_rp * scl / 96.062_rp
    ft = 0.000067_rp * scl / 18.9984_rp
  end subroutine cs_constants

  DEVICE_ATTR
  subroutine cs_residual(x, ta, dic, pt, sit, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, fn, df)
    ! talk_residual: alkalinity residual and its derivative in [H+]
    real(rp), intent(in) :: x, ta, dic, pt, sit
    real(rp), intent(in) :: k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft
    real(rp), intent(out) :: fn, df
    real(rp) :: x1, x2, x3, k12, k12p, k123p, a, a2, da, b, b2, db, c
    real(rp) :: hso4_den, hf_den, kb_den, ksi_den
    x1 = x
    x2 = x1 * x1
    x3 = x2 * x1
    k12 = k1 * k2
    k12p = k1p * k2p
    k123p = k12p * k3p
    a = x3 + k1p * x2 + k12p * x1 + k123p
    a2 = a * a
    da = 3.0_rp * x2 + 2.0_rp * k1p * x1 + k12p
    b = x2 + k1 * x1 + k12
    b2 = b * b
    db = 2.0_rp * x1 + k1
    c = 1.0_rp + st / ks
    kb_den = 1.0_rp + x1 / kb
    ksi_den = 1.0_rp + x1 / ksi
    hso4_den = 1.0_rp + ks * c / x1
    hf_den = 1.0_rp + kf / x1
    fn = k1 * x1 * dic / b &
       + 2.0_rp * dic * k12 / b &
       + bt / kb_den &
       + kw / x1 &
       + pt * k12p * x1 / a &
       + 2.0_rp * pt * k123p / a &
       + sit / ksi_den &
       - x1 / c &
       - st / hso4_den &
       - ft / hf_den &
       - pt * x3 / a &
       - ta
    df = ((k1 * dic * b) - k1 * x1 * dic * db) / b2 &
       - 2.0_rp * dic * k12 * db / b2 &
       - bt / kb / (kb_den * kb_den) &
       - kw / x2 &
       + (pt * k12p * (a - x1 * da)) / a2 &
       - 2.0_rp * pt * k123p * da / a2 &
       - sit / ksi / (ksi_den * ksi_den) &
       - 1.0_rp / c &
       - st * (ks * c / x2) / (hso4_den * hso4_den) &
       - ft * (kf / x2) / (hf_den * hf_den) &
       - pt * x2 * (3.0_rp * a - x1 * da) / a2
  end subroutine cs_residual

  DEVICE_ATTR
  subroutine cs_solve(ta, dic, pt, sit, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, h, ok)
    ! solve_htotal: safeguarded Newton-Raphson (rtsafe) for [H+]; ok = 0
    ! when the root is not bracketed or the final residual is too large
    real(rp), intent(in) :: ta, dic, pt, sit
    real(rp), intent(in) :: k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft
    real(rp), intent(out) :: h
    integer, intent(out) :: ok
    integer :: iter
    real(rp) :: x_lo, x_hi, f_lo, f_hi, xl, xh, x, fn, df, dx, dxold, xtemp, phl, phh
    ok = 0
    h = 0.0_rp
    phh = -ph_hi; phl = -ph_lo
    x_lo = 10.0_rp**phh
    x_hi = 10.0_rp**phl
    call cs_residual(x_lo, ta, dic, pt, sit, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, f_lo, df)
    call cs_residual(x_hi, ta, dic, pt, sit, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, f_hi, df)
    if (f_lo * f_hi > 0.0_rp) return
    if (f_lo < 0.0_rp) then
      xl = x_lo
      xh = x_hi
    else
      xl = x_hi
      xh = x_lo
    endif
    x = 0.5_rp * (x_lo + x_hi)
    dxold = abs(x_hi - x_lo)
    dx = dxold
    call cs_residual(x, ta, dic, pt, sit, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, fn, df)
    do iter = 1, maxit
      if (((x - xh) * df - fn) * ((x - xl) * df - fn) >= 0.0_rp .or. &
          abs(2.0_rp * fn) > abs(dxold * df)) then
        dxold = dx
        dx = 0.5_rp * (xh - xl)
        x = xl + dx
        if (xl == x) exit
      else
        dxold = dx
        dx = fn / df
        xtemp = x
        x = x - dx
        if (xtemp == x) exit
      endif
      if (abs(dx) < xacc * abs(x)) exit
      call cs_residual(x, ta, dic, pt, sit, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, fn, df)
      if (fn < 0.0_rp) then
        xl = x
      else
        xh = x
      endif
    enddo
    if (x <= 0.0_rp) return
    call cs_residual(x, ta, dic, pt, sit, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, fn, df)
    if (abs(fn) > resid_tol * ta) return
    h = x
    ok = 1
  end subroutine cs_solve

  DEVICE_ATTR
  subroutine cs_cell(temp, salt, alk, dic, po4, sio3, rm, beta, eta)
    ! compute_surface_beta_eta for one cell
    real(rp), intent(in) :: temp, salt, alk, dic, po4, sio3, rm
    real(rp), intent(out) :: beta, eta
    real(rp) :: vol_to_mass, ta, tc, pt, sit, h, co2, hco3, co3, iso_q, h2, denom, kbph2
    real(rp) :: k0, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft
    integer :: ok
    beta = 0.0_rp
    eta = 0.0_rp
    if (rm <= 0.5_rp) return
    if (salt < salt_min) return
    if (alk < alk_min .or. dic < dic_min) return
    vol_to_mass = 1.0_rp / (1.0e3_rp * rho_sw)
    ta = alk * vol_to_mass
    tc = dic * vol_to_mass
    pt = max(po4, 0.0_rp) * vol_to_mass
    sit = max(sio3, 0.0_rp) * vol_to_mass
    call cs_constants(temp, salt, k0, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft)
    call cs_solve(ta, tc, pt, sit, k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, h, ok)
    if (ok == 0 .or. h <= 0.0_rp) return
    ! carbonate_speciation
    h2 = h * h
    denom = h2 + k1 * h + k1 * k2
    co2 = tc * h2 / denom
    hco3 = tc * k1 * h / denom
    co3 = tc * k1 * k2 / denom
    if (co2 <= 0.0_rp) return
    ! isocapnic_quotient (Humphreys et al. 2018, Eq. 8)
    kbph2 = (kb + h)**2
    iso_q = ((k1 * co2 * h + 4.0_rp * k1 * k2 * co2 + kw * h + h**3) * kbph2 &
            + kb * bt * h**3) &
           / (k1 * co2 * (2.0_rp * k2 + h) * kbph2)
    if (iso_q <= 0.0_rp) return
    beta = (tc - (hco3 + 2.0_rp * co3) / iso_q) / co2
    eta = 1.0_rp / iso_q
  end subroutine cs_cell

#ifdef USE_CUDA
  attributes(global) &
  subroutine cs_kernel(ntr, ncl, nz, slot, nts, ia, id, ipo, isi, is, ie, js, je, i0, i1, j0, j1, &
                       tp_, t_, rm_, col_, beta_, eta_)
    ! One thread per point of i0:i1 x j0:j1. Scalars come by value; the
    ! per-cell helper takes its arguments by reference (device locals).
    integer(ip), value :: ntr, ncl, nz, slot, nts, ia, id, ipo, isi, is, ie, js, je, i0, i1, j0, j1
    real(pp), device, intent(in) :: tp_(ntr, 0:ncl, nz, 3)
    real(rp), device, intent(in) :: t_(is:ie, js:je, nz, 3, nts)
    real(rp), device, intent(in) :: rm_(is:ie, js:je)
    integer(ip), device, intent(in) :: col_(is:ie, js:je)
    real(rp), device :: beta_(is:ie, js:je), eta_(is:ie, js:je)
    integer(ip) :: i, j, k, ni, nj, col
    real(rp) :: te, sa, al, di, ph, si, rm, b, e
    ni = i1 - i0 + 1; nj = j1 - j0 + 1
    k = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
    if (k >= ni * nj) return
    i = i0 + mod(k, ni); j = j0 + k / ni
    col = col_(i, j)
    te = t_(i, j, nz, slot, itemp); sa = t_(i, j, nz, slot, isalt)
    al = real(tp_(ia, col, nz, slot), rp); di = real(tp_(id, col, nz, slot), rp)
    ph = real(tp_(ipo, col, nz, slot), rp); si = real(tp_(isi, col, nz, slot), rp)
    rm = rm_(i, j)
    call cs_cell(te, sa, al, di, ph, si, rm, b, e)
    beta_(i, j) = b
    eta_(i, j) = e
  end subroutine cs_kernel
#endif

  subroutine cs_surface(tp, ntr, slot, ialk, idic, ipo4, isio3, i0, i1, j0, j1, beta, eta)
    ! beta, eta at the surface level of time slot `slot` over i0:i1 x j0:j1
    ! (T, S from t; the ALT_CO2 ALK/DIC, PO4, SiO3 passive tracers from tp).
    integer(ip), intent(in) :: ntr, slot, ialk, idic, ipo4, isio3, i0, i1, j0, j1
    real(pp), WORK_ATTR intent(in) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    real(rp), WORK_ATTR intent(inout) :: beta(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: eta(decomp%is:decomp%ie, decomp%js:decomp%je)
#ifdef USE_CUDA
    integer(ip) :: npt, nts
    type(dim3) :: grid, block
    npt = (i1 - i0 + 1) * (j1 - j0 + 1)
    if (npt <= 0) return
    nts = int(size(t, 5), ip)
    block = dim3(128, 1, 1)
    grid = dim3((npt + 127) / 128, 1, 1)
    call cs_kernel<<<grid, block, 0, work_stream>>>(ntr, ntpcol, decomp%nz, slot, nts, ialk, idic, ipo4, isio3, &
         decomp%is, decomp%ie, decomp%js, decomp%je, i0, i1, j0, j1, tp, t, rmask, tpcol, beta, eta)
#else
    error stop 'cs_surface: CUDA build required'
#endif
  end subroutine cs_surface

end module carb_sens_mod
