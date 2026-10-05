program test_formulas
  use kinds
  use formulas
  implicit none
  real(rp) :: cs, z, tp, tm, fwd, f1, f2, bn, b, b1, b2, mis
  integer(ip) :: nfail

  nfail = 0

  cs = csf(-1.0_rp, 6.0_rp, 6.0_rp)
  if (abs(cs + 1.0_rp) > 1.0e-12_rp) then
    write(*,*) 'csf(s=-1) should be -1, got', cs
    nfail = nfail + 1
  endif
  cs = csf(0.0_rp, 6.0_rp, 6.0_rp)
  if (abs(cs) > 1.0e-12_rp) then
    write(*,*) 'csf(s=0) should be 0, got', cs
    nfail = nfail + 1
  endif

  z = sm09_z(0.0_rp, 4000.0_rp, 250.0_rp, -1.0_rp, -1.0_rp)
  if (abs(z + 4000.0_rp) > 1.0e-10_rp) then
    write(*,*) 'sm09 bottom should be -h, got', z
    nfail = nfail + 1
  endif
  z = sm09_z(1.0_rp, 4000.0_rp, 250.0_rp, 0.0_rp, 0.0_rp)
  if (abs(z - 1.0_rp) > 1.0e-12_rp) then
    write(*,*) 'sm09 surface should be zeta, got', z
    nfail = nfail + 1
  endif

  ! Uniform layers, linear t: faces must recover the linear interpolant.
  call thickness_aware_faces(1.0_rp, 2.0_rp, 3.0_rp, 4.0_rp, &
                             10.0_rp, 10.0_rp, 10.0_rp, 10.0_rp, &
                             .false., .false., tp, tm)
  if (abs(tp - 2.5_rp) > 1.0e-12_rp .or. abs(tm - 2.5_rp) > 1.0e-12_rp) then
    write(*,*) 'linear column faces', tp, tm
    nfail = nfail + 1
  endif

  call fb_ab3_am4_weights(1_ip, fwd, f1, f2, bn, b, b1, b2)
  if (abs(fwd - 1.0_rp) > 0.0_rp .or. abs(b - 1.0_rp) > 0.0_rp) then
    write(*,*) 'iif=1 weights', fwd, b
    nfail = nfail + 1
  endif
  if (abs(barotropic_pgf(4000._rp, 4000._rp, 0._rp, 0._rp, 0._rp, 0._rp, &
                         0._rp, 0._rp, 0._rp, 0._rp, 0._rp, 0._rp, 1._rp)) > 0._rp) then
    nfail = nfail + 1
  endif

  mis = flux_mismatch(10.0_rp, 2.0_rp, 10.0_rp)
  if (abs(mis) > 1.0e-15_rp) nfail = nfail + 1
  if (abs(apply_mask(5.0_rp, 0.0_rp)) > 0.0_rp) nfail = nfail + 1
  if (abs(harmonic_avg(2.0_rp, -3.0_rp)) > 0.0_rp) nfail = nfail + 1
  if (abs(harmonic_avg(2.0_rp, 2.0_rp) - 2.0_rp) > 1.0e-15_rp) nfail = nfail + 1
  if (abs(split_eos_insitu(1.2_rp, 0.0_rp, -100.0_rp) - 1.2_rp) > 1.0e-15_rp) nfail = nfail + 1
  if (abs(dens_jacobian(1._rp, 1._rp, 5._rp, 5._rp, 0._rp, 0._rp, 1._rp, 1._rp)) > 1.0e-15_rp) nfail = nfail + 1
  if (abs(up3_flux(5._rp, 5._rp, 0._rp, 0._rp, 2._rp) - 10._rp) > 1.0e-14_rp) nfail = nfail + 1
  if (abs(up3_flux(3._rp, 5._rp, 0._rp, 0._rp, 2._rp) - 8._rp) > 1.0e-14_rp) nfail = nfail + 1
  ! Positive flux uses the left curvature: 0.5*(1+3)*6 - (1/6)*2*6 = 10
  if (abs(up3_flux(1._rp, 3._rp, 2._rp, 8._rp, 6._rp) - 10._rp) > 1.0e-14_rp) nfail = nfail + 1
  ! Negative flux uses the right curvature: 0.5*(1+3)*(-6) - (1/6)*8*(-6) = -4
  if (abs(up3_flux(1._rp, 3._rp, 2._rp, 8._rp, -6._rp) + 4._rp) > 1.0e-14_rp) nfail = nfail + 1
  call jackett_split_eos(3.8_rp, 34.5_rp, z, mis)
  if (abs(z) > 5.0_rp) nfail = nfail + 1

  call jackett_alpha_beta(15.0_rp, 35.0_rp, fwd, f1)
  if (fwd <= 0.0_rp .or. f1 <= 0.0_rp) then
    write(*,*) 'alpha, beta should be > 0', fwd, f1
    nfail = nfail + 1
  endif
  if (fwd < 1.0e-5_rp .or. fwd > 5.0e-4_rp) then
    write(*,*) 'alpha out of range', fwd
    nfail = nfail + 1
  endif
  if (abs(lmd_nu_sx(0.0_rp) - 1.0_rp) > 1.0e-15_rp) nfail = nfail + 1
  if (abs(lmd_nu_sx(1.0_rp)) > 1.0e-15_rp) nfail = nfail + 1
  if (abs(lmd_nu_sx(2.0_rp)) > 1.0e-15_rp) nfail = nfail + 1
  if (abs(lmd_gradient_ri(0.0_rp, 0.0_rp, 0.0_rp)) > 0.0_rp) nfail = nfail + 1
  if (abs(lmd_bottom_mult(0.0_rp)) > 1.0e-15_rp) nfail = nfail + 1
  if (abs(lmd_bottom_mult(20.0_rp) - 1.0_rp) > 0.0_rp) nfail = nfail + 1
  call lmd_wscale_ws(2.0_rp, 1.0e-7_rp, 0.01_rp, 20.0_rp, 1.0_rp, f2)
  bn = vonkar * 0.01_rp * 1.0e-6_rp / (1.0e-6_rp + 5.0_rp * vonkar * 2.0_rp * 1.0e-7_rp)
  if (abs(f2 - bn) > 1.0e-14_rp) then
    write(*,*) 'stable ws', f2, bn
    nfail = nfail + 1
  endif
  call lmd_wscale_ws(5.0_rp, -1.0e-6_rp, 0.0_rp, 50.0_rp, 1.0_rp, f2)
  if (f2 <= 0.0_rp) then
    write(*,*) 'convective ws should be > 0', f2
    nfail = nfail + 1
  endif
  if (abs(swr_band_atten(0.58_rp, -1.0_rp / jwt1_mu1, 0.0_rp) - 0.58_rp) > 1.0e-15_rp) nfail = nfail + 1

  if (nfail /= 0) then
    write(*,*) 'test_formulas FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_formulas OK'
end program test_formulas
