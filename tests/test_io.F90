program test_io
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use case_io
  implicit none
  integer(ip) :: nfail
  real(rp) :: expect

  nfail = 0
  call write_toy_case('build', 8_ip, 6_ip, 4_ip)
  call load_case('build/toy.nml')

  if (decomp%nx /= 8 .or. decomp%ny /= 6 .or. decomp%nz /= 4) then
    write(*,*) 'dims', decomp%nx, decomp%ny, decomp%nz
    nfail = nfail + 1
  endif
  if (decomp%periodic_x .or. decomp%periodic_y) then
    write(*,*) 'toy case should be non-periodic'
    nfail = nfail + 1
  endif
  expect = 200.0_rp + 2.0_rp + 2.0_rp
  if (abs(h(1, 1) - expect) > 1.0e-12_rp) then
    write(*,*) 'h(1,1)', h(1, 1), expect
    nfail = nfail + 1
  endif
  if (abs(h(0, 0) - (200.0_rp + 2.0_rp)) > 1.0e-12_rp) then
    write(*,*) 'h(0,0) ghost', h(0, 0)
    nfail = nfail + 1
  endif
  if (rmask(2, 2) > 0.5_rp) then
    write(*,*) 'land point not loaded', rmask(2, 2)
    nfail = nfail + 1
  endif
  if (abs(t(1, 1, 1, 1, 1) - 15.1_rp) > 1.0e-12_rp) then
    write(*,*) 'temp', t(1, 1, 1, 1, 1)
    nfail = nfail + 1
  endif
  if (abs(t(1, 1, 1, 1, 2) - 35.0_rp) > 1.0e-12_rp) then
    write(*,*) 'salt', t(1, 1, 1, 1, 2)
    nfail = nfail + 1
  endif
  if (abs(zeta(1, 1, 1) - 0.01_rp) > 1.0e-12_rp) then
    write(*,*) 'zeta', zeta(1, 1, 1)
    nfail = nfail + 1
  endif
  if (abs(u(1, 1, 1, 1) - 0.05_rp) > 1.0e-12_rp) then
    write(*,*) 'u', u(1, 1, 1, 1)
    nfail = nfail + 1
  endif
  if (abs(start_time - 86400.0_rp) > 1.0e-10_rp) then
    write(*,*) 'ocean_time', start_time
    nfail = nfail + 1
  endif
  if (abs(dt - 900.0_rp) > 0.0_rp .or. abs(hc - 250.0_rp) > 0.0_rp) then
    write(*,*) 'namelist scalars', dt, hc
    nfail = nfail + 1
  endif
  if (abs(zob - 0.01_rp) > 0.0_rp .or. abs(rdrg2 - 0.001_rp) > 1.0e-15_rp) then
    write(*,*) 'drag', zob, rdrg2
    nfail = nfail + 1
  endif
  if (umask(2, 2) > 0.5_rp) then
    write(*,*) 'umask should be 0 on land face'
    nfail = nfail + 1
  endif

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'test_io FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_io OK'
end program test_io
