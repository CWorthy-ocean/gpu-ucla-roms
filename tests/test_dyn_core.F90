program test_dyn_core
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use step_driver, only: run_baroclinic_step
  implicit none
  integer(ip) :: nfail
  real(rp) :: umax, tmax, smax

  nfail = 0
  call init_decomp(12_ip, 12_ip, 8_ip, 2_ip, 1_ip, 1_ip, 0_ip, .true., .true.)
  call init_time_state(8_ip, 20_ip, 50.0_rp, 1_ip, 2_ip)
  call alloc_state(2_ip)
  call init_ideal_box(4000.0_rp, 12000.0_rp, 1.0e-4_rp)
  knew = 1
  iic = 1
  call run_baroclinic_step()

  umax = max(maxval(abs(u(1:decomp%nx+1,1:decomp%ny,:,nnew))), &
             maxval(abs(v(1:decomp%nx,1:decomp%ny+1,:,nnew))))
  if (umax > 1.0e-8_rp) then
    write(*,*) 'rest-state velocity drifted', umax
    nfail = nfail + 1
  endif
  tmax = maxval(abs(t(1:decomp%nx,1:decomp%ny,:,nnew,1) - 15.0_rp))
  smax = maxval(abs(t(1:decomp%nx,1:decomp%ny,:,nnew,2) - 35.0_rp))
  if (tmax > 1.0e-8_rp .or. smax > 1.0e-8_rp) then
    write(*,*) 'uniform tracer drifted', tmax, smax
    nfail = nfail + 1
  endif
  if (maxval(abs(zeta(1:decomp%nx,1:decomp%ny,knew))) > 1.0e-10_rp) then
    write(*,*) 'rest zeta drifted', maxval(abs(zeta(1:decomp%nx,1:decomp%ny,knew)))
    nfail = nfail + 1
  endif

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'test_dyn_core FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_dyn_core OK'
end program test_dyn_core
