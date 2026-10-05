program test_rho_eos
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use geom, only: set_depth
  use rho_eos_mod, only: rho_eos
  use formulas
  implicit none
  integer(ip) :: nfail
  real(rp) :: r1, q1, rmax

  nfail = 0
  call init_decomp(8_ip, 8_ip, 8_ip, 2_ip, 1_ip, 1_ip, 0_ip, .true., .true.)
  call init_time_state(8_ip, 20_ip, 50.0_rp, 1_ip, 2_ip)
  call alloc_state(2_ip)
  call init_ideal_box(4000.0_rp, 12000.0_rp, 1.0e-4_rp)
  knew = 1
  iic = 0
  call set_depth()
  call rho_eos(1_ip)

  call jackett_split_eos(15.0_rp, 35.0_rp, r1, q1)
  if (maxval(abs(rho1(1:decomp%nx,1:decomp%ny,:) - r1)) > 1.0e-10_rp) then
    write(*,*) 'uniform T,S rho1', maxval(abs(rho1 - r1))
    nfail = nfail + 1
  endif
  if (maxval(abs(qp1(1:decomp%nx,1:decomp%ny,:) - q1)) > 1.0e-10_rp) then
    write(*,*) 'uniform T,S qp1', maxval(abs(qp1 - q1))
    nfail = nfail + 1
  endif
  rmax = maxval(abs(rhoA(1:decomp%nx,1:decomp%ny) - rhoA(1,1)))
  if (rmax > 1.0e-12_rp) then
    write(*,*) 'rhoA not horizontally uniform', rmax
    nfail = nfail + 1
  endif
  if (abs(rhoA(1, 1) - r1 / rho0) > 1.0e-6_rp) then
    ! Uniform T,S on a flat column: rhoA ~ in-situ anomaly / rho0, not exactly rho1/rho0
    ! because of the qp1*|z| term. Just require a finite, O(1e-3) value.
    if (abs(rhoA(1, 1)) > 0.1_rp) then
      write(*,*) 'rhoA out of range', rhoA(1, 1)
      nfail = nfail + 1
    endif
  endif

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'test_rho_eos FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_rho_eos OK'
end program test_rho_eos
