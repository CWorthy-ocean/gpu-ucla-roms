program test_prsgrd
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use geom, only: set_depth
  use prsgrd_mod, only: prsgrd
  implicit none
  integer(ip) :: i, j, k, nfail
  real(rp) :: H0, dx, rmax

  H0 = 4000.0_rp
  dx = 12000.0_rp
  nfail = 0

  call init_decomp(12_ip, 12_ip, 16_ip, 2_ip, 1_ip, 1_ip, 0_ip, .true., .true.)
  call init_time_state(16_ip, 20_ip, 50.0_rp, 1_ip, 2_ip)
  call alloc_state(2_ip)
  call init_ideal_box(H0, dx, 1.0e-4_rp)
  knew = 1
  iic = 0
  call set_depth()

  ! Uniform zero density: no 3-D PGF.
  rho1 = 0.0_rp
  qp1 = 0.0_rp
  call prsgrd()
  if (maxval(abs(ru(1:decomp%nx+1,1:decomp%ny,:))) > 1.0e-10_rp) then
    write(*,*) 'zero rho ru', maxval(abs(ru))
    nfail = nfail + 1
  endif
  if (maxval(abs(rv(1:decomp%nx,1:decomp%ny+1,:))) > 1.0e-10_rp) then
    write(*,*) 'zero rho rv', maxval(abs(rv))
    nfail = nfail + 1
  endif

  ! Horizontally uniform stratification on a flat bottom: Jacobian is 0.
  do k = 1, decomp%nz
    rho1(:, :, k) = 0.02_rp * real(decomp%nz - k, rp)
  enddo
  call prsgrd()
  rmax = max(maxval(abs(ru(1:decomp%nx+1,1:decomp%ny,:))), &
             maxval(abs(rv(1:decomp%nx,1:decomp%ny+1,:))))
  if (rmax > 1.0e-8_rp) then
    write(*,*) 'flat+strat ru/rv', rmax
    nfail = nfail + 1
  endif

  ! Sloping topography, rho1 = alpha * z_r (function of z only).
  ! The scheme should keep the residual far below a naive sigma PGF.
  call free_state()
  call alloc_state(2_ip)
  call init_ideal_box(H0, dx, 1.0e-4_rp)
  do j = decomp%js, decomp%je
    do i = decomp%is, decomp%ie
      h(i, j) = 2000.0_rp + 1500.0_rp * real(i, rp) / real(decomp%nx, rp)
    enddo
  enddo
  call set_depth()
  rho1 = 0.0002_rp * z_r
  qp1 = 0.0_rp
  call prsgrd()
  rmax = max(maxval(abs(ru(1:decomp%nx+1,1:decomp%ny,:))), &
             maxval(abs(rv(1:decomp%nx,1:decomp%ny+1,:))))
  if (rmax > 1.0e-2_rp) then
    write(*,*) 'sloping z-only rho residual too large', rmax
    nfail = nfail + 1
  endif

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'test_prsgrd FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_prsgrd OK'
end program test_prsgrd
