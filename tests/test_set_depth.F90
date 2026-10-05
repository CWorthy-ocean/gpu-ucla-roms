program test_set_depth
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use geom
  implicit none
  integer(ip) :: i, j, k, nfail
  real(rp) :: H0, dx, vol, umis, vmis

  H0 = 4000.0_rp
  dx = 1000.0_rp
  call init_decomp(8_ip, 8_ip, 16_ip, 2_ip, 1_ip, 1_ip, 0_ip, .true., .true.)
  call init_time_state(16_ip, 60_ip, 100.0_rp, 1_ip, 2_ip)
  call alloc_state(2_ip)
  call init_ideal_box(H0, dx, 1.0e-4_rp)

  knew = 1
  iic = 0
  call set_depth()
  nfail = 0

  do j = 1, decomp%ny
    do i = 1, decomp%nx
      if (abs(z_w(i, j, 0) + H0) > 1.0e-10_rp) nfail = nfail + 1
      if (abs(z_w(i, j, decomp%nz) - zeta(i, j, knew)) > 1.0e-10_rp) nfail = nfail + 1
      do k = 1, decomp%nz
        if (Hz(i, j, k) <= 0.0_rp) nfail = nfail + 1
        if (abs(Hz(i, j, k) - (z_w(i, j, k) - z_w(i, j, k - 1))) > 1.0e-12_rp) nfail = nfail + 1
      enddo
      if (abs(sum(Hz(i, j, :)) - (H0 + zeta(i, j, knew))) > 1.0e-8_rp) nfail = nfail + 1
    enddo
  enddo

  call set_HUV()
  call omega()
  call check_mode_coupling(umis, vmis, vol)
  if (abs(vol - H0 * dx * dx * real(decomp%nx * decomp%ny, rp)) / vol > 1.0e-10_rp) then
    write(*,*) 'volume mismatch', vol, H0 * dx * dx * real(decomp%nx * decomp%ny, rp)
    nfail = nfail + 1
  endif
  if (maxval(abs(We)) > 1.0e-12_rp) then
    write(*,*) 'We should be ~0 at rest, max=', maxval(abs(We))
    nfail = nfail + 1
  endif

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'test_set_depth FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_set_depth OK'
end program test_set_depth
