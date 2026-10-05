program test_step2d
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use halo_mod
  use geom, only: set_depth
  use step2d_mod, only: step2d_with_halo
  implicit none
  integer(ip) :: i, j, n, nfail
  real(rp) :: H0, dx, vol0, vol1, ru0
  real(rp), allocatable :: rufrc_after1(:,:), rvfrc_after1(:,:)

  H0 = 4000.0_rp
  dx = 12000.0_rp
  nfail = 0

  call init_decomp(16_ip, 16_ip, 8_ip, 2_ip, 1_ip, 1_ip, 0_ip, .true., .true.)
  call init_time_state(8_ip, 20_ip, 50.0_rp, 1_ip, 2_ip)
  call alloc_state(2_ip)
  call init_ideal_box(H0, dx, 1.0e-4_rp)

  ! --- rest state: stay at rest through nfast ---
  knew = 1
  iic = 1
  call halo_fill_2d_s(zeta, STAGGER_RHO, 1_ip)
  call set_depth()
  do n = 1, nfast
    iif = n
    call cycle_fast_index()
    call step2d_with_halo(iif)
  enddo
  if (maxval(abs(zeta(1:decomp%nx,1:decomp%ny,knew))) > 1.0e-12_rp) then
    write(*,*) 'rest zeta drifted', maxval(abs(zeta(1:decomp%nx,1:decomp%ny,knew)))
    nfail = nfail + 1
  endif
  if (maxval(abs(ubar(1:decomp%nx+1,1:decomp%ny,knew))) > 1.0e-12_rp) then
    write(*,*) 'rest ubar drifted', maxval(abs(ubar(1:decomp%nx+1,1:decomp%ny,knew)))
    nfail = nfail + 1
  endif

  ! --- zeta bump: volume conserved; rufrc frozen after iif=1 ---
  call free_state()
  call alloc_state(2_ip)
  call init_ideal_box(H0, dx, 1.0e-4_rp)
  do j = 1, decomp%ny
    do i = 1, decomp%nx
      zeta(i, j, 1) = 0.05_rp * exp(-((real(i, rp) - 8.5_rp)**2 + &
                                      (real(j, rp) - 8.5_rp)**2) / 8.0_rp)
    enddo
  enddo
  knew = 1
  iic = 1
  rufrc = 0.0_rp
  rvfrc = 0.0_rp
  call halo_fill_2d_s(zeta, STAGGER_RHO, 1_ip)
  call set_depth()
  vol0 = sum((h(1:decomp%nx,1:decomp%ny) + zeta(1:decomp%nx,1:decomp%ny,1)) / &
             (pm(1:decomp%nx,1:decomp%ny) * pn(1:decomp%nx,1:decomp%ny)))

  allocate(rufrc_after1(decomp%is:decomp%ie, decomp%js:decomp%je))
  allocate(rvfrc_after1(decomp%is:decomp%ie, decomp%js:decomp%je))
  do n = 1, nfast
    iif = n
    call cycle_fast_index()
    call step2d_with_halo(iif)
    if (n == 1) then
      rufrc_after1 = rufrc
      rvfrc_after1 = rvfrc
      ru0 = maxval(abs(rufrc(1:decomp%nx+1,1:decomp%ny)))
      if (ru0 <= 0.0_rp) then
        write(*,*) 'iif=1 should strip a nonzero PGF into rufrc'
        nfail = nfail + 1
      endif
    endif
  enddo
  if (maxval(abs(rufrc - rufrc_after1)) > 0.0_rp .or. &
      maxval(abs(rvfrc - rvfrc_after1)) > 0.0_rp) then
    write(*,*) 'rufrc/rvfrc changed after iif=1'
    nfail = nfail + 1
  endif
  vol1 = sum((h(1:decomp%nx,1:decomp%ny) + zeta(1:decomp%nx,1:decomp%ny,knew)) / &
             (pm(1:decomp%nx,1:decomp%ny) * pn(1:decomp%nx,1:decomp%ny)))
  if (abs(vol1 - vol0) / vol0 > 1.0e-10_rp) then
    write(*,*) 'volume not conserved', vol0, vol1, abs(vol1-vol0)/vol0
    nfail = nfail + 1
  endif

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'test_step2d FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_step2d OK'
end program test_step2d
