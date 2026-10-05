program test_mpi_halo
  ! Two ranks, periodic in x. Pack stays local; deliver is MPI.
  use kinds
  use comm_mod
  use decomp_mod
  use gpu_state
  use halo_mod
  implicit none
  integer(ip) :: i, j, nfail, gi
  real(rp) :: expect

  call init_comm()
  if (comm_size() /= 2) then
    write(*,*) 'test_mpi_halo needs exactly 2 MPI ranks, got', comm_size()
    call finalize_comm()
    error stop 1
  endif

  call init_decomp(16_ip, 8_ip, 4_ip, 2_ip, 2_ip, 1_ip, comm_rank(), .true., .false.)
  call alloc_state(2_ip)

  if (decomp%nbr_w < 0 .or. decomp%nbr_e < 0) then
    write(*,*) 'rank', comm_rank(), 'expected real MPI neighbors', &
         decomp%nbr_w, decomp%nbr_e
    error stop 1
  endif

  do j = decomp%js, decomp%je
    do i = decomp%is, decomp%ie
      if (i >= 1 .and. i <= decomp%nx .and. j >= 1 .and. j <= decomp%ny) then
        gi = decomp%i0 + i - 1
        zeta(i, j, 1) = real(1000 * gi + j, rp)
      else
        zeta(i, j, 1) = -999.0_rp
      endif
    enddo
  enddo

  call halo_fill_2d_s(zeta, STAGGER_RHO, 1_ip)
  nfail = 0

  do j = 1, decomp%ny
    do i = 1, decomp%halo
      gi = decomp%i0 + decomp%nx + i - 1
      if (gi > decomp%gnx) gi = gi - decomp%gnx
      expect = real(1000 * gi + j, rp)
      if (abs(zeta(decomp%nx + i, j, 1) - expect) > 0.0_rp) nfail = nfail + 1
      gi = decomp%i0 - decomp%halo + i - 1
      if (gi < 1) gi = gi + decomp%gnx
      expect = real(1000 * gi + j, rp)
      if (abs(zeta(i - decomp%halo, j, 1) - expect) > 0.0_rp) nfail = nfail + 1
    enddo
  enddo

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'rank', comm_rank(), 'test_mpi_halo FAILED, nfail=', nfail
    call finalize_comm()
    error stop 1
  endif
  if (comm_rank() == 0) write(*,*) 'test_mpi_halo OK'
  call finalize_comm()
end program test_mpi_halo
