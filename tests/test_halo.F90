program test_halo
  use kinds
  use decomp_mod
  use gpu_state
  use halo_mod
  implicit none
  integer(ip) :: i, j, nfail
  real(rp) :: expect

  call init_decomp(16_ip, 12_ip, 8_ip, 2_ip, 1_ip, 1_ip, 0_ip, .true., .true.)
  call alloc_state(2_ip)

  if (decomp%nbr_w /= NBR_SELF .or. decomp%nbr_e /= NBR_SELF) then
    error stop 'test_halo: expected NBR_SELF on a 1-tile periodic domain'
  endif

  do j = decomp%js, decomp%je
    do i = decomp%is, decomp%ie
      ! Unique interior tag; halo starts dirty.
      if (i >= 1 .and. i <= decomp%nx .and. j >= 1 .and. j <= decomp%ny) then
        zeta(i, j, 1) = real(1000 * i + j, rp)
      else
        zeta(i, j, 1) = -999.0_rp
      endif
    enddo
  enddo

  call halo_fill_2d_s(zeta, STAGGER_RHO, 1_ip)
  nfail = 0

  do j = 1, decomp%ny
    do i = 1, decomp%halo
      expect = real(1000 * i + j, rp)
      if (abs(zeta(decomp%nx + i, j, 1) - expect) > 0.0_rp) nfail = nfail + 1
      expect = real(1000 * (decomp%nx - decomp%halo + i) + j, rp)
      if (abs(zeta(i - decomp%halo, j, 1) - expect) > 0.0_rp) nfail = nfail + 1
    enddo
  enddo
  do i = 1 - decomp%halo, decomp%nx + decomp%halo
    do j = 1, decomp%halo
      ! NS after EW: i may already be wrapped
      if (abs(zeta(i, decomp%ny + j, 1) - zeta(i, j, 1)) > 0.0_rp) nfail = nfail + 1
      if (abs(zeta(i, j - decomp%halo, 1) - zeta(i, decomp%ny - decomp%halo + j, 1)) > 0.0_rp) &
           nfail = nfail + 1
    enddo
  enddo

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'test_halo FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_halo OK'
end program test_halo
