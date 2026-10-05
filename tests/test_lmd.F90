program test_lmd
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use geom, only: set_depth
  use rho_eos_mod, only: rho_eos
  use lmd_mod, only: lmd_vmix
  use formulas
  implicit none
  integer(ip) :: nfail, k, mid
  real(rp) :: akv_max, akt_max, hbl0, hblw, akv_mid, swr0

  nfail = 0
  call init_decomp(12_ip, 12_ip, 8_ip, 2_ip, 1_ip, 1_ip, 0_ip, .true., .true.)
  call init_time_state(8_ip, 20_ip, 50.0_rp, 1_ip, 2_ip)
  call alloc_state(2_ip)
  call init_ideal_box(4000.0_rp, 12000.0_rp, 1.0e-4_rp)
  knew = 1
  iic = 1
  nstp = 1
  nrhs = 1
  call set_depth()
  call rho_eos(1_ip)
  call lmd_vmix(1_ip)

  if (.not. all(Akv(1:decomp%nx, 1:decomp%ny, :) >= 0.0_rp)) then
    write(*,*) 'Akv has NaN or is negative'
    nfail = nfail + 1
  endif
  if (minval(Akv(1:decomp%nx, 1:decomp%ny, :)) < 0.0_rp) then
    write(*,*) 'Akv negative', minval(Akv(1:decomp%nx, 1:decomp%ny, :))
    nfail = nfail + 1
  endif
  if (minval(Akt(1:decomp%nx, 1:decomp%ny, :, :)) < 0.0_rp) then
    write(*,*) 'Akt negative', minval(Akt(1:decomp%nx, 1:decomp%ny, :, :))
    nfail = nfail + 1
  endif
  akv_max = maxval(Akv(1:decomp%nx, 1:decomp%ny, :))
  akt_max = maxval(Akt(1:decomp%nx, 1:decomp%ny, :, itemp))
  if (akv_max <= 0.0_rp .or. akv_max > 1.0_rp) then
    write(*,*) 'rest Akv out of range', akv_max
    nfail = nfail + 1
  endif
  if (akt_max <= 0.0_rp .or. akt_max > 1.0_rp) then
    write(*,*) 'rest Akt out of range', akt_max
    nfail = nfail + 1
  endif

  hbl0 = hbls(1, 1)
  if (hbl0 <= 0.0_rp .or. hbl0 > 4000.0_rp) then
    write(*,*) 'rest hbls out of range', hbl0
    nfail = nfail + 1
  endif
  if (hbbl(1, 1) <= 0.0_rp .or. hbbl(1, 1) > 4000.0_rp) then
    write(*,*) 'rest hbbl out of range', hbbl(1, 1)
    nfail = nfail + 1
  endif
  if (maxval(abs(hbls(1:decomp%nx, 1:decomp%ny) - hbl0)) > 1.0e-10_rp) then
    write(*,*) 'hbls not uniform at rest'
    nfail = nfail + 1
  endif

  if (maxval(abs(swr_frac(1:decomp%nx, 1:decomp%ny, decomp%nz) - 1.0_rp)) > 1.0e-15_rp) then
    write(*,*) 'swr_frac surface should be 1'
    nfail = nfail + 1
  endif
  swr0 = swr_frac(1, 1, 0)
  if (swr0 < 0.0_rp .or. swr0 >= 1.0_rp) then
    write(*,*) 'swr_frac bottom', swr0
    nfail = nfail + 1
  endif
  if (swr_frac(1, 1, decomp%nz - 1) >= 1.0_rp) then
    write(*,*) 'swr_frac should decay below the surface', swr_frac(1, 1, decomp%nz - 1)
    nfail = nfail + 1
  endif
  if (maxval(abs(ghat(1:decomp%nx, 1:decomp%ny, :))) > 0.0_rp) then
    write(*,*) 'ghat should be 0 with no buoyancy flux'
    nfail = nfail + 1
  endif

  ! Stratified rest: interior Ri is large, so mid-column Kv drops toward nuwm.
  do k = 1, decomp%nz
    t(:, :, k, :, 1) = 10.0_rp + 10.0_rp * real(k, rp) / real(decomp%nz, rp)
  enddo
  hbls = 0.0_rp
  hbbl = 0.0_rp
  call rho_eos(1_ip)
  call lmd_vmix(1_ip)
  mid = decomp%nz / 2
  akv_mid = Akv(1, 1, mid)
  if (akv_mid > 5.0e-3_rp) then
    write(*,*) 'stratified interior Akv still in shear-max regime', akv_mid
    nfail = nfail + 1
  endif
  if (akv_mid < 0.0_rp) nfail = nfail + 1

  ! Wind deepens hbl relative to the stratified rest case.
  hbl0 = hbls(1, 1)
  sustr = 1.0e-4_rp
  svstr = 0.0_rp
  hbls = 0.0_rp
  hbbl = 0.0_rp
  call lmd_vmix(1_ip)
  hblw = hbls(1, 1)
  if (hblw <= hbl0) then
    write(*,*) 'wind should deepen hbl', hbl0, hblw
    nfail = nfail + 1
  endif
  if (minval(Akv(1:decomp%nx, 1:decomp%ny, :)) < 0.0_rp) nfail = nfail + 1

  call free_state()
  if (nfail /= 0) then
    write(*,*) 'test_lmd FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_lmd OK'
end program test_lmd
