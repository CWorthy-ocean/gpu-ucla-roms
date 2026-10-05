program test_advect
  use kinds
  use decomp_mod
  use gpu_state
  use formulas
  use advect
  implicit none
  integer(ip) :: i, j, k, nfail, tidx, itrc
  integer(ip) :: nx, ny, nz
  real(rp) :: tconst, uflux, wflux, expect, tplus, tminus
  real(rp) :: curv_l, curv_r, dxi_l, dxi_c, dxi_r

  nfail = 0
  tidx = 1
  itrc = 1
  tconst = 7.0_rp
  uflux = 3.0_rp
  wflux = 2.0_rp

  ! --- periodic box, uniform tracer: interpolant is the constant ---
  call init_decomp(12_ip, 10_ip, 8_ip, 2_ip, 1_ip, 1_ip, 0_ip, .true., .true.)
  call alloc_state(2_ip)
  call init_ideal_box(4000.0_rp, 12000.0_rp, 1.0e-4_rp)
  nx = decomp%nx
  ny = decomp%ny
  nz = decomp%nz
  t(:,:,:,tidx,itrc) = tconst
  FlxU = uflux
  FlxV = -uflux
  We = wflux
  Hz = 10.0_rp
  call tracer_advect_fluxes(tidx, itrc, .true.)
  if (maxval(abs(FX(1:nx+1,1:ny,1:nz) - tconst * uflux)) > 1.0e-12_rp) then
    write(*,*) 'uniform FX', maxval(abs(FX(1:nx+1,1:ny,1:nz) - tconst * uflux))
    nfail = nfail + 1
  endif
  if (maxval(abs(FE(1:nx,1:ny+1,1:nz) + tconst * uflux)) > 1.0e-12_rp) then
    write(*,*) 'uniform FE', maxval(abs(FE(1:nx,1:ny+1,1:nz) + tconst * uflux))
    nfail = nfail + 1
  endif
  if (maxval(abs(Fz(1:nx,1:ny,1:nz-1) - tconst * wflux)) > 1.0e-12_rp) then
    write(*,*) 'uniform C4 Fz', maxval(abs(Fz(1:nx,1:ny,1:nz-1) - tconst * wflux))
    nfail = nfail + 1
  endif
  if (maxval(abs(Fz(1:nx,1:ny,0))) > 0.0_rp .or. maxval(abs(Fz(1:nx,1:ny,nz))) > 0.0_rp) then
    write(*,*) 'C4 top/bottom Fz not closed'
    nfail = nfail + 1
  endif
  call vert_u3_fluxes(tidx, itrc)
  if (maxval(abs(Fz(1:nx,1:ny,1:nz-1) - tconst * wflux)) > 1.0e-12_rp) then
    write(*,*) 'uniform U3 Fz', maxval(abs(Fz(1:nx,1:ny,1:nz-1) - tconst * wflux))
    nfail = nfail + 1
  endif
  call free_state()

  ! --- non-periodic, linear in x/y: curvature is zero, including physical edges ---
  call init_decomp(12_ip, 10_ip, 8_ip, 2_ip, 1_ip, 1_ip, 0_ip, .false., .false.)
  call alloc_state(2_ip)
  call init_ideal_box(4000.0_rp, 12000.0_rp, 1.0e-4_rp)
  nx = decomp%nx
  ny = decomp%ny
  nz = decomp%nz
  do i = decomp%is, decomp%ie
    t(i, :, :, tidx, itrc) = real(i, rp)
  enddo
  FlxU = uflux
  FlxV = 0.0_rp
  We = 0.0_rp
  Hz = 10.0_rp
  call horiz_up3_fluxes(tidx, itrc)
  do k = 1, nz
    do j = 1, ny
      do i = 1, nx + 1
        expect = 0.5_rp * (real(i - 1, rp) + real(i, rp)) * uflux
        if (abs(FX(i, j, k) - expect) > 1.0e-12_rp) then
          write(*,*) 'linear-x FX', i, j, k, FX(i, j, k), expect
          nfail = nfail + 1
        endif
      enddo
    enddo
  enddo

  do j = decomp%js, decomp%je
    t(:, j, :, tidx, itrc) = real(j, rp)
  enddo
  FlxU = 0.0_rp
  FlxV = uflux
  call horiz_up3_fluxes(tidx, itrc)
  do k = 1, nz
    do j = 1, ny + 1
      do i = 1, nx
        expect = 0.5_rp * (real(j - 1, rp) + real(j, rp)) * uflux
        if (abs(FE(i, j, k) - expect) > 1.0e-12_rp) then
          write(*,*) 'linear-y FE', i, j, k, FE(i, j, k), expect
          nfail = nfail + 1
        endif
      enddo
    enddo
  enddo

  ! Linear in z, uniform Hz: interior faces recover the interpolant.
  ! Bottom/top use dlt(0)=0 and surface extrapolation, so they do not.
  do k = 1, nz
    t(:, :, k, tidx, itrc) = real(k, rp)
  enddo
  We = wflux
  Hz = 10.0_rp
  call vert_c4_fluxes(tidx, itrc)
  do k = 2, nz - 2
    expect = (real(k, rp) + 0.5_rp) * wflux
    if (maxval(abs(Fz(1:nx, 1:ny, k) - expect)) > 1.0e-12_rp) then
      write(*,*) 'linear-z C4', k, maxval(abs(Fz(1:nx,1:ny,k) - expect))
      nfail = nfail + 1
    endif
  enddo
  call vert_u3_fluxes(tidx, itrc)
  do k = 2, nz - 2
    expect = (real(k, rp) + 0.5_rp) * wflux
    if (maxval(abs(Fz(1:nx, 1:ny, k) - expect)) > 1.0e-12_rp) then
      write(*,*) 'linear-z U3', k, maxval(abs(Fz(1:nx,1:ny,k) - expect))
      nfail = nfail + 1
    endif
  enddo

  ! Cubic column: C4 is the mean of the two faces; U3 is upwind.
  do k = 1, nz
    t(:, :, k, tidx, itrc) = real(k, rp)**3
  enddo
  We = wflux
  call vert_c4_fluxes(tidx, itrc)
  call thickness_aware_faces(4.0_rp**3, 5.0_rp**3, 6.0_rp**3, 7.0_rp**3, &
                             10.0_rp, 10.0_rp, 10.0_rp, 10.0_rp, &
                             .false., .false., tplus, tminus)
  expect = 0.5_rp * (tplus + tminus) * wflux
  if (abs(Fz(1, 1, 5) - expect) > 1.0e-12_rp) then
    write(*,*) 'cubic C4', Fz(1, 1, 5), expect
    nfail = nfail + 1
  endif
  call vert_u3_fluxes(tidx, itrc)
  expect = tplus * wflux
  if (abs(Fz(1, 1, 5) - expect) > 1.0e-12_rp) then
    write(*,*) 'cubic U3', Fz(1, 1, 5), expect
    nfail = nfail + 1
  endif
  if (abs(tplus - tminus) < 1.0e-12_rp) then
    write(*,*) 'cubic faces should differ'
    nfail = nfail + 1
  endif

  ! LAND_CURV: a land u-face zeros the 3rd-order term on both sides.
  do i = decomp%is, decomp%ie
    t(i, :, :, tidx, itrc) = real(i, rp)**2
  enddo
  umask(5, :) = 0.0_rp
  FlxU = uflux
  FlxV = 0.0_rp
  call horiz_up3_fluxes(tidx, itrc)
  i = 6
  j = 1
  dxi_l = (t(i - 1, j, 1, tidx, itrc) - t(i - 2, j, 1, tidx, itrc)) * umask(i - 1, j)
  dxi_c = (t(i, j, 1, tidx, itrc) - t(i - 1, j, 1, tidx, itrc)) * umask(i, j)
  dxi_r = (t(i + 1, j, 1, tidx, itrc) - t(i, j, 1, tidx, itrc)) * umask(i + 1, j)
  curv_l = (dxi_c - dxi_l) * umask(i - 1, j) * umask(i, j)
  curv_r = (dxi_r - dxi_c) * umask(i, j) * umask(i + 1, j)
  if (abs(curv_l) > 0.0_rp) then
    write(*,*) 'LAND_CURV left curv should be 0', curv_l
    nfail = nfail + 1
  endif
  expect = up3_flux(t(i - 1, j, 1, tidx, itrc), t(i, j, 1, tidx, itrc), &
                    curv_l, curv_r, uflux)
  if (abs(FX(i, j, 1) - expect) > 1.0e-12_rp) then
    write(*,*) 'LAND_CURV FX', FX(i, j, 1), expect
    nfail = nfail + 1
  endif
  ! Interior wet face away from land and the physical rim.
  i = 9
  dxi_l = t(i - 1, j, 1, tidx, itrc) - t(i - 2, j, 1, tidx, itrc)
  dxi_c = t(i, j, 1, tidx, itrc) - t(i - 1, j, 1, tidx, itrc)
  dxi_r = t(i + 1, j, 1, tidx, itrc) - t(i, j, 1, tidx, itrc)
  expect = up3_flux(t(i - 1, j, 1, tidx, itrc), t(i, j, 1, tidx, itrc), &
                    dxi_c - dxi_l, dxi_r - dxi_c, uflux)
  if (abs(FX(i, j, 1) - expect) > 1.0e-12_rp) then
    write(*,*) 'quadratic UP3 FX', FX(i, j, 1), expect
    nfail = nfail + 1
  endif
  call free_state()

  if (nfail /= 0) then
    write(*,*) 'test_advect FAILED, nfail=', nfail
    error stop 1
  endif
  write(*,*) 'test_advect OK'
end program test_advect
