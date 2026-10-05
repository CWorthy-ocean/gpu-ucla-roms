#include "mem_attr.inc"
program test_marbl_sflux
  ! Column-by-column validation of the device MARBL surface-flux kernel
  ! against the unchanged MARBL code running on the host.
  !
  ! Surface tracers, temperature, salinity and the saved surface pH come
  ! from a MiniPac restart (real ocean states); wind, deposition and xco2
  ! are deterministic synthetic fields spanning realistic ranges. Both
  ! paths see bit-identical inputs. Two scenarios: (A) previous pH known
  ! (narrow bracket), (B) no previous pH (wide bracket, growth path).
  !
  ! usage: test_marbl_sflux <restart.nc> <grid.nc> [marbl_in] [ncol_side]
  use kinds
  use decomp_mod
  use arch_mod, only: init_arch, sync_arch
  use marbl_host
  use marbl_sflux_mod
  use netcdf
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  character(len=512) :: rst, grd, cfg, arg
  integer :: nside, ncol, nxi, neta, nzr, nrec, ncid, varid, ierr, m, n, i, j, ii, jj, nwet, stride, cnt
  integer(ip) :: ntr
  integer, allocatable :: maski(:,:), ci(:), cj(:)
  real(rp), allocatable :: g2(:,:), surf(:,:,:), tsurf(:,:), ssurf(:,:), ph0(:,:), pha0(:,:)
  real(pp), WORK_ATTR allocatable :: tr(:,:,:,:,:)
  real(rp), WORK_ATTR allocatable :: ts(:,:,:,:,:), mask(:,:)
  real(rp), allocatable :: ref_flux(:,:), ref_diag(:,:), ref_ph(:), ref_pha(:), trs(:), fl(:), dg(:)
  real(rp) :: u10, dust, iron, nox, nhy, xco2, ph, pha, t0, t1, thost, tgpu
  integer :: scen, nerr_host, nerr_gpu, ncount
  integer(ip) :: is, ie, js, je

  call get_command_argument(1, rst)
  call get_command_argument(2, grd)
  cfg = 'marbl_in'
  if (command_argument_count() >= 3) call get_command_argument(3, cfg)
  nside = 64
  if (command_argument_count() >= 4) then
    call get_command_argument(4, arg); read(arg, *) nside
  endif
  ncol = nside * nside

  call init_decomp(int(nside, ip), int(nside, ip), 1_ip, 2_ip, 1_ip, 1_ip, 0_ip, .false., .false.)
  call init_arch()
  is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je

  call marbl_host_init(100_ip, trim(cfg), .true.)
  ntr = nt_marbl
  call alloc_marbl_sflux(ntr)
  allocate(tr(is:ie, js:je, 1, 3, ntr), ts(is:ie, js:je, 1, 3, 2), mask(is:ie, js:je))
  tr = 0.0_pp; ts = 0.0_rp; mask = 0.0_rp

  ! ---- read the restart surface level
  call chk(nf90_open(trim(rst), NF90_NOWRITE, ncid), 'open rst')
  call dimlen(ncid, 'xi_rho', nxi); call dimlen(ncid, 'eta_rho', neta)
  call dimlen(ncid, 's_rho', nzr); call dimlen(ncid, 'time', nrec)
  write(*,'(a,4i6)') 'restart dims xi,eta,s_rho,time =', nxi, neta, nzr, nrec
  allocate(g2(nxi, neta), surf(nxi, neta, ntr), tsurf(nxi, neta), ssurf(nxi, neta), ph0(nxi, neta), pha0(nxi, neta))
  do m = 1, ntr
    call chk(nf90_inq_varid(ncid, trim(marbl_tracer_names(m)), varid), 'inq '//trim(marbl_tracer_names(m)))
    call chk(nf90_get_var(ncid, varid, g2, start=[1, 1, nzr, nrec], count=[nxi, neta, 1, 1]), 'get tracer')
    surf(:, :, m) = g2
  enddo
  call chk(nf90_inq_varid(ncid, 'temp', varid), 'inq temp')
  call chk(nf90_get_var(ncid, varid, tsurf, start=[1, 1, nzr, nrec], count=[nxi, neta, 1, 1]), 'get temp')
  call chk(nf90_inq_varid(ncid, 'salt', varid), 'inq salt')
  call chk(nf90_get_var(ncid, varid, ssurf, start=[1, 1, nzr, nrec], count=[nxi, neta, 1, 1]), 'get salt')
  call chk(nf90_inq_varid(ncid, 'MARBL_PH_SURF', varid), 'inq ph')
  call chk(nf90_get_var(ncid, varid, ph0, start=[1, 1, nrec], count=[nxi, neta, 1]), 'get ph')
  call chk(nf90_inq_varid(ncid, 'MARBL_PH_SURF_ALT_CO2', varid), 'inq pha')
  call chk(nf90_get_var(ncid, varid, pha0, start=[1, 1, nrec], count=[nxi, neta, 1]), 'get pha')
  call chk(nf90_close(ncid), 'close rst')
  allocate(maski(nxi, neta))
  call chk(nf90_open(trim(grd), NF90_NOWRITE, ncid), 'open grid')
  call chk(nf90_inq_varid(ncid, 'mask_rho', varid), 'inq mask')
  call chk(nf90_get_var(ncid, varid, maski), 'get mask')
  call chk(nf90_close(ncid), 'close grid')

  ! ---- choose columns: every stride-th wet point, interior of the grid
  nwet = 0
  do j = 2, neta - 1
    do i = 2, nxi - 1
      if (maski(i, j) == 1 .and. ph0(i, j) > 0.0_rp) nwet = nwet + 1
    enddo
  enddo
  stride = max(1, nwet / ncol)
  allocate(ci(ncol), cj(ncol))
  cnt = 0; n = 0
  do j = 2, neta - 1
    do i = 2, nxi - 1
      if (maski(i, j) == 1 .and. ph0(i, j) > 0.0_rp) then
        n = n + 1
        if (mod(n - 1, stride) == 0 .and. cnt < ncol) then
          cnt = cnt + 1; ci(cnt) = i; cj(cnt) = j
        endif
      endif
    enddo
  enddo
  ncol = cnt
  write(*,'(a,i0,a,i0,a,i0)') 'wet points ', nwet, '  stride ', stride, '  columns used ', ncol

  ! ---- fill the device-side arrays (column n -> local (ii,jj))
  do n = 1, ncol
    ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
    i = ci(n); j = cj(n)
    do m = 1, ntr
      tr(ii, jj, 1, 1, m) = real(surf(i, j, m), pp)
    enddo
    ts(ii, jj, 1, 1, itemp) = tsurf(i, j)
    ts(ii, jj, 1, 1, isalt) = ssurf(i, j)
    mask(ii, jj) = 1.0_rp
    call synth(n, u10, dust, iron, nox, nhy, xco2)
    sf_u10sqr(ii, jj) = u10; sf_dust(ii, jj) = dust; sf_iron(ii, jj) = iron
    sf_nox(ii, jj) = nox; sf_nhy(ii, jj) = nhy; sf_xco2(ii, jj) = xco2
    sf_xco2_alt(ii, jj) = 284.7_rp
  enddo

  allocate(ref_flux(ntr, ncol), ref_diag(MSF_NDIAG, ncol), ref_ph(ncol), ref_pha(ncol), trs(ntr), fl(ntr), dg(MSF_NDIAG))

  do scen = 1, 2
    if (scen == 1) then
      write(*,'(/a)') '=== scenario A: previous pH from restart'
    else
      write(*,'(/a)') '=== scenario B: no previous pH (wide bracket)'
    endif
    ! host reference
    nerr_host = 0
    call cpu_time(t0)
    do n = 1, ncol
      ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
      i = ci(n); j = cj(n)
      do m = 1, ntr
        trs(m) = real(tr(ii, jj, 1, 1, m), rp)     ! exactly what the kernel reads
      enddo
      if (scen == 1) then
        ph = ph0(i, j); pha = pha0(i, j)
      else
        ph = 0.0_rp; pha = 0.0_rp
      endif
      call synth(n, u10, dust, iron, nox, nhy, xco2)
      call marbl_host_sflux_column(trs, u10, tsurf(i, j), ssurf(i, j), xco2, 284.7_rp, dust, iron, &
                                   nox, nhy, ph, pha, fl, dg, ierr)
      if (ierr /= 0) nerr_host = nerr_host + 1
      ref_flux(:, n) = fl; ref_diag(:, n) = dg; ref_ph(n) = ph; ref_pha(n) = pha
    enddo
    call cpu_time(t1)
    thost = t1 - t0

    ! device
    do n = 1, ncol
      ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
      if (scen == 1) then
        sf_ph(ii, jj) = ph0(ci(n), cj(n)); sf_ph_alt(ii, jj) = pha0(ci(n), cj(n))
      else
        sf_ph(ii, jj) = 0.0_rp; sf_ph_alt(ii, jj) = 0.0_rp
      endif
    enddo
    call marbl_sflux_compute(msf, ntr, 1_ip, 1_ip, 1_ip, tr, 2_ip, ts, mask)   ! warm-up
    call sync_arch()
    ! re-seed pH (the kernel updated it) and time a second launch
    do n = 1, ncol
      ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
      if (scen == 1) then
        sf_ph(ii, jj) = ph0(ci(n), cj(n)); sf_ph_alt(ii, jj) = pha0(ci(n), cj(n))
      else
        sf_ph(ii, jj) = 0.0_rp; sf_ph_alt(ii, jj) = 0.0_rp
      endif
    enddo
    call sync_arch()
    call cpu_time(t0)
    call marbl_sflux_compute(msf, ntr, 1_ip, 1_ip, 1_ip, tr, 2_ip, ts, mask)
    call sync_arch()
    call cpu_time(t1)
    tgpu = t1 - t0
    nerr_gpu = 0
    do n = 1, ncol
      ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
      if (sf_err(ii, jj) /= 0) nerr_gpu = nerr_gpu + 1
    enddo
    write(*,'(a,i0,a,f9.4,a,f9.5,a,i0,a,i0)') 'columns ', ncol, '  host ', thost, ' s   gpu ', tgpu, &
         ' s   solver errors host/gpu ', nerr_host, '/', nerr_gpu

    ! compare
    write(*,'(/a)') 'quantity            max|ref|        max|diff|     max rel      n(rel>1e-13)  n(rel>1e-10)'
    ncount = 0
    do m = 1, ntr
      if (maxval(abs(ref_flux(m, :))) == 0.0_rp) cycle
      call report('flux '//marbl_tracer_names(m), ref_flux(m, :), m, 0)
    enddo
    do m = 1, MSF_NDIAG
      if (maxval(abs(ref_diag(m, :))) == 0.0_rp) cycle
      call report('diag '//msf_diag_name(m), ref_diag(m, :), m, 1)
    enddo
    call report('saved pH', ref_ph, 0, 2)
    call report('saved pH alt', ref_pha, 0, 3)
    write(*,'(a,i0)') 'quantities with rel diff > 1e-10 somewhere: ', ncount
  enddo
  write(*,*) 'test_marbl_sflux done'

contains

  subroutine chk(st, where)
    integer, intent(in) :: st
    character(len=*), intent(in) :: where
    if (st /= nf90_noerr) then
      write(*,*) 'netcdf error at ', where, ': ', trim(nf90_strerror(st))
      error stop 1
    endif
  end subroutine chk

  subroutine dimlen(nc, name, n)
    integer, intent(in) :: nc
    character(len=*), intent(in) :: name
    integer, intent(out) :: n
    integer :: did
    call chk(nf90_inq_dimid(nc, name, did), 'dimid '//name)
    call chk(nf90_inquire_dimension(nc, did, len=n), 'dimlen '//name)
  end subroutine dimlen

  ! deterministic pseudo-random forcing for column n
  subroutine synth(n, u10, dust, iron, nox, nhy, xco2)
    integer, intent(in) :: n
    real(rp), intent(out) :: u10, dust, iron, nox, nhy, xco2
    real(rp) :: r(6)
    integer :: k
    do k = 1, 6
      r(k) = fract(sin(real(n, rp) * (12.9898_rp + 7.0_rp * k)) * 43758.5453_rp)
    enddo
    u10  = 0.5_rp + 150.0_rp * r(1)        ! m2/s2  (0.7 .. 12 m/s)
    dust = 1.0e-9_rp * r(2)                ! kg/m2/s
    iron = 1.0e-8_rp * r(3)                ! mmol/m2/s
    nox  = 2.0e-10_rp * r(4)               ! mmol/m2/s
    nhy  = 2.0e-10_rp * r(5)               ! mmol/m2/s
    xco2 = 380.0_rp + 40.0_rp * r(6)       ! ppm
  end subroutine synth

  real(rp) function fract(x)
    real(rp), intent(in) :: x
    fract = x - floor(x)
  end function fract

  subroutine report(label, ref, m, kind)
    character(len=*), intent(in) :: label
    real(rp), intent(in) :: ref(:)
    integer, intent(in) :: m, kind
    real(rp) :: dmax, rmax, amax, d, r, v
    integer :: n, n13, n10, ii, jj
    dmax = 0.0_rp; rmax = 0.0_rp; amax = 0.0_rp; n13 = 0; n10 = 0
    do n = 1, ncol
      ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
      select case (kind)
      case (0); v = sf_flux(ii, jj, m)
      case (1); v = sf_diag(ii, jj, m)
      case (2); v = sf_ph(ii, jj)
      case default; v = sf_ph_alt(ii, jj)
      end select
      d = abs(v - ref(n))
      r = d / max(abs(ref(n)), 1.0e-300_rp)
      amax = max(amax, abs(ref(n))); dmax = max(dmax, d); rmax = max(rmax, r)
      if (r > 1.0e-13_rp) n13 = n13 + 1
      if (r > 1.0e-10_rp) n10 = n10 + 1
    enddo
    if (n10 > 0) ncount = ncount + 1
    write(*,'(a18,3es14.4,2i14)') adjustl(label), amax, dmax, rmax, n13, n10
  end subroutine report

end program test_marbl_sflux
