#include "mem_attr.inc"
program test_marbl_interior
  ! Column-by-column validation of the ported MARBL interior path against
  ! the unchanged MARBL code (host), on full MiniPac water columns from the
  ! UP3 restart. Two comparisons:
  !   host port  the device routines compiled for the host, run column by
  !              column on the login node (algorithm check, no GPU needed)
  !   device     the K1/K2 kernels (needs a GPU)
  ! Shortwave and dust are deterministic synthetic fields (some columns
  ! dark). Both paths see bit-identical inputs.
  !
  ! usage: test_marbl_interior <restart.nc> <grid.nc> [marbl_in] [ncol_side] [host|gpu]
  use kinds
  use decomp_mod
#ifdef USE_CUDA
  use arch_mod, only: init_arch, sync_arch
#endif
  use time_state, only: set_scoord, Cs_w, Cs_r, theta_s, theta_b, hc
  use formulas, only: sm09_z
  use marbl_host
  use marbl_sflux_mod, only: alloc_marbl_sflux, sf_dust
  use marbl_interior_mod
  use netcdf
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  character(len=512) :: rst, grd, cfg, arg, mode
  integer :: nside, ncol, nxi, neta, nz, nrec, ncid, varid, ierr, m, n, i, j, k, kk, ii, jj, nwet, stride, cnt
  integer(ip) :: ntr, nzi
  integer, allocatable :: maski(:,:), ci(:), cj(:)
  real(rp), allocatable :: g3(:,:,:), g2(:,:), hgrd(:,:), zeta(:,:)
  real(rp), allocatable :: col_tr(:,:,:), col_t(:,:), col_s(:,:), col_ph(:,:), col_pha(:,:)   ! ROMS k order
  real(rp), allocatable :: col_zw(:,:), col_zr(:,:), col_hz(:,:)
  real(rp), allocatable :: ref_tend(:,:,:), ref_diag(:,:,:), ref_ph(:,:), ref_pha(:,:)
  real(rp), allocatable :: prt_tend(:,:,:), prt_diag(:,:,:), prt_ph(:,:), prt_pha(:,:)
  real(rp), allocatable :: dz(:), zw(:), zt(:), pot(:), sal(:), prs(:), trm(:,:), ph(:), pha(:), tnd(:,:), dgm(:,:)
  real(pp), WORK_ATTR allocatable :: tr(:,:,:,:,:)
  real(rp), WORK_ATTR allocatable :: ts(:,:,:,:,:), mask(:,:), z_r3(:,:,:), z_w3(:,:,:), hz3(:,:,:)
  real(rp) :: qsw, dust, t0, t1, thost, tport, tk1, tk2, ds, s_w, s_r, dtm
  integer :: nerr_host, nerr_port, nerr_gpu
  integer(ip) :: is, ie, js, je
  logical :: do_gpu

  call get_command_argument(1, rst)
  call get_command_argument(2, grd)
  cfg = 'marbl_in'
  if (command_argument_count() >= 3) call get_command_argument(3, cfg)
  nside = 48
  if (command_argument_count() >= 4) then
    call get_command_argument(4, arg); read(arg, *) nside
  endif
  mode = 'gpu'
  if (command_argument_count() >= 5) call get_command_argument(5, mode)
  do_gpu = (trim(mode) == 'gpu')
  ncol = nside * nside
  dtm = 3600.0_rp

  ! ---- restart dimensions
  call chk(nf90_open(trim(rst), NF90_NOWRITE, ncid), 'open rst')
  call dimlen(ncid, 'xi_rho', nxi); call dimlen(ncid, 'eta_rho', neta)
  call dimlen(ncid, 's_rho', nz); call dimlen(ncid, 'time', nrec)
  write(*,'(a,4i6)') 'restart dims xi,eta,s_rho,time =', nxi, neta, nz, nrec
  nzi = nz

  call init_decomp(int(nside, ip), int(nside, ip), nzi, 2_ip, 1_ip, 1_ip, 0_ip, .false., .false.)
#ifdef USE_CUDA
  if (do_gpu) call init_arch()
#else
  do_gpu = .false.
#endif
  is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
  theta_s = 6.0_rp; theta_b = 6.0_rp; hc = 250.0_rp
  call set_scoord(nzi)

  call marbl_host_init(nzi, trim(cfg), .true.)
  ntr = nt_marbl
  call set_marbl_par()
  write(*,'(a,i0,a,i0,a,i0,a,es12.4)') 'par: ntr ', pr%ntr, ' nauto ', pr%nauto, ' nzoo ', pr%nzoo, &
       ' mass_to_vol ', pr%mass_to_vol
  call alloc_marbl_sflux(ntr)
  call alloc_marbl_interior(nzi, ntr, .true.)

  ! ---- grid: h, mask
  allocate(maski(nxi, neta), hgrd(nxi, neta), zeta(nxi, neta), g2(nxi, neta), g3(nxi, neta, nz))
  call chk(nf90_open(trim(grd), NF90_NOWRITE, varid), 'open grid')
  call chk(nf90_inq_varid(varid, 'mask_rho', m), 'inq mask'); call chk(nf90_get_var(varid, m, maski), 'get mask')
  call chk(nf90_inq_varid(varid, 'h', m), 'inq h');            call chk(nf90_get_var(varid, m, hgrd), 'get h')
  call chk(nf90_close(varid), 'close grid')
  call chk(nf90_inq_varid(ncid, 'zeta', varid), 'inq zeta')
  call chk(nf90_get_var(ncid, varid, zeta, start=[1, 1, nrec], count=[nxi, neta, 1]), 'get zeta')

  ! ---- choose columns: every stride-th wet interior point
  nwet = 0
  do j = 2, neta - 1
    do i = 2, nxi - 1
      if (maski(i, j) == 1) nwet = nwet + 1
    enddo
  enddo
  stride = max(1, nwet / ncol)
  allocate(ci(ncol), cj(ncol))
  cnt = 0; n = 0
  do j = 2, neta - 1
    do i = 2, nxi - 1
      if (maski(i, j) == 1) then
        n = n + 1
        if (mod(n - 1, stride) == 0 .and. cnt < ncol) then
          cnt = cnt + 1; ci(cnt) = i; cj(cnt) = j
        endif
      endif
    enddo
  enddo
  ncol = cnt
  write(*,'(a,i0,a,i0,a,i0)') 'wet points ', nwet, '  stride ', stride, '  columns used ', ncol

  ! ---- column data from the restart (ROMS k order, 1 = bottom)
  allocate(col_tr(ntr, nz, ncol), col_t(nz, ncol), col_s(nz, ncol), col_ph(nz, ncol), col_pha(nz, ncol))
  allocate(col_zw(0:nz, ncol), col_zr(nz, ncol), col_hz(nz, ncol))
  do m = 1, ntr
    call chk(nf90_inq_varid(ncid, trim(marbl_tracer_names(m)), varid), 'inq '//trim(marbl_tracer_names(m)))
    call chk(nf90_get_var(ncid, varid, g3, start=[1, 1, 1, nrec], count=[nxi, neta, nz, 1]), 'get tracer')
    do n = 1, ncol
      col_tr(m, :, n) = g3(ci(n), cj(n), :)
    enddo
  enddo
  call chk(nf90_inq_varid(ncid, 'temp', varid), 'inq temp')
  call chk(nf90_get_var(ncid, varid, g3, start=[1, 1, 1, nrec], count=[nxi, neta, nz, 1]), 'get temp')
  do n = 1, ncol
    col_t(:, n) = g3(ci(n), cj(n), :)
  enddo
  call chk(nf90_inq_varid(ncid, 'salt', varid), 'inq salt')
  call chk(nf90_get_var(ncid, varid, g3, start=[1, 1, 1, nrec], count=[nxi, neta, nz, 1]), 'get salt')
  do n = 1, ncol
    col_s(:, n) = g3(ci(n), cj(n), :)
  enddo
  call chk(nf90_inq_varid(ncid, 'MARBL_PH_3D', varid), 'inq ph3d')
  call chk(nf90_get_var(ncid, varid, g3, start=[1, 1, 1, nrec], count=[nxi, neta, nz, 1]), 'get ph3d')
  do n = 1, ncol
    col_ph(:, n) = g3(ci(n), cj(n), :)
  enddo
  call chk(nf90_inq_varid(ncid, 'MARBL_PH_3D_ALT_CO2', varid), 'inq ph3da')
  call chk(nf90_get_var(ncid, varid, g3, start=[1, 1, 1, nrec], count=[nxi, neta, nz, 1]), 'get ph3da')
  do n = 1, ncol
    col_pha(:, n) = g3(ci(n), cj(n), :)
  enddo
  call chk(nf90_close(ncid), 'close rst')
  deallocate(g3)

  ! ---- geometry (SM09 transform, as the model)
  ds = 1.0_rp / real(nz, rp)
  do n = 1, ncol
    i = ci(n); j = cj(n)
    do k = 0, nz
      s_w = ds * real(k - nz, rp)
      col_zw(k, n) = sm09_z(zeta(i, j), hgrd(i, j), hc, s_w, Cs_w(k))
    enddo
    do k = 1, nz
      s_r = ds * (real(k - nz, rp) - 0.5_rp)
      col_zr(k, n) = sm09_z(zeta(i, j), hgrd(i, j), hc, s_r, Cs_r(k))
      col_hz(k, n) = col_zw(k, n) - col_zw(k - 1, n)
    enddo
  enddo

  ! ---- device-side arrays (column n -> local (ii,jj)); kind pp for tracers
  allocate(tr(is:ie, js:je, nz, 1, ntr), ts(is:ie, js:je, nz, 1, 2), mask(is:ie, js:je))
  allocate(z_r3(is:ie, js:je, nz), z_w3(is:ie, js:je, 0:nz), hz3(is:ie, js:je, nz))
  tr = 0.0_pp; ts = 0.0_rp; mask = 0.0_rp; z_r3 = 0.0_rp; z_w3 = 0.0_rp; hz3 = 1.0_rp
  call seed_device()

  allocate(ref_tend(ntr, nz, ncol), ref_diag(ND_IT, nz, ncol), ref_ph(nz, ncol), ref_pha(nz, ncol))
  allocate(prt_tend(ntr, nz, ncol), prt_diag(ND_IT, nz, ncol), prt_ph(nz, ncol), prt_pha(nz, ncol))
  allocate(dz(nz), zw(nz), zt(nz), pot(nz), sal(nz), prs(nz), trm(ntr, nz), ph(nz), pha(nz), tnd(ntr, nz), dgm(ND_IT, nz))

  ! ---- host MARBL reference
  nerr_host = 0
  call cpu_time(t0)
  do n = 1, ncol
    call column_inputs(n)
    ph = ref_ph(:, n); pha = ref_pha(:, n)      ! seeded by column_inputs
    call marbl_host_interior_column(nzi, dz, zw, zt, pot, sal, prs, qsw, dust, trm, ph, pha, tnd, dgm, ierr)
    if (ierr /= 0) nerr_host = nerr_host + 1
    ref_tend(:, :, n) = tnd; ref_diag(:, :, n) = dgm; ref_ph(:, n) = ph; ref_pha(:, n) = pha
  enddo
  call cpu_time(t1)
  thost = t1 - t0

#ifndef USE_CUDA
  ! ---- host run of the port
  nerr_port = 0
  call cpu_time(t0)
  do n = 1, ncol
    call column_inputs(n)
    ph = prt_ph(:, n); pha = prt_pha(:, n)
    call marbl_interior_column_host(nzi, dz, zw, zt, pot, sal, prs, qsw, dust, trm, ph, pha, tnd, dgm, ierr)
    if (ierr /= 0) nerr_port = nerr_port + 1
    prt_tend(:, :, n) = tnd; prt_diag(:, :, n) = dgm; prt_ph(:, n) = ph; prt_pha(:, n) = pha
  enddo
  call cpu_time(t1)
  tport = t1 - t0
  write(*,'(/a,i0,a,f9.3,a,f9.3,a,i0,a,i0)') 'columns ', ncol, '  MARBL host ', thost, ' s   port host ', tport, &
       ' s   solver errors MARBL/port ', nerr_host, '/', nerr_port
  write(*,'(/a)') '=== host port vs MARBL'
  call compare(prt_tend, prt_diag, prt_ph, prt_pha)
  write(*,'(/a)') 'column 1 detail (MARBL k, input pH, MARBL pH, port pH, MARBL pH diag, port pH diag, MARBL O2prod, port O2prod)'
  do k = 1, nz, 11
    kk = nz - k + 1
    write(*,'(i4,3f12.7,2f12.7,2es14.6)') k, col_ph(kk, 1), ref_ph(k, 1), prt_ph(k, 1), ref_diag(14, k, 1), &
         prt_diag(14, k, 1), ref_diag(7, k, 1), prt_diag(7, k, 1)
  enddo
#endif

#ifdef USE_CUDA
  if (do_gpu) then
    call sync_arch()
    call cpu_time(t0)
    call marbl_carbonate_compute(ntr, nzi, 1_ip, 1_ip, tr, 2_ip, ts, z_r3, mask)
    call sync_arch()
    call cpu_time(t1)
    tk1 = t1 - t0
    call cpu_time(t0)
    call marbl_interior_compute(ntr, nzi, 1_ip, 1_ip, tr, 2_ip, ts, z_r3, z_w3, hz3, mask, sf_dust, dtm, .true.)
    call sync_arch()
    call cpu_time(t1)
    tk2 = t1 - t0
    nerr_gpu = 0
    do n = 1, ncol
      ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
      nerr_gpu = nerr_gpu + it_err(ii, jj)
    enddo
    write(*,'(/a,f9.5,a,f9.5,a,i0)') 'device: carbonate kernel ', tk1, ' s   interior kernel ', tk2, &
         ' s   solver failures ', nerr_gpu
    ! gather device results in column layout (MARBL k order)
    do n = 1, ncol
      ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
      do k = 1, nz
        kk = nz - k + 1
        prt_tend(:, k, n) = it_tend(ii, jj, kk, 1:ntr)
        prt_diag(:, k, n) = it_diag(ii, jj, kk, :)
        prt_ph(k, n) = it_ph(ii, jj, kk)
        prt_pha(k, n) = it_pha(ii, jj, kk)
      enddo
    enddo
    write(*,'(/a)') '=== device vs MARBL'
    call compare(prt_tend, prt_diag, prt_ph, prt_pha)
    ! timing of a second launch (warm)
    call seed_device()
    call sync_arch()
    call cpu_time(t0)
    call marbl_carbonate_compute(ntr, nzi, 1_ip, 1_ip, tr, 2_ip, ts, z_r3, mask)
    call sync_arch()
    call cpu_time(t1)
    tk1 = t1 - t0
    call cpu_time(t0)
    call marbl_interior_compute(ntr, nzi, 1_ip, 1_ip, tr, 2_ip, ts, z_r3, z_w3, hz3, mask, sf_dust, dtm, .false.)
    call sync_arch()
    call cpu_time(t1)
    tk2 = t1 - t0
    write(*,'(a,f9.5,a,f9.5,a)') 'device (warm, no diagnostics): carbonate ', tk1, ' s   interior ', tk2, ' s'
  endif
#endif
  write(*,*) 'test_marbl_interior done'

contains

  subroutine chk(st, where)
    integer, intent(in) :: st
    character(len=*), intent(in) :: where
    if (st /= nf90_noerr) then
      write(*,*) 'netcdf error at ', where, ': ', trim(nf90_strerror(st))
      error stop 1
    endif
  end subroutine chk

  subroutine dimlen(nc, name, len)
    integer, intent(in) :: nc
    character(len=*), intent(in) :: name
    integer, intent(out) :: len
    integer :: did
    call chk(nf90_inq_dimid(nc, name, did), 'dimid '//name)
    call chk(nf90_inquire_dimension(nc, did, len=len), 'dimlen '//name)
  end subroutine dimlen

  ! deterministic synthetic forcing for column n
  subroutine synth(n, qsw, dust)
    integer, intent(in) :: n
    real(rp), intent(out) :: qsw, dust
    real(rp) :: r1, r2
    r1 = fract(sin(real(n, rp) * 12.9898_rp) * 43758.5453_rp)
    r2 = fract(sin(real(n, rp) * 19.9898_rp) * 43758.5453_rp)
    qsw = 300.0_rp * r1
    if (mod(n, 7) == 0) qsw = 0.0_rp          ! night
    dust = 1.0e-9_rp * r2
  end subroutine synth

  real(rp) function fract(x)
    real(rp), intent(in) :: x
    fract = x - floor(x)
  end function fract

  ! MARBL-order inputs for column n, exactly what the device reads (kind pp
  ! tracers converted back to rp). Seeds the pH columns of both result sets.
  subroutine column_inputs(n)
    integer, intent(in) :: n
    integer :: k, kk, m
    ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
    do k = 1, nz
      kk = nz - k + 1
      dz(k) = col_hz(kk, n)
      zw(k) = -col_zw(kk - 1, n)
      zt(k) = -col_zr(kk, n)
      pot(k) = col_t(kk, n)
      sal(k) = col_s(kk, n)
      prs(k) = -col_zr(kk, n) * 0.1_rp
      do m = 1, ntr
        trm(m, k) = real(tr(ii, jj, kk, 1, m), rp)
      enddo
      ref_ph(k, n) = col_ph(kk, n); ref_pha(k, n) = col_pha(kk, n)
      prt_ph(k, n) = col_ph(kk, n); prt_pha(k, n) = col_pha(kk, n)
    enddo
    call synth(n, qsw, dust)
  end subroutine column_inputs

  subroutine seed_device()
    integer :: n, k, m
    do n = 1, ncol
      ii = mod(n - 1, nside) + 1; jj = (n - 1) / nside + 1
      do k = 1, nz
        do m = 1, ntr
          tr(ii, jj, k, 1, m) = real(col_tr(m, k, n), pp)
        enddo
        ts(ii, jj, k, 1, itemp) = col_t(k, n)
        ts(ii, jj, k, 1, isalt) = col_s(k, n)
        z_r3(ii, jj, k) = col_zr(k, n)
        hz3(ii, jj, k) = col_hz(k, n)
        it_ph(ii, jj, k) = col_ph(k, n)
        it_pha(ii, jj, k) = col_pha(k, n)
      enddo
      do k = 0, nz
        z_w3(ii, jj, k) = col_zw(k, n)
      enddo
      mask(ii, jj) = 1.0_rp
      call synth(n, qsw, dust)
      it_qsw(ii, jj) = qsw
      sf_dust(ii, jj) = dust
    enddo
    it_err = 0
  end subroutine seed_device

  subroutine compare(ptend, pdiag, pph, ppha)
    real(rp), intent(in) :: ptend(:,:,:), pdiag(:,:,:), pph(:,:), ppha(:,:)
    character(len=16), parameter :: dname(ND_IT) = [character(len=16) :: 'PAR_avg', 'NITRIF', 'DENITRIF', &
         'photoC_TOT', 'POC_FLUX_IN', 'POC_REMIN_DIC', 'O2_PRODUCTION', 'O2_CONSUMPTION', 'Fe_scavenge', &
         'insitu_temp', 'CO3', 'H2CO3', 'co3_sat_calc', 'pH_3D', 'DOC_prod', 'POC_PROD']
    integer :: m, nbad
    nbad = 0
    write(*,'(a)') 'quantity              max|ref|      max|diff|   diff/scale  n(>1e-13)  n(>1e-10)  n(>1e-6)'
    do m = 1, ntr
      call report('tend '//marbl_tracer_names(m), ref_tend(m, :, :), ptend(m, :, :), nbad)
    enddo
    do m = 1, ND_IT
      call report('diag '//dname(m), ref_diag(m, :, :), pdiag(m, :, :), nbad)
    enddo
    call report('pH_3D saved', ref_ph, pph, nbad)
    call report('pH_3D_ALT saved', ref_pha, ppha, nbad)
    write(*,'(a,i0)') 'quantities with diff/scale > 1e-10 somewhere: ', nbad
  end subroutine compare

  subroutine report(label, ref, val, nbad)
    character(len=*), intent(in) :: label
    real(rp), intent(in) :: ref(:,:), val(:,:)
    integer, intent(inout) :: nbad
    real(rp) :: amax, dmax, d
    integer :: n13, n10, n6, a, b
    amax = maxval(abs(ref))
    dmax = 0.0_rp; n13 = 0; n10 = 0; n6 = 0
    do b = 1, size(ref, 2)
      do a = 1, size(ref, 1)
        d = abs(val(a, b) - ref(a, b))
        dmax = max(dmax, d)
        if (amax > 0.0_rp) then
          if (d > 1.0e-13_rp * amax) n13 = n13 + 1
          if (d > 1.0e-10_rp * amax) n10 = n10 + 1
          if (d > 1.0e-6_rp * amax) n6 = n6 + 1
        else if (d > 0.0_rp) then
          n13 = n13 + 1; n10 = n10 + 1; n6 = n6 + 1
        endif
      enddo
    enddo
    if (n10 > 0) nbad = nbad + 1
    if (amax > 0.0_rp) then
      write(*,'(a20,3es14.4,3i10)') adjustl(label), amax, dmax, dmax / amax, n13, n10, n6
    else
      write(*,'(a20,2es14.4,a14,3i10)') adjustl(label), amax, dmax, '   (ref zero)', n13, n10, n6
    endif
  end subroutine report

end program test_marbl_interior
