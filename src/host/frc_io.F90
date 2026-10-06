#include "mem_attr.inc"
module frc_io
  ! Flux-file or COARE 3.0 bulk surface forcing; two-level time
  ! interpolation. Records are read on the host into managed buffers;
  ! the interpolation onto the grid and the bulk fluxes run as device
  ! kernels (the 2x-coarse upsampling path stays on the host). No rivers / BGC.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use nc_io
  use bulk_coare, only: calc_bulk_coare
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  private

  integer, parameter :: max_frc = 64
  character(len=512), public :: case_frcfiles(max_frc)
  integer,            public :: case_nfrc = 0
  logical,            public :: interp_flux_frc = .false.

  real(rp), parameter :: Cp = 3985.0_rp
  real(rp), parameter :: cmday2ms = 0.01_rp / 86400.0_rp

  type :: series2d
    character(len=64)  :: vname = ''
    character(len=64)  :: tname = ''
    character(len=1)   :: gtype = 'r'
    integer :: ntot = 0, it1 = 1, it2 = 2, n1 = 0, n2 = 0, time_axis = 3
    integer :: fi0 = 1, fj0 = 1   ! file start of this tile's window (on-grid fields)
    logical :: ready = .false.
    ! climatology: cycle_length attribute of the time axis (<= 0: none);
    ! t1e/t2e are the shifted record times of the current bracket
    real(rp) :: cycle = -1.0_rp, t1e = 0.0_rp, t2e = 0.0_rp
    integer, allocatable :: ifile(:), irec(:)
    character(len=512), allocatable :: paths(:)
    real(rp), allocatable :: times(:)
    real(rp), WORK_ATTR allocatable :: buf(:,:,:)   ! spatial (n1,n2,2)
  end type series2d

  type(series2d) :: s_sustr, s_svstr, s_shflx, s_swflx, s_swrad
  type(series2d) :: s_uwnd, s_vwnd, s_tair, s_qair, s_prate, s_lwrad, s_bswrad
  ! MARBL surface forcing (dust/iron/nox/nhy climatologies, atmospheric xCO2)
  type(series2d) :: s_dust, s_iron, s_nox, s_nhy, s_xco2, s_xco2a
  logical, public :: bgc_frc_on = .false.
  real(rp), WORK_ATTR allocatable :: uwnd(:,:), vwnd(:,:), qair(:,:), prate(:,:), lwrad_in(:,:)

  public :: read_forcing_namelist
  public :: init_forcing
  public :: set_forces
  public :: init_bgc_forcing
  public :: set_bgc_forces
  public :: cyc_bracket
  public :: uwnd, vwnd
  public :: write_toy_flux
  public :: write_toy_bulk

contains

  subroutine read_forcing_namelist(unit, nml_dir)
    integer,          intent(in) :: unit
    character(len=*), intent(in) :: nml_dir
    integer :: ios, i
    character(len=512) :: frcfiles(max_frc), frcfile(max_frc)
    logical :: interp_bulk_frc, check_bulk_frc_units
    namelist /forcing_files/ frcfiles, frcfile
    namelist /surf_frc_settings/ interp_flux_frc, interp_bulk_frc, check_bulk_frc_units
    namelist /flux_frc_settings/ interp_flux_frc
    namelist /bulk_frc_settings/ interp_bulk_frc, check_bulk_frc_units

    frcfiles = ''; frcfile = ''
    interp_bulk_frc = .false.; check_bulk_frc_units = .false.
    interp_flux_frc = .false.
    rewind(unit)
    read(unit, nml=forcing_files, iostat=ios)
    rewind(unit)
    read(unit, nml=surf_frc_settings, iostat=ios)
    rewind(unit)
    read(unit, nml=flux_frc_settings, iostat=ios)
    rewind(unit)
    read(unit, nml=bulk_frc_settings, iostat=ios)

    case_nfrc = 0
    do i = 1, max_frc
      if (len_trim(frcfiles(i)) > 0) then
        case_nfrc = case_nfrc + 1
        case_frcfiles(case_nfrc) = resolve_one(nml_dir, frcfiles(i))
      elseif (len_trim(frcfile(i)) > 0) then
        case_nfrc = case_nfrc + 1
        case_frcfiles(case_nfrc) = resolve_one(nml_dir, frcfile(i))
      endif
    enddo
  end subroutine read_forcing_namelist

  subroutine init_forcing()
    integer :: nmiss
    have_flux = .false.
    have_bulk = .false.
    if (case_nfrc < 1) return
    call bind_series(s_uwnd, 'uwnd', 'time', 'r')
    call bind_series(s_vwnd, 'vwnd', 'time', 'r')
    call bind_series(s_tair, 'Tair', 'time', 'r')
    call bind_series(s_qair, 'qair', 'time', 'r')
    call bind_series(s_prate, 'rain', 'time', 'r')
    call bind_series(s_lwrad, 'lwrad', 'time', 'r')
    call bind_series(s_bswrad, 'swrad', 'time', 'r')
    have_bulk = s_uwnd%ready .and. s_vwnd%ready .and. s_tair%ready .and. s_qair%ready
    if (have_bulk) then
      if (.not. allocated(uwnd)) then
        allocate(uwnd(decomp%is:decomp%ie, decomp%js:decomp%je))
        allocate(vwnd(decomp%is:decomp%ie, decomp%js:decomp%je))
        allocate(qair(decomp%is:decomp%ie, decomp%js:decomp%je))
        allocate(prate(decomp%is:decomp%ie, decomp%js:decomp%je))
        allocate(lwrad_in(decomp%is:decomp%ie, decomp%js:decomp%je))
      endif
      uwnd = 0.0_rp; vwnd = 0.0_rp; tair = 0.0_rp
      qair = 0.0_rp; prate = 0.0_rp; lwrad_in = 0.0_rp
      write(*,'(a,i0,a,2es14.6)') 'bulk/COARE forcing, Tair records = ', s_tair%ntot, &
           '  days ', s_tair%times(1), s_tair%times(s_tair%ntot)
      return
    endif
    nmiss = 0
    if (s_uwnd%ready .or. s_vwnd%ready .or. s_tair%ready .or. s_qair%ready) then
      if (.not. s_uwnd%ready) nmiss = nmiss + 1
      if (.not. s_vwnd%ready) nmiss = nmiss + 1
      if (.not. s_tair%ready) nmiss = nmiss + 1
      if (.not. s_qair%ready) nmiss = nmiss + 1
      write(*,'(a,i0,a)') 'ERROR: incomplete bulk forcing (', nmiss, &
           ' of uwnd/vwnd/Tair/qair missing); flux fallback is disabled'
      error stop 'init_forcing: bulk is the default; will not use flux forcing'
    endif
    call bind_series(s_sustr, 'sustr', 'sms_time', 'u')
    call bind_series(s_svstr, 'svstr', 'sms_time', 'v')
    call bind_series(s_shflx, 'shflux', 'shf_time', 'r')
    call bind_series(s_swflx, 'swflux', 'swf_time', 'r')
    call bind_series(s_swrad, 'swrad', 'srf_time', 'r')
    have_flux = s_sustr%ready .and. s_svstr%ready .and. s_shflx%ready
    if (have_flux) write(*,'(a,i0,a)') 'flux forcing (no bulk vars): ', case_nfrc, ' files'
  end subroutine init_forcing

  subroutine set_forces()
    integer(ip) :: i, j
    real(rp) :: tneed
    logical, save :: logged_bulk = .false.
    if (have_bulk) then
      tneed = frc_modtime()
      call fill_r(s_uwnd, tneed, uwnd)
      call fill_r(s_vwnd, tneed, vwnd)
      call fill_r(s_tair, tneed, tair)
      call fill_r(s_qair, tneed, qair)
      if (s_prate%ready) call fill_r(s_prate, tneed, prate)
      if (s_lwrad%ready) call fill_r(s_lwrad, tneed, lwrad_in)
      if (s_bswrad%ready) then
        call fill_r(s_bswrad, tneed, srflx)
        call divide_2d(srflx, rho0_case * Cp)
      else
        call set_2d(srflx, 0.0_rp)
      endif
      if (.not. logged_bulk) then
        call sync_arch()
        write(*,'(a,es14.6,a,2es14.6,a,2i8)') 'bulk interp tdays=', tneed, &
             '  Tair times=', s_tair%times(s_tair%it1), s_tair%times(s_tair%it2), &
             '  recs=', s_tair%it1, s_tair%it2
        write(*,'(a,2es14.6,a,2es14.6)') 'bulk Tair range=', &
             minval(tair), maxval(tair), '  uwnd range=', minval(uwnd), maxval(uwnd)
        logged_bulk = .true.
      endif
      call calc_bulk_coare(uwnd, vwnd, tair, qair, prate, lwrad_in)
      return
    endif
    if (.not. have_flux) return
    tneed = frc_modtime()
    call fill_u(s_sustr, tneed, sustr)
    call fill_v(s_svstr, tneed, svstr)
    call fill_r(s_shflx, tneed, stflx(:,:,itemp))
    sustr = sustr / rho0_case
    svstr = svstr / rho0_case
    do j = decomp%js, decomp%je
      do i = decomp%is, decomp%ie
        stflx(i, j, itemp) = stflx(i, j, itemp) / (rho0_case * Cp)
        stflx(i, j, isalt) = 0.0_rp
      enddo
    enddo
    if (s_swflx%ready) then
      call fill_r(s_swflx, tneed, swflx)
      swflx = -swflx * cmday2ms
    endif
    if (s_swrad%ready) then
      call fill_r(s_swrad, tneed, srflx)
      srflx = srflx / (rho0_case * Cp)
    endif
  end subroutine set_forces

  subroutine bind_series(s, vname, tname, gtype)
    type(series2d),   intent(inout) :: s
    character(len=*), intent(in)    :: vname, tname, gtype
    integer :: i, ncid, nxi, neta, nrec, tax, k, ntot
    real(rp), allocatable :: tloc(:)
    real(rp) :: cyc
    logical :: lcyc
    s%vname = trim(vname)
    s%tname = trim(tname)
    s%gtype = gtype
    s%ready = .false.
    s%ntot = 0
    ntot = 0
    do i = 1, case_nfrc
      call nc_open_read(case_frcfiles(i), ncid)
      if (nc_has_var(ncid, vname)) then
        call nc_forcing_layout(ncid, vname, tname, nxi, neta, nrec, tax)
        ntot = ntot + max(nrec, 1)
        if (s%n1 == 0) then
          s%n1 = nxi; s%n2 = neta; s%time_axis = tax
          ! a field on the global model grid: this tile reads only its window
          if ((gtype == 'r' .and. nxi == decomp%gnx + 2 .and. neta == decomp%gny + 2) .or. &
              (gtype == 'u' .and. nxi == decomp%gnx + 1 .and. neta == decomp%gny) .or. &
              (gtype == 'v' .and. nxi == decomp%gnx .and. neta == decomp%gny + 1) .or. &
              (gtype == 'r' .and. nxi == decomp%gnx .and. neta == decomp%gny)) then
            s%n1 = nxi - int(decomp%gnx) + int(decomp%nx)
            s%n2 = neta - int(decomp%gny) + int(decomp%ny)
            s%fi0 = int(decomp%i0); s%fj0 = int(decomp%j0)
          endif
        endif
      endif
      call nc_close(ncid)
    enddo
    if (ntot < 1) return
    allocate(s%times(ntot), s%ifile(ntot), s%irec(ntot), s%paths(ntot))
    s%ntot = 0
    do i = 1, case_nfrc
      call nc_open_read(case_frcfiles(i), ncid)
      if (nc_has_var(ncid, vname)) then
        call nc_forcing_layout(ncid, vname, tname, nxi, neta, nrec, tax)
        nrec = max(nrec, 1)
        allocate(tloc(nrec))
        if (nc_has_var(ncid, tname)) then
          call nc_get_1d(ncid, tname, tloc)
        else
          tloc = 0.0_rp
        endif
        do k = 1, nrec
          s%ntot = s%ntot + 1
          s%times(s%ntot) = tloc(k)
          s%ifile(s%ntot) = i
          s%irec(s%ntot) = k
          s%paths(s%ntot) = case_frcfiles(i)
        enddo
        deallocate(tloc)
      endif
      call nc_close(ncid)
    enddo
    if (s%ntot < 1) return
    call sort_catalog(s)
    ! climatology? cycle_length on the time axis of the first file with the variable
    s%cycle = -1.0_rp
    do i = 1, case_nfrc
      call nc_open_read(case_frcfiles(i), ncid)
      if (nc_has_var(ncid, vname) .and. nc_has_var(ncid, tname)) then
        call nc_get_att_real(ncid, tname, 'cycle_length', cyc, lcyc)
        if (lcyc .and. cyc > 0.0_rp) s%cycle = cyc
        call nc_close(ncid)
        exit
      endif
      call nc_close(ncid)
    enddo
    if (s%cycle > 0.0_rp) write(*,'(3a,f10.3)') 'forcing ', trim(vname), ' is cyclic, cycle_length =', s%cycle
    s%ready = .true.
    allocate(s%buf(s%n1, s%n2, 2))
    s%buf = 0.0_rp
    s%it1 = 1
    s%it2 = min(2, s%ntot)
    call read_catalog_rec(s, s%it1, 1)
    call read_catalog_rec(s, s%it2, 2)
  end subroutine bind_series

  subroutine sort_catalog(s)
    type(series2d), intent(inout) :: s
    integer :: i, j, ifile, irec
    real(rp) :: tswap
    character(len=512) :: pswap
    do i = 2, s%ntot
      tswap = s%times(i)
      ifile = s%ifile(i)
      irec = s%irec(i)
      pswap = s%paths(i)
      j = i - 1
      do while (j >= 1)
        if (s%times(j) <= tswap) exit
        s%times(j + 1) = s%times(j)
        s%ifile(j + 1) = s%ifile(j)
        s%irec(j + 1) = s%irec(j)
        s%paths(j + 1) = s%paths(j)
        j = j - 1
      enddo
      s%times(j + 1) = tswap
      s%ifile(j + 1) = ifile
      s%irec(j + 1) = irec
      s%paths(j + 1) = pswap
    enddo
  end subroutine sort_catalog

  subroutine read_catalog_rec(s, idx, slot)
    type(series2d), intent(inout) :: s
    integer,        intent(in)    :: idx, slot
    integer :: ncid
    real(rp), allocatable :: tmp(:,:)
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (idx < 1 .or. idx > s%ntot) return
    ! Read into host memory, then one bulk copy into the managed buffer:
    ! netCDF writing straight into managed memory faults page by page.
    allocate(tmp(s%n1, s%n2))
    call nc_open_read(s%paths(idx), ncid)
    call nc_read_forcing_2d(ncid, s%vname, tmp, s%n1, s%n2, s%irec(idx), s%time_axis, s%fi0, s%fj0)
    call nc_close(ncid)
#ifdef USE_CUDA
    ierr = cudaMemcpy(s%buf(:,:,slot), tmp, s%n1 * s%n2, cudaMemcpyHostToDevice)
    if (ierr /= 0) error stop 'read_catalog_rec: cudaMemcpy'
#else
    s%buf(:,:,slot) = tmp
#endif
    deallocate(tmp)
  end subroutine read_catalog_rec

  ! Climatology bracket, as the CPU model (find_new_record_cycle): the
  ! record times are shifted by whole cycles so that the first shifted time
  ! above tneed is record i2; i1 is the one before it (wrapping to the last
  ! record of the previous cycle).
  subroutine cyc_bracket(times, ntot, cycle, tneed, i1, i2, t1, t2)
    integer,  intent(in)  :: ntot
    real(rp), intent(in)  :: times(ntot), cycle, tneed
    integer,  intent(out) :: i1, i2
    real(rp), intent(out) :: t1, t2
    integer :: icycle, k
    if (ntot < 2) then
      i1 = 1; i2 = 1; t1 = tneed; t2 = tneed
      return
    endif
    icycle = floor((tneed - times(ntot)) / cycle) + 1
    i2 = ntot
    do k = 1, ntot
      if (times(k) + real(icycle, rp) * cycle > tneed) then
        i2 = k
        exit
      endif
    enddo
    t2 = times(i2) + real(icycle, rp) * cycle
    if (i2 == 1) then
      i1 = ntot
      t1 = times(ntot) + real(icycle - 1, rp) * cycle
    else
      i1 = i2 - 1
      t1 = times(i1) + real(icycle, rp) * cycle
    endif
  end subroutine cyc_bracket

  subroutine advance_series(s, tneed)
    type(series2d), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    integer :: rec, i, old1, old2, i1, i2
    if (.not. s%ready .or. s%ntot < 2) return
    if (s%cycle > 0.0_rp) then
      call cyc_bracket(s%times, s%ntot, s%cycle, tneed, i1, i2, s%t1e, s%t2e)
      if (i1 /= s%it1) then
        s%it1 = i1
        call read_catalog_rec(s, i1, 1)
      endif
      if (i2 /= s%it2) then
        s%it2 = i2
        call read_catalog_rec(s, i2, 2)
      endif
      return
    endif
    if (s%times(s%it2) >= tneed .and. s%times(s%it1) <= tneed) return
    rec = 1
    do i = 1, s%ntot - 1
      if (s%times(i) <= tneed .and. s%times(i + 1) >= tneed) then
        rec = i
        exit
      endif
      if (s%times(i) <= tneed) rec = i
    enddo
    rec = max(1, min(rec, s%ntot - 1))
    if (rec == s%it1 .and. rec + 1 == s%it2) return
    old1 = s%it1; old2 = s%it2
    s%it1 = rec
    s%it2 = rec + 1
    if (s%it1 /= old1) call read_catalog_rec(s, s%it1, 1)
    if (s%it2 /= old2) call read_catalog_rec(s, s%it2, 2)
  end subroutine advance_series

  subroutine weights(s, tneed, c1, c2)
    type(series2d), intent(in)  :: s
    real(rp),       intent(in)  :: tneed
    real(rp),       intent(out) :: c1, c2
    real(rp) :: dtm
    real(rp) :: t1, t2
    if (s%ntot < 2) then
      c1 = 1.0_rp; c2 = 0.0_rp
      return
    endif
    if (s%cycle > 0.0_rp) then
      t1 = s%t1e; t2 = s%t2e
    else
      t1 = s%times(s%it1); t2 = s%times(s%it2)
    endif
    dtm = t2 - t1
    if (abs(dtm) < 1.0e-12_rp) then
      c1 = 1.0_rp; c2 = 0.0_rp
    else
      c1 = (t2 - tneed) / dtm
      c2 = (tneed - t1) / dtm
    endif
  end subroutine weights

  subroutine fill_r(s, tneed, dest)
    type(series2d), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp) :: c1, c2
    real(rp), allocatable :: tmp(:,:)
    integer :: i0, j0
    if (.not. s%ready) return
    call advance_series(s, tneed)
    call weights(s, tneed, c1, c2)
    if (on_grid(s%n1, s%n2, 'r', i0, j0)) then
      call interp_on_grid(s%buf, s%n1, s%n2, c1, c2, dest, i0, j0)
    else
      allocate(tmp(s%n1, s%n2))
      call sync_arch()
      tmp = c1 * s%buf(:,:,1) + c2 * s%buf(:,:,2)
      where (is_nan(tmp)) tmp = 0.0_rp
      call map_field(tmp, s%n1, s%n2, dest, 'r')
    endif
  end subroutine fill_r

  subroutine fill_u(s, tneed, dest)
    type(series2d), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp) :: c1, c2
    real(rp), allocatable :: tmp(:,:)
    integer :: i0, j0
    if (.not. s%ready) return
    call advance_series(s, tneed)
    call weights(s, tneed, c1, c2)
    if (on_grid(s%n1, s%n2, 'u', i0, j0)) then
      call interp_on_grid(s%buf, s%n1, s%n2, c1, c2, dest, i0, j0)
    else
      allocate(tmp(s%n1, s%n2))
      call sync_arch()
      tmp = c1 * s%buf(:,:,1) + c2 * s%buf(:,:,2)
      where (is_nan(tmp)) tmp = 0.0_rp
      call map_field(tmp, s%n1, s%n2, dest, 'u')
    endif
  end subroutine fill_u

  subroutine fill_v(s, tneed, dest)
    type(series2d), intent(inout) :: s
    real(rp),       intent(in)    :: tneed
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp) :: c1, c2
    real(rp), allocatable :: tmp(:,:)
    integer :: i0, j0
    if (.not. s%ready) return
    call advance_series(s, tneed)
    call weights(s, tneed, c1, c2)
    if (on_grid(s%n1, s%n2, 'v', i0, j0)) then
      call interp_on_grid(s%buf, s%n1, s%n2, c1, c2, dest, i0, j0)
    else
      allocate(tmp(s%n1, s%n2))
      call sync_arch()
      tmp = c1 * s%buf(:,:,1) + c2 * s%buf(:,:,2)
      where (is_nan(tmp)) tmp = 0.0_rp
      call map_field(tmp, s%n1, s%n2, dest, 'v')
    endif
  end subroutine fill_v

  logical function on_grid(n1, n2, gtype, i0, j0)
    ! The file field is on the model stencil (the direct-copy cases of
    ! map_field): dest(i0+i-1, j0+j-1) = src(i, j).
    integer,          intent(in)  :: n1, n2
    character(len=*), intent(in)  :: gtype
    integer,          intent(out) :: i0, j0
    on_grid = .true.
    i0 = 1; j0 = 1
    if (gtype == 'r' .and. n1 == decomp%nx + 2 .and. n2 == decomp%ny + 2) then
      i0 = 0; j0 = 0
    elseif (gtype == 'r' .and. n1 == decomp%nx .and. n2 == decomp%ny) then
      continue
    elseif (gtype == 'u' .and. n1 == decomp%nx + 1 .and. n2 == decomp%ny) then
      continue
    elseif (gtype == 'v' .and. n1 == decomp%nx .and. n2 == decomp%ny + 1) then
      continue
    else
      on_grid = .false.
    endif
  end function on_grid

  subroutine interp_on_grid(b, n1, n2, c1, c2, dest, i0, j0)
    ! Device time interpolation of two buffered records onto the grid,
    ! NaN (missing) -> 0, as the host tmp/where/map_field path.
    integer,  intent(in) :: n1, n2, i0, j0
    real(rp), WORK_ATTR intent(in) :: b(n1, n2, 2)
    real(rp), intent(in) :: c1, c2
    real(rp), WORK_ATTR intent(inout) :: dest(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer :: i, j
    real(rp) :: x
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, n2
      do i = 1, n1
        x = c1 * b(i, j, 1) + c2 * b(i, j, 2)
        if (x /= x) x = 0.0_rp
        dest(i0 + i - 1, j0 + j - 1) = x
      enddo
    enddo
  end subroutine interp_on_grid

  ! MARBL surface forcing series. Names/units as the CPU model
  ! (bgc_shared_vars / marbl_driver): dust kg/m2/s, iron nmol/cm2/s ->
  ! mmol/m2/s (x0.01), nox/nhy kg N/m2/s -> mmol/m2/s (x71394.200220751),
  ! xco2_air / xco2_air_alt ppm. Missing series leave the field as is.
  subroutine init_bgc_forcing()
    bgc_frc_on = .false.
    if (case_nfrc < 1) return
    call bind_series(s_dust, 'dust', 'dust_time', 'r')
    call bind_series(s_iron, 'iron', 'iron_time', 'r')
    call bind_series(s_nox, 'nox', 'nox_time', 'r')
    call bind_series(s_nhy, 'nhy', 'nhy_time', 'r')
    call bind_series(s_xco2, 'xco2_air', 'xco2_time', 'r')
    call bind_series(s_xco2a, 'xco2_air_alt', 'xco2_time', 'r')
    bgc_frc_on = s_dust%ready .or. s_iron%ready .or. s_nox%ready .or. s_nhy%ready .or. s_xco2%ready
    write(*,'(a,6l2)') 'MARBL forcing series dust iron nox nhy xco2 xco2_alt ready =', s_dust%ready, &
         s_iron%ready, s_nox%ready, s_nhy%ready, s_xco2%ready, s_xco2a%ready
  end subroutine init_bgc_forcing

  subroutine set_bgc_forces(tneed, dust, iron, nox, nhy, xco2, xco2a)
    real(rp), intent(in) :: tneed
    real(rp), WORK_ATTR intent(inout) :: dust(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: iron(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: nox(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: nhy(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: xco2(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: xco2a(decomp%is:decomp%ie, decomp%js:decomp%je)
    if (.not. bgc_frc_on) return
    if (s_dust%ready) call fill_r(s_dust, tneed, dust)
    if (s_iron%ready) then
      call fill_r(s_iron, tneed, iron)
      call scale_2d(iron, 0.01_rp)
    endif
    if (s_nox%ready) then
      call fill_r(s_nox, tneed, nox)
      call scale_2d(nox, 71394.200220751_rp)
    endif
    if (s_nhy%ready) then
      call fill_r(s_nhy, tneed, nhy)
      call scale_2d(nhy, 71394.200220751_rp)
    endif
    if (s_xco2%ready) call fill_r(s_xco2, tneed, xco2)
    if (s_xco2a%ready) call fill_r(s_xco2a, tneed, xco2a)
  end subroutine set_bgc_forces

  subroutine scale_2d(a, f)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: f
    integer :: i, j, is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        a(i, j) = a(i, j) * f
      enddo
    enddo
  end subroutine scale_2d

  subroutine set_2d(a, val)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: val
    integer :: i, j, is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        a(i, j) = val
      enddo
    enddo
  end subroutine set_2d

  subroutine divide_2d(a, d)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: d
    integer :: i, j, is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = js, je
      do i = is, ie
        a(i, j) = a(i, j) / d
      enddo
    enddo
  end subroutine divide_2d

  subroutine map_field(src, n1, n2, dest, gtype)
    integer,          intent(in)    :: n1, n2
    real(rp),         intent(in)    :: src(n1, n2)
    real(rp),         intent(inout) :: dest(decomp%is:, decomp%js:)
    character(len=*), intent(in)    :: gtype
    integer :: i0, j0, nxi, neta, i, j
    real(rp), allocatable :: fine(:,:)

    if (gtype == 'u') then
      nxi = decomp%nx + 1; neta = decomp%ny
    elseif (gtype == 'v') then
      nxi = decomp%nx; neta = decomp%ny + 1
    else
      nxi = decomp%nx + 2; neta = decomp%ny + 2
    endif

    ! File on the LLm+2 rho / matching u,v stencil.
    if (gtype == 'r' .and. n1 == decomp%nx + 2 .and. n2 == decomp%ny + 2) then
      dest(0:decomp%nx+1, 0:decomp%ny+1) = src
      return
    endif
    if (gtype == 'r' .and. n1 == decomp%nx .and. n2 == decomp%ny) then
      dest(1:decomp%nx, 1:decomp%ny) = src
      return
    endif
    if (gtype == 'u' .and. n1 == decomp%nx + 1 .and. n2 == decomp%ny) then
      dest(1:decomp%nx+1, 1:decomp%ny) = src
      return
    endif
    if (gtype == 'v' .and. n1 == decomp%nx .and. n2 == decomp%ny + 1) then
      dest(1:decomp%nx, 1:decomp%ny+1) = src
      return
    endif

    ! 2x coarse (standalone flux file).
    allocate(fine(nxi, neta))
    call upsample2(src, n1, n2, fine, nxi, neta, gtype)
    if (gtype == 'r') then
      i0 = 0; j0 = 0
      if (nxi == decomp%nx) then
        i0 = 1; j0 = 1
      endif
      do j = 1, neta
        do i = 1, nxi
          dest(i0 + i - 1, j0 + j - 1) = fine(i, j)
        enddo
      enddo
    elseif (gtype == 'u') then
      dest(1:nxi, 1:neta) = fine
    else
      dest(1:nxi, 1:neta) = fine
    endif
    deallocate(fine)
  end subroutine map_field

  subroutine upsample2(c, ncx, ncy, f, nfx, nfy, gtype)
    integer,          intent(in)  :: ncx, ncy, nfx, nfy
    real(rp),         intent(in)  :: c(ncx, ncy)
    real(rp),         intent(out) :: f(nfx, nfy)
    character(len=*), intent(in)  :: gtype
    integer :: i, j, ic, jc
    real(rp) :: xl, yl, xi, yj
    do j = 1, nfy
      do i = 1, nfx
        if (gtype == 'u') then
          xi = 0.5_rp * real(i, rp) + 0.5_rp
          yj = 0.5_rp * real(j, rp) + 0.25_rp
        elseif (gtype == 'v') then
          xi = 0.5_rp * real(i, rp) + 0.25_rp
          yj = 0.5_rp * real(j, rp) + 0.5_rp
        else
          xi = 0.5_rp * real(i, rp) + 0.25_rp
          yj = 0.5_rp * real(j, rp) + 0.25_rp
        endif
        ic = max(1, min(ncx - 1, int(floor(xi))))
        jc = max(1, min(ncy - 1, int(floor(yj))))
        xl = xi - real(ic, rp)
        yl = yj - real(jc, rp)
        xl = min(1.0_rp, max(0.0_rp, xl))
        yl = min(1.0_rp, max(0.0_rp, yl))
        f(i, j) = (1.0_rp - yl) * ((1.0_rp - xl) * nan0(c(ic, jc)) + xl * nan0(c(ic + 1, jc))) + &
                  yl * ((1.0_rp - xl) * nan0(c(ic, jc + 1)) + xl * nan0(c(ic + 1, jc + 1)))
      enddo
    enddo
  end subroutine upsample2

  real(rp) function nan0(x)
    real(rp), intent(in) :: x
    if (is_nan(x)) then
      nan0 = 0.0_rp
    else
      nan0 = x
    endif
  end function nan0

  function find_var_file(vname, tname) result(path)
    character(len=*), intent(in) :: vname
    character(len=*), intent(in), optional :: tname
    character(len=512) :: path, fallback
    integer :: i, ncid
    path = ''
    fallback = ''
    do i = 1, case_nfrc
      call nc_open_read(case_frcfiles(i), ncid)
      if (nc_has_var(ncid, vname)) then
        if (present(tname)) then
          if (nc_has_var(ncid, tname)) then
            path = case_frcfiles(i)
            call nc_close(ncid)
            return
          elseif (len_trim(fallback) == 0) then
            fallback = case_frcfiles(i)
          endif
        else
          path = case_frcfiles(i)
          call nc_close(ncid)
          return
        endif
      endif
      call nc_close(ncid)
    enddo
    if (len_trim(path) == 0) path = fallback
  end function find_var_file

  function resolve_one(dir, path) result(out)
    character(len=*), intent(in) :: dir, path
    character(len=512) :: out
    if (len_trim(path) == 0) then
      out = ''
    elseif (path(1:1) == '/') then
      out = trim(path)
    else
      out = trim(dir)//'/'//trim(path)
    endif
  end function resolve_one

  subroutine write_toy_flux(path, nxi, neta, t0, t1)
    character(len=*), intent(in) :: path
    integer,          intent(in) :: nxi, neta
    real(rp),         intent(in) :: t0, t1
    call write_toy_flux_recs(path, nxi, neta, t0, t1)
  end subroutine write_toy_flux

  subroutine write_toy_flux_recs(path, nxi, neta, t0, t1)
    character(len=*), intent(in) :: path
    integer,          intent(in) :: nxi, neta
    real(rp),         intent(in) :: t0, t1
    integer :: ncid, d_xi, d_eta, d_xiu, d_etav, d_t, id
    real(rp), allocatable :: au(:,:,:), av(:,:,:), ar(:,:,:)
    allocate(au(nxi - 1, neta, 2), av(nxi, neta - 1, 2), ar(nxi, neta, 2))
    au = 0.1_rp
    av = 0.0_rp
    ar = 0.0_rp
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'sms_time', 2)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    d_xiu = nc_def_dim(ncid, 'xi_u', nxi - 1)
    d_etav = nc_def_dim(ncid, 'eta_v', neta - 1)
    id = nc_def_var(ncid, 'sms_time', (/d_t/))
    id = nc_def_var(ncid, 'shf_time', (/d_t/))
    id = nc_def_var(ncid, 'swf_time', (/d_t/))
    id = nc_def_var(ncid, 'srf_time', (/d_t/))
    id = nc_def_var(ncid, 'sustr', (/d_xiu, d_eta, d_t/))
    id = nc_def_var(ncid, 'svstr', (/d_xi, d_etav, d_t/))
    id = nc_def_var(ncid, 'shflux', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'swflux', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'swrad', (/d_xi, d_eta, d_t/))
    call nc_enddef(ncid)
    call nc_put_1d(ncid, 'sms_time', (/ t0, t1 /))
    call nc_put_1d(ncid, 'shf_time', (/ t0, t1 /))
    call nc_put_1d(ncid, 'swf_time', (/ t0, t1 /))
    call nc_put_1d(ncid, 'srf_time', (/ t0, t1 /))
    call nc_put_3d(ncid, 'sustr', au)
    call nc_put_3d(ncid, 'svstr', av)
    call nc_put_3d(ncid, 'shflux', ar)
    call nc_put_3d(ncid, 'swflux', ar)
    call nc_put_3d(ncid, 'swrad', ar)
    call nc_close(ncid)
    deallocate(au, av, ar)
  end subroutine write_toy_flux_recs

  subroutine write_toy_bulk(path, nxi, neta, t0, t1)
    character(len=*), intent(in) :: path
    integer,          intent(in) :: nxi, neta
    real(rp),         intent(in) :: t0, t1
    integer :: ncid, d_xi, d_eta, d_t, id
    real(rp), allocatable :: ar(:,:,:)
    allocate(ar(nxi, neta, 2))
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'time', 2)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    id = nc_def_var(ncid, 'time', (/d_t/))
    id = nc_def_var(ncid, 'uwnd', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'vwnd', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'Tair', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'qair', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'rain', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'lwrad', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'swrad', (/d_xi, d_eta, d_t/))
    call nc_enddef(ncid)
    call nc_put_1d(ncid, 'time', (/ t0, t1 /))
    ar = 5.0_rp;   call nc_put_3d(ncid, 'uwnd', ar)
    ar = 0.0_rp;   call nc_put_3d(ncid, 'vwnd', ar)
    ar = 15.0_rp;  call nc_put_3d(ncid, 'Tair', ar)
    ar = 0.01_rp;  call nc_put_3d(ncid, 'qair', ar)
    ar = 0.0_rp;   call nc_put_3d(ncid, 'rain', ar)
    ar = 350.0_rp; call nc_put_3d(ncid, 'lwrad', ar)
    ar = 200.0_rp; call nc_put_3d(ncid, 'swrad', ar)
    call nc_close(ncid)
    deallocate(ar)
  end subroutine write_toy_bulk

end module frc_io
