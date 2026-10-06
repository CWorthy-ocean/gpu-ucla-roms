module case_io
  ! Namelist + grid/init NetCDF. Host-side; one rank reads the whole tile.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use geom, only: setup_metrics, set_depth, dump_vic
  use nc_io
  use host_buf, only: host_read_2d, host_read_3d
  use frc_io, only: read_forcing_namelist, init_forcing, set_forces
  use bry_io, only: init_bry, set_bry_all
  use tide_io, only: read_tide_namelist, init_tides, set_tides
  use his_io, only: read_output_namelist, set_output_root, init_output
  use comm_mod, only: comm_rank, comm_size
  use formulas, only: rho0_eos => rho0
  use passive_mod, only: ntp, tp, read_passive_namelist, alloc_passive, init_passive, passive_name
  use marbl_gpu, only: marbl_on, read_marbl_namelist, marbl_gpu_setup_tracers, marbl_gpu_init_forcing, &
                       marbl_gpu_load_state
  implicit none
  private
  logical :: passive_loaded = .false.

  character(len=512), public :: case_nml = ''
  character(len=512), public :: case_grdname = ''
  character(len=512), public :: case_inifile = ''
  character(len=256), public :: case_title = 'roms_gpu'
  character(len=512), public :: case_output_root = './output'
  integer(ip), public :: case_llm = 0
  integer(ip), public :: case_mmm = 0
  integer(ip), public :: case_nz = 0
  integer(ip), public :: case_npx = 1
  integer(ip), public :: case_npy = 1
  integer(ip), public :: case_record = 1
  integer(ip), public :: case_nrrec = -1
  logical,     public :: case_exact_restart = .false.
  logical :: case_have_prev = .false.
  logical,     public :: case_periodic_x = .false.
  logical,     public :: case_periodic_y = .false.
  real(rp),    public :: case_v_sponge = 0.0_rp

  public :: read_namelist
  public :: load_case
  public :: report_loaded_case
  public :: write_toy_case
  public :: sanitize_land

contains

  subroutine read_namelist(path)
    character(len=*), intent(in) :: path
    integer :: unit, ios
    character(len=512) :: msg, nml_dir

    integer(ip) :: llm, mmm, nz, n, np_xi, np_eta, nt_passive, nt_bgc, ntrc_bio
    integer(ip) :: nsub_x, nsub_e, ndtfast_in, ntimes_in, ninfo, nrrec
    real(rp) :: dt_in, theta_s_in, theta_b_in, hc_in
    real(rp) :: zob_in, rdrg_in, rdrg2_in, visc2_in, akv_bak_in, rho0_in
    real(rp) :: akt_bak_in(2), tnu2(64)
    character(len=512) :: grdname, inifile, ininame, title, output_root_name

    namelist /param_settings/ np_xi, np_eta, nsub_x, nsub_e, llm, mmm, nz, n, &
                              nt_passive, nt_bgc, ntrc_bio
    namelist /grid_settings/ grdname
    namelist /initial_conditions/ inifile, ininame, nrrec
    namelist /simulation_name_settings/ output_root_name, title
    namelist /ubind_settings/ ubind
    real(rp) :: v_sponge
    namelist /v_sponge_settings/ v_sponge

    llm = 32; mmm = 32; nz = 0; n = 0
    np_xi = 1; np_eta = 1; nsub_x = 1; nsub_e = 1
    nt_passive = 0; nt_bgc = 0; ntrc_bio = 0; nrrec = -1
    ininame = ''
    dt_in = 100.0_rp; ndtfast_in = 60; ntimes_in = 1; ninfo = 1
    theta_s_in = 6.0_rp; theta_b_in = 6.0_rp; hc_in = 250.0_rp
    zob_in = 0.01_rp; rdrg_in = 0.0_rp; rdrg2_in = 0.0_rp; visc2_in = 0.0_rp
    akv_bak_in = 0.0_rp; akt_bak_in = 0.0_rp; tnu2 = 0.0_rp; rho0_in = 1025.0_rp
    grdname = ''; inifile = ''
    title = 'roms_gpu'; output_root_name = './output'

    case_nml = trim(path)
    nml_dir = dirname(path)

    open(newunit=unit, file=trim(path), status='old', action='read', iostat=ios, iomsg=msg)
    if (ios /= 0) then
      write(*,*) 'cannot open namelist ', trim(path), ': ', trim(msg)
      error stop 'read_namelist'
    endif

    rewind(unit); read(unit, nml=param_settings, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      write(*,*) 'PARAM_SETTINGS: ', trim(msg)
      error stop 'read_namelist'
    endif

    rewind(unit); read(unit, nml=grid_settings, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      write(*,*) 'GRID_SETTINGS: ', trim(msg)
      error stop 'read_namelist'
    endif

    rewind(unit); read(unit, nml=initial_conditions, iostat=ios, iomsg=msg)
    if (ios /= 0) then
      write(*,*) 'INITIAL_CONDITIONS: ', trim(msg)
      error stop 'read_namelist'
    endif

    call read_s_coord(unit, theta_s_in, theta_b_in, hc_in)
    call read_time_stepping(unit, dt_in, ndtfast_in, ntimes_in, ninfo)
    ninfo_steps = max(ninfo, 1)
    call read_optional_scalars(unit, zob_in, rdrg_in, rdrg2_in, visc2_in, &
                               akv_bak_in, akt_bak_in, rho0_in)
    rewind(unit); read(unit, nml=ubind_settings, iostat=ios)
    v_sponge = 0.0_rp
    rewind(unit); read(unit, nml=v_sponge_settings, iostat=ios)
    if (ios /= 0) v_sponge = 0.0_rp
    case_v_sponge = v_sponge
    call read_forcing_namelist(unit, nml_dir)
    call read_tide_namelist(unit)
    call read_output_namelist(unit)
    call read_passive_namelist(unit, nt_passive)
    call read_marbl_namelist(unit, nml_dir, nt_bgc)

    rewind(unit); read(unit, nml=simulation_name_settings, iostat=ios)
    close(unit)

    if (nz <= 0) nz = n
    if (nz <= 0) nz = 16
    if (len_trim(inifile) == 0) inifile = ininame
    case_llm = llm
    case_mmm = mmm
    case_nz = nz
    case_npx = np_xi
    case_npy = np_eta
    case_grdname = resolve_path(nml_dir, grdname)
    case_inifile = resolve_path(nml_dir, inifile)
    case_nrrec = nrrec
    rho0_case = rho0_in
    if (abs(rho0_in - rho0_eos) > 1.0e-6_rp) then
      write(*,'(a,f10.4,a,f10.4,a)') 'case_io: namelist rho0 =', rho0_in, &
           ' but build uses rho0 =', rho0_eos, '; rebuild with -DROMS_RHO0=<value>'
      error stop 1
    endif
    case_title = trim(title)
    case_output_root = resolve_path(nml_dir, output_root_name)
    theta_s = theta_s_in
    theta_b = theta_b_in
    hc = hc_in
    zob = zob_in
    rdrg = rdrg_in
    rdrg2 = rdrg2_in
    visc2 = visc2_in
    akv_bak = akv_bak_in
    akt_bak = akt_bak_in

    call apply_decomp(case_periodic_x, case_periodic_y)
    call init_time_state(nz, ndtfast_in, dt_in, ntimes_in, 2_ip)
  end subroutine read_namelist

  subroutine apply_sponge()
    ! CPU set_nudgcof (SPONGE, UV_VIS2, TS_DIF2, not PACIFIC): linear ramp
    ! over isp = sponge_size+1 points from every non-periodic edge.
    integer(ip), parameter :: sponge_size = 15_ip
    integer(ip) :: i, j, ig, jg, isp, ibnd, itrc, i_lo, i_hi, j_lo, j_hi
    real(rp), allocatable :: wrk(:,:)

    if (case_v_sponge <= 0.0_rp) return
    isp = sponge_size + 1_ip
    i_lo = lbound(visc2_r, 1); i_hi = ubound(visc2_r, 1)
    j_lo = lbound(visc2_r, 2); j_hi = ubound(visc2_r, 2)
    allocate(wrk(i_lo:i_hi, j_lo:j_hi))
    do j = j_lo, j_hi
      jg = decomp%j0 + j - 1_ip
      do i = i_lo, i_hi
        ig = decomp%i0 + i - 1_ip
        ibnd = isp
        if (.not. decomp%periodic_x) ibnd = min(ibnd, ig, decomp%gnx + 1_ip - ig)
        if (.not. decomp%periodic_y) ibnd = min(ibnd, jg, decomp%gny + 1_ip - jg)
        wrk(i, j) = real(isp - ibnd, rp) / real(isp, rp)
      enddo
    enddo

    visc2_r = visc2_r + case_v_sponge * wrk
    do j = j_lo + 1, j_hi
      do i = i_lo + 1, i_hi
        visc2_p(i, j) = visc2_p(i, j) + 0.25_rp * case_v_sponge * &
             (wrk(i, j) + wrk(i - 1, j) + wrk(i, j - 1) + wrk(i - 1, j - 1))
      enddo
    enddo
    do itrc = 1, size(diff2, 3)
      diff2(:, :, itrc) = diff2(:, :, itrc) + case_v_sponge * wrk
    enddo
    deallocate(wrk)
    if (comm_rank() == 0) write(*,'(a,i3,a,es12.4)') 'Sponge layer = ', sponge_size, &
         ' interior grid points, v_sponge =', case_v_sponge
  end subroutine apply_sponge

  subroutine apply_decomp(periodic_x, periodic_y)
    logical, intent(in) :: periodic_x, periodic_y
    integer(ip) :: npx, npy, nproc
    nproc = comm_size()
    npx = case_npx
    npy = case_npy
    if (nproc == 1) then
      if (npx * npy > 1 .and. comm_rank() == 0) then
        write(*,'(a,2i4,a)') 'note: namelist np_xi,np_eta =', npx, npy, &
             ' but 1 MPI rank; using 1 x 1'
      endif
      npx = 1_ip
      npy = 1_ip
    elseif (nproc /= npx * npy) then
      write(*,*) 'MPI size', nproc, '/= np_xi*np_eta', npx * npy
      error stop 'apply_decomp'
    endif
    call init_decomp(case_llm, case_mmm, case_nz, 2_ip, npx, npy, comm_rank(), &
                     periodic_x, periodic_y)
  end subroutine apply_decomp

  subroutine read_s_coord(unit, theta_s_in, theta_b_in, hc_in)
    integer, intent(in) :: unit
    real(rp), intent(inout) :: theta_s_in, theta_b_in, hc_in
    integer :: ios
    real(rp) :: theta_s, theta_b, hc
    namelist /s_coord/ theta_s, theta_b, hc
    theta_s = theta_s_in; theta_b = theta_b_in; hc = hc_in
    rewind(unit)
    read(unit, nml=s_coord, iostat=ios)
    if (ios == 0) then
      theta_s_in = theta_s
      theta_b_in = theta_b
      hc_in = hc
    endif
  end subroutine read_s_coord

  subroutine read_time_stepping(unit, dt_in, ndtfast_in, ntimes_in, ninfo)
    integer, intent(in) :: unit
    real(rp), intent(inout) :: dt_in
    integer(ip), intent(inout) :: ndtfast_in, ntimes_in, ninfo
    integer :: ios
    real(rp) :: dt
    integer(ip) :: ndtfast, ntimes
    namelist /time_stepping/ dt, ndtfast, ntimes, ninfo
    dt = dt_in; ndtfast = ndtfast_in; ntimes = ntimes_in
    rewind(unit)
    read(unit, nml=time_stepping, iostat=ios)
    if (ios /= 0) then
      write(*,*) 'TIME_STEPPING missing or invalid'
      error stop 'read_namelist'
    endif
    dt_in = dt
    ndtfast_in = ndtfast
    ntimes_in = ntimes
  end subroutine read_time_stepping

  subroutine read_optional_scalars(unit, zob_in, rdrg_in, rdrg2_in, visc2_in, &
                                   akv_bak_in, akt_bak_in, rho0_in)
    integer, intent(in) :: unit
    real(rp), intent(inout) :: zob_in, rdrg_in, rdrg2_in, visc2_in, akv_bak_in
    real(rp), intent(inout) :: akt_bak_in(2), rho0_in
    integer :: ios
    real(rp) :: rdrg, rdrg2, zob, visc2, akv_bak, akt_bak(2), tnu2(64), rho0
    namelist /bottom_drag_settings/ rdrg, rdrg2, zob
    namelist /vertical_mixing_settings/ akv_bak, akt_bak
    namelist /lateral_visc_settings/ visc2
    namelist /tracer_diff2/ tnu2
    namelist /rho0_settings/ rho0
    rdrg = rdrg_in; rdrg2 = rdrg2_in; zob = zob_in
    visc2 = visc2_in; akv_bak = akv_bak_in; akt_bak = akt_bak_in
    tnu2 = 0.0_rp; rho0 = rho0_in
    rewind(unit); read(unit, nml=bottom_drag_settings, iostat=ios)
    rewind(unit); read(unit, nml=vertical_mixing_settings, iostat=ios)
    rewind(unit); read(unit, nml=lateral_visc_settings, iostat=ios)
    rewind(unit); read(unit, nml=tracer_diff2, iostat=ios)
    rewind(unit); read(unit, nml=rho0_settings, iostat=ios)
    zob_in = zob; rdrg_in = rdrg; rdrg2_in = rdrg2; visc2_in = visc2
    akv_bak_in = akv_bak; akt_bak_in = akt_bak
    rho0_in = rho0
  end subroutine read_optional_scalars

  subroutine load_case(nml_path)
    character(len=*), intent(in) :: nml_path
    integer :: ncid, xi_rho, eta_rho, xi_u, eta_v, s_rho
    integer :: i0, j0, nxi, neta
    integer(ip) :: k
    logical :: had_f, had_lat
    real(rp), parameter :: deg2rad = 3.14159265358979323846_rp / 180.0_rp
    real(rp), parameter :: omega_earth = 4.0_rp * 3.14159265358979323846_rp / (23.9344699_rp * 3600.0_rp)

    call read_namelist(nml_path)
    if (decomp%npx * decomp%npy > 1) then
      write(*,*) 'tiled NetCDF I/O is not wired; run MiniPac on 1 MPI rank'
      error stop 'load_case'
    endif
    call alloc_state(nt)
    if (marbl_on) call marbl_gpu_setup_tracers()
    call alloc_passive()

    call nc_open_read(case_grdname, ncid)
    xi_rho = nc_dim_len(ncid, 'xi_rho')
    eta_rho = nc_dim_len(ncid, 'eta_rho')
    call infer_layout(xi_rho, eta_rho, i0, j0, nxi, neta)
    write(*,'(a,a)') 'grid file: ', trim(case_grdname)
    write(*,'(a,2i6,a,2i4)') '  xi_rho, eta_rho =', xi_rho, eta_rho, '  mapped i0,j0 =', i0, j0

    call host_read_2d(ncid, 'h', h(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta)
    call host_read_2d(ncid, 'pm', pm(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta)
    call host_read_2d(ncid, 'pn', pn(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta)
    if (nc_has_var(ncid, 'mask_rho')) then
      call host_read_2d(ncid, 'mask_rho', rmask(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta)
    else
      rmask = 1.0_rp
    endif
    ! As roms_bugfix with SPHERICAL: f comes from lat_rho and the sidereal
    ! day, not from the grid file (whose f may use a 24 h day).
    had_lat = nc_has_var(ncid, 'lat_rho')
    if (had_lat) then
      call host_read_2d(ncid, 'lat_rho', latr(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta)
    endif
    had_f = nc_has_var(ncid, 'f') .and. .not. had_lat
    if (had_f) then
      call host_read_2d(ncid, 'f', f(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta)
    endif
    if (nc_has_var(ncid, 'lon_rho')) then
      call host_read_2d(ncid, 'lon_rho', lonr(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta)
    endif
    if (nc_has_var(ncid, 'angle')) then
      call host_read_2d(ncid, 'angle', angler(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta)
    endif
    call nc_close(ncid)
    if (.not. had_f) then
      f(i0:i0+nxi-1, j0:j0+neta-1) = omega_earth * &
           sin(deg2rad * latr(i0:i0+nxi-1, j0:j0+neta-1))
    endif

    call sanitize_land()
    call pad_physical_halo()
    call setup_metrics()
    visc2_r = visc2
    visc2_p = visc2
    diff2 = 0.0_rp
    diff2(:, :, 1) = 0.0_rp
    if (size(diff2, 3) >= 2) diff2(:, :, 2) = 0.0_rp
    call apply_sponge()

    call nc_open_read(case_inifile, ncid)
    write(*,'(a,a)') 'init file: ', trim(case_inifile)
    xi_u = nc_dim_len(ncid, 'xi_u')
    eta_v = nc_dim_len(ncid, 'eta_v')
    s_rho = nc_dim_len(ncid, 's_rho')
    if (s_rho /= decomp%nz) then
      write(*,*) 's_rho', s_rho, '/= namelist nz', decomp%nz
      error stop 'load_case'
    endif
    call load_init_records(ncid, i0, j0, nxi, neta, xi_u, eta_v)
    call nc_close(ncid)

    call sanitize_land()
    call pad_physical_halo()
    do k = 2, 4
      zeta(:, :, k) = zeta(:, :, 1)
      ubar(:, :, k) = ubar(:, :, 1)
      vbar(:, :, k) = vbar(:, :, 1)
    enddo
    ! Slot 3 is the nnew workspace. CPU leaves it at 0 after get_init;
    ! Orlanski vin on the first step is that zero, not a copy of nstp.
    ! Slot 2 is n-1 only when both records were restored; otherwise copy
    ! the current state into the AB/AM slots.
    if (.not. case_have_prev) then
      u(:, :, :, 2) = u(:, :, :, 1)
      v(:, :, :, 2) = v(:, :, :, 1)
      t(:, :, :, 2, 1) = t(:, :, :, 1, 1)
      t(:, :, :, 2, 2) = t(:, :, :, 1, 2)
    endif
    knew = 1
    nstp = 1
    nrhs = 1
    tdays = start_time * sec2day
    call dump_vic('after load')
    if (.not. passive_loaded) call init_passive()
    if (marbl_on .and. .not. passive_loaded) error stop 'MARBL: initial file lacks the MARBL tracers'
    call init_forcing()
    if (marbl_on) call marbl_gpu_init_forcing()
    call init_bry()
    call init_tides()
    frc_slot = 0
    call set_forces()
    call set_bry_all()
    call set_tides()
    call set_output_root(case_output_root, case_title)
    call init_output()
  end subroutine load_case

  subroutine load_init_records(ncid, i0, j0, nxi, neta, xi_u, eta_v)
    ! UCLA EXACT_RESTART soft policy: if two consecutive records exist
    ! (ocean_time differs by dt, time_step consecutive when present) then
    ! restore 3-D u/v/t into slots 2 then 1. Coupling DU/DV_avg* from the
    ! last record are required to keep the exact (non-forward) first step.
    integer, intent(in) :: ncid, i0, j0, nxi, neta, xi_u, eta_v
    integer :: nrecs, rec, ibuf(8)
    real(rp) :: tbuf(1), time_bak, tscale
    logical :: times_ok, steps_ok, have_coup, has_tstep
    character(len=64) :: units

    case_exact_restart = .false.
    case_have_prev = .false.
    forw_start = 1
    ntstart = 1
    rec = 1
    nrecs = 1
    tscale = 1.0_rp
    have_coup = .false.
    has_tstep = .false.

    if (nc_has_var(ncid, 'ocean_time')) then
      nrecs = nc_var_nrecs(ncid, 'ocean_time')
      if (case_nrrec > 0) then
        rec = case_nrrec
        if (rec > nrecs) then
          write(*,*) 'requested restart record', rec, 'exceeds nrecs', nrecs
          error stop 'load_case'
        endif
      else
        rec = nrecs
      endif
      units = ''
      call nc_get_att_text(ncid, 'ocean_time', 'units', units)
      if (units(1:3) == 'day' .or. units(1:3) == 'Day') tscale = 86400.0_rp
      call nc_get_1d(ncid, 'ocean_time', tbuf, rec)
      start_time = tbuf(1) * tscale
      time = start_time

      times_ok = .false.
      steps_ok = .true.
      if (rec >= 2) then
        call nc_get_1d(ncid, 'ocean_time', tbuf, rec - 1)
        time_bak = tbuf(1) * tscale
        times_ok = (abs(start_time - time_bak - dt) < 0.01_rp * dt)
        if (.not. times_ok) then
          write(*,'(a,2i4,a)') 'WARNING: Exact restart is requested, but ', &
               rec - 1, rec, ' are not consecutive time steps; using forward start.'
        endif
      else
        write(*,'(a)') 'WARNING: Exact restart is requested, but the initial file ' // &
             'does not contain sufficient records.'
      endif

      has_tstep = nc_has_var(ncid, 'time_step')
      if (has_tstep) then
        call nc_get_ints(ncid, 'time_step', ibuf, rec)
        ntstart = ibuf(1) + 1
        if (times_ok) then
          ! CPU get_init: ntstart = time_step(n-1)+1, then require
          ! time_step(n) == ntstart, i.e. the two records differ by 1.
          call nc_get_ints(ncid, 'time_step', ibuf, rec - 1)
          steps_ok = (ibuf(1) + 1 == ntstart - 1)
          if (.not. steps_ok) then
            write(*,'(a,2i4,a)') 'WARNING: Exact restart is requested, but ', &
                 rec - 1, rec, ' are not consecutive time_step values; using forward start.'
          endif
        endif
      endif

      have_coup = has_coupling_fields(ncid)
      if (times_ok .and. steps_ok) then
        write(*,'(a,i0,a)') 'exact-restart: reading previous 3-D record ', rec - 1, ' into slot 2'
        call load_3d_record(ncid, rec - 1, 2, i0, j0, nxi, neta, xi_u, eta_v)
        call load_rufrc_slot(ncid, rec - 1, 2, i0, j0, nxi, neta, xi_u, eta_v)
        case_have_prev = .true.
      endif
      if (times_ok .and. steps_ok .and. have_coup .and. has_tstep) then
        case_exact_restart = .true.
        forw_start = 1
      else
        forw_start = ntstart
        if (times_ok .and. steps_ok .and. .not. have_coup) then
          write(*,'(a)') 'WARNING: Exact restart cancelled: DU/DV_avg coupling fields missing.'
        endif
      endif
    else
      rec = 1
    endif
    case_record = rec

    write(*,'(a,i0,a,i0)') 'exact-restart: reading current record ', rec, ' into slot 1; nrecs=', nrecs
    call load_2d_record(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
    call load_3d_record(ncid, rec, 1, i0, j0, nxi, neta, xi_u, eta_v)
    call load_rufrc_slot(ncid, rec, 1, i0, j0, nxi, neta, xi_u, eta_v)
    if (have_coup .or. has_coupling_fields(ncid)) then
      call load_coupling_fields(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
    endif
    if (case_exact_restart) then
      write(*,'(a,f12.4,a,i0,a,i0)') 'get_init :: Exact restart from day =', &
           start_time * sec2day, ' rec =', rec, ' ntstart=', ntstart
    elseif (nc_has_var(ncid, 'time_step')) then
      write(*,'(a,f12.4,a,i0,a,i0)') 'get_init :: Approximate, single-step restart from day =', &
           start_time * sec2day, ' rec =', rec, ' ntstart=', ntstart
    else
      write(*,'(a,f12.4,a,i0)') 'get_init :: Read initial conditions for day =', &
           start_time * sec2day, ' record =', rec
    endif
  end subroutine load_init_records

  logical function has_coupling_fields(ncid) result(ok)
    integer, intent(in) :: ncid
    ok = nc_has_var(ncid, 'DU_avg1') .and. nc_has_var(ncid, 'DV_avg1') .and. &
         nc_has_var(ncid, 'DU_avg2') .and. nc_has_var(ncid, 'DV_avg2') .and. &
         nc_has_var(ncid, 'DU_avg_bak') .and. nc_has_var(ncid, 'DV_avg_bak')
  end function has_coupling_fields

  subroutine load_2d_record(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
    integer, intent(in) :: ncid, rec, i0, j0, nxi, neta, xi_u, eta_v
    call host_read_2d(ncid, 'zeta', zeta(i0:i0+nxi-1, j0:j0+neta-1, 1), nxi, neta, rec)
    call host_read_2d(ncid, 'ubar', ubar(i0+1:i0+xi_u, j0:j0+neta-1, 1), xi_u, neta, rec)
    call host_read_2d(ncid, 'vbar', vbar(i0:i0+nxi-1, j0+1:j0+eta_v, 1), nxi, eta_v, rec)
    if (nc_has_var(ncid, 'hbls')) then
      call host_read_2d(ncid, 'hbls', hbls(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta, rec)
    endif
    if (nc_has_var(ncid, 'hbbl')) then
      call host_read_2d(ncid, 'hbbl', hbbl(i0:i0+nxi-1, j0:j0+neta-1), nxi, neta, rec)
    endif
  end subroutine load_2d_record

  subroutine load_3d_record(ncid, rec, tindx, i0, j0, nxi, neta, xi_u, eta_v)
    integer, intent(in) :: ncid, rec, tindx, i0, j0, nxi, neta, xi_u, eta_v
    integer :: itp
    real(rp), allocatable :: tmp(:,:,:)
    call host_read_3d(ncid, 'u', u(i0+1:i0+xi_u, j0:j0+neta-1, :, tindx), &
                      xi_u, neta, decomp%nz, rec)
    call host_read_3d(ncid, 'v', v(i0:i0+nxi-1, j0+1:j0+eta_v, :, tindx), &
                      nxi, eta_v, decomp%nz, rec)
    call host_read_3d(ncid, 'temp', t(i0:i0+nxi-1, j0:j0+neta-1, :, tindx, 1), &
                      nxi, neta, decomp%nz, rec)
    call host_read_3d(ncid, 'salt', t(i0:i0+nxi-1, j0:j0+neta-1, :, tindx, 2), &
                      nxi, neta, decomp%nz, rec)
    ! Passive tracers, when the file carries them (float32 in restarts).
    if (ntp > 0) then
      if (nc_has_var(ncid, trim(passive_name(1)))) then
        allocate(tmp(nxi, neta, decomp%nz))
        do itp = 1, ntp
          if (.not. nc_has_var(ncid, trim(passive_name(itp)))) exit
          call nc_read_3d(ncid, trim(passive_name(itp)), tmp, nxi, neta, decomp%nz, rec)
          tp(itp, i0:i0+nxi-1, j0:j0+neta-1, :, tindx) = real(tmp, pp)
        enddo
        deallocate(tmp)
        passive_loaded = .true.
      endif
    endif
    if (marbl_on .and. tindx == 1) call marbl_gpu_load_state(ncid, rec, i0, j0, nxi, neta)
  end subroutine load_3d_record

  subroutine load_coupling_fields(ncid, rec, i0, j0, nxi, neta, xi_u, eta_v)
    integer, intent(in) :: ncid, rec, i0, j0, nxi, neta, xi_u, eta_v
    call host_read_2d(ncid, 'DU_avg1', DU_avg1(i0+1:i0+xi_u, j0:j0+neta-1), xi_u, neta, rec)
    call host_read_2d(ncid, 'DV_avg1', DV_avg1(i0:i0+nxi-1, j0+1:j0+eta_v), nxi, eta_v, rec)
    call host_read_2d(ncid, 'DU_avg2', DU_avg2(i0+1:i0+xi_u, j0:j0+neta-1), xi_u, neta, rec)
    call host_read_2d(ncid, 'DV_avg2', DV_avg2(i0:i0+nxi-1, j0+1:j0+eta_v), nxi, eta_v, rec)
    call host_read_2d(ncid, 'DU_avg_bak', DU_avg_bak(i0+1:i0+xi_u, j0:j0+neta-1), xi_u, neta, rec)
    call host_read_2d(ncid, 'DV_avg_bak', DV_avg_bak(i0:i0+nxi-1, j0+1:j0+eta_v), nxi, eta_v, rec)
    DU_avg1 = DU_avg1 * umask
    DV_avg1 = DV_avg1 * vmask
    DU_avg2 = DU_avg2 * umask
    DV_avg2 = DV_avg2 * vmask
    DU_avg_bak = DU_avg_bak * umask
    DV_avg_bak = DV_avg_bak * vmask
  end subroutine load_coupling_fields

  subroutine load_rufrc_slot(ncid, rec, tindx, i0, j0, nxi, neta, xi_u, eta_v)
    integer, intent(in) :: ncid, rec, tindx, i0, j0, nxi, neta, xi_u, eta_v
    character(len=16) :: uname, vname
    if (nc_has_var(ncid, 'rufrc_bak')) then
      uname = 'rufrc_bak'
    elseif (nc_has_var(ncid, 'rufrc')) then
      uname = 'rufrc'
    else
      return
    endif
    if (nc_has_var(ncid, 'rvfrc_bak')) then
      vname = 'rvfrc_bak'
    elseif (nc_has_var(ncid, 'rvfrc')) then
      vname = 'rvfrc'
    else
      return
    endif
    call host_read_2d(ncid, trim(uname), rufrc_bak(i0+1:i0+xi_u, j0:j0+neta-1, tindx), &
                      xi_u, neta, rec)
    call host_read_2d(ncid, trim(vname), rvfrc_bak(i0:i0+nxi-1, j0+1:j0+eta_v, tindx), &
                      nxi, eta_v, rec)
    rufrc_bak(:, :, tindx) = rufrc_bak(:, :, tindx) * umask
    rvfrc_bak(:, :, tindx) = rvfrc_bak(:, :, tindx) * vmask
  end subroutine load_rufrc_slot

  subroutine infer_layout(xi_rho, eta_rho, i0, j0, nxi, neta)
    integer, intent(in)  :: xi_rho, eta_rho
    integer, intent(out) :: i0, j0, nxi, neta
    nxi = xi_rho
    neta = eta_rho
    if (xi_rho == case_llm + 2 .and. eta_rho == case_mmm + 2) then
      i0 = 0; j0 = 0
      case_periodic_x = .false.
      case_periodic_y = .false.
    elseif (xi_rho == case_llm .and. eta_rho == case_mmm) then
      i0 = 1; j0 = 1
      case_periodic_x = .true.
      case_periodic_y = .true.
    else
      write(*,*) 'grid dims', xi_rho, eta_rho, 'do not match LLm,MMm', case_llm, case_mmm
      error stop 'infer_layout'
    endif
    ! decomp already created as non-periodic default; re-init if periodic.
    if (case_periodic_x .or. case_periodic_y) then
      call free_state()
      call apply_decomp(case_periodic_x, case_periodic_y)
      call alloc_state(nt)
    endif
  end subroutine infer_layout

  subroutine sanitize_land()
    integer(ip) :: i, j, k, n
    do j = decomp%js, decomp%je
      do i = decomp%is, decomp%ie
        if (is_nan(rmask(i, j))) rmask(i, j) = 0.0_rp
        if (rmask(i, j) < 0.5_rp) then
          rmask(i, j) = 0.0_rp
          if (is_nan(h(i, j)) .or. h(i, j) <= 0.0_rp) h(i, j) = 1.0_rp
          if (is_nan(pm(i, j)) .or. pm(i, j) <= 0.0_rp) pm(i, j) = 1.0_rp
          if (is_nan(pn(i, j)) .or. pn(i, j) <= 0.0_rp) pn(i, j) = 1.0_rp
          ! CPU setup_grid writes f from lat_rho unmasked; Coriolis uses it on land.
          zeta(i, j, :) = 0.0_rp
          hbls(i, j) = 0.0_rp
          hbbl(i, j) = 0.0_rp
          t(i, j, :, :, :) = 0.0_rp
        endif
        if (is_nan(h(i, j)) .or. h(i, j) <= 0.0_rp) h(i, j) = 1.0_rp
        if (is_nan(pm(i, j)) .or. pm(i, j) <= 0.0_rp) pm(i, j) = 1.0e-4_rp
        if (is_nan(pn(i, j)) .or. pn(i, j) <= 0.0_rp) pn(i, j) = 1.0e-4_rp
        if (is_nan(f(i, j))) f(i, j) = 0.0_rp
      enddo
    enddo
    do n = 1, 3
      do k = 1, decomp%nz
        do j = decomp%js, decomp%je
          do i = decomp%is, decomp%ie
            if (is_nan(u(i, j, k, n))) u(i, j, k, n) = 0.0_rp
            if (is_nan(v(i, j, k, n))) v(i, j, k, n) = 0.0_rp
          enddo
        enddo
      enddo
    enddo
    where (is_nan(ubar)) ubar = 0.0_rp
    where (is_nan(vbar)) vbar = 0.0_rp
    where (is_nan(zeta)) zeta = 0.0_rp
    where (is_nan(t)) t = 0.0_rp
  end subroutine sanitize_land

  subroutine pad_physical_halo()
    integer(ip) :: i, j, is, ie, js, je, nx, ny
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    nx = decomp%nx; ny = decomp%ny
    if (decomp%edge_w) then
      do j = js, je
        do i = is, -1
          call copy_col(i, 0, j)
        enddo
      enddo
    endif
    if (decomp%edge_e) then
      do j = js, je
        do i = nx + 2, ie
          call copy_col(i, nx + 1, j)
        enddo
      enddo
    endif
    if (decomp%edge_s) then
      do i = is, ie
        do j = js, -1
          call copy_row_metrics(i, j, 0)
          call zero_row_state(i, j)
        enddo
      enddo
    endif
    if (decomp%edge_n) then
      do i = is, ie
        do j = ny + 2, je
          call copy_row_metrics(i, j, ny + 1)
          call zero_row_state(i, j)
        enddo
      enddo
    endif
  contains
    subroutine copy_col(idst, isrc, j)
      integer(ip), intent(in) :: idst, isrc, j
      h(idst, j) = h(isrc, j); pm(idst, j) = pm(isrc, j); pn(idst, j) = pn(isrc, j)
      rmask(idst, j) = rmask(isrc, j); f(idst, j) = f(isrc, j)
      lonr(idst, j) = lonr(isrc, j); latr(idst, j) = latr(isrc, j); angler(idst, j) = angler(isrc, j)
      zeta(idst, j, :) = zeta(isrc, j, :)
      ubar(idst, j, :) = ubar(isrc, j, :)
      vbar(idst, j, :) = vbar(isrc, j, :)
      u(idst, j, :, :) = u(isrc, j, :, :)
      v(idst, j, :, :) = v(isrc, j, :, :)
      t(idst, j, :, :, :) = t(isrc, j, :, :, :)
      hbls(idst, j) = hbls(isrc, j); hbbl(idst, j) = hbbl(isrc, j)
    end subroutine copy_col
    subroutine copy_row_metrics(i, jdst, jsrc)
      integer(ip), intent(in) :: i, jdst, jsrc
      h(i, jdst) = h(i, jsrc); pm(i, jdst) = pm(i, jsrc); pn(i, jdst) = pn(i, jsrc)
      f(i, jdst) = f(i, jsrc)
      lonr(i, jdst) = lonr(i, jsrc); latr(i, jdst) = latr(i, jsrc); angler(i, jdst) = angler(i, jsrc)
    end subroutine copy_row_metrics
    subroutine zero_row_state(i, jdst)
      integer(ip), intent(in) :: i, jdst
      rmask(i, jdst) = 0.0_rp
      umask(i, jdst) = 0.0_rp
      vmask(i, jdst) = 0.0_rp
      zeta(i, jdst, :) = 0.0_rp
      ubar(i, jdst, :) = 0.0_rp
      vbar(i, jdst, :) = 0.0_rp
      u(i, jdst, :, :) = 0.0_rp
      v(i, jdst, :, :) = 0.0_rp
      t(i, jdst, :, :, :) = 0.0_rp
      hbls(i, jdst) = 0.0_rp
      hbbl(i, jdst) = 0.0_rp
    end subroutine zero_row_state
  end subroutine pad_physical_halo

  subroutine report_loaded_case()
    integer(ip) :: nwet, i, j, k
    real(rp) :: hmin, hmax, tmin, tmax, zmax, umax
    nwet = count(rmask(1:decomp%nx, 1:decomp%ny) > 0.5_rp)
    hmin = huge(1.0_rp); hmax = -huge(1.0_rp)
    tmin = huge(1.0_rp); tmax = -huge(1.0_rp)
    zmax = 0.0_rp; umax = 0.0_rp
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) > 0.5_rp) then
          hmin = min(hmin, h(i, j))
          hmax = max(hmax, h(i, j))
          zmax = max(zmax, abs(zeta(i, j, 1)))
          do k = 1, decomp%nz
            tmin = min(tmin, t(i, j, k, 1, 1))
            tmax = max(tmax, t(i, j, k, 1, 1))
          enddo
        endif
      enddo
    enddo
    umax = 0.0_rp
    ic_u_i = 0; ic_u_j = 0; ic_u_k = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx + 1
        if (umask(i, j) <= 0.5_rp) cycle
        do k = 1, decomp%nz
          if (abs(u(i, j, k, 1)) > umax) then
            umax = abs(u(i, j, k, 1))
            ic_u_i = i; ic_u_j = j; ic_u_k = k
          endif
        enddo
      enddo
    enddo
    ic_u_val = umax
    write(*,'(a)') 'loaded case'
    write(*,'(a,a)') '  title  = ', trim(case_title)
    write(*,'(a,3i6)') '  nx,ny,nz = ', decomp%nx, decomp%ny, decomp%nz
    write(*,'(a,i8)') '  wet rho = ', nwet
    write(*,'(a,2es14.6)') '  h min,max = ', hmin, hmax
    write(*,'(a,2es14.6)') '  T min,max = ', tmin, tmax
    write(*,'(a,es14.6)') '  max |zeta| = ', zmax
    write(*,'(a,es14.6,a,3i6)') '  max |u|    = ', umax, ' at i,j,k =', ic_u_i, ic_u_j, ic_u_k
    write(*,'(a,es14.6)') '  ocean_time = ', start_time
    write(*,'(a,2i8,a,l1)') '  ntstart, forw_start = ', ntstart, forw_start, &
         '  exact = ', case_exact_restart
    write(*,'(a,3es12.4)') '  dt, theta_s, hc = ', dt, theta_s, hc
    iic = 0
    knew = 1
    call set_depth()
    write(*,'(a,es14.6)') '  max |Hz|   = ', maxval(abs(Hz(1:decomp%nx, 1:decomp%ny, :)), &
         mask=spread(rmask(1:decomp%nx, 1:decomp%ny) > 0.5_rp, 3, decomp%nz))
  end subroutine report_loaded_case

  subroutine write_toy_case(dir, llm, mmm, nz)
    character(len=*), intent(in) :: dir
    integer(ip),      intent(in) :: llm, mmm, nz
    integer :: ncid, d_xi, d_eta, d_xiu, d_etav, d_s, d_t, nxi, neta
    integer :: i, j, k
    integer :: id_h, id_pm, id_pn, id_mask, id_f, id_z, id_ub, id_vb, id_u, id_v, id_t, id_s, id_ot
    real(rp), allocatable :: a2(:,:), au(:,:), av(:,:), a3(:,:,:), au3(:,:,:), av3(:,:,:)
    integer :: unit

    nxi = llm + 2
    neta = mmm + 2
    allocate(a2(nxi, neta), au(nxi - 1, neta), av(nxi, neta - 1))
    allocate(a3(nxi, neta, nz), au3(nxi - 1, neta, nz), av3(nxi, neta - 1, nz))

    do j = 1, neta
      do i = 1, nxi
        a2(i, j) = 200.0_rp + real(i + j, rp)
      enddo
    enddo
    call nc_create(trim(dir)//'/toy_grid.nc', ncid)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    d_xiu = nc_def_dim(ncid, 'xi_u', nxi - 1)
    d_etav = nc_def_dim(ncid, 'eta_v', neta - 1)
    id_h = nc_def_var(ncid, 'h', (/d_xi, d_eta/))
    id_pm = nc_def_var(ncid, 'pm', (/d_xi, d_eta/))
    id_pn = nc_def_var(ncid, 'pn', (/d_xi, d_eta/))
    id_mask = nc_def_var(ncid, 'mask_rho', (/d_xi, d_eta/))
    id_f = nc_def_var(ncid, 'f', (/d_xi, d_eta/))
    call nc_enddef(ncid)
    call nc_put_2d(ncid, 'h', a2)
    a2 = 1.0_rp / 12000.0_rp
    call nc_put_2d(ncid, 'pm', a2)
    call nc_put_2d(ncid, 'pn', a2)
    a2 = 1.0_rp
    a2(3, 3) = 0.0_rp
    call nc_put_2d(ncid, 'mask_rho', a2)
    a2 = 1.0e-4_rp
    call nc_put_2d(ncid, 'f', a2)
    call nc_close(ncid)

    a2 = 0.01_rp
    a2(3, 3) = 0.0_rp
    au = 0.02_rp
    av = -0.03_rp
    do k = 1, nz
      a3(:, :, k) = 15.0_rp + 0.1_rp * real(k, rp)
      au3(:, :, k) = 0.05_rp
      av3(:, :, k) = -0.04_rp
    enddo
    call nc_create(trim(dir)//'/toy_init.nc', ncid)
    d_t = nc_def_dim(ncid, 'ocean_time', 1)
    d_s = nc_def_dim(ncid, 's_rho', nz)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    d_xiu = nc_def_dim(ncid, 'xi_u', nxi - 1)
    d_etav = nc_def_dim(ncid, 'eta_v', neta - 1)
    id_ot = nc_def_var(ncid, 'ocean_time', (/d_t/))
    id_z = nc_def_var(ncid, 'zeta', (/d_xi, d_eta/))
    id_ub = nc_def_var(ncid, 'ubar', (/d_xiu, d_eta/))
    id_vb = nc_def_var(ncid, 'vbar', (/d_xi, d_etav/))
    id_u = nc_def_var(ncid, 'u', (/d_xiu, d_eta, d_s/))
    id_v = nc_def_var(ncid, 'v', (/d_xi, d_etav, d_s/))
    id_t = nc_def_var(ncid, 'temp', (/d_xi, d_eta, d_s/))
    id_s = nc_def_var(ncid, 'salt', (/d_xi, d_eta, d_s/))
    call nc_enddef(ncid)
    call nc_put_1d(ncid, 'ocean_time', (/ 86400.0_rp /))
    call nc_put_2d(ncid, 'zeta', a2)
    call nc_put_2d(ncid, 'ubar', au)
    call nc_put_2d(ncid, 'vbar', av)
    call nc_put_3d(ncid, 'u', au3)
    call nc_put_3d(ncid, 'v', av3)
    call nc_put_3d(ncid, 'temp', a3)
    a3 = 35.0_rp
    call nc_put_3d(ncid, 'salt', a3)
    call nc_close(ncid)

    open(newunit=unit, file=trim(dir)//'/toy.nml', status='replace')
    write(unit,'(a)') '&param_settings'
    write(unit,'(a,i0)') '    llm = ', llm
    write(unit,'(a,i0)') '    mmm = ', mmm
    write(unit,'(a)')    '    np_eta = 1'
    write(unit,'(a)')    '    np_xi = 1'
    write(unit,'(a)')    '    nt_bgc = 0'
    write(unit,'(a)')    '    nt_passive = 0'
    write(unit,'(a,i0)') '    nz = ', nz
    write(unit,'(a)')    '/'
    write(unit,'(a)') '&grid_settings'
    write(unit,'(a)') "    grdname = 'toy_grid.nc'"
    write(unit,'(a)') '/'
    write(unit,'(a)') '&initial_conditions'
    write(unit,'(a)') "    inifile = 'toy_init.nc'"
    write(unit,'(a)') '/'
    write(unit,'(a)') '&s_coord'
    write(unit,'(a)') '    hc = 250.0'
    write(unit,'(a)') '    theta_b = 6.0'
    write(unit,'(a)') '    theta_s = 6.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&time_stepping'
    write(unit,'(a)') '    dt = 900.0'
    write(unit,'(a)') '    ndtfast = 20'
    write(unit,'(a)') '    ninfo = 1'
    write(unit,'(a)') '    ntimes = 3'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&bottom_drag_settings'
    write(unit,'(a)') '    rdrg = 0.0'
    write(unit,'(a)') '    rdrg2 = 0.001'
    write(unit,'(a)') '    zob = 0.01'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&vertical_mixing_settings'
    write(unit,'(a)') '    akt_bak = 0.0, 0.0'
    write(unit,'(a)') '    akv_bak = 0.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&lateral_visc_settings'
    write(unit,'(a)') '    visc2 = 0.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&tracer_diff2'
    write(unit,'(a)') '    tnu2 = 0.0, 0.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&rho0_settings'
    write(unit,'(a)') '    rho0 = 1000.0'
    write(unit,'(a)') '/'
    write(unit,'(a)') '&simulation_name_settings'
    write(unit,'(a)') "    output_root_name = './output'"
    write(unit,'(a)') "    title = 'toy_case'"
    write(unit,'(a)') '/'
    close(unit)
    deallocate(a2, au, av, a3, au3, av3)
  end subroutine write_toy_case

  function dirname(path) result(dir)
    character(len=*), intent(in) :: path
    character(len=512) :: dir
    integer :: n
    n = index(path, '/', back=.true.)
    if (n > 0) then
      dir = path(1:n-1)
    else
      dir = '.'
    endif
  end function dirname

  function resolve_path(dir, path) result(out)
    character(len=*), intent(in) :: dir, path
    character(len=512) :: out
    if (len_trim(path) == 0) then
      out = ''
    elseif (path(1:1) == '/') then
      out = trim(path)
    else
      out = trim(dir)//'/'//trim(path)
    endif
  end function resolve_path

end module case_io
