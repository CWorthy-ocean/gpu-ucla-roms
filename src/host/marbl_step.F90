#include "mem_attr.inc"
module marbl_gpu
  ! MARBL inside the GPU model: configuration, tracer/state setup, the
  ! per-call driver (surface fluxes, carbonate chemistry, interior
  ! tendencies) and the restart/history hooks. Mirrors the CPU driver
  ! marbl_driver.F90: MARBL runs every marbl_timestep_ratio steps at the
  ! end of the tracer step on the nnew slot, surface flux first (applied to
  ! the top layer), then the interior tendencies, with ifrac = 0 and
  ! atmospheric pressure 1 atm.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use arch_mod, only: sync_arch
  use nc_io
  use frc_io, only: init_bgc_forcing, set_bgc_forces, uwnd, vwnd, bgc_frc_on
  use passive_mod, only: ntp, tp, set_passive_marbl, passive_marbl, passive_name
  use marbl_host
  use marbl_sflux_mod
  use marbl_interior_mod
  use io_server, only: io_stage_2d, io_stage_3d
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  private

  logical, public :: marbl_on = .false.
  character(len=512), public :: marbl_config_file = 'marbl_in'
  real(rp), public :: marbl_timestep = 3600.0_rp
  integer(ip), public :: marbl_timestep_ratio = 1
  real(rp), public :: xco2_default = 284.7_rp       ! ppm when no xco2 forcing (CPU default)
  integer(ip) :: ncalls = 0
  real(rp), parameter :: Cp = 3985.0_rp

  ! history diagnostics (2-D, surface flux module slots)
  integer, parameter :: NHIS = 7
  character(len=16), parameter :: his_name(NHIS) = [character(len=16) :: &
       'PH', 'pCO2SURF', 'DpCO2', 'FG_CO2', 'FG_ALT_CO2', 'PH_ALT_CO2', 'pCO2SURF_ALT_CO2']
  integer, parameter :: his_slot(NHIS) = [D_PH, D_PCO2, D_DPCO2, D_FGCO2, D_FGCO2_ALT, D_PH_ALT, D_PCO2_ALT]
  ! history diagnostics from the interior carbonate kernel: 2-D saturation
  ! depths and 3-D carbonate fields (CPU names, mmol/m^3 and m)
  integer, parameter :: NHIS2I = 2, NHIS3 = 4
  character(len=16), parameter :: his2i_name(NHIS2I) = [character(len=16) :: 'zsatcalc', 'zsatarag']
  character(len=16), parameter :: his3_name(NHIS3) = [character(len=16) :: &
       'CO3', 'CO3_ALT_CO2', 'co3_sat_calc', 'co3_sat_arag']

  public :: read_marbl_namelist
  public :: marbl_gpu_setup_tracers
  public :: marbl_gpu_init_forcing
  public :: marbl_gpu_load_state
  public :: marbl_gpu_step
  public :: marbl_gpu_nhis
  public :: marbl_gpu_nhis3
  public :: marbl_gpu_define_his
  public :: marbl_gpu_stage_his
  public :: marbl_gpu_define_rst
  public :: marbl_gpu_write_rst

contains

  subroutine read_marbl_namelist(unit, nml_dir, nt_bgc)
    ! &marbl_biogeochemistry_settings as the CPU model; MARBL is on when
    ! nt_bgc > 0 in param_settings.
    integer, intent(in) :: unit
    character(len=*), intent(in) :: nml_dir
    integer(ip), intent(in) :: nt_bgc
    integer :: ios
    character(len=512) :: cfg
    real(rp) :: tstep
    character(len=32) :: marbl_tracers_to_write(64)
    character(len=400) :: marbl_diagnostics_to_write(64)
    namelist /marbl_biogeochemistry_settings/ marbl_config_file, marbl_timestep, &
         marbl_tracers_to_write, marbl_diagnostics_to_write
    marbl_on = (nt_bgc > 0)
    if (.not. marbl_on) return
    marbl_config_file = 'marbl_in'
    marbl_timestep = 3600.0_rp
    marbl_tracers_to_write = ''
    marbl_diagnostics_to_write = ''
    rewind(unit)
    read(unit, nml=marbl_biogeochemistry_settings, iostat=ios)
    if (ios > 0) error stop 'marbl_biogeochemistry_settings: read error'
    cfg = marbl_config_file
    if (cfg(1:1) /= '/') marbl_config_file = trim(nml_dir)//'/'//trim(cfg)
    tstep = marbl_timestep
    write(*,'(a,i0,a,a,a,f8.1)') 'MARBL on: nt_bgc = ', nt_bgc, '  config ', trim(marbl_config_file), &
         '  requested timestep ', tstep
  end subroutine read_marbl_namelist

  ! Initialise MARBL (unchanged settings/init code), publish the tracer
  ! names to the passive module and allocate the device arrays. Called
  ! after read_namelist (decomp, dt known), before alloc_passive.
  subroutine marbl_gpu_setup_tracers()
    if (.not. marbl_on) return
    call marbl_host_init(decomp%nz, trim(marbl_config_file), .true.)
    call set_passive_marbl(marbl_tracer_names)
    call set_marbl_par()
    call alloc_marbl_sflux(ntp)
    call alloc_marbl_interior(decomp%nz, ntp, .false.)
    marbl_timestep_ratio = max(1_ip, int(marbl_timestep / dt, ip))
    marbl_timestep = dt * real(marbl_timestep_ratio, rp)
    sf_xco2 = xco2_default
    sf_xco2_alt = xco2_default
    write(*,'(a,i0,a,f8.1,a)') 'MARBL timestep ratio ', marbl_timestep_ratio, '  (', marbl_timestep, ' s)'
  end subroutine marbl_gpu_setup_tracers

  subroutine marbl_gpu_init_forcing()
    if (.not. marbl_on) return
    call init_bgc_forcing()
  end subroutine marbl_gpu_init_forcing

  ! Saved state from the initial/restart file (CPU names). Missing
  ! variables leave the pH at 0 = "no previous solution".
  subroutine marbl_gpu_load_state(ncid, rec, i0, j0, nxi, neta, fi0, fj0)
    integer, intent(in) :: ncid, rec, i0, j0, nxi, neta, fi0, fj0
    real(rp), allocatable :: t2(:,:), t3(:,:,:)
    if (.not. marbl_on) return
    allocate(t2(nxi, neta), t3(nxi, neta, decomp%nz))
    if (nc_has_var(ncid, 'MARBL_PH_SURF')) then
      call nc_read_2d(ncid, 'MARBL_PH_SURF', t2, nxi, neta, rec, i0f=fi0, j0f=fj0)
      sf_ph(i0:i0+nxi-1, j0:j0+neta-1) = t2
    endif
    if (nc_has_var(ncid, 'MARBL_PH_SURF_ALT_CO2')) then
      call nc_read_2d(ncid, 'MARBL_PH_SURF_ALT_CO2', t2, nxi, neta, rec, i0f=fi0, j0f=fj0)
      sf_ph_alt(i0:i0+nxi-1, j0:j0+neta-1) = t2
    endif
    if (nc_has_var(ncid, 'MARBL_PH_3D')) then
      call nc_read_3d(ncid, 'MARBL_PH_3D', t3, nxi, neta, decomp%nz, rec, i0f=fi0, j0f=fj0)
      it_ph(i0:i0+nxi-1, j0:j0+neta-1, :) = t3
    endif
    if (nc_has_var(ncid, 'MARBL_PH_3D_ALT_CO2')) then
      call nc_read_3d(ncid, 'MARBL_PH_3D_ALT_CO2', t3, nxi, neta, decomp%nz, rec, i0f=fi0, j0f=fj0)
      it_pha(i0:i0+nxi-1, j0:j0+neta-1, :) = t3
    endif
    deallocate(t2, t3)
    write(*,'(a)') 'MARBL saved state (surface and 3-D pH) loaded where present'
  end subroutine marbl_gpu_load_state

  ! One MARBL call: at the end of the tracer step (after the passive
  ! corrector and its open boundaries), as the CPU driver.
  subroutine marbl_gpu_step()
    integer(ip) :: i, j, m, nx, ny, nz, nn, islot
    real(rp) :: dtm, rc, tfrc
    if (.not. marbl_on) return
    if (mod(iic, marbl_timestep_ratio) /= 0) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    nn = nnew
    dtm = marbl_timestep
    rc = rho0_case * Cp
    ! forcing at the corrector's set_forces time (CPU: '1/2 fwd', n+dt/2);
    ! by now the step driver has moved frc_slot on to the n+3/2 boundary time.
    islot = frc_slot
    frc_slot = 1
    tfrc = frc_modtime()
    frc_slot = islot
    call set_bgc_forces(tfrc, sf_dust, sf_iron, sf_nox, sf_nhy, sf_xco2, sf_xco2_alt)
    ! wind speed squared (bulk winds on the device) and shortwave in W/m2
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        sf_u10sqr(i, j) = (uwnd(i, j)**2) + (vwnd(i, j)**2)
        it_qsw(i, j) = srflx(i, j) * rc
      enddo
    enddo
    ! surface fluxes, applied to the top layer of slot nnew
    call marbl_sflux_compute(msf, ntp, nz, nz, nn, tp, nt, t, rmask)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do m = 1, ntp
      do j = 1, ny
        do i = 1, nx
          if (rmask(i, j) /= 0.0_rp) then
            tp(m, i, j, nz, nn) = real(real(tp(m, i, j, nz, nn), rp) + &
                 sf_flux(i, j, m) * dtm / Hz(i, j, nz), pp)
          endif
        enddo
      enddo
    enddo
    ! interior: carbonate chemistry, then the column ecosystem/sinking sweep
    call marbl_carbonate_compute(ntp, nz, nn, 3_ip, tp, nt, t, z_r, rmask)
    call marbl_zsat_compute(nz, z_r, z_w, rmask)
    call marbl_interior_compute(ntp, nz, nn, 3_ip, tp, nt, t, z_r, z_w, Hz, rmask, sf_dust, dtm, .false.)
    ncalls = ncalls + 1
    if (ncalls <= 3 .or. mod(ncalls, 24) == 0) then
      call sync_arch()
      write(*,'(a,i0,a,i0,a,i0,a,i0)') 'MARBL call ', ncalls, ' at iic ', iic, &
           '  surface solver failures ', count(sf_err(1:nx, 1:ny) /= 0), &
           '  interior solver failures ', sum(it_err(1:nx, 1:ny))
    endif
  end subroutine marbl_gpu_step

  ! Debug: NaN census of the surface-flux inputs/outputs after the first
  ! call, plus one sample point.

  integer function marbl_gpu_nhis()
    ! number of 2-D MARBL history fields (staging capacity)
    marbl_gpu_nhis = 0
    if (marbl_on) marbl_gpu_nhis = NHIS + NHIS2I
  end function marbl_gpu_nhis

  integer function marbl_gpu_nhis3()
    ! number of 3-D (nz-level) MARBL history fields (staging capacity)
    marbl_gpu_nhis3 = 0
    if (marbl_on) marbl_gpu_nhis3 = NHIS3
  end function marbl_gpu_nhis3

  subroutine marbl_gpu_define_his(ncid, d_xi, d_eta, d_s, d_t)
    integer, intent(in) :: ncid, d_xi, d_eta, d_s, d_t
    integer :: n, id
    if (.not. marbl_on) return
    do n = 1, NHIS
      id = nc_def_var_f(ncid, trim(his_name(n)), (/d_xi, d_eta, d_t/))
    enddo
    do n = 1, NHIS2I
      id = nc_def_var_f(ncid, trim(his2i_name(n)), (/d_xi, d_eta, d_t/))
    enddo
    do n = 1, NHIS3
      id = nc_def_var_f(ncid, trim(his3_name(n)), (/d_xi, d_eta, d_s, d_t/))
    enddo
  end subroutine marbl_gpu_define_his

  subroutine marbl_gpu_stage_his(i0, i1, j0, j1, di, dj)
    integer, intent(in) :: i0, i1, j0, j1, di, dj
    integer :: n, nz
    if (.not. marbl_on) return
    nz = decomp%nz
    do n = 1, NHIS
      call io_stage_2d(trim(his_name(n)), sf_diag(decomp%is, decomp%js, his_slot(n)), i0, i1, j0, j1, di, dj)
    enddo
    call io_stage_2d(trim(his2i_name(1)), it_zsatc(decomp%is, decomp%js), i0, i1, j0, j1, di, dj)
    call io_stage_2d(trim(his2i_name(2)), it_zsata(decomp%is, decomp%js), i0, i1, j0, j1, di, dj)
    call io_stage_3d(trim(his3_name(1)), it_co3(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
    call io_stage_3d(trim(his3_name(2)), it_co3a(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
    call io_stage_3d(trim(his3_name(3)), it_co3sat(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
    call io_stage_3d(trim(his3_name(4)), it_co3sata(decomp%is, decomp%js, 1), 1, nz, i0, i1, j0, j1, di, dj)
  end subroutine marbl_gpu_stage_his

  subroutine marbl_gpu_define_rst(ncid, d_xi, d_eta, d_s, d_t)
    integer, intent(in) :: ncid, d_xi, d_eta, d_s, d_t
    integer :: id
    if (.not. marbl_on) return
    id = nc_def_var(ncid, 'MARBL_PH_SURF', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'MARBL_PH_SURF_ALT_CO2', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'MARBL_PH_3D', (/d_xi, d_eta, d_s, d_t/))
    id = nc_def_var(ncid, 'MARBL_PH_3D_ALT_CO2', (/d_xi, d_eta, d_s, d_t/))
  end subroutine marbl_gpu_define_rst

  subroutine marbl_gpu_write_rst(ncid, rec, i0, i1, j0, j1, fi0, fj0)
    ! fi0, fj0: 1-based global file offsets of this tile's first interior point
    integer, intent(in) :: ncid, rec, i0, i1, j0, j1, fi0, fj0
    real(rp), allocatable :: t2(:,:), t3(:,:,:)
    integer :: nxi, neta
    if (.not. marbl_on) return
    nxi = i1 - i0 + 1; neta = j1 - j0 + 1
    call sync_arch()
    allocate(t2(nxi, neta), t3(nxi, neta, decomp%nz))
    t2 = sf_ph(i0:i1, j0:j1);     call nc_put_2d_rec(ncid, 'MARBL_PH_SURF', t2, rec, fi0, fj0)
    t2 = sf_ph_alt(i0:i1, j0:j1); call nc_put_2d_rec(ncid, 'MARBL_PH_SURF_ALT_CO2', t2, rec, fi0, fj0)
    t3 = it_ph(i0:i1, j0:j1, :);  call nc_put_3d_rec(ncid, 'MARBL_PH_3D', t3, rec, fi0, fj0)
    t3 = it_pha(i0:i1, j0:j1, :); call nc_put_3d_rec(ncid, 'MARBL_PH_3D_ALT_CO2', t3, rec, fi0, fj0)
    deallocate(t2, t3)
  end subroutine marbl_gpu_write_rst

end module marbl_gpu
