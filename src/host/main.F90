program roms_gpu
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use step_driver
  use case_io, only: load_case, report_loaded_case
  use his_io, only: maybe_wrt_output
  use arch_mod, only: init_arch, arch_name
  use comm_mod, only: init_comm, finalize_comm, comm_rank, comm_size, is_io_rank
  use io_server, only: io_server_run, io_finalize
  implicit none

  integer(ip) :: gnx, gny, nz, halo, npx, npy, rank
  integer(ip) :: ndtfast_in, ntimes_in, nt_in, istep
  real(rp) :: dt_in, H0, dx, f0
  character(len=32) :: mode
  character(len=512) :: nml_path

  mode = 'geom'
  if (command_argument_count() >= 1) call get_command_argument(1, mode)
  call init_comm()
  if (is_io_rank()) then
    ! Output server: no GPU context, no model state. Serves until QUIT.
    call io_server_run()
    call finalize_comm()
    stop
  endif
  call init_arch()

  ! Ideal-box defaults stay one tile. A loaded case uses namelist
  ! np_xi/np_eta when comm_size matches, otherwise 1 x 1.
  gnx = 32; gny = 32; nz = 16; halo = 2
  npx = 1; npy = 1; rank = comm_rank()
  ndtfast_in = 60; ntimes_in = 3; nt_in = 2
  dt_in = 100.0_rp
  H0 = 4000.0_rp; dx = 12000.0_rp; f0 = 1.0e-4_rp

  if (trim(mode) == 'load' .or. trim(mode) == 'run') then
    nml_path = 'namelist.nml'
    if (command_argument_count() >= 2) call get_command_argument(2, nml_path)
    call load_case(nml_path)
    write(*,'(a,a)') 'roms_gpu  architecture = ', trim(arch_name())
    write(*,'(a,2i6)') '  MPI rank, size = ', comm_rank(), comm_size()
    write(*,'(a)') 'roms_gpu  (loaded case)'
    write(*,'(a,3i6)') '  local nx,ny,nz = ', decomp%nx, decomp%ny, decomp%nz
    write(*,'(a,2i6)') '  nfast, ndtfast = ', nfast, ndtfast
    write(*,'(a,4l2)') '  edges W E S N  = ', decomp%edge_w, decomp%edge_e, &
                        decomp%edge_s, decomp%edge_n
    write(*,'(a,3l2,a,4l2)') '  flux,bulk,bry = ', have_flux, have_bulk, have_bry, &
                        '  OBC WENS =', obc_west, obc_east, obc_south, obc_north
    write(*,'(a,2l2)') '  pot,bry tides = ', pot_tides, bry_tides
    call report_loaded_case()
    if (trim(mode) == 'run') then
      do istep = 1, max(ntimes, 1)
        iic = ntstart + istep - 1
        call run_baroclinic_step()
        call maybe_wrt_output()
      enddo
      call io_finalize()
    endif
  else
  call init_decomp(gnx, gny, nz, halo, npx, npy, rank, .true., .true.)
  call init_time_state(nz, ndtfast_in, dt_in, ntimes_in, nt_in)
  call alloc_state(nt)
  call init_ideal_box(H0, dx, f0)

  write(*,'(a,a)') 'roms_gpu  architecture = ', trim(arch_name())
  write(*,'(a,2i6)') '  MPI rank, size = ', comm_rank(), comm_size()
  write(*,'(a,3i6)') '  local nx,ny,nz = ', decomp%nx, decomp%ny, decomp%nz
  write(*,'(a,2i6)') '  nfast, ndtfast = ', nfast, ndtfast
  write(*,'(a,4l2)') '  edges W E S N  = ', decomp%edge_w, decomp%edge_e, &
                      decomp%edge_s, decomp%edge_n
  write(*,'(a,4i4)') '  nbrs  W E S N  = ', decomp%nbr_w, decomp%nbr_e, &
                      decomp%nbr_s, decomp%nbr_n

  if (trim(mode) == 'geom') then
    call run_geom_only()
  elseif (trim(mode) == 'baro') then
    call run_barotropic_only()
  elseif (trim(mode) == 'prsgrd') then
    call run_prsgrd_only()
  elseif (trim(mode) == 'advect') then
    call run_advect_only()
  elseif (trim(mode) == 'step') then
    call run_baroclinic_step()
  elseif (trim(mode) == 'lmd') then
    call run_lmd_only()
  else
    write(*,*) 'usage: roms_gpu [geom|baro|prsgrd|advect|step|lmd|load|run [namelist]]'
    error stop
  endif
  endif

  call io_finalize()
  call free_state()
  call finalize_comm()
end program roms_gpu
