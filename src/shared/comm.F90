module comm_mod
  ! MPI wrapper for halo deliver and the optional I/O server.
  ! World = compute ranks plus, when the job has >= 2 ranks and
  ! ROMS_GPU_IO_SERVER is not '0', one I/O server on the last world rank.
  ! Compute ranks exchange halos on compute_comm (comm_rank/comm_size
  ! refer to it); the server only receives staged output records over
  ! the world communicator (see io_server). One compute rank is a no-op
  ! for halos (NBR_SELF / NBR_NONE stay in halo_mod). Real neighbor ranks
  ! (>= 0) go through host-staged MPI_Sendrecv of the same pack buffers.
  ! CUDA graphs stay off when comm_size > 1 because deliver is host MPI.
  use kinds
  implicit none
#ifdef USE_MPI
  include 'mpif.h'
#endif
  private

  integer(ip) :: my_rank = 0
  integer(ip) :: nproc = 1
  integer :: world_rank = 0, world_size = 1
  integer :: world_comm = 0, compute_comm = 0
  integer :: io_rank_w = -1          ! world rank of the I/O server, -1 = none
  logical :: io_rank_flag = .false.
  logical :: ready = .false.

  public :: init_comm
  public :: finalize_comm
  public :: comm_rank
  public :: comm_size
  public :: comm_exchange2
  public :: comm_exchange3
  public :: have_io_server
  public :: is_io_rank
  public :: io_server_rank
  public :: comm_world
  public :: comm_world_rank

contains

  subroutine init_comm()
#ifdef USE_MPI
    integer :: ierr, color, stat, r, n
    logical :: flag
    character(len=8) :: val
    call MPI_Initialized(flag, ierr)
    if (.not. flag) then
      call MPI_Init(ierr)
      if (ierr /= MPI_SUCCESS) error stop 'init_comm: MPI_Init'
    endif
    world_comm = MPI_COMM_WORLD
    call MPI_Comm_rank(world_comm, world_rank, ierr)
    call MPI_Comm_size(world_comm, world_size, ierr)
    io_rank_w = -1
    if (world_size >= 2) then
      call get_environment_variable('ROMS_GPU_IO_SERVER', val, status=stat)
      if (.not. (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '0')) &
           io_rank_w = world_size - 1
    endif
    io_rank_flag = (world_rank == io_rank_w)
    if (io_rank_w >= 0) then
      color = merge(1, 0, io_rank_flag)
      call MPI_Comm_split(world_comm, color, world_rank, compute_comm, ierr)
      if (ierr /= MPI_SUCCESS) error stop 'init_comm: MPI_Comm_split'
    else
      compute_comm = world_comm
    endif
    call MPI_Comm_rank(compute_comm, r, ierr)
    call MPI_Comm_size(compute_comm, n, ierr)
    my_rank = int(r, ip)
    nproc = int(n, ip)
#endif
    ready = .true.
  end subroutine init_comm

  subroutine finalize_comm()
#ifdef USE_MPI
    integer :: ierr
    logical :: flag
    call MPI_Finalized(flag, ierr)
    if (.not. flag) call MPI_Finalize(ierr)
#endif
    ready = .false.
    my_rank = 0
    nproc = 1
    io_rank_w = -1
    io_rank_flag = .false.
  end subroutine finalize_comm

  integer(ip) function comm_rank()
    comm_rank = my_rank
  end function comm_rank

  integer(ip) function comm_size()
    comm_size = nproc
  end function comm_size

  logical function have_io_server()
    have_io_server = (io_rank_w >= 0)
  end function have_io_server

  logical function is_io_rank()
    is_io_rank = io_rank_flag
  end function is_io_rank

  integer function io_server_rank()
    io_server_rank = io_rank_w
  end function io_server_rank

  integer function comm_world()
    comm_world = world_comm
  end function comm_world

  integer function comm_world_rank()
    comm_world_rank = world_rank
  end function comm_world_rank

  integer function mpi_dest(nbr)
    integer(ip), intent(in) :: nbr
#ifdef USE_MPI
    if (nbr >= 0) then
      mpi_dest = int(nbr)
    else
      mpi_dest = MPI_PROC_NULL
    endif
#else
    if (nbr >= 0) error stop 'comm: MPI neighbor but this build has no USE_MPI'
    mpi_dest = -1
#endif
  end function mpi_dest

  subroutine comm_exchange2(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    real(rp), intent(in)    :: send_lo(:,:), send_hi(:,:)
    real(rp), intent(out)   :: recv_lo(:,:), recv_hi(:,:)
#ifdef USE_MPI
    integer :: n, dest_lo, dest_hi, ierr, stat(MPI_STATUS_SIZE)
    real(rp), allocatable :: s_lo(:), s_hi(:), r_lo(:), r_hi(:)
    if (.not. ready) call init_comm()
    n = size(send_lo)
    if (size(send_hi) /= n .or. size(recv_lo) /= n .or. size(recv_hi) /= n) &
         error stop 'comm_exchange2: buffer size mismatch'
    allocate(s_lo(n), s_hi(n), r_lo(n), r_hi(n))
    s_lo = reshape(send_lo, (/n/))
    s_hi = reshape(send_hi, (/n/))
    r_lo = 0.0_rp
    r_hi = 0.0_rp
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    call MPI_Sendrecv(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 101, &
                      r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 101, &
                      compute_comm, stat, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_exchange2: west/east slo'
    call MPI_Sendrecv(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 102, &
                      r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 102, &
                      compute_comm, stat, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_exchange2: east/west shi'
    recv_lo = reshape(r_lo, shape(recv_lo))
    recv_hi = reshape(r_hi, shape(recv_hi))
    deallocate(s_lo, s_hi, r_lo, r_hi)
#else
    recv_lo = 0.0_rp
    recv_hi = 0.0_rp
    if (nbr_lo >= 0 .or. nbr_hi >= 0) &
         error stop 'comm_exchange2: real MPI neighbor but this build has no USE_MPI'
#endif
  end subroutine comm_exchange2

  subroutine comm_exchange3(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    real(rp), intent(in)    :: send_lo(:,:,:), send_hi(:,:,:)
    real(rp), intent(out)   :: recv_lo(:,:,:), recv_hi(:,:,:)
#ifdef USE_MPI
    integer :: n, dest_lo, dest_hi, ierr, stat(MPI_STATUS_SIZE)
    real(rp), allocatable :: s_lo(:), s_hi(:), r_lo(:), r_hi(:)
    if (.not. ready) call init_comm()
    n = size(send_lo)
    if (size(send_hi) /= n .or. size(recv_lo) /= n .or. size(recv_hi) /= n) &
         error stop 'comm_exchange3: buffer size mismatch'
    allocate(s_lo(n), s_hi(n), r_lo(n), r_hi(n))
    s_lo = reshape(send_lo, (/n/))
    s_hi = reshape(send_hi, (/n/))
    r_lo = 0.0_rp
    r_hi = 0.0_rp
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    call MPI_Sendrecv(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 201, &
                      r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 201, &
                      compute_comm, stat, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_exchange3: slo'
    call MPI_Sendrecv(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 202, &
                      r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 202, &
                      compute_comm, stat, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_exchange3: shi'
    recv_lo = reshape(r_lo, shape(recv_lo))
    recv_hi = reshape(r_hi, shape(recv_hi))
    deallocate(s_lo, s_hi, r_lo, r_hi)
#else
    recv_lo = 0.0_rp
    recv_hi = 0.0_rp
    if (nbr_lo >= 0 .or. nbr_hi >= 0) &
         error stop 'comm_exchange3: real MPI neighbor but this build has no USE_MPI'
#endif
  end subroutine comm_exchange3

end module comm_mod
