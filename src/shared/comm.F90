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
  ! GPU-aware MPI (Cray MPICH + libmpi_gtl_cuda): device buffers go straight
  ! to MPI_Sendrecv when MPICH_GPU_SUPPORT_ENABLED=1 is set for the job.
  logical :: comm_gpu_direct = .false.
  integer :: io_rank_w = -1          ! world rank of the first I/O server rank, -1 = none
  integer :: io_nserv = 0            ! number of I/O server (writer) ranks, the last world ranks
  logical :: io_rank_flag = .false.
  logical :: ready = .false.

  public :: init_comm
  public :: finalize_comm
  public :: comm_rank
  public :: comm_size
  public :: comm_exchange2
  public :: comm_sendrecv_raw
  public :: comm_sendrecv_dev
  public :: comm_post_dev, comm_wait_dev
  public :: comm_min_real
  public :: comm_reduce_maxsum
  public :: comm_gpu_direct
  public :: comm_exchange3
  public :: have_io_server
  public :: is_io_rank
  public :: io_server_rank
  public :: io_server_count, io_server_index, comm_writers
  public :: comm_world
  public :: comm_world_rank
  public :: comm_world_size
  public :: compute_barrier
  public :: comm_compute

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
    io_nserv = 0
    if (world_size >= 2) then
      call get_environment_variable('ROMS_GPU_IO_SERVER', val, status=stat)
      if (.not. (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '0')) then
        ! ROMS_GPU_IO_SERVERS = number of writer ranks (default 1); they are
        ! the last world ranks and share the output files (parallel NetCDF
        ! when more than one).
        ! Rule: the compute ranks are np_xi x np_eta from the namelist and
        ! every extra launched rank is a writer. ROMS_GPU_IO_SERVERS overrides.
        io_nserv = 1
        n = namelist_tiles()
        if (n >= 1 .and. world_size > n) io_nserv = world_size - n
        call get_environment_variable('ROMS_GPU_IO_SERVERS', val, status=stat)
        if (stat == 0 .and. len_trim(val) > 0) then
          read(val, *, iostat=stat) n
          if (stat == 0 .and. n >= 1 .and. n < world_size) io_nserv = n
        endif
        io_rank_w = world_size - io_nserv
      endif
    endif
    io_rank_flag = (world_rank >= io_rank_w .and. io_rank_w >= 0)
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
    call get_environment_variable('MPICH_GPU_SUPPORT_ENABLED', val, status=stat)
    comm_gpu_direct = (stat == 0 .and. len_trim(val) > 0 .and. val(1:1) == '1')
    if (my_rank == 0 .and. .not. io_rank_flag) &
         write(*,'(a,l1)') 'comm: GPU-direct halo exchange (MPICH_GPU_SUPPORT_ENABLED) = ', comm_gpu_direct
#endif
    ready = .true.
  end subroutine init_comm

  integer function namelist_tiles() result(ntile)
    ! np_xi * np_eta from the namelist named on the command line (argument
    ! 2 of 'run' / 'load'), by a plain text scan; 0 when unavailable.
    character(len=512) :: path, line
    character(len=16) :: mode
    integer :: unit, ios, npx, npy, k
    ntile = 0
    if (command_argument_count() < 2) return
    call get_command_argument(1, mode)
    if (trim(mode) /= 'run' .and. trim(mode) /= 'load') return
    call get_command_argument(2, path)
    open(newunit=unit, file=trim(path), status='old', action='read', iostat=ios)
    if (ios /= 0) return
    npx = 0; npy = 0
    do
      read(unit, '(a)', iostat=ios) line
      if (ios /= 0) exit
      k = index(line, '!')
      if (k > 0) line = line(1:k - 1)
      call grab(line, 'np_xi', npx)
      call grab(line, 'np_eta', npy)
    enddo
    close(unit)
    if (npx >= 1 .and. npy >= 1) ntile = npx * npy
  contains
    subroutine grab(l, key, v)
      character(len=*), intent(in) :: l, key
      integer, intent(inout) :: v
      integer :: i, j, ios2
      i = index(l, key)
      if (i == 0) return
      if (i > 1) then
        if (l(i-1:i-1) /= ' ' .and. l(i-1:i-1) /= ',') return
      endif
      j = i + len(key)
      if (j > len(l)) return
      if (l(j:j) /= ' ' .and. l(j:j) /= '=') return
      j = index(l(i:), '=')
      if (j == 0) return
      read(l(i + j:), *, iostat=ios2) v
      if (ios2 /= 0) v = 0
    end subroutine grab
  end function namelist_tiles

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

  integer function comm_world_size()
    comm_world_size = world_size
  end function comm_world_size

  integer function comm_compute()
    ! The communicator of the compute ranks (world when there is no server).
    comm_compute = compute_comm
  end function comm_compute

  subroutine compute_barrier()
    ! Barrier over the compute ranks (no-op on one rank).
#ifdef USE_MPI
    integer :: ierr
    if (comm_size() > 1) call MPI_Barrier(compute_comm, ierr)
#endif
  end subroutine compute_barrier

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

  integer function io_server_count()
    io_server_count = io_nserv
  end function io_server_count

  integer function io_server_index()
    ! 0 .. nserv-1 on a writer rank, -1 elsewhere
    io_server_index = -1
    if (io_rank_flag) io_server_index = world_rank - io_rank_w
  end function io_server_index

  integer function comm_writers()
    ! The writer ranks' own communicator (the split gives it to colour 1).
    comm_writers = compute_comm
  end function comm_writers

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

  subroutine comm_sendrecv_raw(nbr_lo, nbr_hi, n, s_lo, s_hi, r_lo, r_hi)
    ! The two neighbor exchanges on contiguous host buffers of n values
    ! (no allocation or reshape; the caller stages device data into pinned
    ! memory). A missing neighbor is MPI_PROC_NULL: its receive buffer is
    ! left untouched and the caller does not read it.
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer,     intent(in) :: n
    real(rp),    intent(in)    :: s_lo(*), s_hi(*)
    real(rp),    intent(inout) :: r_lo(*), r_hi(*)
#ifdef USE_MPI
    integer :: dest_lo, dest_hi, ierr, req(4), stats(MPI_STATUS_SIZE, 4)
    if (.not. ready) call init_comm()
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    ! Both directions in flight at once: receives first, then sends, one wait.
    call MPI_Irecv(r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 301,     compute_comm, req(1), ierr)
    call MPI_Irecv(r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 302, compute_comm, req(2), ierr)
    call MPI_Isend(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 301,     compute_comm, req(3), ierr)
    call MPI_Isend(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 302, compute_comm, req(4), ierr)
    call MPI_Waitall(4, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_sendrecv_raw: waitall'
#else
    if (nbr_lo >= 0 .or. nbr_hi >= 0) error stop 'comm_sendrecv_raw: MPI neighbor without USE_MPI'
#endif
  end subroutine comm_sendrecv_raw

  subroutine comm_sendrecv_dev(nbr_lo, nbr_hi, n, s_lo, s_hi, r_lo, r_hi)
    ! As comm_sendrecv_raw on DEVICE buffers (GPU-aware MPI); the caller has
    ! synchronized the stream that packed them.
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer,     intent(in) :: n
#ifdef USE_CUDA
    real(rp), device, intent(in)    :: s_lo(*), s_hi(*)
    real(rp), device, intent(inout) :: r_lo(*), r_hi(*)
#else
    real(rp), intent(in)    :: s_lo(*), s_hi(*)
    real(rp), intent(inout) :: r_lo(*), r_hi(*)
#endif
#ifdef USE_MPI
    integer :: dest_lo, dest_hi, ierr, req(4), stats(MPI_STATUS_SIZE, 4)
    if (.not. ready) call init_comm()
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    ! Both directions in flight at once: receives first, then sends, one wait.
    call MPI_Irecv(r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 311,     compute_comm, req(1), ierr)
    call MPI_Irecv(r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 312, compute_comm, req(2), ierr)
    call MPI_Isend(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, 311,     compute_comm, req(3), ierr)
    call MPI_Isend(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, 312, compute_comm, req(4), ierr)
    call MPI_Waitall(4, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_sendrecv_dev: waitall'
#else
    if (nbr_lo >= 0 .or. nbr_hi >= 0) error stop 'comm_sendrecv_dev: MPI neighbor without USE_MPI'
#endif
  end subroutine comm_sendrecv_dev

  subroutine comm_post_dev(nbr_lo, nbr_hi, n, s_lo, s_hi, r_lo, r_hi, tag_lo, tag_hi, req)
    ! Non-blocking form of comm_sendrecv_dev: posts the four requests with
    ! the caller's tags (distinct from the blocking exchanges, which use
    ! 301/302 and 311/312, so an exchange in flight cannot be matched by
    ! one posted later) and returns; comm_wait_dev completes them. The
    ! buffers must stay untouched until then.
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer,     intent(in) :: n, tag_lo, tag_hi
#ifdef USE_CUDA
    real(rp), device, intent(in)    :: s_lo(*), s_hi(*)
    real(rp), device, intent(inout) :: r_lo(*), r_hi(*)
#else
    real(rp), intent(in)    :: s_lo(*), s_hi(*)
    real(rp), intent(inout) :: r_lo(*), r_hi(*)
#endif
    integer, intent(out) :: req(4)
#ifdef USE_MPI
    integer :: dest_lo, dest_hi, ierr
    if (.not. ready) call init_comm()
    dest_lo = mpi_dest(nbr_lo)
    dest_hi = mpi_dest(nbr_hi)
    call MPI_Irecv(r_hi, n, MPI_DOUBLE_PRECISION, dest_hi, tag_lo, compute_comm, req(1), ierr)
    call MPI_Irecv(r_lo, n, MPI_DOUBLE_PRECISION, dest_lo, tag_hi, compute_comm, req(2), ierr)
    call MPI_Isend(s_lo, n, MPI_DOUBLE_PRECISION, dest_lo, tag_lo, compute_comm, req(3), ierr)
    call MPI_Isend(s_hi, n, MPI_DOUBLE_PRECISION, dest_hi, tag_hi, compute_comm, req(4), ierr)
#else
    req = 0
    if (nbr_lo >= 0 .or. nbr_hi >= 0) error stop 'comm_post_dev: MPI neighbor without USE_MPI'
#endif
  end subroutine comm_post_dev

  real(rp) function comm_min_real(x)
    ! Minimum of x over the compute ranks (x itself without MPI).
    real(rp), intent(in) :: x
#ifdef USE_MPI
    integer :: ierr
    real(rp) :: y
    if (.not. ready) call init_comm()
    call MPI_Allreduce(x, y, 1, MPI_DOUBLE_PRECISION, MPI_MIN, compute_comm, ierr)
    comm_min_real = y
#else
    comm_min_real = x
#endif
  end function comm_min_real

  subroutine comm_reduce_maxsum(n, x, xmax, xsum)
    ! Max and sum of x(1:n) over the compute ranks, on compute rank 0
    ! (x itself without MPI).
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: xmax(n), xsum(n)
#ifdef USE_MPI
    integer :: ierr
    if (.not. ready) call init_comm()
    call MPI_Reduce(x, xmax, n, MPI_DOUBLE_PRECISION, MPI_MAX, 0, compute_comm, ierr)
    call MPI_Reduce(x, xsum, n, MPI_DOUBLE_PRECISION, MPI_SUM, 0, compute_comm, ierr)
#else
    xmax = x; xsum = x
#endif
  end subroutine comm_reduce_maxsum

  subroutine comm_wait_dev(req)
    integer, intent(inout) :: req(4)
#ifdef USE_MPI
    integer :: ierr, stats(MPI_STATUS_SIZE, 4)
    call MPI_Waitall(4, req, stats, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'comm_wait_dev: waitall'
#endif
  end subroutine comm_wait_dev

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
