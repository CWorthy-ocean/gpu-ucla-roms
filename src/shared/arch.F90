module arch_mod
  ! CPU vs GPU. Working fields live on this architecture (managed
  ! storage when USE_CUDA is on). The host only launches and does I/O.
  ! GPU: one work stream. step2d/halo CUF launches go on it so a CUDA
  ! graph can capture {step2d interiors, halo_fill}.
  use kinds
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  private

  integer, parameter, public :: ARCH_CPU = 0
  integer, parameter, public :: ARCH_GPU = 1
  integer, public :: arch = ARCH_CPU
  logical, public :: capturing = .false.

#ifdef USE_CUDA
  integer, parameter :: NGRAPH = 3
  integer(kind=cuda_stream_kind), public :: work_stream
  logical :: stream_ready = .false.
  type(cudaGraph) :: graphs(NGRAPH)
  type(cudaGraphExec) :: execs(NGRAPH)
  logical :: graph_ready(NGRAPH) = .false.
#endif

  public :: init_arch
  public :: select_device
  public :: on_gpu
  public :: sync_arch
  public :: arch_name
  public :: reset_graphs
  public :: graph_begin
  public :: graph_end
  public :: graph_launch
  public :: graph_is_ready

contains

  subroutine init_arch()
#ifdef USE_CUDA
    integer :: ierr
    arch = ARCH_GPU
    if (.not. stream_ready) then
      ierr = cudaStreamCreate(work_stream)
      if (ierr /= 0) then
        write(*,*) 'cudaStreamCreate failed', ierr
        error stop 'init_arch'
      endif
      stream_ready = .true.
    endif
#else
    arch = ARCH_CPU
#endif
  end subroutine init_arch

  subroutine select_device(rank)
    ! One GPU per compute rank: device = rank modulo the visible device count.
    integer, intent(in) :: rank
#ifdef USE_CUDA
    integer :: ierr, ndev, dev
    ierr = cudaGetDeviceCount(ndev)
    if (ierr /= 0 .or. ndev < 1) error stop 'select_device: no CUDA device'
    dev = mod(rank, ndev)
    ierr = cudaSetDevice(dev)
    if (ierr /= 0) error stop 'select_device: cudaSetDevice'
    write(*,'(a,i0,a,i0,a,i0)') 'arch: rank ', rank, ' -> GPU ', dev, ' of ', ndev
#endif
  end subroutine select_device

  logical function on_gpu()
    on_gpu = (arch == ARCH_GPU)
  end function on_gpu

  function arch_name() result(s)
    character(len=16) :: s
    if (arch == ARCH_GPU) then
      s = 'GPU'
    else
      s = 'CPU'
    endif
  end function arch_name

  subroutine sync_arch()
#ifdef USE_CUDA
    integer :: ierr
    if (capturing) return
    ierr = cudaDeviceSynchronize()
    if (ierr /= 0) then
      write(*,*) 'cudaDeviceSynchronize failed', ierr
      error stop 'sync_arch'
    endif
#endif
  end subroutine sync_arch

  subroutine reset_graphs()
#ifdef USE_CUDA
    integer :: i, ierr
    do i = 1, NGRAPH
      if (graph_ready(i)) then
        ierr = cudaGraphExecDestroy(execs(i))
        ierr = cudaGraphDestroy(graphs(i))
        graph_ready(i) = .false.
      endif
    enddo
#endif
    capturing = .false.
  end subroutine reset_graphs

  logical function graph_is_ready(id)
    integer, intent(in) :: id
#ifdef USE_CUDA
    graph_is_ready = graph_ready(id)
#else
    graph_is_ready = .false.
#endif
  end function graph_is_ready

  subroutine graph_begin()
#ifdef USE_CUDA
    integer :: ierr
    capturing = .true.
    ierr = cudaStreamBeginCapture(work_stream, 0)
    if (ierr /= 0) then
      write(*,*) 'cudaStreamBeginCapture failed', ierr
      error stop 'graph_begin'
    endif
#else
    capturing = .false.
#endif
  end subroutine graph_begin

  subroutine graph_end(id)
    integer, intent(in) :: id
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaStreamEndCapture(work_stream, graphs(id))
    if (ierr /= 0) then
      write(*,*) 'cudaStreamEndCapture failed', ierr, 'section', id
      error stop 'graph_end'
    endif
    ierr = cudaGraphInstantiate(execs(id), graphs(id), 0)
    if (ierr /= 0) then
      write(*,*) 'cudaGraphInstantiate failed', ierr, 'section', id
      error stop 'graph_end'
    endif
    graph_ready(id) = .true.
    capturing = .false.
#else
    capturing = .false.
#endif
  end subroutine graph_end

  subroutine graph_launch(id)
    integer, intent(in) :: id
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaGraphLaunch(execs(id), work_stream)
    if (ierr /= 0) then
      write(*,*) 'cudaGraphLaunch failed', ierr, 'section', id
      error stop 'graph_launch'
    endif
    ierr = cudaStreamSynchronize(work_stream)
    if (ierr /= 0) then
      write(*,*) 'cudaStreamSynchronize failed', ierr
      error stop 'graph_launch'
    endif
#endif
  end subroutine graph_launch

end module arch_mod
