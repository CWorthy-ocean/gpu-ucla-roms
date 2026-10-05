#include "mem_attr.inc"
module io_server
  ! Asynchronous history output.
  !
  ! Compute rank: each output field is gathered on the GPU (on the work
  ! stream, so it is ordered after the step that produced it) into a
  ! float32 device buffer and copied asynchronously into one of two
  ! pinned host record buffers; a CUDA event marks the record complete.
  ! io_poll (called every step) sends completed records to the I/O
  ! server rank with non-blocking MPI, or, without a server, writes them
  ! itself. The host never reads managed memory here, so no page
  ! migrations happen and the next step starts at once.
  !
  ! I/O rank: io_server_run loops on commands (OPEN file, PUT record,
  ! QUIT) and writes with netCDF. File creation (metadata) stays on the
  ! compute rank; the server only opens, appends records and closes.
  ! Commands for one record travel together, in order, so a file switch
  ! can never overtake a record that belongs to the previous file.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use nc_io
  use comm_mod
#ifdef USE_CUDA
  use cudafor
  use arch_mod, only: work_stream
#endif
  implicit none
#ifdef USE_MPI
  include 'mpif.h'
#endif
  private

  integer, parameter, public :: IO_KIND_HIS = 1, IO_KIND_RST = 2
  integer, parameter :: CMD_OPEN = 1, CMD_PUT = 2, CMD_QUIT = 3
  integer, parameter :: TAG_CMD = 900, TAG_STR = 901, TAG_TIME = 902, TAG_DATA = 910
  integer, parameter :: MAXV = 96, NAMELEN = 16, PATHLEN = 512   ! MARBL: 32 tracers + diagnostics
  integer, parameter :: NHDR = 8 + 4 * MAXV
  integer, parameter :: NSET = 2

  type rec_t
    logical :: used = .false., sent = .false.
    integer :: fkind = 0, rec = 0, nfile = 0, nvars = 0, nfloat = 0
    real(rp) :: time = 0.0_rp
    character(len=PATHLEN) :: open_path = ''
    character(len=NAMELEN) :: names(MAXV) = ''
    integer :: n1(MAXV) = 0, n2(MAXV) = 0, n3(MAXV) = 0, nrank(MAXV) = 0, off(MAXV) = 0
    integer :: nreq = 0
    integer :: reqs(MAXV) = 0
  end type rec_t

  type(rec_t) :: pend(NSET)
  integer :: cur = 0                 ! set being filled
  integer :: hcap = 0, dcap = 0
#ifdef USE_CUDA
  real(4), pinned, allocatable :: hpin1(:), hpin2(:)
  real(4), device, allocatable :: dstage(:)
  type(cudaEvent) :: ev(NSET)
#else
  real(4), allocatable :: hpin1(:), hpin2(:), dstage(:)
#endif
  ! Compute-side fallback (no server): current file per kind.
  character(len=PATHLEN) :: fname_sync(2) = ''
  ! Server side.
  real(4), allocatable :: rbuf(:)
  integer :: ncid_open(2) = -1

  public :: io_init
  public :: io_stage_begin
  public :: io_stage_2d
  public :: io_stage_3d
  public :: io_stage_3d_p
  public :: io_stage_omega
  public :: io_stage_end
  public :: io_poll
  public :: io_flush
  public :: io_finalize
  public :: io_server_run

contains

  subroutine io_init(cap_record, cap_field)
    ! cap_record: floats per staged record; cap_field: floats in the
    ! largest single field.
    integer, intent(in) :: cap_record, cap_field
#ifdef USE_CUDA
    integer :: ierr, i
#endif
    if (hcap > 0) return
    hcap = max(cap_record, 1)
    dcap = max(cap_field, 1)
    allocate(hpin1(hcap), hpin2(hcap), dstage(dcap))
#ifdef USE_CUDA
    do i = 1, NSET
      ierr = cudaEventCreateWithFlags(ev(i), cudaEventDisableTiming)
      if (ierr /= 0) error stop 'io_init: cudaEventCreate'
    enddo
#endif
    if (have_io_server()) then
      write(*,'(a,i0,a,f6.2,a)') 'io: asynchronous output to server rank ', &
           io_server_rank(), '  (staging ', 2.0 * real(hcap) * 4.0 / 2.0**30, ' GB pinned)'
    else
      write(*,'(a)') 'io: no server rank; staged records are written by the compute rank'
    endif
  end subroutine io_init

  subroutine io_stage_begin(fkind, rec, nfile, time, open_path)
    integer, intent(in) :: fkind, rec, nfile
    real(rp), intent(in) :: time
    character(len=*), intent(in) :: open_path
    integer :: idx
    if (hcap == 0) error stop 'io_stage_begin: io_init not called'
    idx = mod(cur, NSET) + 1
    ! The set must be free: dispatched and its sends complete.
    if (pend(idx)%used .and. .not. pend(idx)%sent) call dispatch(idx, .true.)
    call wait_sends(idx)
    pend(idx) = rec_t()
    pend(idx)%used = .true.
    pend(idx)%fkind = fkind
    pend(idx)%rec = rec
    pend(idx)%nfile = nfile
    pend(idx)%time = time
    pend(idx)%open_path = open_path
    cur = idx
  end subroutine io_stage_begin

  subroutine add_var(name, n1, n2, n3, nrank, off)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n1, n2, n3, nrank
    integer, intent(out) :: off
    integer :: iv, n
    type(rec_t), pointer :: r
    r => null()
    iv = pend(cur)%nvars + 1
    if (iv > MAXV) error stop 'io_stage: too many variables (raise MAXV)'
    n = n1 * n2 * n3
    off = pend(cur)%nfloat + 1
    if (off + n - 1 > hcap) error stop 'io_stage: record exceeds staging capacity'
    if (n > dcap) error stop 'io_stage: field exceeds device staging capacity'
    pend(cur)%nvars = iv
    pend(cur)%names(iv) = name
    pend(cur)%n1(iv) = n1
    pend(cur)%n2(iv) = n2
    pend(cur)%n3(iv) = n3
    pend(cur)%nrank(iv) = nrank
    pend(cur)%off(iv) = off
    pend(cur)%nfloat = pend(cur)%nfloat + n
  end subroutine add_var

  subroutine copy_to_host(off, n)
    ! dstage(1:n) -> pinned set `cur` at off, asynchronously on work_stream.
    integer, intent(in) :: off, n
#ifdef USE_CUDA
    integer :: ierr
    if (cur == 1) then
      ierr = cudaMemcpyAsync(hpin1(off:off+n-1), dstage(1:n), n, cudaMemcpyDeviceToHost, work_stream)
    else
      ierr = cudaMemcpyAsync(hpin2(off:off+n-1), dstage(1:n), n, cudaMemcpyDeviceToHost, work_stream)
    endif
    if (ierr /= 0) then
      write(*,*) 'io_stage: cudaMemcpyAsync failed', ierr
      error stop 'io_stage'
    endif
#else
    if (cur == 1) then
      hpin1(off:off+n-1) = dstage(1:n)
    else
      hpin2(off:off+n-1) = dstage(1:n)
    endif
#endif
  end subroutine copy_to_host

  subroutine io_stage_2d(name, a, i0, i1, j0, j1)
    ! a is the first element of a (is:ie, js:je) slab (sequence association).
    character(len=*), intent(in) :: name
    real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer, intent(in) :: i0, i1, j0, j1
    integer :: i, j, ni, nj, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    call add_var(name, ni, nj, 1, 2, off)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
    do j = j0, j1
      do i = i0, i1
        dstage((j - j0) * ni + (i - i0) + 1) = real(a(i, j), 4)
      enddo
    enddo
    call copy_to_host(off, ni * nj)
  end subroutine io_stage_2d

  subroutine io_stage_3d(name, a, k0, k1, i0, i1, j0, j1)
    ! a is the first element of a (is:ie, js:je, k0:k1) slab.
    character(len=*), intent(in) :: name
    integer, intent(in) :: k0, k1, i0, i1, j0, j1
    real(rp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, k0:k1)
    integer :: i, j, k, ni, nj, nk, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nk = k1 - k0 + 1
    call add_var(name, ni, nj, nk, 3, off)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*,0,work_stream>>>
#endif
    do k = k0, k1
      do j = j0, j1
        do i = i0, i1
          dstage((k - k0) * ni * nj + (j - j0) * ni + (i - i0) + 1) = real(a(i, j, k), 4)
        enddo
      enddo
    enddo
    call copy_to_host(off, ni * nj * nk)
  end subroutine io_stage_3d

  subroutine io_stage_3d_p(name, a, k0, k1, i0, i1, j0, j1)
    ! As io_stage_3d for a real(pp) (passive tracer) slab.
    character(len=*), intent(in) :: name
    integer, intent(in) :: k0, k1, i0, i1, j0, j1
    real(pp), WORK_ATTR intent(in) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, k0:k1)
    integer :: i, j, k, ni, nj, nk, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nk = k1 - k0 + 1
    call add_var(name, ni, nj, nk, 3, off)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*,0,work_stream>>>
#endif
    do k = k0, k1
      do j = j0, j1
        do i = i0, i1
          dstage((k - k0) * ni * nj + (j - j0) * ni + (i - i0) + 1) = real(a(i, j, k), 4)
        enddo
      enddo
    enddo
    call copy_to_host(off, ni * nj * nk)
  end subroutine io_stage_3d_p

  subroutine io_stage_omega(name, i0, i1, j0, j1)
    ! omega = pm*pn*(We+Wi) on w-levels 0:nz (as the former host put_omega).
    character(len=*), intent(in) :: name
    integer, intent(in) :: i0, i1, j0, j1
    integer :: i, j, k, ni, nj, nz, off
    ni = i1 - i0 + 1
    nj = j1 - j0 + 1
    nz = decomp%nz
    call add_var(name, ni, nj, nz + 1, 3, off)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*,0,work_stream>>>
#endif
    do k = 0, nz
      do j = j0, j1
        do i = i0, i1
          dstage(k * ni * nj + (j - j0) * ni + (i - i0) + 1) = &
               real(pm(i, j) * pn(i, j) * (We(i, j, k) + Wi(i, j, k)), 4)
        enddo
      enddo
    enddo
    call copy_to_host(off, ni * nj * (nz + 1))
  end subroutine io_stage_omega

  subroutine io_stage_end()
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaEventRecord(ev(cur), work_stream)
    if (ierr /= 0) error stop 'io_stage_end: cudaEventRecord'
#endif
    pend(cur)%sent = .false.
  end subroutine io_stage_end

  logical function set_ready(idx)
    integer, intent(in) :: idx
#ifdef USE_CUDA
    integer :: ierr
    ierr = cudaEventQuery(ev(idx))
    if (ierr == 0) then
      set_ready = .true.
    elseif (ierr == cudaErrorNotReady) then
      set_ready = .false.
    else
      write(*,*) 'io_poll: cudaEventQuery failed', ierr
      error stop 'io_poll'
    endif
#else
    set_ready = .true.
#endif
  end function set_ready

  subroutine io_poll()
    ! Called every step: hand over records whose GPU copies have landed.
    integer :: i
    do i = 1, NSET
      if (pend(i)%used .and. .not. pend(i)%sent) then
        if (set_ready(i)) call dispatch(i, .false.)
      endif
    enddo
  end subroutine io_poll

  subroutine io_flush()
    integer :: i
    do i = 1, NSET
      if (pend(i)%used .and. .not. pend(i)%sent) call dispatch(i, .true.)
    enddo
    do i = 1, NSET
      call wait_sends(i)
    enddo
  end subroutine io_flush

  subroutine io_finalize()
#ifdef USE_MPI
    integer :: hdr(NHDR), ierr
#endif
    if (hcap == 0) return
    call io_flush()
#ifdef USE_MPI
    if (have_io_server() .and. .not. is_io_rank()) then
      hdr = 0
      hdr(1) = CMD_QUIT
      call MPI_Send(hdr, NHDR, MPI_INTEGER, io_server_rank(), TAG_CMD, comm_world(), ierr)
    endif
#endif
    hcap = 0
  end subroutine io_finalize

  subroutine dispatch(idx, block)
    integer, intent(in) :: idx
    logical, intent(in) :: block
#ifdef USE_CUDA
    integer :: ierr
    if (block) then
      ierr = cudaEventSynchronize(ev(idx))
      if (ierr /= 0) error stop 'io: cudaEventSynchronize'
    endif
#endif
    if (have_io_server()) then
      call send_set(idx)
    else
      call write_set_local(idx)
    endif
    pend(idx)%sent = .true.
  end subroutine dispatch

  subroutine wait_sends(idx)
    integer, intent(in) :: idx
#ifdef USE_MPI
    integer :: ierr
    if (pend(idx)%nreq > 0) then
      call MPI_Waitall(pend(idx)%nreq, pend(idx)%reqs, MPI_STATUSES_IGNORE, ierr)
      if (ierr /= MPI_SUCCESS) error stop 'io: MPI_Waitall'
    endif
#endif
    pend(idx)%nreq = 0
  end subroutine wait_sends

  subroutine pack_hdr(r, hdr)
    type(rec_t), intent(in) :: r
    integer, intent(out) :: hdr(NHDR)
    integer :: iv, b
    hdr = 0
    hdr(1) = CMD_PUT
    hdr(2) = r%fkind
    hdr(3) = r%rec
    hdr(4) = r%nvars
    hdr(6) = r%nfile
    do iv = 1, r%nvars
      b = 8 + 4 * (iv - 1)
      hdr(b + 1) = r%n1(iv)
      hdr(b + 2) = r%n2(iv)
      hdr(b + 3) = r%n3(iv)
      hdr(b + 4) = r%nrank(iv)
    enddo
  end subroutine pack_hdr

  subroutine send_set(idx)
    integer, intent(in) :: idx
#ifdef USE_MPI
    integer :: hdr(NHDR), ierr, iv, n, dest, comm
    character(len=NAMELEN * MAXV) :: namebuf
    dest = io_server_rank()
    comm = comm_world()
    if (len_trim(pend(idx)%open_path) > 0) then
      hdr = 0
      hdr(1) = CMD_OPEN
      hdr(2) = pend(idx)%fkind
      call MPI_Send(hdr, NHDR, MPI_INTEGER, dest, TAG_CMD, comm, ierr)
      call MPI_Send(pend(idx)%open_path, PATHLEN, MPI_CHARACTER, dest, TAG_STR, comm, ierr)
    endif
    call pack_hdr(pend(idx), hdr)
    namebuf = ''
    do iv = 1, pend(idx)%nvars
      namebuf((iv - 1) * NAMELEN + 1:iv * NAMELEN) = pend(idx)%names(iv)
    enddo
    call MPI_Send(hdr, NHDR, MPI_INTEGER, dest, TAG_CMD, comm, ierr)
    call MPI_Send(namebuf, NAMELEN * MAXV, MPI_CHARACTER, dest, TAG_STR, comm, ierr)
    call MPI_Send(pend(idx)%time, 1, MPI_DOUBLE_PRECISION, dest, TAG_TIME, comm, ierr)
    do iv = 1, pend(idx)%nvars
      n = pend(idx)%n1(iv) * pend(idx)%n2(iv) * pend(idx)%n3(iv)
      if (idx == 1) then
        call MPI_Isend(hpin1(pend(idx)%off(iv)), n, MPI_REAL4, dest, TAG_DATA + iv, comm, &
                       pend(idx)%reqs(iv), ierr)
      else
        call MPI_Isend(hpin2(pend(idx)%off(iv)), n, MPI_REAL4, dest, TAG_DATA + iv, comm, &
                       pend(idx)%reqs(iv), ierr)
      endif
      if (ierr /= MPI_SUCCESS) error stop 'io: MPI_Isend'
    enddo
    pend(idx)%nreq = pend(idx)%nvars
#else
    error stop 'io: server requested without MPI'
#endif
  end subroutine send_set

  subroutine write_record(ncid, r, buf)
    ! Write one staged record from a float32 buffer (compute fallback and
    ! server share this).
    integer, intent(in) :: ncid
    type(rec_t), intent(in) :: r
    real(4), intent(in) :: buf(*)
    integer :: iv, off
    call nc_put_1d_rec(ncid, 'ocean_time', r%time, r%rec)
    do iv = 1, r%nvars
      off = r%off(iv)
      if (r%nrank(iv) == 2) then
        call nc_put_2d_rec_f(ncid, trim(r%names(iv)), buf(off), r%n1(iv), r%n2(iv), r%rec)
      else
        call nc_put_3d_rec_f(ncid, trim(r%names(iv)), buf(off), r%n1(iv), r%n2(iv), r%n3(iv), r%rec)
      endif
    enddo
  end subroutine write_record

  subroutine write_set_local(idx)
    integer, intent(in) :: idx
    integer :: ncid, k
    k = pend(idx)%fkind
    if (len_trim(pend(idx)%open_path) > 0) fname_sync(k) = pend(idx)%open_path
    if (len_trim(fname_sync(k)) == 0) error stop 'io: record without a file'
    call nc_open_write(trim(fname_sync(k)), ncid)
    if (idx == 1) then
      call write_record(ncid, pend(idx), hpin1)
    else
      call write_record(ncid, pend(idx), hpin2)
    endif
    call nc_close(ncid)
    call report_written(pend(idx))
  end subroutine write_set_local

  subroutine report_written(r)
    type(rec_t), intent(in) :: r
    if (r%fkind == IO_KIND_HIS) then
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote history t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    else
      write(*,'(a,es14.6,a,i0,a,i0)') 'wrote restart t = ', r%time, &
           '  rec = ', r%rec, '  file ', r%nfile
    endif
    flush(6)
  end subroutine report_written

  subroutine io_server_run()
    ! I/O rank main loop. Returns after CMD_QUIT.
#ifdef USE_MPI
    integer :: hdr(NHDR), ierr, stat(MPI_STATUS_SIZE), src, comm, iv, off, n, b, k, ntot
    integer :: c0, c1, c2, rate
    character(len=PATHLEN) :: path
    character(len=NAMELEN * MAXV) :: namebuf
    type(rec_t) :: r
    src = 0                      ! compute root is world rank 0
    comm = comm_world()
    write(*,'(a,i0)') 'io_server: ready on world rank ', comm_world_rank()
    flush(6)
    do
      call MPI_Recv(hdr, NHDR, MPI_INTEGER, src, TAG_CMD, comm, stat, ierr)
      if (ierr /= MPI_SUCCESS) error stop 'io_server: MPI_Recv cmd'
      select case (hdr(1))
      case (CMD_OPEN)
        call MPI_Recv(path, PATHLEN, MPI_CHARACTER, src, TAG_STR, comm, stat, ierr)
        k = hdr(2)
        if (ncid_open(k) >= 0) call nc_close(ncid_open(k))
        call nc_open_write(trim(path), ncid_open(k))
        write(*,'(a,a)') 'io_server: opened ', trim(path)
        flush(6)
      case (CMD_PUT)
        r = rec_t()
        r%fkind = hdr(2)
        r%rec = hdr(3)
        r%nvars = hdr(4)
        r%nfile = hdr(6)
        call MPI_Recv(namebuf, NAMELEN * MAXV, MPI_CHARACTER, src, TAG_STR, comm, stat, ierr)
        call MPI_Recv(r%time, 1, MPI_DOUBLE_PRECISION, src, TAG_TIME, comm, stat, ierr)
        ntot = 0
        do iv = 1, r%nvars
          b = 8 + 4 * (iv - 1)
          r%n1(iv) = hdr(b + 1); r%n2(iv) = hdr(b + 2); r%n3(iv) = hdr(b + 3)
          r%nrank(iv) = hdr(b + 4)
          r%names(iv) = namebuf((iv - 1) * NAMELEN + 1:iv * NAMELEN)
          r%off(iv) = ntot + 1
          ntot = ntot + r%n1(iv) * r%n2(iv) * r%n3(iv)
        enddo
        if (allocated(rbuf)) then
          if (size(rbuf) < ntot) deallocate(rbuf)
        endif
        if (.not. allocated(rbuf)) allocate(rbuf(ntot))
        ! Receive the whole record first so the compute rank's sends
        ! complete at memory speed, then write.
        call system_clock(c0, rate)
        do iv = 1, r%nvars
          off = r%off(iv)
          n = r%n1(iv) * r%n2(iv) * r%n3(iv)
          call MPI_Recv(rbuf(off), n, MPI_REAL4, src, TAG_DATA + iv, comm, stat, ierr)
          if (ierr /= MPI_SUCCESS) error stop 'io_server: MPI_Recv data'
        enddo
        k = r%fkind
        if (ncid_open(k) < 0) error stop 'io_server: record before file open'
        call system_clock(c1)
        call write_record(ncid_open(k), r, rbuf)
        call nc_sync(ncid_open(k))
        call system_clock(c2)
        call report_written(r)
        write(*,'(a,f7.2,a,f7.2,a,f7.1,a)') 'io_server:   recv ', real(c1 - c0) / real(rate), &
             ' s  write ', real(c2 - c1) / real(rate), ' s  (', real(ntot) * 4.0 / 2.0**20, ' MB)'
      case (CMD_QUIT)
        do k = 1, 2
          if (ncid_open(k) >= 0) call nc_close(ncid_open(k))
          ncid_open(k) = -1
        enddo
        write(*,'(a)') 'io_server: done'
        flush(6)
        exit
      case default
        error stop 'io_server: unknown command'
      end select
    enddo
#endif
  end subroutine io_server_run

end module io_server
