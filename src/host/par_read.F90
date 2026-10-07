#include "mem_attr.inc"
module par_read
  ! Collective, batched reads of tile windows from one NetCDF file.
  !
  ! A tile window read straight from a contiguous (classic) file costs one
  ! short row read per (j, k), and the rate collapses as tiles shrink. Here
  ! the compute ranks instead read whole variable records contiguously, one
  ! variable per rank per batch, and redistribute each tile's window over
  ! MPI. The bytes that land in a tile are exactly the bytes the window read
  ! would have produced.
  !
  ! Usage (collective; every compute rank runs the same call sequence):
  !   call pr_begin()                 ! phase 1: pr_read_* only record requests
  !   ... pr_read_2d / pr_read_3d ...
  !   call pr_flush(ncid)             ! parallel reads + redistribution
  !   call pr_take()                  ! phase 2: pr_read_* hand the data out
  !   ... the same pr_read_2d / pr_read_3d sequence ...
  !   call pr_end()
  use kinds
  use netcdf
  use nc_io, only: nc_check
  use comm_mod, only: comm_rank, comm_size, comm_compute
  use decomp_mod, only: decomp
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
#ifdef USE_MPI
  include 'mpif.h'
#endif
  private
  public :: pr_begin, pr_flush, pr_take, pr_end, pr_taking
  public :: pr_read_2d, pr_read_3d
  public :: pr_store_2d, pr_store_3d, pr_store_3d_pt

  integer, parameter :: MAXQ = 512
  type :: ent_t
    character(len=64) :: name = ''
    integer :: n1 = 0, n2 = 0, n3 = 1, i0f = 1, j0f = 1, rec = 0
    real(rp), allocatable :: buf(:)
  end type ent_t
  type :: sb_t
    real(rp), allocatable :: b(:)
  end type sb_t
  type(ent_t), save :: q(MAXQ)
  integer :: nq = 0, inext = 0, mode = 0   ! mode 0 idle, 1 queueing, 2 taking
  ! device staging for the stores (grown to the largest window)
#ifdef USE_CUDA
  real(rp), device, allocatable :: dstage(:)
#else
  real(rp), allocatable :: dstage(:)
#endif
  integer :: dcap = 0
  real(8) :: t_read = 0.0d0, t_comm = 0.0d0

contains

  subroutine pr_begin()
    if (mode /= 0) error stop 'pr_begin: a read set is already open'
    mode = 1
    nq = 0
  end subroutine pr_begin

  subroutine pr_take()
    if (mode /= 1) error stop 'pr_take: pr_flush not done'
    mode = 2
    inext = 0
  end subroutine pr_take

  subroutine pr_end()
    integer :: k
    if (mode == 2 .and. inext /= nq) error stop 'pr_end: not every queued variable was taken'
    do k = 1, nq
      if (allocated(q(k)%buf)) deallocate(q(k)%buf)
      q(k) = ent_t()
    enddo
    nq = 0
    inext = 0
    mode = 0
    if (allocated(dstage)) deallocate(dstage)
    dcap = 0
  end subroutine pr_end

  logical function pr_taking()
    pr_taking = (mode == 2)
  end function pr_taking

  subroutine enqueue(name, n1, n2, n3, rec, i0f, j0f)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n1, n2, n3, rec, i0f, j0f
    if (nq >= MAXQ) error stop 'par_read: too many queued variables (raise MAXQ)'
    nq = nq + 1
    q(nq)%name = name
    q(nq)%n1 = n1; q(nq)%n2 = n2; q(nq)%n3 = n3
    q(nq)%rec = rec; q(nq)%i0f = i0f; q(nq)%j0f = j0f
  end subroutine enqueue

  integer function take_next(name, n1, n2, n3) result(k)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n1, n2, n3
    inext = inext + 1
    k = inext
    if (k > nq) error stop 'par_read: more variables taken than were queued'
    if (trim(q(k)%name) /= trim(name) .or. q(k)%n1 /= n1 .or. q(k)%n2 /= n2 .or. q(k)%n3 /= n3) then
      write(*,'(5a)') 'par_read: phase-2 call ', trim(name), ' does not match queued ', trim(q(k)%name)
      error stop 'par_read: the two phases must issue the same sequence of reads'
    endif
    if (.not. allocated(q(k)%buf)) error stop 'par_read: variable has no data (pr_flush not run?)'
  end function take_next

  subroutine pr_read_2d(ncid, name, dest, n1, n2, rec, i0f, j0f)
    ! Window (i0f..i0f+n1-1, j0f..j0f+n2-1) of a 2-D (or 2-D + time) variable.
    integer,          intent(in)    :: ncid, n1, n2
    character(len=*), intent(in)    :: name
    real(rp),         intent(inout) :: dest(n1, n2)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, i0, j0, k, i, j
    r = 0; i0 = 1; j0 = 1
    if (present(rec)) r = rec
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, 1, r, i0, j0)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, 1)
      do j = 1, n2
        do i = 1, n1
          dest(i, j) = q(k)%buf((j - 1) * n1 + i)
        enddo
      enddo
      deallocate(q(k)%buf)
    else
      error stop 'pr_read_2d: call pr_begin first'
    endif
  end subroutine pr_read_2d

  subroutine pr_read_3d(ncid, name, dest, n1, n2, n3, rec, i0f, j0f)
    integer,          intent(in)    :: ncid, n1, n2, n3
    character(len=*), intent(in)    :: name
    real(rp),         intent(inout) :: dest(n1, n2, n3)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, i0, j0, k, i, j, kk
    r = 0; i0 = 1; j0 = 1
    if (present(rec)) r = rec
    if (present(i0f)) i0 = i0f
    if (present(j0f)) j0 = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, n3, r, i0, j0)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, n3)
      do kk = 1, n3
        do j = 1, n2
          do i = 1, n1
            dest(i, j, kk) = q(k)%buf(((kk - 1) * n2 + (j - 1)) * n1 + i)
          enddo
        enddo
      enddo
      deallocate(q(k)%buf)
    else
      error stop 'pr_read_3d: call pr_begin first'
    endif
  end subroutine pr_read_3d

  subroutine read_full(ncid, e, fbuf, gx, gy)
    ! One whole record of a variable, contiguous: fbuf(gx, gy, e%n3).
    integer, intent(in) :: ncid
    type(ent_t), intent(in) :: e
    real(rp), allocatable, intent(out) :: fbuf(:)
    integer, intent(out) :: gx, gy
    integer :: varid, ndims, xtype, dimids(nf90_max_var_dims), dlen(nf90_max_var_dims), k
    integer, allocatable :: start(:), cnt(:), ibuf(:)
    call nc_check(nf90_inq_varid(ncid, trim(e%name), varid), 'var '//trim(e%name))
    call nc_check(nf90_inquire_variable(ncid, varid, xtype=xtype, ndims=ndims, dimids=dimids), &
                  'ndims '//trim(e%name))
    dlen = 1
    do k = 1, ndims
      call nc_check(nf90_inquire_dimension(ncid, dimids(k), len=dlen(k)), 'dim '//trim(e%name))
    enddo
    gx = dlen(1); gy = dlen(2)
    if (e%n3 == 1) then
      if (e%rec > 0 .and. ndims >= 3) then
        allocate(start(3), cnt(3)); start = (/1, 1, e%rec/); cnt = (/gx, gy, 1/)
      else
        allocate(start(2), cnt(2)); start = (/1, 1/); cnt = (/gx, gy/)
      endif
    else
      if (dlen(3) < e%n3) error stop 'par_read: variable has fewer levels than requested'
      if (e%rec > 0 .and. ndims >= 4) then
        allocate(start(4), cnt(4)); start = (/1, 1, 1, e%rec/); cnt = (/gx, gy, e%n3, 1/)
      else
        allocate(start(3), cnt(3)); start = (/1, 1, 1/); cnt = (/gx, gy, e%n3/)
      endif
    endif
    if (e%i0f + e%n1 - 1 > gx .or. e%j0f + e%n2 - 1 > gy) error stop 'par_read: window outside the file extent'
    allocate(fbuf(gx * gy * e%n3))
    if (xtype == nf90_int .or. xtype == nf90_short .or. xtype == nf90_byte) then
      allocate(ibuf(gx * gy * e%n3))
      call nc_check(nf90_get_var(ncid, varid, ibuf, start=start, count=cnt), 'get '//trim(e%name))
      fbuf = real(ibuf, rp)
      deallocate(ibuf)
    else
      call nc_check(nf90_get_var(ncid, varid, fbuf, start=start, count=cnt), 'get '//trim(e%name))
    endif
    deallocate(start, cnt)
  end subroutine read_full

  subroutine pack_window(fbuf, gx, gy, n3, i0f, j0f, n1, n2, out)
    integer, intent(in) :: gx, gy, n3, i0f, j0f, n1, n2
    real(rp), intent(in) :: fbuf(gx, gy, n3)
    real(rp), intent(out) :: out(n1, n2, n3)
    integer :: k
    do k = 1, n3
      out(:, :, k) = fbuf(i0f:i0f + n1 - 1, j0f:j0f + n2 - 1, k)
    enddo
  end subroutine pack_window

  subroutine pr_flush(ncid)
    ! Collective: batch b hands variable b*nc + rank + 1 to this rank, which
    ! reads it whole and sends every tile its window; all ranks receive
    ! their windows of the batch's variables.
    integer, intent(in) :: ncid
    integer :: nc, me, nb, b, iv, jv, jv0, jv1, t, n, gx, gy, k
    integer(8) :: c0, c1, c2, crate
    real(rp), allocatable :: fbuf(:)
#ifdef USE_MPI
    integer :: comm, ierr, ntot, src
    integer, allocatable :: win(:,:,:), mine(:,:), rreq(:), sreq(:), nqs(:)
    type(sb_t), allocatable :: sb(:)
#endif
    if (mode /= 1) error stop 'pr_flush: not in the queueing phase'
    nc = comm_size(); me = comm_rank()
#ifdef USE_MPI
    comm = comm_compute()
    allocate(nqs(0:nc - 1))
    call MPI_Allgather(nq, 1, MPI_INTEGER, nqs, 1, MPI_INTEGER, comm, ierr)
    if (any(nqs /= nq)) error stop 'par_read: ranks queued different numbers of variables'
    deallocate(nqs)
    if (nq == 0) return
    allocate(mine(4, nq), win(4, nq, 0:nc - 1))
    do k = 1, nq
      mine(:, k) = (/q(k)%n1, q(k)%n2, q(k)%i0f, q(k)%j0f/)
    enddo
    call MPI_Allgather(mine, 4 * nq, MPI_INTEGER, win, 4 * nq, MPI_INTEGER, comm, ierr)
    deallocate(mine)
    nb = (nq + nc - 1) / nc
    allocate(rreq(nc), sreq(0:nc - 1), sb(0:nc - 1))
    do b = 0, nb - 1
      iv = b * nc + me + 1
      jv0 = b * nc + 1
      jv1 = min(nq, (b + 1) * nc)
      call system_clock(c0, crate)
      if (iv <= nq) call read_full(ncid, q(iv), fbuf, gx, gy)
      call system_clock(c1, crate)
      t_read = t_read + real(c1 - c0, 8) / real(crate, 8)
      rreq = MPI_REQUEST_NULL
      do jv = jv0, jv1
        n = q(jv)%n1 * q(jv)%n2 * q(jv)%n3
        allocate(q(jv)%buf(n))
        src = jv - 1 - b * nc
        call MPI_Irecv(q(jv)%buf, n, MPI_REAL8, src, jv, comm, rreq(jv - jv0 + 1), ierr)
      enddo
      sreq = MPI_REQUEST_NULL
      if (iv <= nq) then
        do t = 0, nc - 1
          n = win(1, iv, t) * win(2, iv, t) * q(iv)%n3
          allocate(sb(t)%b(n))
          call pack_window(fbuf, gx, gy, q(iv)%n3, win(3, iv, t), win(4, iv, t), &
                           win(1, iv, t), win(2, iv, t), sb(t)%b)
          call MPI_Isend(sb(t)%b, n, MPI_REAL8, t, iv, comm, sreq(t), ierr)
        enddo
      endif
      call MPI_Waitall(jv1 - jv0 + 1, rreq, MPI_STATUSES_IGNORE, ierr)
      if (iv <= nq) call MPI_Waitall(nc, sreq, MPI_STATUSES_IGNORE, ierr)
      call system_clock(c2, crate)
      t_comm = t_comm + real(c2 - c1, 8) / real(crate, 8)
      if (iv <= nq) then
        do t = 0, nc - 1
          deallocate(sb(t)%b)
        enddo
        deallocate(fbuf)
      endif
    enddo
    deallocate(rreq, sreq, sb, win)
    write(*,'(a,i0,a,i0,a,f7.2,a,f7.2,a)') 'par_read: rank ', me, '  vars ', nq, '  whole-variable reads ', t_read, &
         ' s  redistribution ', t_comm, ' s'
    t_read = 0.0d0; t_comm = 0.0d0
#else
    do iv = 1, nq
      call read_full(ncid, q(iv), fbuf, gx, gy)
      n = q(iv)%n1 * q(iv)%n2 * q(iv)%n3
      allocate(q(iv)%buf(n))
      call pack_window(fbuf, gx, gy, q(iv)%n3, q(iv)%i0f, q(iv)%j0f, q(iv)%n1, q(iv)%n2, q(iv)%buf)
      deallocate(fbuf)
    enddo
#endif
  end subroutine pr_flush

  subroutine stage_to_device(k, n)
    ! The taken window k (host) into the device staging buffer.
    integer, intent(in) :: k, n
#ifdef USE_CUDA
    integer :: ierr
#endif
    if (n > dcap) then
      if (allocated(dstage)) deallocate(dstage)
      allocate(dstage(n))
      dcap = n
    endif
#ifdef USE_CUDA
    ierr = cudaMemcpy(dstage, q(k)%buf, n, cudaMemcpyHostToDevice)
    if (ierr /= 0) error stop 'par_read: cudaMemcpy to the staging buffer failed'
#else
    dstage(1:n) = q(k)%buf(1:n)
#endif
    deallocate(q(k)%buf)
  end subroutine stage_to_device

  subroutine pr_store_2d(ncid, name, a, i0, j0, n1, n2, rec, i0f, j0f)
    ! Window (i0f.., j0f..) of the file into a(i0:i0+n1-1, j0:j0+n2-1) on the
    ! device (a is the first element of a (is:ie, js:je) field).
    integer,          intent(in) :: ncid, i0, j0, n1, n2
    character(len=*), intent(in) :: name
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, fi, fj, k, i, j, ia, ja
    r = 0; fi = 1; fj = 1
    if (present(rec)) r = rec
    if (present(i0f)) fi = i0f
    if (present(j0f)) fj = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, 1, r, fi, fj)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, 1)
      call stage_to_device(k, n1 * n2)
      ia = i0; ja = j0
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, n2
        do i = 1, n1
          a(ia + i - 1, ja + j - 1) = dstage((j - 1) * n1 + i)
        enddo
      enddo
#ifdef USE_CUDA
      if (cudaDeviceSynchronize() /= 0) error stop 'pr_store_2d: kernel failed'
#endif
    else
      error stop 'pr_store_2d: call pr_begin first'
    endif
  end subroutine pr_store_2d

  subroutine pr_store_3d(ncid, name, a, i0, j0, n1, n2, n3, rec, i0f, j0f)
    integer,          intent(in) :: ncid, i0, j0, n1, n2, n3
    character(len=*), intent(in) :: name
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, n3)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, fi, fj, k, i, j, kk, ia, ja
    r = 0; fi = 1; fj = 1
    if (present(rec)) r = rec
    if (present(i0f)) fi = i0f
    if (present(j0f)) fj = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, n3, r, fi, fj)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, n3)
      call stage_to_device(k, n1 * n2 * n3)
      ia = i0; ja = j0
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do kk = 1, n3
        do j = 1, n2
          do i = 1, n1
            a(ia + i - 1, ja + j - 1, kk) = dstage(((kk - 1) * n2 + (j - 1)) * n1 + i)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      if (cudaDeviceSynchronize() /= 0) error stop 'pr_store_3d: kernel failed'
#endif
    else
      error stop 'pr_store_3d: call pr_begin first'
    endif
  end subroutine pr_store_3d

  subroutine pr_store_3d_pt(ncid, name, a, ntr, itr, i0, j0, n1, n2, n3, rec, i0f, j0f)
    ! Tracer itr of a (ntr, is:ie, js:je, n3) passive array.
    integer,          intent(in) :: ncid, ntr, itr, i0, j0, n1, n2, n3
    character(len=*), intent(in) :: name
    real(pp), WORK_ATTR intent(inout) :: a(ntr, decomp%is:decomp%ie, decomp%js:decomp%je, n3)
    integer,          intent(in), optional :: rec, i0f, j0f
    integer :: r, fi, fj, k, i, j, kk, ia, ja, it
    r = 0; fi = 1; fj = 1
    if (present(rec)) r = rec
    if (present(i0f)) fi = i0f
    if (present(j0f)) fj = j0f
    if (mode == 1) then
      call enqueue(name, n1, n2, n3, r, fi, fj)
    elseif (mode == 2) then
      k = take_next(name, n1, n2, n3)
      call stage_to_device(k, n1 * n2 * n3)
      ia = i0; ja = j0; it = itr
#ifdef USE_CUDA
      !$cuf kernel do(3) <<<*,*>>>
#endif
      do kk = 1, n3
        do j = 1, n2
          do i = 1, n1
            a(it, ia + i - 1, ja + j - 1, kk) = real(dstage(((kk - 1) * n2 + (j - 1)) * n1 + i), pp)
          enddo
        enddo
      enddo
#ifdef USE_CUDA
      if (cudaDeviceSynchronize() /= 0) error stop 'pr_store_3d_pt: kernel failed'
#endif
    else
      error stop 'pr_store_3d_pt: call pr_begin first'
    endif
  end subroutine pr_store_3d_pt

end module par_read
