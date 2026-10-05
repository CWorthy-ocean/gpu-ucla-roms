#include "mem_attr.inc"
#include "cuf_launch.inc"
module halo_mod
  ! Pack/unpack launch on the GPU work stream. NBR_SELF deliver is a
  ! device copy so {step2d, halo_fill} can live in a CUDA graph.
  ! EW then NS so corners ride along (ROMS do_corners).
  use kinds
  use decomp_mod, only: decomp
  use time_state, only: fast, nt
  use comm_mod, only: comm_exchange2, comm_exchange3
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private

  real(rp), WORK_ATTR allocatable :: ew_slo(:,:,:), ew_shi(:,:,:)
  real(rp), WORK_ATTR allocatable :: ew_rlo(:,:,:), ew_rhi(:,:,:)
  real(rp), WORK_ATTR allocatable :: ns_slo(:,:,:), ns_shi(:,:,:)
  real(rp), WORK_ATTR allocatable :: ns_rlo(:,:,:), ns_rhi(:,:,:)
  real(rp), WORK_ATTR allocatable :: work2d(:,:)
  real(rp), WORK_ATTR allocatable :: work3d(:,:,:)
  real(rp), WORK_ATTR allocatable :: work3dw(:,:,:)

  public :: halo_fill_2d
  public :: halo_fill_2d_n
  public :: halo_fill_2d_s
  public :: halo_fill_2d_st
  public :: halo_fill_2d_sw
  public :: halo_fill_3d
  public :: halo_fill_3d_s
  public :: halo_fill_3d_t
  public :: halo_fill_3d_w
  public :: halo_fill_3d_w_s
  public :: halo_ensure
  public :: halo_free

contains

  subroutine ensure_bufs()
    integer(ip) :: h, is, ie, js, je, nz
    if (allocated(ew_slo)) then
      if (lbound(ew_slo, 2) == decomp%js .and. ubound(ew_slo, 2) == decomp%je .and. &
          ubound(ew_slo, 3) == decomp%nz) return
      call halo_free()
    endif
    h = decomp%halo
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    nz = decomp%nz
    allocate(ew_slo(h, js:je, 0:nz), ew_shi(h, js:je, 0:nz))
    allocate(ew_rlo(h, js:je, 0:nz), ew_rhi(h, js:je, 0:nz))
    allocate(ns_slo(is:ie, h, 0:nz), ns_shi(is:ie, h, 0:nz))
    allocate(ns_rlo(is:ie, h, 0:nz), ns_rhi(is:ie, h, 0:nz))
    allocate(work2d(is:ie, js:je))
    allocate(work3d(is:ie, js:je, 1:nz))
    allocate(work3dw(is:ie, js:je, 0:nz))
    ew_slo = 0.0_rp; ew_shi = 0.0_rp; ew_rlo = 0.0_rp; ew_rhi = 0.0_rp
    ns_slo = 0.0_rp; ns_shi = 0.0_rp; ns_rlo = 0.0_rp; ns_rhi = 0.0_rp
    work2d = 0.0_rp; work3d = 0.0_rp; work3dw = 0.0_rp
  end subroutine ensure_bufs

  subroutine halo_ensure()
    ! Nothing to exchange on a tile without neighbours: no buffers.
    if (no_neighbors()) return
    call ensure_bufs()
  end subroutine halo_ensure

  subroutine halo_free()
    if (allocated(ew_slo)) deallocate(ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (allocated(ns_slo)) deallocate(ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (allocated(work2d)) deallocate(work2d)
    if (allocated(work3d)) deallocate(work3d)
    if (allocated(work3dw)) deallocate(work3dw)
  end subroutine halo_free

  subroutine halo_fill_2d(a, stagger)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip), intent(in) :: stagger
    integer(ip) :: i1, i2, j1, j2
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(stagger, i1, i2, j1, j2)
    call exchange_ew_2d(a, i1, i2, j1, j2)
    call exchange_ns_2d(a, i1, i2, j1, j2)
  end subroutine halo_fill_2d

  subroutine halo_fill_2d_n(a, stagger)
    ! Halo a(:,:,fast%knew). The slot is read from managed memory so a
    ! captured graph can replay after the host advances knew.
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    integer(ip), intent(in) :: stagger
    integer(ip) :: i1, i2, j1, j2
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(stagger, i1, i2, j1, j2)
    call exchange_ew_2d_n(a, i1, i2, j1, j2)
    call exchange_ns_2d_n(a, i1, i2, j1, j2)
  end subroutine halo_fill_2d_n

  subroutine halo_fill_3d(a, stagger)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz)
    integer(ip), intent(in) :: stagger
    integer(ip) :: i1, i2, j1, j2
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(stagger, i1, i2, j1, j2)
    call exchange_ew_3d(a, i1, i2, j1, j2, 1, decomp%nz)
    call exchange_ns_3d(a, i1, i2, j1, j2, 1, decomp%nz)
  end subroutine halo_fill_3d

  subroutine halo_fill_3d_w(a, stagger)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    integer(ip), intent(in) :: stagger
    integer(ip) :: i1, i2, j1, j2
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(stagger, i1, i2, j1, j2)
    call exchange_ew_3dw(a, i1, i2, j1, j2, 0, decomp%nz)
    call exchange_ns_3dw(a, i1, i2, j1, j2, 0, decomp%nz)
  end subroutine halo_fill_3d_w

  ! Slot fills copy one time/tracer level onto an explicit-shape work
  ! array, then reuse halo_fill_2d/3d. Assumed-shape managed dummies
  ! become host descriptors in nvfortran and are illegal in CUF.
  ! The copies are kernels on work_stream, ordered after the caller's
  ! kernels; host loops here would race them on managed memory.
  subroutine halo_fill_2d_s(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, n, is, ie, js, je
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    CUF_DO2
    do j = js, je
      do i = is, ie
        work2d(i, j) = a(i, j, n)
      enddo
    enddo
    call halo_fill_2d(work2d, stagger)
    CUF_DO2
    do j = js, je
      do i = is, ie
        a(i, j, n) = work2d(i, j)
      enddo
    enddo
  end subroutine halo_fill_2d_s

  subroutine halo_fill_2d_st(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, nt)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, n, is, ie, js, je
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    CUF_DO2
    do j = js, je
      do i = is, ie
        work2d(i, j) = a(i, j, n)
      enddo
    enddo
    call halo_fill_2d(work2d, stagger)
    CUF_DO2
    do j = js, je
      do i = is, ie
        a(i, j, n) = work2d(i, j)
      enddo
    enddo
  end subroutine halo_fill_2d_st

  subroutine halo_fill_2d_sw(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, n, is, ie, js, je
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    CUF_DO2
    do j = js, je
      do i = is, ie
        work2d(i, j) = a(i, j, n)
      enddo
    enddo
    call halo_fill_2d(work2d, stagger)
    CUF_DO2
    do j = js, je
      do i = is, ie
        a(i, j, n) = work2d(i, j)
      enddo
    enddo
  end subroutine halo_fill_2d_sw

  subroutine halo_fill_3d_s(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 3)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, k, n, is, ie, js, je, k0, k1
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    k0 = 1; k1 = decomp%nz
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          work3d(i, j, k) = a(i, j, k, n)
        enddo
      enddo
    enddo
    call halo_fill_3d(work3d, stagger)
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          a(i, j, k, n) = work3d(i, j, k)
        enddo
      enddo
    enddo
  end subroutine halo_fill_3d_s

  subroutine halo_fill_3d_t(a, stagger, slot, itrc)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 1:decomp%nz, 3, nt)
    integer(ip), intent(in) :: stagger, slot, itrc
    integer(ip) :: i, j, k, n, itr, is, ie, js, je, k0, k1
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    itr = itrc
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    k0 = 1; k1 = decomp%nz
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          work3d(i, j, k) = a(i, j, k, n, itr)
        enddo
      enddo
    enddo
    call halo_fill_3d(work3d, stagger)
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          a(i, j, k, n, itr) = work3d(i, j, k)
        enddo
      enddo
    enddo
  end subroutine halo_fill_3d_t

  subroutine halo_fill_3d_w_s(a, stagger, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, 0:decomp%nz, 2)
    integer(ip), intent(in) :: stagger, slot
    integer(ip) :: i, j, k, n, is, ie, js, je, k0, k1
    if (no_neighbors()) return
    call ensure_bufs()
    n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    k0 = 0; k1 = decomp%nz
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          work3dw(i, j, k) = a(i, j, k, n)
        enddo
      enddo
    enddo
    call halo_fill_3d_w(work3dw, stagger)
    CUF_DO3
    do k = k0, k1
      do j = js, je
        do i = is, ie
          a(i, j, k, n) = work3dw(i, j, k)
        enddo
      enddo
    enddo
  end subroutine halo_fill_3d_w_s

  logical function no_neighbors()
    ! Single non-periodic tile: every halo point is a physical-edge ghost
    ! set by the boundary code, so a fill has nothing to deliver.
    no_neighbors = decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE .and. &
                   decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE
  end function no_neighbors

  subroutine interior_bounds(stagger, i1, i2, j1, j2)
    integer(ip), intent(in)  :: stagger
    integer(ip), intent(out) :: i1, i2, j1, j2
    i1 = 1
    j1 = 1
    i2 = decomp%nx
    j2 = decomp%ny
    if (stagger == STAGGER_U) i2 = decomp%nx + 1
    if (stagger == STAGGER_V) j2 = decomp%ny + 1
  end subroutine interior_bounds

  subroutine exchange_ew_2d(a, i1, i2, j1, j2)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:)
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: h, j, m
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    CUF_DO2
    do j = j1, j2
      do m = 1, h
        ew_slo(m, j, 0) = a(i1 + m - 1, j)
        ew_shi(m, j, 0) = a(i2 - h + m, j)
      enddo
    enddo
    call deliver2_ew(decomp%nbr_w, decomp%nbr_e)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          a(i1 - h + m - 1, j) = ew_rlo(m, j, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          a(i2 + m, j) = ew_rhi(m, j, 0)
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ew_2d

  subroutine exchange_ns_2d(a, i1, i2, j1, j2)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:)
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: h, i, m, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO2
    do m = 1, h
      do i = ilo, ihi
        ns_slo(i, m, 0) = a(i, j1 + m - 1)
        ns_shi(i, m, 0) = a(i, j2 - h + m)
      enddo
    enddo
    call deliver2_ns(decomp%nbr_s, decomp%nbr_n)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          a(i, j1 - h + m - 1) = ns_rlo(i, m, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          a(i, j2 + m) = ns_rhi(i, m, 0)
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ns_2d

  subroutine exchange_ew_2d_n(a, i1, i2, j1, j2)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: h, j, m, n
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    CUF_DO2
    do j = j1, j2
      do m = 1, h
        n = fast%knew
        ew_slo(m, j, 0) = a(i1 + m - 1, j, n)
        ew_shi(m, j, 0) = a(i2 - h + m, j, n)
      enddo
    enddo
    call deliver2_ew(decomp%nbr_w, decomp%nbr_e)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          n = fast%knew
          a(i1 - h + m - 1, j, n) = ew_rlo(m, j, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          n = fast%knew
          a(i2 + m, j, n) = ew_rhi(m, j, 0)
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ew_2d_n

  subroutine exchange_ns_2d_n(a, i1, i2, j1, j2)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: h, i, m, n, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO2
    do m = 1, h
      do i = ilo, ihi
        n = fast%knew
        ns_slo(i, m, 0) = a(i, j1 + m - 1, n)
        ns_shi(i, m, 0) = a(i, j2 - h + m, n)
      enddo
    enddo
    call deliver2_ns(decomp%nbr_s, decomp%nbr_n)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          n = fast%knew
          a(i, j1 - h + m - 1, n) = ns_rlo(i, m, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          n = fast%knew
          a(i, j2 + m, n) = ns_rhi(i, m, 0)
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ns_2d_n

  subroutine exchange_ew_3d(a, i1, i2, j1, j2, k0, k1)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1
    integer(ip) :: h, j, m, k
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k)
          ew_shi(m, j, k) = a(i2 - h + m, j, k)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ew_3d

  subroutine exchange_ns_3d(a, i1, i2, j1, j2, k0, k1)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1
    integer(ip) :: h, i, m, k, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k)
          ns_shi(i, m, k) = a(i, j2 - h + m, k)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ns_3d

  subroutine exchange_ew_3dw(a, i1, i2, j1, j2, k0, k1)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, 0:)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1
    integer(ip) :: h, j, m, k
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k)
          ew_shi(m, j, k) = a(i2 - h + m, j, k)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ew_3dw

  subroutine exchange_ns_3dw(a, i1, i2, j1, j2, k0, k1)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, 0:)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1
    integer(ip) :: h, i, m, k, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k)
          ns_shi(i, m, k) = a(i, j2 - h + m, k)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ns_3dw

  subroutine exchange_ew_2d_s(a, i1, i2, j1, j2, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, slot
    integer(ip) :: h, j, m, n
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    n = slot
    CUF_DO2
    do j = j1, j2
      do m = 1, h
        ew_slo(m, j, 0) = a(i1 + m - 1, j, n)
        ew_shi(m, j, 0) = a(i2 - h + m, j, n)
      enddo
    enddo
    call deliver2_ew(decomp%nbr_w, decomp%nbr_e)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          a(i1 - h + m - 1, j, n) = ew_rlo(m, j, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          a(i2 + m, j, n) = ew_rhi(m, j, 0)
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ew_2d_s

  subroutine exchange_ns_2d_s(a, i1, i2, j1, j2, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, slot
    integer(ip) :: h, i, m, n, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    n = slot
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO2
    do m = 1, h
      do i = ilo, ihi
        ns_slo(i, m, 0) = a(i, j1 + m - 1, n)
        ns_shi(i, m, 0) = a(i, j2 - h + m, n)
      enddo
    enddo
    call deliver2_ns(decomp%nbr_s, decomp%nbr_n)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          a(i, j1 - h + m - 1, n) = ns_rlo(i, m, 0)
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          a(i, j2 + m, n) = ns_rhi(i, m, 0)
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ns_2d_s

  subroutine exchange_ew_3d_s(a, i1, i2, j1, j2, k0, k1, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot
    integer(ip) :: h, j, m, k, n
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    n = slot
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k, n)
          ew_shi(m, j, k) = a(i2 - h + m, j, k, n)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k, n) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k, n) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ew_3d_s

  subroutine exchange_ns_3d_s(a, i1, i2, j1, j2, k0, k1, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot
    integer(ip) :: h, i, m, k, n, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    n = slot
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k, n)
          ns_shi(i, m, k) = a(i, j2 - h + m, k, n)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k, n) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k, n) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ns_3d_s

  subroutine exchange_ew_3d_t(a, i1, i2, j1, j2, k0, k1, slot, itrc)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :, :, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot, itrc
    integer(ip) :: h, j, m, k, n, itr
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    n = slot
    itr = itrc
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k, n, itr)
          ew_shi(m, j, k) = a(i2 - h + m, j, k, n, itr)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k, n, itr) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k, n, itr) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ew_3d_t

  subroutine exchange_ns_3d_t(a, i1, i2, j1, j2, k0, k1, slot, itrc)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, :, :, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot, itrc
    integer(ip) :: h, i, m, k, n, itr, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    n = slot
    itr = itrc
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k, n, itr)
          ns_shi(i, m, k) = a(i, j2 - h + m, k, n, itr)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k, n, itr) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k, n, itr) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ns_3d_t

  subroutine exchange_ew_3dw_s(a, i1, i2, j1, j2, k0, k1, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, 0:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot
    integer(ip) :: h, j, m, k, n
    if (decomp%nbr_w == NBR_NONE .and. decomp%nbr_e == NBR_NONE) return
    h = decomp%halo
    n = slot
    CUF_DO3
    do k = k0, k1
      do j = j1, j2
        do m = 1, h
          ew_slo(m, j, k) = a(i1 + m - 1, j, k, n)
          ew_shi(m, j, k) = a(i2 - h + m, j, k, n)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo, ew_shi, ew_rlo, ew_rhi)
    if (decomp%nbr_w /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i1 - h + m - 1, j, k, n) = ew_rlo(m, j, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do j = j1, j2
          do m = 1, h
            a(i2 + m, j, k, n) = ew_rhi(m, j, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ew_3dw_s

  subroutine exchange_ns_3dw_s(a, i1, i2, j1, j2, k0, k1, slot)
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:, decomp%js:, 0:, :)
    integer(ip), intent(in) :: i1, i2, j1, j2, k0, k1, slot
    integer(ip) :: h, i, m, k, n, ilo, ihi
    if (decomp%nbr_s == NBR_NONE .and. decomp%nbr_n == NBR_NONE) return
    h = decomp%halo
    n = slot
    ilo = i1 - h
    ihi = i2 + h
    CUF_DO3
    do k = k0, k1
      do m = 1, h
        do i = ilo, ihi
          ns_slo(i, m, k) = a(i, j1 + m - 1, k, n)
          ns_shi(i, m, k) = a(i, j2 - h + m, k, n)
        enddo
      enddo
    enddo
    call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo, ns_shi, ns_rlo, ns_rhi)
    if (decomp%nbr_s /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j1 - h + m - 1, k, n) = ns_rlo(i, m, k)
          enddo
        enddo
      enddo
    endif
    if (decomp%nbr_n /= NBR_NONE) then
      CUF_DO3
      do k = k0, k1
        do m = 1, h
          do i = ilo, ihi
            a(i, j2 + m, k, n) = ns_rhi(i, m, k)
          enddo
        enddo
      enddo
    endif
    call sync_arch()
  end subroutine exchange_ns_3dw_s

  ! Host-side sections of a managed array are legal. A CUF kernel on a
  ! section is not: nvfortran materializes a host temp. Index the parent
  ! pack buffers (k = 0 for 2-D) instead of passing ew_slo(:,:,0).
  subroutine deliver2_ew(nbr_lo, nbr_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer(ip) :: i, j, lo, hi, h, js, je
    if (nbr_lo == NBR_NONE .and. nbr_hi == NBR_NONE) return
    if (nbr_lo >= 0 .or. nbr_hi >= 0) then
      call sync_arch()
      call comm_exchange2(nbr_lo, nbr_hi, ew_slo(:,:,0), ew_shi(:,:,0), &
                          ew_rlo(:,:,0), ew_rhi(:,:,0))
      return
    endif
    h = decomp%halo
    js = decomp%js
    je = decomp%je
    lo = nbr_lo
    hi = nbr_hi
    CUF_DO2
    do j = js, je
      do i = 1, h
        ew_rlo(i, j, 0) = 0.0_rp
        ew_rhi(i, j, 0) = 0.0_rp
        if (lo == NBR_SELF .and. hi == NBR_SELF) then
          ew_rhi(i, j, 0) = ew_slo(i, j, 0)
          ew_rlo(i, j, 0) = ew_shi(i, j, 0)
        elseif (lo == NBR_SELF) then
          ew_rlo(i, j, 0) = ew_shi(i, j, 0)
        elseif (hi == NBR_SELF) then
          ew_rhi(i, j, 0) = ew_slo(i, j, 0)
        endif
      enddo
    enddo
  end subroutine deliver2_ew

  subroutine deliver2_ns(nbr_lo, nbr_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer(ip) :: i, m, lo, hi, h, is, ie
    if (nbr_lo == NBR_NONE .and. nbr_hi == NBR_NONE) return
    if (nbr_lo >= 0 .or. nbr_hi >= 0) then
      call sync_arch()
      call comm_exchange2(nbr_lo, nbr_hi, ns_slo(:,:,0), ns_shi(:,:,0), &
                          ns_rlo(:,:,0), ns_rhi(:,:,0))
      return
    endif
    h = decomp%halo
    is = decomp%is
    ie = decomp%ie
    lo = nbr_lo
    hi = nbr_hi
    CUF_DO2
    do m = 1, h
      do i = is, ie
        ns_rlo(i, m, 0) = 0.0_rp
        ns_rhi(i, m, 0) = 0.0_rp
        if (lo == NBR_SELF .and. hi == NBR_SELF) then
          ns_rhi(i, m, 0) = ns_slo(i, m, 0)
          ns_rlo(i, m, 0) = ns_shi(i, m, 0)
        elseif (lo == NBR_SELF) then
          ns_rlo(i, m, 0) = ns_shi(i, m, 0)
        elseif (hi == NBR_SELF) then
          ns_rhi(i, m, 0) = ns_slo(i, m, 0)
        endif
      enddo
    enddo
  end subroutine deliver2_ns

  subroutine deliver3(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    real(rp), WORK_ATTR intent(inout) :: send_lo(:,:,:), send_hi(:,:,:)
    real(rp), WORK_ATTR intent(inout) :: recv_lo(:,:,:), recv_hi(:,:,:)
    integer(ip) :: i, j, k, n1, n2, n3, lo, hi
    if (nbr_lo == NBR_NONE .and. nbr_hi == NBR_NONE) return
    if (nbr_lo >= 0 .or. nbr_hi >= 0) then
      call sync_arch()
      call comm_exchange3(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi)
      return
    endif
    n1 = size(send_lo, 1)
    n2 = size(send_lo, 2)
    n3 = size(send_lo, 3)
    lo = nbr_lo
    hi = nbr_hi
    CUF_DO3
    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          recv_lo(i, j, k) = 0.0_rp
          recv_hi(i, j, k) = 0.0_rp
          if (lo == NBR_SELF .and. hi == NBR_SELF) then
            recv_hi(i, j, k) = send_lo(i, j, k)
            recv_lo(i, j, k) = send_hi(i, j, k)
          elseif (lo == NBR_SELF) then
            recv_lo(i, j, k) = send_hi(i, j, k)
          elseif (hi == NBR_SELF) then
            recv_hi(i, j, k) = send_lo(i, j, k)
          endif
        enddo
      enddo
    enddo
  end subroutine deliver3

end module halo_mod
