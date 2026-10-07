#include "mem_attr.inc"
#include "cuf_launch.inc"
module halo_mod
  ! Pack/unpack launch on the GPU work stream. NBR_SELF deliver is a
  ! device copy so {step2d, halo_fill} can live in a CUDA graph.
  ! EW then NS so corners ride along (ROMS do_corners).
  use kinds
  use decomp_mod, only: decomp
  use time_state, only: fast, nt
  use comm_mod, only: comm_exchange2, comm_exchange3, comm_sendrecv_raw, comm_sendrecv_dev, comm_gpu_direct
  use gpu_state, only: Hz, z_r, z_w, FlxU, FlxV, We, Wi, u, v, t, rho1, qp1, rho, bvf, rhoA, rhoS, ru, rv, &
                       umask, vmask, rmask, sustr, svstr, stflx, srflx, f, hbls, hbbl, Akv, Akt, ghat, &
                       rufrc, rvfrc, wrk3a, wrk3b, ubar, vbar, zeta
#ifdef USE_CUDA
  use cudafor
#endif
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private

  ! Exchange buffers are plain device arrays (not managed): the pack and
  ! unpack kernels fill them, and the MPI path moves them with cudaMemcpy
  ! to and from pinned host memory, so no page migrates between the host
  ! and the device on an exchange.
#ifdef USE_CUDA
  real(rp), device, allocatable :: ew_slo(:,:,:), ew_shi(:,:,:)
  real(rp), device, allocatable :: ew_rlo(:,:,:), ew_rhi(:,:,:)
  real(rp), device, allocatable :: ns_slo(:,:,:), ns_shi(:,:,:)
  real(rp), device, allocatable :: ns_rlo(:,:,:), ns_rhi(:,:,:)
#else
  real(rp), allocatable :: ew_slo(:,:,:), ew_shi(:,:,:)
  real(rp), allocatable :: ew_rlo(:,:,:), ew_rhi(:,:,:)
  real(rp), allocatable :: ns_slo(:,:,:), ns_shi(:,:,:)
  real(rp), allocatable :: ns_rlo(:,:,:), ns_rhi(:,:,:)
#endif
  real(rp), WORK_ATTR allocatable :: work2d(:,:)
  real(rp), WORK_ATTR allocatable :: work3d(:,:,:)
  real(rp), WORK_ATTR allocatable :: work3dw(:,:,:)
  ! Group exchange state and buffers (device; grown on demand).
  integer, parameter :: HG_MAX = 40
  integer :: hg_n = 0, hg_nlev_tot = 0
  integer :: hg_id(HG_MAX), hg_slot(HG_MAX), hg_itr(HG_MAX), hg_lev0(HG_MAX), hg_nlev(HG_MAX)
#ifdef USE_CUDA
  real(rp), device, allocatable :: gew_slo(:,:,:), gew_shi(:,:,:), gew_rlo(:,:,:), gew_rhi(:,:,:)
  real(rp), device, allocatable :: gns_slo(:,:,:), gns_shi(:,:,:), gns_rlo(:,:,:), gns_rhi(:,:,:)
#else
  real(rp), allocatable :: gew_slo(:,:,:), gew_shi(:,:,:), gew_rlo(:,:,:), gew_rhi(:,:,:)
  real(rp), allocatable :: gns_slo(:,:,:), gns_shi(:,:,:), gns_rlo(:,:,:), gns_rhi(:,:,:)
#endif
  integer :: hg_cap = 0
#ifdef USE_CUDA
  ! Pinned host staging for the MPI path: packed device buffers are copied
  ! here asynchronously on work_stream, exchanged, and copied back. No
  ! managed-memory page migration and no per-call allocation.
  real(rp), pinned, allocatable :: hs_lo(:), hs_hi(:), hr_lo(:), hr_hi(:)
  integer :: hbuf_n = 0
#endif
  ! All-tracer passive exchange buffers: (h, js:je, nz*ntr) and (is:ie, h, nz*ntr)
#ifdef USE_CUDA
  real(rp), device, allocatable :: tew_slo(:,:,:), tew_shi(:,:,:), tew_rlo(:,:,:), tew_rhi(:,:,:)
  real(rp), device, allocatable :: tns_slo(:,:,:), tns_shi(:,:,:), tns_rlo(:,:,:), tns_rhi(:,:,:)
#else
  real(rp), allocatable :: tew_slo(:,:,:), tew_shi(:,:,:), tew_rlo(:,:,:), tew_rhi(:,:,:)
  real(rp), allocatable :: tns_slo(:,:,:), tns_shi(:,:,:), tns_rlo(:,:,:), tns_rhi(:,:,:)
#endif

  public :: halo_fill_2d
  public :: halo_fill_2d_n
  public :: halo_fill_uv_n
  public :: hg_begin, hg_add, hg_end
  ! Field ids for the group exchange (hg_*): several fields in one
  ! east-west and one north-south round. All fields use the rho-cell
  ! exchange bounds (see interior_bounds), so staggers may be mixed.
  integer, parameter, public :: F_HZ = 1, F_ZR = 2, F_ZW = 3, F_FLXU = 4, F_FLXV = 5, F_WE = 6, F_WI = 7, &
       F_U = 8, F_V = 9, F_T = 10, F_RHO1 = 11, F_QP1 = 12, F_RHO = 13, F_BVF = 14, F_RHOA = 15, F_RHOS = 16, &
       F_RU = 17, F_RV = 18, F_UMASK = 19, F_VMASK = 20, F_RMASK = 21, F_SUSTR = 22, F_SVSTR = 23, &
       F_STFLX = 24, F_SRFLX = 25, F_F = 26, F_HBLS = 27, F_HBBL = 28, F_AKV = 29, F_AKT = 30, F_GHAT = 31, &
       F_RUFRC = 32, F_RVFRC = 33, F_WRK3A = 34, F_WRK3B = 35, F_UBAR = 36, F_VBAR = 37, F_ZETA = 38
  public :: halo_fill_2d_s
  public :: halo_fill_2d_st
  public :: halo_fill_2d_sw
  public :: halo_fill_3d
  public :: halo_fill_3d_s
  public :: halo_fill_3d_t
  public :: halo_fill_3d_tp
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
    if (allocated(tew_slo)) deallocate(tew_slo, tew_shi, tew_rlo, tew_rhi)
    if (allocated(tns_slo)) deallocate(tns_slo, tns_shi, tns_rlo, tns_rhi)
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

  subroutine halo_fill_uv_n(u2, v2)
    ! ubar and vbar at slot fast%knew in ONE exchange per direction: both are
    ! packed as levels 0 and 1 of the 3-D buffers (the rho-cell exchange
    ! bounds are the same for every stagger). Pure copies.
    real(rp), WORK_ATTR intent(inout) :: u2(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    real(rp), WORK_ATTR intent(inout) :: v2(decomp%is:decomp%ie, decomp%js:decomp%je, 4)
    integer(ip) :: i1, i2, j1, j2, h, i, j, m, n, ilo, ihi
    if (no_neighbors()) return
    call ensure_bufs()
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    h = decomp%halo
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      CUF_DO2
      do j = j1, j2
        do m = 1, h
          n = fast%knew
          ew_slo(m, j, 0) = u2(i1 + m - 1, j, n)
          ew_shi(m, j, 0) = u2(i2 - h + m, j, n)
          ew_slo(m, j, 1) = v2(i1 + m - 1, j, n)
          ew_shi(m, j, 1) = v2(i2 - h + m, j, n)
        enddo
      enddo
      call deliver3(decomp%nbr_w, decomp%nbr_e, ew_slo(:, :, 0:1), ew_shi(:, :, 0:1), &
                    ew_rlo(:, :, 0:1), ew_rhi(:, :, 0:1))
      if (decomp%nbr_w /= NBR_NONE) then
        CUF_DO2
        do j = j1, j2
          do m = 1, h
            n = fast%knew
            u2(i1 - h + m - 1, j, n) = ew_rlo(m, j, 0)
            v2(i1 - h + m - 1, j, n) = ew_rlo(m, j, 1)
          enddo
        enddo
      endif
      if (decomp%nbr_e /= NBR_NONE) then
        CUF_DO2
        do j = j1, j2
          do m = 1, h
            n = fast%knew
            u2(i2 + m, j, n) = ew_rhi(m, j, 0)
            v2(i2 + m, j, n) = ew_rhi(m, j, 1)
          enddo
        enddo
      endif
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      ilo = i1 - h; ihi = i2 + h
      CUF_DO2
      do m = 1, h
        do i = ilo, ihi
          n = fast%knew
          ns_slo(i, m, 0) = u2(i, j1 + m - 1, n)
          ns_shi(i, m, 0) = u2(i, j2 - h + m, n)
          ns_slo(i, m, 1) = v2(i, j1 + m - 1, n)
          ns_shi(i, m, 1) = v2(i, j2 - h + m, n)
        enddo
      enddo
      call deliver3(decomp%nbr_s, decomp%nbr_n, ns_slo(:, :, 0:1), ns_shi(:, :, 0:1), &
                    ns_rlo(:, :, 0:1), ns_rhi(:, :, 0:1))
      if (decomp%nbr_s /= NBR_NONE) then
        CUF_DO2
        do m = 1, h
          do i = ilo, ihi
            n = fast%knew
            u2(i, j1 - h + m - 1, n) = ns_rlo(i, m, 0)
            v2(i, j1 - h + m - 1, n) = ns_rlo(i, m, 1)
          enddo
        enddo
      endif
      if (decomp%nbr_n /= NBR_NONE) then
        CUF_DO2
        do m = 1, h
          do i = ilo, ihi
            n = fast%knew
            u2(i, j2 + m, n) = ns_rhi(i, m, 0)
            v2(i, j2 + m, n) = ns_rhi(i, m, 1)
          enddo
        enddo
      endif
    endif
  end subroutine halo_fill_uv_n

  ! ------------------------------------------------------------ group exchange
  subroutine hg_begin()
    hg_n = 0; hg_nlev_tot = 0
  end subroutine hg_begin

  subroutine hg_add(id, slot, itr)
    ! Register a field (optionally a time slot and a tracer/level index).
    integer, intent(in) :: id
    integer(ip), intent(in), optional :: slot, itr
    integer :: nl
    if (no_neighbors()) return
    hg_n = hg_n + 1
    if (hg_n > HG_MAX) error stop 'hg_add: too many fields in one group'
    hg_id(hg_n) = id
    hg_slot(hg_n) = 1; hg_itr(hg_n) = 1
    if (present(slot)) hg_slot(hg_n) = int(slot)
    if (present(itr)) hg_itr(hg_n) = int(itr)
    select case (id)
    case (F_ZW, F_WE, F_WI, F_BVF, F_AKV, F_AKT, F_GHAT)
      nl = int(decomp%nz) + 1
    case (F_HZ, F_ZR, F_FLXU, F_FLXV, F_U, F_V, F_T, F_RHO1, F_QP1, F_RHO, F_RU, F_RV, F_WRK3A, F_WRK3B)
      nl = int(decomp%nz)
    case default
      nl = 1
    end select
    hg_lev0(hg_n) = hg_nlev_tot
    hg_nlev(hg_n) = nl
    hg_nlev_tot = hg_nlev_tot + nl
  end subroutine hg_add

  subroutine hg_end()
    ! One east-west round for every registered field, then one north-south
    ! round (including the east-west halo columns), as the single fills do.
    integer(ip) :: i1, i2, j1, j2, h, is, ie, js, je, k
    integer :: L
    if (no_neighbors() .or. hg_n == 0) then
      hg_n = 0; hg_nlev_tot = 0
      return
    endif
    call ensure_bufs()
    h = decomp%halo; is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    L = hg_nlev_tot
    if (hg_cap < L) then
      if (allocated(gew_slo)) deallocate(gew_slo, gew_shi, gew_rlo, gew_rhi, gns_slo, gns_shi, gns_rlo, gns_rhi)
      allocate(gew_slo(h, js:je, L), gew_shi(h, js:je, L), gew_rlo(h, js:je, L), gew_rhi(h, js:je, L))
      allocate(gns_slo(is:ie, h, L), gns_shi(is:ie, h, L), gns_rlo(is:ie, h, L), gns_rhi(is:ie, h, L))
      gew_slo = 0.0_rp; gew_shi = 0.0_rp; gew_rlo = 0.0_rp; gew_rhi = 0.0_rp
      gns_slo = 0.0_rp; gns_shi = 0.0_rp; gns_rlo = 0.0_rp; gns_rhi = 0.0_rp
      hg_cap = L
    endif
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      do k = 1, hg_n
        call hg_dispatch(k, 1, i1, i2, j1, j2)
      enddo
      call deliver3(decomp%nbr_w, decomp%nbr_e, gew_slo(:, :, 1:L), gew_shi(:, :, 1:L), &
                    gew_rlo(:, :, 1:L), gew_rhi(:, :, 1:L))
      do k = 1, hg_n
        call hg_dispatch(k, 2, i1, i2, j1, j2)
      enddo
    endif
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      do k = 1, hg_n
        call hg_dispatch(k, 3, i1, i2, j1, j2)
      enddo
      call deliver3(decomp%nbr_s, decomp%nbr_n, gns_slo(:, :, 1:L), gns_shi(:, :, 1:L), &
                    gns_rlo(:, :, 1:L), gns_rhi(:, :, 1:L))
      do k = 1, hg_n
        call hg_dispatch(k, 4, i1, i2, j1, j2)
      enddo
    endif
    hg_n = 0; hg_nlev_tot = 0
  end subroutine hg_end

  subroutine hg_dispatch(k, phase, i1, i2, j1, j2)
    ! Resolve the field id to its slab (first element + level range) and run
    ! phase 1: pack EW, 2: unpack EW, 3: pack NS, 4: unpack NS.
    integer, intent(in) :: k, phase
    integer(ip), intent(in) :: i1, i2, j1, j2
    integer(ip) :: is, js, nz, sl, it
    is = decomp%is; js = decomp%js; nz = decomp%nz
    sl = hg_slot(k); it = hg_itr(k)
    select case (hg_id(k))
    case (F_HZ);    call hg_slab(Hz(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_ZR);    call hg_slab(z_r(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_ZW);    call hg_slab(z_w(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_FLXU);  call hg_slab(FlxU(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_FLXV);  call hg_slab(FlxV(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_WE);    call hg_slab(We(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_WI);    call hg_slab(Wi(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_U);     call hg_slab(u(is, js, 1, sl), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_V);     call hg_slab(v(is, js, 1, sl), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_T);     call hg_slab(t(is, js, 1, sl, it), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RHO1);  call hg_slab(rho1(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_QP1);   call hg_slab(qp1(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RHO);   call hg_slab(rho(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_BVF);   call hg_slab(bvf(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RHOA);  call hg_slab(rhoA(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_RHOS);  call hg_slab(rhoS(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_RU);    call hg_slab(ru(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RV);    call hg_slab(rv(is, js, 1), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_UMASK); call hg_slab(umask(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_VMASK); call hg_slab(vmask(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_RMASK); call hg_slab(rmask(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_SUSTR); call hg_slab(sustr(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_SVSTR); call hg_slab(svstr(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_STFLX); call hg_slab(stflx(is, js, it), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_SRFLX); call hg_slab(srflx(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_F);     call hg_slab(f(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_HBLS);  call hg_slab(hbls(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_HBBL);  call hg_slab(hbbl(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_AKV);   call hg_slab(Akv(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_AKT);   call hg_slab(Akt(is, js, 0, it), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_GHAT);  call hg_slab(ghat(is, js, 0), 0_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_RUFRC); call hg_slab(rufrc(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_RVFRC); call hg_slab(rvfrc(is, js), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    ! wrk3a/wrk3b are (0:nz) arrays that the single fill exchanged as nz
    ! levels from level 0 (sequence association); keep exactly that slab.
    case (F_WRK3A); call hg_slab(wrk3a(is, js, 0), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_WRK3B); call hg_slab(wrk3b(is, js, 0), 1_ip, nz, k, phase, i1, i2, j1, j2)
    case (F_UBAR);  call hg_slab(ubar(is, js, sl), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_VBAR);  call hg_slab(vbar(is, js, sl), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case (F_ZETA);  call hg_slab(zeta(is, js, sl), 1_ip, 1_ip, k, phase, i1, i2, j1, j2)
    case default
      error stop 'hg_dispatch: unknown field id'
    end select
  end subroutine hg_dispatch

  subroutine hg_slab(a, kb, ke, k, phase, i1, i2, j1, j2)
    ! Pack / unpack one slab a(is:ie, js:je, kb:ke) at group level offset
    ! hg_lev0(k) (pure copies).
    integer(ip), intent(in) :: kb, ke, i1, i2, j1, j2
    integer, intent(in) :: k, phase
    real(rp), WORK_ATTR intent(inout) :: a(decomp%is:decomp%ie, decomp%js:decomp%je, kb:ke)
    integer(ip) :: i, j, m, kk, h, l0, ilo, ihi
    h = decomp%halo
    l0 = hg_lev0(k)
    ilo = i1 - h; ihi = i2 + h
    select case (phase)
    case (1)
      CUF_DO3
      do kk = kb, ke
        do j = j1, j2
          do m = 1, h
            gew_slo(m, j, l0 + kk - kb + 1) = a(i1 + m - 1, j, kk)
            gew_shi(m, j, l0 + kk - kb + 1) = a(i2 - h + m, j, kk)
          enddo
        enddo
      enddo
    case (2)
      if (decomp%nbr_w /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do j = j1, j2
            do m = 1, h
              a(i1 - h + m - 1, j, kk) = gew_rlo(m, j, l0 + kk - kb + 1)
            enddo
          enddo
        enddo
      endif
      if (decomp%nbr_e /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do j = j1, j2
            do m = 1, h
              a(i2 + m, j, kk) = gew_rhi(m, j, l0 + kk - kb + 1)
            enddo
          enddo
        enddo
      endif
    case (3)
      CUF_DO3
      do kk = kb, ke
        do m = 1, h
          do i = ilo, ihi
            gns_slo(i, m, l0 + kk - kb + 1) = a(i, j1 + m - 1, kk)
            gns_shi(i, m, l0 + kk - kb + 1) = a(i, j2 - h + m, kk)
          enddo
        enddo
      enddo
    case (4)
      if (decomp%nbr_s /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do m = 1, h
            do i = ilo, ihi
              a(i, j1 - h + m - 1, kk) = gns_rlo(i, m, l0 + kk - kb + 1)
            enddo
          enddo
        enddo
      endif
      if (decomp%nbr_n /= NBR_NONE) then
        CUF_DO3
        do kk = kb, ke
          do m = 1, h
            do i = ilo, ihi
              a(i, j2 + m, kk) = gns_rhi(i, m, l0 + kk - kb + 1)
            enddo
          enddo
        enddo
      endif
    end select
  end subroutine hg_slab

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

  subroutine halo_fill_3d_tp(a, ntr, slot)
    ! Passive tracers, tracer-innermost layout a(ntr, is:ie, js:je, nz, 3):
    ! one exchange of all tracers in slot `slot` at rho points. Pure copies,
    ! so a tiled run reproduces the single-tile interior bit for bit. The
    ! combined buffer index is kk = (k-1)*ntr + itr; east-west first, then
    ! north-south including the east-west halo columns (as halo_fill_3d).
    real(pp), WORK_ATTR intent(inout) :: a(ntr, decomp%is:decomp%ie, decomp%js:decomp%je, decomp%nz, 3)
    integer(ip), intent(in) :: ntr, slot
    integer(ip) :: i, j, k, kk, m, itr, n, h, nz, nk, i1, i2, j1, j2, ilo, ihi, is, ie, js, je
    if (no_neighbors()) return
    if (ntr <= 0) return
    h = decomp%halo; nz = decomp%nz; nk = nz * ntr; n = slot
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call interior_bounds(STAGGER_RHO, i1, i2, j1, j2)
    if (allocated(tew_slo)) then
      if (ubound(tew_slo, 3) /= nk .or. lbound(tew_slo, 2) /= js .or. ubound(tew_slo, 2) /= je .or. &
          lbound(tns_slo, 1) /= is .or. ubound(tns_slo, 1) /= ie) then
        deallocate(tew_slo, tew_shi, tew_rlo, tew_rhi, tns_slo, tns_shi, tns_rlo, tns_rhi)
      endif
    endif
    if (.not. allocated(tew_slo)) then
      allocate(tew_slo(h, js:je, nk), tew_shi(h, js:je, nk), tew_rlo(h, js:je, nk), tew_rhi(h, js:je, nk))
      allocate(tns_slo(is:ie, h, nk), tns_shi(is:ie, h, nk), tns_rlo(is:ie, h, nk), tns_rhi(is:ie, h, nk))
      tew_slo = 0.0_rp; tew_shi = 0.0_rp; tew_rlo = 0.0_rp; tew_rhi = 0.0_rp
      tns_slo = 0.0_rp; tns_shi = 0.0_rp; tns_rlo = 0.0_rp; tns_rhi = 0.0_rp
    endif
    ! ---- east-west
    if (decomp%nbr_w /= NBR_NONE .or. decomp%nbr_e /= NBR_NONE) then
      CUF_DO3
      do kk = 1, nk
        do j = j1, j2
          do m = 1, h
            k = (kk - 1) / ntr + 1
            itr = kk - (k - 1) * ntr
            tew_slo(m, j, kk) = real(a(itr, i1 + m - 1, j, k, n), rp)
            tew_shi(m, j, kk) = real(a(itr, i2 - h + m, j, k, n), rp)
          enddo
        enddo
      enddo
      call deliver3(decomp%nbr_w, decomp%nbr_e, tew_slo, tew_shi, tew_rlo, tew_rhi)
      if (decomp%nbr_w /= NBR_NONE) then
        CUF_DO3
        do kk = 1, nk
          do j = j1, j2
            do m = 1, h
              k = (kk - 1) / ntr + 1
              itr = kk - (k - 1) * ntr
              a(itr, i1 - h + m - 1, j, k, n) = real(tew_rlo(m, j, kk), pp)
            enddo
          enddo
        enddo
      endif
      if (decomp%nbr_e /= NBR_NONE) then
        CUF_DO3
        do kk = 1, nk
          do j = j1, j2
            do m = 1, h
              k = (kk - 1) / ntr + 1
              itr = kk - (k - 1) * ntr
              a(itr, i2 + m, j, k, n) = real(tew_rhi(m, j, kk), pp)
            enddo
          enddo
        enddo
      endif
    endif
    ! ---- north-south, including the east-west halo columns
    if (decomp%nbr_s /= NBR_NONE .or. decomp%nbr_n /= NBR_NONE) then
      ilo = i1 - h; ihi = i2 + h
      CUF_DO3
      do kk = 1, nk
        do m = 1, h
          do i = ilo, ihi
            k = (kk - 1) / ntr + 1
            itr = kk - (k - 1) * ntr
            tns_slo(i, m, kk) = real(a(itr, i, j1 + m - 1, k, n), rp)
            tns_shi(i, m, kk) = real(a(itr, i, j2 - h + m, k, n), rp)
          enddo
        enddo
      enddo
      call deliver3(decomp%nbr_s, decomp%nbr_n, tns_slo, tns_shi, tns_rlo, tns_rhi)
      if (decomp%nbr_s /= NBR_NONE) then
        CUF_DO3
        do kk = 1, nk
          do m = 1, h
            do i = ilo, ihi
              k = (kk - 1) / ntr + 1
              itr = kk - (k - 1) * ntr
              a(itr, i, j1 - h + m - 1, k, n) = real(tns_rlo(i, m, kk), pp)
            enddo
          enddo
        enddo
      endif
      if (decomp%nbr_n /= NBR_NONE) then
        CUF_DO3
        do kk = 1, nk
          do m = 1, h
            do i = ilo, ihi
              k = (kk - 1) / ntr + 1
              itr = kk - (k - 1) * ntr
              a(itr, i, j2 + m, k, n) = real(tns_rhi(i, m, kk), pp)
            enddo
          enddo
        enddo
      endif
    endif
  end subroutine halo_fill_3d_tp

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
    ! Exchange bounds are the rho-cell range for every stagger. A u (v)
    ! face on a seam is the interior point 1 of the east (north) tile and
    ! the point nx+1 (ny+1) of the west (south) tile: the kernels compute it
    ! on the east/north side (uv_rhs stops at nx toward a neighbor), so the
    ! exchange carries it, with one more point beyond, to the west/south
    ! tile as its halo a(nx+1), a(nx+2). Periodic wrap gives a(nx+1) = a(1).
    ! The physical boundary rows (j = 0 at a south edge, ny+1 at a north
    ! edge) are computed by the boundary routines with tangential neighbours
    ! (u2dbc/v2dbc/u3dbc/v3dbc/t3dbc radiation), so an east-west seam on an
    ! open edge needs them exchanged too. The north-south round already
    ! covers the columns 0 and nx+1 (it packs i1-h .. i2+h).
    i1 = 1
    j1 = 1
    i2 = decomp%nx
    j2 = decomp%ny
    if (decomp%edge_s) j1 = 0
    if (decomp%edge_n) j2 = decomp%ny + 1
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
  end subroutine exchange_ns_3dw_s

  ! Host-side sections of a managed array are legal. A CUF kernel on a
  ! section is not: nvfortran materializes a host temp. Index the parent
  ! pack buffers (k = 0 for 2-D) instead of passing ew_slo(:,:,0).
  subroutine deliver2_ew(nbr_lo, nbr_hi)
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    integer(ip) :: i, j, lo, hi, h, js, je
    if (nbr_lo == NBR_NONE .and. nbr_hi == NBR_NONE) return
    if (nbr_lo >= 0 .or. nbr_hi >= 0) then
#ifdef USE_CUDA
      call deliver_dev(nbr_lo, nbr_hi, ew_slo(:,:,0:0), ew_shi(:,:,0:0), ew_rlo(:,:,0:0), ew_rhi(:,:,0:0))
#else
      call sync_arch()
      call comm_exchange2(nbr_lo, nbr_hi, ew_slo(:,:,0), ew_shi(:,:,0), &
                          ew_rlo(:,:,0), ew_rhi(:,:,0))
#endif
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
#ifdef USE_CUDA
      call deliver_dev(nbr_lo, nbr_hi, ns_slo(:,:,0:0), ns_shi(:,:,0:0), ns_rlo(:,:,0:0), ns_rhi(:,:,0:0))
#else
      call sync_arch()
      call comm_exchange2(nbr_lo, nbr_hi, ns_slo(:,:,0), ns_shi(:,:,0), &
                          ns_rlo(:,:,0), ns_rhi(:,:,0))
#endif
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
#ifdef USE_CUDA
    real(rp), device, intent(inout), contiguous :: send_lo(:,:,:), send_hi(:,:,:)
    real(rp), device, intent(inout), contiguous :: recv_lo(:,:,:), recv_hi(:,:,:)
#else
    real(rp), intent(inout), contiguous :: send_lo(:,:,:), send_hi(:,:,:)
    real(rp), intent(inout), contiguous :: recv_lo(:,:,:), recv_hi(:,:,:)
#endif
    integer(ip) :: i, j, k, n1, n2, n3, lo, hi
    if (nbr_lo == NBR_NONE .and. nbr_hi == NBR_NONE) return
    if (nbr_lo >= 0 .or. nbr_hi >= 0) then
#ifdef USE_CUDA
      call deliver_dev(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi)
#else
      call sync_arch()
      call comm_exchange3(nbr_lo, nbr_hi, send_lo, send_hi, recv_lo, recv_hi)
#endif
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

#ifdef USE_CUDA
  subroutine deliver_dev(nbr_lo, nbr_hi, s_lo, s_hi, r_lo, r_hi)
    ! MPI delivery of packed device buffers through pinned host memory,
    ! stream-ordered after the pack kernels and before the unpack kernels.
    integer(ip), intent(in) :: nbr_lo, nbr_hi
    real(rp), device, intent(inout), contiguous :: s_lo(:,:,:), s_hi(:,:,:)
    real(rp), device, intent(inout), contiguous :: r_lo(:,:,:), r_hi(:,:,:)
    integer :: n, ierr
    n = size(s_lo)
    if (size(s_hi) /= n .or. size(r_lo) /= n .or. size(r_hi) /= n) &
         error stop 'deliver_dev: buffer size mismatch'
    if (comm_gpu_direct) then
      ! GPU-aware MPI: the pack kernels (work_stream) must have completed;
      ! the blocking Sendrecv returns with r_lo/r_hi filled on the device.
      ierr = cudaStreamSynchronize(work_stream)
      if (ierr /= 0) error stop 'deliver_dev: stream sync'
      call comm_sendrecv_dev(nbr_lo, nbr_hi, n, s_lo, s_hi, r_lo, r_hi)
      return
    endif
    if (hbuf_n < n) then
      if (allocated(hs_lo)) deallocate(hs_lo, hs_hi, hr_lo, hr_hi)
      allocate(hs_lo(n), hs_hi(n), hr_lo(n), hr_hi(n))
      hbuf_n = n
    endif
    ! Synchronous copies: the legacy default stream orders them after the
    ! pack kernels and before the unpack kernels on work_stream (a blocking
    ! stream). cudaMemcpyAsync does not resolve for these managed dummies.
    ierr = cudaMemcpy(hs_lo, s_lo, n)
    if (ierr /= 0) error stop 'deliver_dev: D2H lo'
    ierr = cudaMemcpy(hs_hi, s_hi, n)
    if (ierr /= 0) error stop 'deliver_dev: D2H hi'
    call comm_sendrecv_raw(nbr_lo, nbr_hi, n, hs_lo, hs_hi, hr_lo, hr_hi)
    ierr = cudaMemcpy(r_lo, hr_lo, n)
    if (ierr /= 0) error stop 'deliver_dev: H2D lo'
    ierr = cudaMemcpy(r_hi, hr_hi, n)
    if (ierr /= 0) error stop 'deliver_dev: H2D hi'
  end subroutine deliver_dev
#endif

end module halo_mod
