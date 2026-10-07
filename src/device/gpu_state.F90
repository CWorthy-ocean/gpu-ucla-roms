#include "mem_attr.inc"
module gpu_state
  ! One working copy per field, local tile + halo. USE_CUDA: managed
  ! (host I/O and device kernels share it). CPU: ordinary allocatables.
  ! NetCDF talks only to host buffers in host_buf, not these arrays.
  use kinds
  use decomp_mod, only: decomp
  use arch_mod, only: init_arch, reset_graphs
  implicit none
  private

  ! Prognostic
  real(rp), WORK_ATTR public, allocatable :: zeta(:,:,:)          ! (i,j,4)
  real(rp), WORK_ATTR public, allocatable :: ubar(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: vbar(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: u(:,:,:,:)           ! (i,j,k,3)
  real(rp), WORK_ATTR public, allocatable :: v(:,:,:,:)
  real(rp), WORK_ATTR public, allocatable :: t(:,:,:,:,:)         ! (i,j,k,3,nt)

  ! Geometry / fluxes
  real(rp), WORK_ATTR public, allocatable :: Hz(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: Hz_u(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: Hz_v(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: z_r(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: z_w(:,:,:)
  ! (The zeta=0 geometry copies Hz0/z_r0/z_w0 were never read; removed.)
  real(rp), WORK_ATTR public, allocatable :: FlxU(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: FlxV(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: We(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: Wi(:,:,:)

  ! Grid (constant after init)
  real(rp), WORK_ATTR public, allocatable :: h(:,:)
  real(rp), WORK_ATTR public, allocatable :: hinv(:,:)
  real(rp), WORK_ATTR public, allocatable :: pm(:,:), pn(:,:)
  real(rp), WORK_ATTR public, allocatable :: dn_u(:,:), dm_v(:,:)
  real(rp), WORK_ATTR public, allocatable :: dm_r(:,:), dn_r(:,:)
  real(rp), WORK_ATTR public, allocatable :: rmask(:,:), umask(:,:), vmask(:,:)
  real(rp), WORK_ATTR public, allocatable :: f(:,:)
  real(rp), WORK_ATTR public, allocatable :: lonr(:,:), latr(:,:), angler(:,:)
  real(rp), WORK_ATTR public, allocatable :: swflx(:,:)
  real(rp), WORK_ATTR public, allocatable :: tair(:,:)
  real(rp), WORK_ATTR public, allocatable :: pmask(:,:), dm_p(:,:), dn_p(:,:)
  real(rp), WORK_ATTR public, allocatable :: dm_u(:,:), dn_v(:,:)
  real(rp), WORK_ATTR public, allocatable :: pmon_u(:,:), pnom_v(:,:)
  real(rp), WORK_ATTR public, allocatable :: dndx(:,:), dmde(:,:)
  real(rp), WORK_ATTR public, allocatable :: Akv(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: Akt(:,:,:,:)
  real(rp), WORK_ATTR public, allocatable :: visc2_r(:,:), visc2_p(:,:)
  real(rp), WORK_ATTR public, allocatable :: diff2(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: sustr(:,:), svstr(:,:)
  ! Rho-point surface stress (BULK_FRC); only lmd_kpp uses it, for ustar.
  real(rp), WORK_ATTR public, allocatable :: sustr_r(:,:), svstr_r(:,:)
  real(rp), WORK_ATTR public, allocatable :: stflx(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: r_D(:,:)
  real(rp), WORK_ATTR public, allocatable :: bvf(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: hbls(:,:), hbbl(:,:)
  real(rp), WORK_ATTR public, allocatable :: ghat(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: swr_frac(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: srflx(:,:)
  real(rp), public :: zob = 0.01_rp
  real(rp), public :: rdrg = 0.0_rp
  real(rp), public :: rdrg2 = 0.0_rp
  real(rp), public :: visc2 = 0.0_rp
  real(rp), public :: akv_bak = 0.0_rp
  real(rp), public :: akt_bak(2) = 0.0_rp
  real(rp), public :: ubind = 0.1_rp
  real(rp), public :: rho0_case = 1025.0_rp
  logical,  public :: obc_west = .false.
  logical,  public :: obc_east = .false.
  logical,  public :: obc_south = .false.
  logical,  public :: obc_north = .false.
  logical,  public :: have_flux = .false.
  logical,  public :: have_bry = .false.
  logical,  public :: have_bulk = .false.
  logical,  public :: pot_tides = .false.
  logical,  public :: bry_tides = .false.
  real(rp), WORK_ATTR public, allocatable :: ptide(:,:)

  ! Open-boundary forcing (along-edge, host-filled, kernels consume).
  real(rp), WORK_ATTR public, allocatable :: zeta_west(:), ubar_west(:), vbar_west(:)
  real(rp), WORK_ATTR public, allocatable :: zeta_east(:), ubar_east(:), vbar_east(:)
  real(rp), WORK_ATTR public, allocatable :: zeta_south(:), ubar_south(:), vbar_south(:)
  real(rp), WORK_ATTR public, allocatable :: zeta_north(:), ubar_north(:), vbar_north(:)
  real(rp), WORK_ATTR public, allocatable :: u_west(:,:), v_west(:,:), t_west(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: u_east(:,:), v_east(:,:), t_east(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: u_south(:,:), v_south(:,:), t_south(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: u_north(:,:), v_north(:,:), t_north(:,:,:)

  ! Mode coupling (first-class; restart must include these)
  real(rp), WORK_ATTR public, allocatable :: rufrc(:,:), rvfrc(:,:)
  real(rp), WORK_ATTR public, allocatable :: rufrc_bak(:,:,:), rvfrc_bak(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: DU_avg1(:,:), DV_avg1(:,:)
  real(rp), WORK_ATTR public, allocatable :: DU_avg2(:,:), DV_avg2(:,:)
  real(rp), WORK_ATTR public, allocatable :: DU_avg_bak(:,:), DV_avg_bak(:,:)
  real(rp), WORK_ATTR public, allocatable :: zt_avg1(:,:)
  real(rp), WORK_ATTR public, allocatable :: rho(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: rho1(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: qp1(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: rhoA(:,:), rhoS(:,:)
  real(rp), WORK_ATTR public, allocatable :: ru(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: rv(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: P(:,:,:)

  ! Persistent step2d workspace (allocated once, not per iif)
  real(rp), WORK_ATTR public, allocatable :: Drhs(:,:), DUon(:,:), DVom(:,:)
  real(rp), WORK_ATTR public, allocatable :: rubar(:,:), rvbar(:,:)
  real(rp), WORK_ATTR public, allocatable :: zeta_new(:,:), Dnew(:,:)
  real(rp), WORK_ATTR public, allocatable :: zwrk(:,:), rzeta(:,:), rzeta2(:,:), rzetaSA(:,:)
  ! FX/FE/Fz (per-tracer flux fallback) alias wrk3a/wrk3b/col_wc.
  ! Persistent 3-D physics workspace (managed; CUF kernels write these)
  real(rp), WORK_ATTR public, allocatable :: Hz_bak(:,:,:), Hz_fwd(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: wrk2a(:,:), wrk2b(:,:), wrk2c(:,:), wrk2d(:,:)
  real(rp), WORK_ATTR public, allocatable :: wrk2e(:,:), wrk2f(:,:)
  ! 3-D scratch pool (allocated on first use); see the lifetime notes in
  ! GPU_REVIEW_2026-10-02.md section 3 (M1).
  real(rp), WORK_ATTR public, allocatable :: wrk3a(:,:,:), wrk3b(:,:,:), wrk3c(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: wrk3d(:,:,:), wrk3e(:,:,:), wrk3f(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: col_fc(:,:,:), col_cf(:,:,:)
  real(rp), WORK_ATTR public, allocatable :: col_dc(:,:,:), col_wc(:,:,:)

  public :: alloc_state
  public :: ensure_wrk3
  public :: free_state
  public :: init_ideal_box

  ! Wet columns of this tile's interior (1:nx, 1:ny), j-major order: the
  ! column kernels run over these only. Land columns keep their (zero)
  ! values, which is what the full-tile kernels leave there too.
  integer(ip), public :: nwet = 0
#ifdef USE_CUDA
  integer(ip), device, allocatable, public :: wet_i(:), wet_j(:)
#else
  integer(ip), allocatable, public :: wet_i(:), wet_j(:)
#endif
  public :: build_wet_list


contains

  subroutine build_wet_list()
    ! Called once the mask is final (after fill_static_halos / setup_metrics).
    integer(ip) :: i, j, n
    integer(ip), allocatable :: hi(:), hj(:)
    n = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) > 0.5_rp) n = n + 1
      enddo
    enddo
    allocate(hi(max(n, 1)), hj(max(n, 1)))
    n = 0
    do j = 1, decomp%ny
      do i = 1, decomp%nx
        if (rmask(i, j) > 0.5_rp) then
          n = n + 1; hi(n) = i; hj(n) = j
        endif
      enddo
    enddo
    nwet = n
    if (allocated(wet_i)) deallocate(wet_i, wet_j)
    allocate(wet_i(max(n, 1)), wet_j(max(n, 1)))
    wet_i = hi; wet_j = hj
    deallocate(hi, hj)
  end subroutine build_wet_list

  subroutine alloc_state(nt)
    integer(ip), intent(in) :: nt
    integer(ip) :: is, ie, js, je, nz

    call init_arch()
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    nz = decomp%nz

    allocate(zeta(is:ie,js:je,4), ubar(is:ie,js:je,4), vbar(is:ie,js:je,4))
    allocate(u(is:ie,js:je,nz,3), v(is:ie,js:je,nz,3))
    allocate(t(is:ie,js:je,nz,3,nt))
    allocate(Hz(is:ie,js:je,nz), Hz_u(is:ie,js:je,nz), Hz_v(is:ie,js:je,nz))
    allocate(z_r(is:ie,js:je,nz), z_w(is:ie,js:je,0:nz))
    allocate(FlxU(is:ie,js:je,nz), FlxV(is:ie,js:je,nz))
    allocate(We(is:ie,js:je,0:nz), Wi(is:ie,js:je,0:nz))
    allocate(h(is:ie,js:je), hinv(is:ie,js:je))
    allocate(pm(is:ie,js:je), pn(is:ie,js:je))
    allocate(dn_u(is:ie,js:je), dm_v(is:ie,js:je))
    allocate(dm_r(is:ie,js:je), dn_r(is:ie,js:je))
    allocate(rmask(is:ie,js:je), umask(is:ie,js:je), vmask(is:ie,js:je))
    allocate(f(is:ie,js:je), lonr(is:ie,js:je), latr(is:ie,js:je), angler(is:ie,js:je))
    allocate(swflx(is:ie,js:je), tair(is:ie,js:je))
    allocate(pmask(is:ie,js:je), dm_p(is:ie,js:je), dn_p(is:ie,js:je))
    allocate(dm_u(is:ie,js:je), dn_v(is:ie,js:je))
    allocate(pmon_u(is:ie,js:je), pnom_v(is:ie,js:je))
    allocate(dndx(is:ie,js:je), dmde(is:ie,js:je))
    allocate(Akv(is:ie,js:je,0:nz), Akt(is:ie,js:je,0:nz,2))
    allocate(visc2_r(is:ie,js:je), visc2_p(is:ie,js:je))
    allocate(diff2(is:ie,js:je,nt))
    allocate(sustr(is:ie,js:je), svstr(is:ie,js:je), stflx(is:ie,js:je,nt))
    allocate(sustr_r(is:ie,js:je), svstr_r(is:ie,js:je))
    sustr_r = 0.0_rp; svstr_r = 0.0_rp
    allocate(r_D(is:ie,js:je), bvf(is:ie,js:je,0:nz))
    allocate(hbls(is:ie,js:je), hbbl(is:ie,js:je), srflx(is:ie,js:je))
    allocate(ghat(is:ie,js:je,0:nz), swr_frac(is:ie,js:je,0:nz))
    allocate(rufrc(is:ie,js:je), rvfrc(is:ie,js:je))
    allocate(rufrc_bak(is:ie,js:je,2), rvfrc_bak(is:ie,js:je,2))
    allocate(DU_avg1(is:ie,js:je), DV_avg1(is:ie,js:je))
    allocate(DU_avg2(is:ie,js:je), DV_avg2(is:ie,js:je))
    allocate(DU_avg_bak(is:ie,js:je), DV_avg_bak(is:ie,js:je))
    allocate(zt_avg1(is:ie,js:je), rho(is:ie,js:je,nz))
    allocate(rho1(is:ie,js:je,nz), qp1(is:ie,js:je,nz))
    allocate(rhoA(is:ie,js:je), rhoS(is:ie,js:je))
    allocate(ru(is:ie,js:je,nz), rv(is:ie,js:je,nz), P(is:ie,js:je,nz))
    allocate(Drhs(is:ie,js:je), DUon(is:ie,js:je), DVom(is:ie,js:je))
    allocate(rubar(is:ie,js:je), rvbar(is:ie,js:je))
    allocate(zeta_new(is:ie,js:je), Dnew(is:ie,js:je))
    allocate(zwrk(is:ie,js:je), rzeta(is:ie,js:je), rzeta2(is:ie,js:je), rzetaSA(is:ie,js:je))
    allocate(Hz_bak(is:ie,js:je,nz), Hz_fwd(is:ie,js:je,nz))
    allocate(wrk2a(is:ie,js:je), wrk2b(is:ie,js:je), wrk2c(is:ie,js:je), wrk2d(is:ie,js:je))
    allocate(wrk2e(is:ie,js:je), wrk2f(is:ie,js:je))
    allocate(col_fc(is:ie,js:je,0:nz), col_cf(is:ie,js:je,0:nz))
    allocate(col_dc(is:ie,js:je,0:nz), col_wc(is:ie,js:je,0:nz))
    Hz_bak = 0.0_rp; Hz_fwd = 0.0_rp
    wrk2a = 0.0_rp; wrk2b = 0.0_rp; wrk2c = 0.0_rp; wrk2d = 0.0_rp
    wrk2e = 0.0_rp; wrk2f = 0.0_rp
    col_fc = 0.0_rp; col_cf = 0.0_rp; col_dc = 0.0_rp; col_wc = 0.0_rp
    allocate(zeta_west(js:je), ubar_west(js:je), vbar_west(js:je))
    allocate(zeta_east(js:je), ubar_east(js:je), vbar_east(js:je))
    allocate(zeta_south(is:ie), ubar_south(is:ie), vbar_south(is:ie))
    allocate(zeta_north(is:ie), ubar_north(is:ie), vbar_north(is:ie))
    allocate(u_west(js:je,nz), v_west(js:je,nz), t_west(js:je,nz,nt))
    allocate(u_east(js:je,nz), v_east(js:je,nz), t_east(js:je,nz,nt))
    allocate(u_south(is:ie,nz), v_south(is:ie,nz), t_south(is:ie,nz,nt))
    allocate(u_north(is:ie,nz), v_north(is:ie,nz), t_north(is:ie,nz,nt))
    allocate(ptide(is:ie,js:je))
    ptide = 0.0_rp
    have_bulk = .false.
    zeta_west = 0.0_rp; ubar_west = 0.0_rp; vbar_west = 0.0_rp
    zeta_east = 0.0_rp; ubar_east = 0.0_rp; vbar_east = 0.0_rp
    zeta_south = 0.0_rp; ubar_south = 0.0_rp; vbar_south = 0.0_rp
    zeta_north = 0.0_rp; ubar_north = 0.0_rp; vbar_north = 0.0_rp
    u_west = 0.0_rp; v_west = 0.0_rp; t_west = 0.0_rp
    u_east = 0.0_rp; v_east = 0.0_rp; t_east = 0.0_rp
    u_south = 0.0_rp; v_south = 0.0_rp; t_south = 0.0_rp
    u_north = 0.0_rp; v_north = 0.0_rp; t_north = 0.0_rp
    obc_west = .false.; obc_east = .false.; obc_south = .false.; obc_north = .false.
    have_flux = .false.; have_bry = .false.

    zeta = 0.0_rp; ubar = 0.0_rp; vbar = 0.0_rp
    u = 0.0_rp; v = 0.0_rp; t = 0.0_rp
    Hz = 0.0_rp; z_r = 0.0_rp; z_w = 0.0_rp
    FlxU = 0.0_rp; FlxV = 0.0_rp; We = 0.0_rp; Wi = 0.0_rp
    rmask = 1.0_rp; umask = 1.0_rp; vmask = 1.0_rp; pmask = 1.0_rp
    lonr = 0.0_rp; latr = 0.0_rp; angler = 0.0_rp
    swflx = 0.0_rp; tair = 0.0_rp
    Akv = 0.0_rp; Akt = 0.0_rp
    visc2_r = 0.0_rp; visc2_p = 0.0_rp; diff2 = 0.0_rp
    sustr = 0.0_rp; svstr = 0.0_rp; stflx = 0.0_rp; r_D = 0.0_rp
    bvf = 0.0_rp
    hbls = 0.0_rp; hbbl = 0.0_rp; srflx = 0.0_rp
    ghat = 0.0_rp; swr_frac = 0.0_rp
    zob = 0.01_rp; akv_bak = 0.0_rp; akt_bak = 0.0_rp
    rufrc = 0.0_rp; rvfrc = 0.0_rp
    rufrc_bak = 0.0_rp; rvfrc_bak = 0.0_rp
    DU_avg1 = 0.0_rp; DV_avg1 = 0.0_rp
    DU_avg2 = 0.0_rp; DV_avg2 = 0.0_rp
    DU_avg_bak = 0.0_rp; DV_avg_bak = 0.0_rp
    zt_avg1 = 0.0_rp; rho = 0.0_rp
    rho1 = 0.0_rp; qp1 = 0.0_rp
    rhoA = 0.0_rp; rhoS = 0.0_rp
    ru = 0.0_rp; rv = 0.0_rp; P = 0.0_rp
  end subroutine alloc_state

  subroutine free_state()
    if (allocated(zeta)) deallocate(zeta, ubar, vbar, u, v, t)
    if (allocated(Hz)) deallocate(Hz, Hz_u, Hz_v, z_r, z_w)
    if (allocated(FlxU)) deallocate(FlxU, FlxV, We, Wi)
    if (allocated(h)) deallocate(h, hinv, pm, pn, dn_u, dm_v, dm_r, dn_r)
    if (allocated(rmask)) deallocate(rmask, umask, vmask, f, lonr, latr, angler, swflx, tair, &
                                    pmask, dm_p, dn_p, dm_u, dn_v, pmon_u, pnom_v, dndx, dmde, &
                                    Akv, Akt, visc2_r, visc2_p, diff2, &
                                    sustr, svstr, sustr_r, svstr_r, stflx, r_D, bvf, &
                                    hbls, hbbl, srflx, ghat, swr_frac)
    if (allocated(rufrc)) deallocate(rufrc, rvfrc, rufrc_bak, rvfrc_bak)
    if (allocated(DU_avg1)) deallocate(DU_avg1, DV_avg1, DU_avg2, DV_avg2, &
                                       DU_avg_bak, DV_avg_bak, zt_avg1, rho, rho1, qp1, &
                                       rhoA, rhoS, ru, rv, P)
    if (allocated(Drhs)) deallocate(Drhs, DUon, DVom, rubar, rvbar, &
                                    zeta_new, Dnew, zwrk, rzeta, rzeta2, rzetaSA)
    if (allocated(Hz_bak)) deallocate(Hz_bak, Hz_fwd, wrk2a, wrk2b, wrk2c, wrk2d, &
                                      wrk2e, wrk2f, col_fc, col_cf, col_dc, col_wc)
    if (allocated(wrk3a)) deallocate(wrk3a, wrk3b, wrk3c, wrk3d, wrk3e, wrk3f)
    if (allocated(zeta_west)) deallocate(zeta_west, ubar_west, vbar_west, &
         zeta_east, ubar_east, vbar_east, zeta_south, ubar_south, vbar_south, &
         zeta_north, ubar_north, vbar_north, u_west, v_west, t_west, u_east, v_east, t_east, &
         u_south, v_south, t_south, u_north, v_north, t_north)
    if (allocated(ptide)) deallocate(ptide)
    call reset_graphs()
  end subroutine free_state

  subroutine init_ideal_box(H0, dx, f0)
    ! Flat-bottom periodic box for halo / set_depth / omega tests.
    real(rp), intent(in) :: H0, dx, f0
    integer(ip) :: i, j, k, is, ie, js, je

    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    do j = js, je
      do i = is, ie
        h(i, j) = H0
        hinv(i, j) = 1.0_rp / (H0 + 1.0_rp)  ! overwritten once hc is known
        pm(i, j) = 1.0_rp / dx
        pn(i, j) = 1.0_rp / dx
        dm_r(i, j) = dx
        dn_r(i, j) = dx
        dn_u(i, j) = dx
        dm_v(i, j) = dx
        dm_u(i, j) = dx
        dn_v(i, j) = dx
        dm_p(i, j) = dx
        dn_p(i, j) = dx
        pmon_u(i, j) = 1.0_rp
        pnom_v(i, j) = 1.0_rp
        dndx(i, j) = 0.0_rp
        dmde(i, j) = 0.0_rp
        pmask(i, j) = 1.0_rp
        f(i, j) = f0
        rmask(i, j) = 1.0_rp
        umask(i, j) = 1.0_rp
        vmask(i, j) = 1.0_rp
        zeta(i, j, :) = 0.0_rp
      enddo
    enddo
    do k = 1, 3
      t(:, :, :, k, 1) = 15.0_rp
      t(:, :, :, k, 2) = 35.0_rp
    enddo
  end subroutine init_ideal_box

  subroutine ensure_wrk3()
    integer(ip) :: is, ie, js, je, nz
    if (allocated(wrk3a)) return
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    nz = decomp%nz
    ! 0:nz so the pool can also hold w-level fields (lmd Kt/Ks).
    allocate(wrk3a(is:ie,js:je,0:nz), wrk3b(is:ie,js:je,0:nz), wrk3c(is:ie,js:je,0:nz))
    allocate(wrk3d(is:ie,js:je,0:nz), wrk3e(is:ie,js:je,0:nz), wrk3f(is:ie,js:je,0:nz))
    wrk3a = 0.0_rp; wrk3b = 0.0_rp; wrk3c = 0.0_rp
    wrk3d = 0.0_rp; wrk3e = 0.0_rp; wrk3f = 0.0_rp
  end subroutine ensure_wrk3

end module gpu_state
