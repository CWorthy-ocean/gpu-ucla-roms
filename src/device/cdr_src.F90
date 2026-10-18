#include "mem_attr.inc"
#include "cuf_launch.inc"
module cdr_src_mod
  ! CDR releases (UCLA cdr_frc, parameterized releases with cdr_trcflx) on
  ! the device. A release spreads its tracer flux [mmol/s] over cells and
  ! levels with weights w(k) (a Gaussian in the horizontal and the vertical,
  ! normalized so the weights of one release sum to 1 over the domain);
  ! cdr_io builds them on the host. As the CPU (step3d_t, per tracer after
  ! the horizontal advection), the corrector adds to the thickness-weighted
  ! tracer of slot nnew
  !   t += dt*pm*pn*w(k)*flux(release, tracer)
  ! for every release of the cell, in release order, with the flux at the
  ! corrector's forcing time (n+1/2): c1*f1 + c2*f2 of two records, or the
  ! earlier record without time interpolation (c1 = 1, c2 = 0).
  ! The CPU adds to every tracer at every level. Where the flux of a tracer
  ! is 0 in both records, or the weight of a level is 0 in every profile of
  ! the cell, its addition is an exact zero, so those are skipped: the
  ! kernel visits only the active tracers (nonzero flux in either record)
  ! and each cell's levels from its first to its last nonzero weight.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  implicit none
  private
  public :: cdr_on, cdr_set_lists, cdr_set_record, cdr_copy_upper, cdr_set_weights, cdr_apply_tp

  integer, parameter :: CDR_NKB = 4                    ! chunks of levels per (cell, tracer)
  logical :: cdr_on = .false.
  integer(ip) :: ncc = 0, ncq = 0, nrel = 0, ntrf = 0   ! cells, profiles, releases, passive tracers
  integer(ip) :: nact = 0                              ! active tracers
  real(rp) :: w_cor(2) = (/1.0_rp, 0.0_rp/)
  real(rp), allocatable :: h_flx(:,:,:)                ! host copy of cd_flx

  ! Cells of this tile with a release: local (i, j), passive column, the
  ! levels cc_k0 .. cc_k1 with a nonzero weight, and their profiles
  ! cq_p0(c) .. cq_p0(c+1)-1 (release order): release cq_rel and weights
  ! cq_w(k, q). cd_flx(itp, release, slot): two flux records of each
  ! passive tracer, slot 1 the earlier. ca_itr(1:nact): the active tracers.
#ifdef USE_CUDA
  integer(ip), device, allocatable :: cc_i(:), cc_j(:), cc_col(:), cc_k0(:), cc_k1(:), cc_p0(:), cq_rel(:)
  integer(ip), device, allocatable :: ca_itr(:)
  real(rp),    device, allocatable :: cq_w(:,:), cd_flx(:,:,:)
#else
  integer(ip), allocatable :: cc_i(:), cc_j(:), cc_col(:), cc_k0(:), cc_k1(:), cc_p0(:), cq_rel(:)
  integer(ip), allocatable :: ca_itr(:)
  real(rp),    allocatable :: cq_w(:,:), cd_flx(:,:,:)
#endif

contains

  subroutine cdr_set_lists(nc, ci, cj, ccol, cp0, nq, qrel, qw, nr, nt_p)
    ! Copies the host-built cells and profiles (cdr_io) to the device and
    ! allocates the flux records (filled by cdr_set_record).
    integer(ip), intent(in) :: nc, nq, nr, nt_p
    integer(ip), intent(in) :: ci(:), cj(:), ccol(:), cp0(:), qrel(:)
    real(rp),    intent(in) :: qw(:,:)
    integer(ip), allocatable :: k0(:), k1(:)
    integer(ip) :: c, q, k, nz
    ncc = nc; ncq = nq; nrel = nr; ntrf = nt_p; nz = decomp%nz
    allocate(cc_i(max(nc, 1)), cc_j(max(nc, 1)), cc_col(max(nc, 1)), cc_p0(nc + 1))
    allocate(cc_k0(max(nc, 1)), cc_k1(max(nc, 1)))
    allocate(cq_rel(max(nq, 1)), cq_w(nz, max(nq, 1)))
    allocate(cd_flx(max(nt_p, 1), max(nr, 1), 2), h_flx(max(nt_p, 1), max(nr, 1), 2))
    allocate(ca_itr(max(nt_p, 1)))
    cc_i = 0; cc_j = 0; cc_col = 0; cc_k0 = 1; cc_k1 = 0; cq_rel = 1; cq_w = 0.0_rp
    cd_flx = 0.0_rp; h_flx = 0.0_rp; ca_itr = 1; nact = 0
    cc_p0 = cp0(1:nc + 1)
    if (nc > 0) then
      cc_i = ci(1:nc); cc_j = cj(1:nc); cc_col = ccol(1:nc)
      allocate(k0(nc), k1(nc))
      k0 = nz + 1; k1 = 0
      do c = 1, nc
        do q = cp0(c), cp0(c + 1) - 1
          do k = 1, nz
            if (qw(k, q) /= 0.0_rp) then
              k0(c) = min(k0(c), k); k1(c) = max(k1(c), k)
            endif
          enddo
        enddo
      enddo
      cc_k0 = k0; cc_k1 = k1
      deallocate(k0, k1)
    endif
    if (nq > 0) then
      cq_rel = qrel(1:nq); cq_w = qw(:, 1:nq)
    endif
    cdr_on = .true.
  end subroutine cdr_set_lists

  subroutine set_active()
    ! The tracers with a nonzero flux in either record, from the host copy.
    integer(ip) :: itr, act(max(ntrf, 1))
    nact = 0
    do itr = 1, ntrf
      if (any(h_flx(itr, 1:nrel, 1:2) /= 0.0_rp)) then
        nact = nact + 1
        act(nact) = itr
      endif
    enddo
    if (nact > 0) ca_itr(1:nact) = act(1:nact)
  end subroutine set_active

  subroutine cdr_set_record(slot, flx)
    ! One flux record (passive tracer, release) into slot 1 or 2. The device
    ! is drained first: kernels already queued may still read the slot.
    integer,  intent(in) :: slot
    real(rp), intent(in) :: flx(:,:)
    call sync_arch()
    h_flx(1:ntrf, 1:nrel, slot) = flx(1:ntrf, 1:nrel)
    cd_flx(1:ntrf, 1:nrel, slot) = flx(1:ntrf, 1:nrel)
    call set_active()
    call sync_arch()
  end subroutine cdr_set_record

  subroutine cdr_copy_upper()
    ! The bracket moved on by one record: the upper slot becomes the lower.
    call sync_arch()
    h_flx(:, :, 1) = h_flx(:, :, 2)
    cd_flx(:, :, 1) = cd_flx(:, :, 2)
    call set_active()
    call sync_arch()
  end subroutine cdr_copy_upper

  subroutine cdr_set_weights(c1, c2)
    ! Time weights of the two records at the corrector's forcing time.
    real(rp), intent(in) :: c1, c2
    w_cor(1) = c1; w_cor(2) = c2
  end subroutine cdr_set_weights

  subroutine cdr_apply_tp(tp, ntr, n_new, dtp)
    ! The releases into the passive tracers of slot n_new (thickness-weighted
    ! after the corrector's horizontal pass). One thread per (active tracer
    ! and cell, chunk of levels), the (tracer, cell) pairs across the warp.
    ! At each level the cell's releases are added in release order, as the
    ! CPU's cidx loop does, each as t + ((dt*pm*pn)*w)*flux; the fluxes of
    ! the first two releases are formed once per thread.
    integer(ip), intent(in) :: ntr
    real(pp), WORK_ATTR intent(inout) :: tp(ntr, 0:ntpcol, decomp%nz, 3)
    integer(ip), intent(in) :: n_new
    real(rp),    intent(in) :: dtp
    integer(ip) :: c, k, i, j, itr, q, a, m, nm, na, nz, nn, col, kb, kch, nkb, k0, k1, p0, p1, r
    real(rp) :: c1, c2, dtl, cff, val, f, f1, f2
    if (.not. cdr_on .or. ncc == 0 .or. nact == 0 .or. ntr <= 0) return
    na = nact; nm = nact * ncc; nz = decomp%nz; nn = n_new; dtl = dtp
    kch = (nz + CDR_NKB - 1) / CDR_NKB; nkb = (nz + kch - 1) / kch
    c1 = w_cor(1); c2 = w_cor(2)
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*, (32, 4, 1), 0, work_stream>>>
#endif
    do kb = 1, nkb
      do m = 1, nm
        a = mod(m - 1, na) + 1
        c = (m - 1) / na + 1
        col = cc_col(c)
        if (col == 0) cycle
        k0 = max((kb - 1) * kch + 1, cc_k0(c))
        k1 = min(kb * kch, nz, cc_k1(c))
        if (k0 > k1) cycle
        itr = ca_itr(a)
        i = cc_i(c); j = cc_j(c)
        cff = dtl * pm(i, j) * pn(i, j)
        p0 = cc_p0(c); p1 = cc_p0(c + 1) - 1
        r = cq_rel(p0)
        f1 = c1 * cd_flx(itr, r, 1) + c2 * cd_flx(itr, r, 2)
        f2 = 0.0_rp
        if (p1 > p0) then
          r = cq_rel(p0 + 1)
          f2 = c1 * cd_flx(itr, r, 1) + c2 * cd_flx(itr, r, 2)
        endif
        do k = k0, k1
          val = real(tp(itr, col, k, nn), rp)
          val = val + cff * cq_w(k, p0) * f1
          if (p1 > p0) val = val + cff * cq_w(k, p0 + 1) * f2
          do q = p0 + 2, p1
            r = cq_rel(q)
            f = c1 * cd_flx(itr, r, 1) + c2 * cd_flx(itr, r, 2)
            val = val + cff * cq_w(k, q) * f
          enddo
          tp(itr, col, k, nn) = real(val, pp)
        enddo
      enddo
    enddo
  end subroutine cdr_apply_tp

end module cdr_src_mod
