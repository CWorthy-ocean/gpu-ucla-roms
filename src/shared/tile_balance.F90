module tile_balance
  ! Tensor-product tile cuts that balance a per-cell cost over npx x npy
  ! tiles: one set of x cuts shared by every row of tiles and one set of y
  ! cuts shared by every column of tiles, so the Cartesian halo exchange,
  ! the neighbor logic and the per-tile file offsets are unchanged. The
  ! result is deterministic from the cost field, so every rank computes it.
  !
  ! Method: 2-D prefix sums make any tile's cost four lookups. Seed the cuts
  ! from quantiles of the column / row sums, then alternate: with the y cuts
  ! fixed choose the x cuts that minimize the largest tile cost (contiguous
  ! partition, exact by binary search on the bound with a greedy pass), then
  ! the y cuts with the x cuts fixed, until nothing moves. Every tile is at
  ! least minw cells wide and high.
  use kinds
  implicit none
  private
  public :: balance_cuts, uniform_cuts, strip_cost

contains

  subroutine uniform_cuts(g, np, cut)
    ! Equal widths, remainder spread over the first tiles.
    integer(ip), intent(in)  :: g, np
    integer(ip), intent(out) :: cut(0:np)
    integer(ip) :: k
    cut(0) = 0
    do k = 1, np
      cut(k) = cut(k - 1) + g / np + merge(1_ip, 0_ip, k <= mod(g, np))
    enddo
  end subroutine uniform_cuts

  subroutine balance_cuts(cost, gnx, gny, npx, npy, minw, xcut, ycut)
    integer(ip), intent(in)  :: gnx, gny, npx, npy, minw
    real(rp),    intent(in)  :: cost(gnx, gny)
    integer(ip), intent(out) :: xcut(0:npx), ycut(0:npy)
    real(rp), allocatable :: p(:,:)
    integer(ip) :: i, j, sweep
    integer(ip) :: oldx(0:npx), oldy(0:npy)

    if (npx * minw > gnx .or. npy * minw > gny) &
         error stop 'tile_balance: too many tiles for the grid (minimum tile width)'
    allocate(p(0:gnx, 0:gny))
    p = 0.0_rp
    do j = 1, gny
      do i = 1, gnx
        p(i, j) = p(i - 1, j) + p(i, j - 1) - p(i - 1, j - 1) + max(cost(i, j), 0.0_rp)
      enddo
    enddo
    call seed_cuts(p(1:gnx, gny), gnx, npx, minw, xcut)
    call seed_cuts(p(gnx, 1:gny), gny, npy, minw, ycut)
    do sweep = 1, 20
      oldx = xcut; oldy = ycut
      call best_cuts(p, gnx, gny, npx, npy, minw, ycut, .true.,  xcut)
      call best_cuts(p, gnx, gny, npx, npy, minw, xcut, .false., ycut)
      if (all(xcut == oldx) .and. all(ycut == oldy)) exit
    enddo
    deallocate(p)
  end subroutine balance_cuts

  subroutine seed_cuts(cum, g, np, minw, cut)
    ! Quantiles of a cumulative cost along one axis, then the width floor.
    integer(ip), intent(in)  :: g, np, minw
    real(rp),    intent(in)  :: cum(g)
    integer(ip), intent(out) :: cut(0:np)
    integer(ip) :: k, i
    real(rp) :: target
    cut(0) = 0
    cut(np) = g
    i = 1
    do k = 1, np - 1
      target = cum(g) * real(k, rp) / real(np, rp)
      do while (i < g .and. cum(i) < target)
        i = i + 1
      enddo
      cut(k) = i
    enddo
    call enforce_width(g, np, minw, cut)
  end subroutine seed_cuts

  subroutine enforce_width(g, np, minw, cut)
    integer(ip), intent(in)    :: g, np, minw
    integer(ip), intent(inout) :: cut(0:np)
    integer(ip) :: k
    do k = 1, np - 1
      cut(k) = max(cut(k), cut(k - 1) + minw)
    enddo
    do k = np - 1, 1, -1
      cut(k) = min(cut(k), cut(k + 1) - minw)
    enddo
  end subroutine enforce_width

  function strip_cost(p, gnx, gny, a, b, c, d) result(t)
    ! Cost of cells a+1:b (x) by c+1:d (y) from the prefix sum.
    integer(ip), intent(in) :: gnx, gny, a, b, c, d
    real(rp),    intent(in) :: p(0:gnx, 0:gny)
    real(rp) :: t
    t = p(b, d) - p(a, d) - p(b, c) + p(a, c)
  end function strip_cost

  subroutine best_cuts(p, gnx, gny, npx, npy, minw, fixed, along_x, cut)
    ! With the cuts of the other axis fixed, choose this axis' cuts so that
    ! the largest tile cost is minimal: binary search on the bound; a bound
    ! is feasible when a greedy pass (each strip as wide as the bound allows,
    ! leaving room for the remaining strips) reaches the end.
    integer(ip), intent(in)  :: gnx, gny, npx, npy, minw
    real(rp),    intent(in)  :: p(0:gnx, 0:gny)
    logical,     intent(in)  :: along_x
    integer(ip), intent(in)  :: fixed(0:)
    integer(ip), intent(out) :: cut(0:)
    integer(ip) :: g, np, nf, it
    real(rp) :: lo, hi, mid, total
    integer(ip), allocatable :: trial(:)
    if (along_x) then
      g = gnx; np = npx; nf = npy
    else
      g = gny; np = npy; nf = npx
    endif
    allocate(trial(0:np))
    total = p(gnx, gny)
    lo = 0.0_rp
    hi = total
    do it = 1, 64
      mid = 0.5_rp * (lo + hi)
      if (feasible(mid, trial)) then
        hi = mid
      else
        lo = mid
      endif
      if (hi - lo <= 1.0e-9_rp * max(total, 1.0_rp)) exit
    enddo
    if (.not. feasible(hi, trial)) then
      ! only the width floor can make a bound infeasible: fall back to it
      call seed_cuts(merge_cum(), g, np, minw, trial)
    endif
    cut(0:np) = trial
    deallocate(trial)

  contains

    function merge_cum() result(c)
      real(rp) :: c(g)
      integer(ip) :: i
      do i = 1, g
        if (along_x) then
          c(i) = p(i, gny)
        else
          c(i) = p(gnx, i)
        endif
      enddo
    end function merge_cum

    function tile_max(a, b) result(t)
      ! Largest cost over the fixed strips of the other axis for cells a+1:b.
      integer(ip), intent(in) :: a, b
      real(rp) :: t
      integer(ip) :: s
      t = 0.0_rp
      do s = 1, nf
        if (along_x) then
          t = max(t, strip_cost(p, gnx, gny, a, b, fixed(s - 1), fixed(s)))
        else
          t = max(t, strip_cost(p, gnx, gny, fixed(s - 1), fixed(s), a, b))
        endif
      enddo
    end function tile_max

    logical function feasible(bound, c)
      real(rp),    intent(in)  :: bound
      integer(ip), intent(out) :: c(0:np)
      integer(ip) :: k, a, b, bmax
      c(0) = 0
      a = 0
      do k = 1, np - 1
        bmax = g - (np - k) * minw
        b = a + minw
        if (b > bmax) then
          feasible = .false.; return
        endif
        if (tile_max(a, b) > bound) then
          feasible = .false.; return
        endif
        do while (b < bmax)
          if (tile_max(a, b + 1) > bound) exit
          b = b + 1
        enddo
        c(k) = b
        a = b
      enddo
      c(np) = g
      feasible = (tile_max(a, g) <= bound) .and. (g - a >= minw)
    end function feasible
  end subroutine best_cuts

end module tile_balance
