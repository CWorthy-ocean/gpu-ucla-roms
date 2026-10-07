module decomp_mod
  ! One rank <-> one GPU <-> one horizontal tile, even when npx=npy=1.
  ! Kernels use local i=1:nx, j=1:ny. Physical OBCs test edge_*.
  use kinds
  implicit none
  private

  type, public :: decomp_t
    integer(ip) :: nx = 0, ny = 0, nz = 0
    integer(ip) :: halo = 2
    integer(ip) :: is = 0, ie = 0, js = 0, je = 0   ! allocated bounds
    integer(ip) :: gnx = 0, gny = 0                 ! global interior
    integer(ip) :: i0 = 1, j0 = 1                   ! global index of local (1,1)
    integer(ip) :: inode = 0, jnode = 0
    integer(ip) :: npx = 1, npy = 1
    integer(ip) :: rank = 0
    logical     :: edge_w = .true., edge_e = .true.
    logical     :: edge_s = .true., edge_n = .true.
    integer(ip) :: nbr_w = NBR_NONE, nbr_e = NBR_NONE
    integer(ip) :: nbr_s = NBR_NONE, nbr_n = NBR_NONE
    logical     :: periodic_x = .false., periodic_y = .false.
  end type decomp_t

  type(decomp_t), public, save :: decomp

  public :: init_decomp
  public :: in_interior
  public :: in_u_interior
  public :: in_v_interior

contains

  subroutine init_decomp(gnx, gny, nz, halo, npx, npy, rank, &
                         periodic_x, periodic_y, xcut, ycut)
    ! xcut(0:npx), ycut(0:npy): global interior indices of the tile cuts
    ! (tile k covers xcut(k-1)+1 : xcut(k)); absent = equal widths (the
    ! remainder spread over the first tiles).
    integer(ip), intent(in) :: gnx, gny, nz, halo, npx, npy, rank
    logical,     intent(in) :: periodic_x, periodic_y
    integer(ip), intent(in), optional :: xcut(0:npx), ycut(0:npy)
    integer(ip) :: inode, jnode, nx, ny, i0, j0, k

    if (halo < 2) error stop 'init_decomp: halo must be >= 2 (U3/UP3)'

    inode = mod(rank, npx)
    jnode = rank / npx
    if (present(xcut)) then
      nx = xcut(inode + 1) - xcut(inode)
      i0 = xcut(inode) + 1
    else
      nx = gnx / npx + merge(1_ip, 0_ip, inode + 1 <= mod(gnx, npx))
      i0 = 1
      do k = 0, inode - 1
        i0 = i0 + gnx / npx + merge(1_ip, 0_ip, k + 1 <= mod(gnx, npx))
      enddo
    endif
    if (present(ycut)) then
      ny = ycut(jnode + 1) - ycut(jnode)
      j0 = ycut(jnode) + 1
    else
      ny = gny / npy + merge(1_ip, 0_ip, jnode + 1 <= mod(gny, npy))
      j0 = 1
      do k = 0, jnode - 1
        j0 = j0 + gny / npy + merge(1_ip, 0_ip, k + 1 <= mod(gny, npy))
      enddo
    endif
    if (nx < 2 * halo + 2 .or. ny < 2 * halo + 2) error stop 'init_decomp: tile too small for the halo'

    decomp%nx = nx
    decomp%ny = ny
    decomp%nz = nz
    decomp%halo = halo
    ! Extra +1 so u lives at i=1:nx+1 and v at j=1:ny+1 on the same allocation.
    decomp%is = 1 - halo
    decomp%ie = nx + 1 + halo
    decomp%js = 1 - halo
    decomp%je = ny + 1 + halo
    decomp%gnx = gnx
    decomp%gny = gny
    decomp%i0 = i0
    decomp%j0 = j0
    decomp%inode = inode
    decomp%jnode = jnode
    decomp%npx = npx
    decomp%npy = npy
    decomp%rank = rank
    decomp%periodic_x = periodic_x
    decomp%periodic_y = periodic_y

    decomp%edge_w = (inode == 0) .and. (.not. periodic_x)
    decomp%edge_e = (inode == npx - 1) .and. (.not. periodic_x)
    decomp%edge_s = (jnode == 0) .and. (.not. periodic_y)
    decomp%edge_n = (jnode == npy - 1) .and. (.not. periodic_y)

    decomp%nbr_w = neighbor(inode - 1, jnode, npx, npy, periodic_x, .false., rank)
    decomp%nbr_e = neighbor(inode + 1, jnode, npx, npy, periodic_x, .false., rank)
    decomp%nbr_s = neighbor(inode, jnode - 1, npx, npy, .false., periodic_y, rank)
    decomp%nbr_n = neighbor(inode, jnode + 1, npx, npy, .false., periodic_y, rank)
  end subroutine init_decomp

  integer(ip) function neighbor(in, jn, npx, npy, per_x, per_y, rank) result(nbr)
    integer(ip), intent(in) :: in, jn, npx, npy, rank
    logical,     intent(in) :: per_x, per_y
    integer(ip) :: i, j

    i = in
    j = jn
    if (per_x) then
      if (i < 0) i = npx - 1
      if (i >= npx) i = 0
    endif
    if (per_y) then
      if (j < 0) j = npy - 1
      if (j >= npy) j = 0
    endif
    if (i < 0 .or. i >= npx .or. j < 0 .or. j >= npy) then
      nbr = NBR_NONE
      return
    endif
    nbr = j * npx + i
    if (nbr == rank) nbr = NBR_SELF
  end function neighbor

  logical function in_interior(i, j)
    integer(ip), intent(in) :: i, j
    in_interior = (i >= 1 .and. i <= decomp%nx .and. &
                   j >= 1 .and. j <= decomp%ny)
  end function in_interior

  logical function in_u_interior(i, j)
    integer(ip), intent(in) :: i, j
    in_u_interior = (i >= 1 .and. i <= decomp%nx + 1 .and. &
                     j >= 1 .and. j <= decomp%ny)
  end function in_u_interior

  logical function in_v_interior(i, j)
    integer(ip), intent(in) :: i, j
    in_v_interior = (i >= 1 .and. i <= decomp%nx .and. &
                     j >= 1 .and. j <= decomp%ny + 1)
  end function in_v_interior

end module decomp_mod
