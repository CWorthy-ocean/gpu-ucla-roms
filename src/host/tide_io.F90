#include "mem_attr.inc"
#ifdef USE_CUDA
! Rounded multiply: keeps a*b from contracting into an FMA on the device,
! so the tide sums match the (no-FMA) host arithmetic bit for bit.
#define TMUL(a, b) __dmul_rn(a, b)
#else
#define TMUL(a, b) ((a) * (b))
#endif
module tide_io
  ! File tides: omega + ssh/u/v/pot Re/Im. Adds to climatological bry
  ! and fills ptide for prsgrd. No analytical constituents.
  use kinds
  use decomp_mod
  use gpu_state
  use time_state
  use frc_io, only: case_frcfiles, case_nfrc
  use nc_io
  use arch_mod, only: sync_arch
#ifdef USE_CUDA
  use cudadevice, only: __dmul_rn
#endif
  implicit none
  private

  integer :: ntides = 0
  logical :: ana_tides = .false.
  logical :: want_pot = .false.
  logical :: want_bry = .false.
  logical :: tides_ready = .false.
  real(rp), allocatable :: ftide(:)
  ! Managed: set_tides sums them on the device every step.
  real(rp), WORK_ATTR allocatable :: ztide_r(:,:,:), ztide_i(:,:,:)
  real(rp), WORK_ATTR allocatable :: utide_r(:,:,:), utide_i(:,:,:)
  real(rp), WORK_ATTR allocatable :: vtide_r(:,:,:), vtide_i(:,:,:)
  real(rp), WORK_ATTR allocatable :: ptide_r(:,:,:), ptide_i(:,:,:)

  public :: read_tide_namelist
  public :: init_tides
  public :: set_tides
  public :: write_toy_tides

contains

  subroutine read_tide_namelist(unit)
    integer, intent(in) :: unit
    integer :: ios
    integer :: ntides
    logical :: pot_tides, bry_tides, ana_tides
    namelist /tides_settings/ ntides, pot_tides, bry_tides, ana_tides
    namelist /tidal_frc_settings/ ntides, pot_tides, bry_tides, ana_tides
    ntides = 0
    pot_tides = .false.
    bry_tides = .false.
    ana_tides = .false.
    rewind(unit)
    read(unit, nml=tides_settings, iostat=ios)
    if (ios /= 0) then
      rewind(unit)
      read(unit, nml=tidal_frc_settings, iostat=ios)
    endif
    call apply_tide_flags(ntides, pot_tides, bry_tides, ana_tides)
  end subroutine read_tide_namelist

  subroutine apply_tide_flags(n, pot, bry, ana)
    integer, intent(in) :: n
    logical, intent(in) :: pot, bry, ana
    ntides = n
    ana_tides = ana
    want_pot = pot
    want_bry = bry
    pot_tides = pot
    bry_tides = bry
  end subroutine apply_tide_flags

  subroutine init_tides()
    integer :: ncid, n1, n2, n3, nrec, it, is, ie, js, je
    character(len=512) :: path
    tides_ready = .false.
    pot_tides = want_pot
    bry_tides = want_bry
    if (.not. pot_tides .and. .not. bry_tides) return
    if (ana_tides) then
      write(*,*) 'ana_tides is not implemented'
      error stop 'init_tides'
    endif
    path = find_tide_file()
    if (len_trim(path) == 0) then
      write(*,*) 'tide file with omega not found'
      error stop 'init_tides'
    endif
    is = decomp%is; ie = decomp%ie
    js = decomp%js; je = decomp%je
    call nc_open_read(path, ncid)
    nrec = nc_var_nrecs(ncid, 'omega')
    if (ntides < 1) ntides = nrec
    ntides = min(ntides, max(nrec, 1))
    allocate(ftide(ntides))
    call nc_get_1d(ncid, 'omega', ftide)
    if (bry_tides) then
      allocate(ztide_r(is:ie,js:je,ntides), ztide_i(is:ie,js:je,ntides))
      allocate(utide_r(is:ie,js:je,ntides), utide_i(is:ie,js:je,ntides))
      allocate(vtide_r(is:ie,js:je,ntides), vtide_i(is:ie,js:je,ntides))
      ztide_r = 0.0_rp; ztide_i = 0.0_rp
      utide_r = 0.0_rp; utide_i = 0.0_rp
      vtide_r = 0.0_rp; vtide_i = 0.0_rp
    endif
    if (pot_tides) then
      allocate(ptide_r(is:ie,js:je,ntides), ptide_i(is:ie,js:je,ntides))
      ptide_r = 0.0_rp; ptide_i = 0.0_rp
    endif
    do it = 1, ntides
      if (bry_tides) then
        call nc_var_shape(ncid, 'ssh_Re', n1, n2, n3, nrec)
        call load2(ncid, 'ssh_Re', n1, n2, it, ztide_r(:,:,it), 'r')
        call load2(ncid, 'ssh_Im', n1, n2, it, ztide_i(:,:,it), 'r')
        call nc_var_shape(ncid, 'u_Re', n1, n2, n3, nrec)
        call load2(ncid, 'u_Re', n1, n2, it, utide_r(:,:,it), 'u')
        call load2(ncid, 'u_Im', n1, n2, it, utide_i(:,:,it), 'u')
        call nc_var_shape(ncid, 'v_Re', n1, n2, n3, nrec)
        call load2(ncid, 'v_Re', n1, n2, it, vtide_r(:,:,it), 'v')
        call load2(ncid, 'v_Im', n1, n2, it, vtide_i(:,:,it), 'v')
      endif
      if (pot_tides) then
        call nc_var_shape(ncid, 'pot_Re', n1, n2, n3, nrec)
        call load2(ncid, 'pot_Re', n1, n2, it, ptide_r(:,:,it), 'r')
        call load2(ncid, 'pot_Im', n1, n2, it, ptide_i(:,:,it), 'r')
      endif
    enddo
    call nc_close(ncid)
    if (bry_tides) then
      obc_west = .true.; obc_east = .true.
      obc_south = .true.; obc_north = .true.
    endif
    tides_ready = .true.
    write(*,'(a,i0,a,2l2)') 'tides: n = ', ntides, '  pot,bry =', pot_tides, bry_tides
  end subroutine init_tides

  subroutine set_tides()
    ! One small device kernel per constituent and target (c, s are
    ! kernel arguments), in the same order as the host loops it replaces.
    integer(ip) :: it, i, j, nx, ny, ip, is, ie, js, je, n
    real(rp) :: omT, c, s, z0, v0
    logical :: want_dump
    if (.not. tides_ready) return
    nx = decomp%nx; ny = decomp%ny
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    ip = iprb_pt(1)
    z0 = 0.0_rp; v0 = 0.0_rp
    want_dump = (probe_on .or. iic == ntstart) .and. ip >= is .and. ip <= ie
    if (want_dump) then
      call sync_arch()
      z0 = zeta_north(ip)
      v0 = vbar_north(ip)
    endif
    if (pot_tides) then
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = js, je
        do i = is, ie
          ptide(i, j) = 0.0_rp
        enddo
      enddo
    endif
    do it = 1, ntides
      omT = ftide(it) * (time + 0.5_rp * dt)
      c = cos(omT)
      s = sin(omT)
      n = it
      if (bry_tides) then
#ifdef USE_CUDA
        !$cuf kernel do(1) <<<*,*>>>
#endif
        do j = 0, ny + 1
          zeta_west(j) = zeta_west(j) + TMUL(ztide_r(0, j, n), c) - TMUL(ztide_i(0, j, n), s)
          zeta_east(j) = zeta_east(j) + TMUL(ztide_r(nx + 1, j, n), c) - TMUL(ztide_i(nx + 1, j, n), s)
          ubar_west(j) = ubar_west(j) + TMUL(utide_r(1, j, n), c) - TMUL(utide_i(1, j, n), s)
          ubar_east(j) = ubar_east(j) + TMUL(utide_r(nx + 1, j, n), c) - TMUL(utide_i(nx + 1, j, n), s)
          vbar_west(j) = vbar_west(j) + TMUL(vtide_r(0, j, n), c) - TMUL(vtide_i(0, j, n), s)
          vbar_east(j) = vbar_east(j) + TMUL(vtide_r(nx + 1, j, n), c) - TMUL(vtide_i(nx + 1, j, n), s)
        enddo
#ifdef USE_CUDA
        !$cuf kernel do(1) <<<*,*>>>
#endif
        do i = 0, nx + 1
          zeta_south(i) = zeta_south(i) + TMUL(ztide_r(i, 0, n), c) - TMUL(ztide_i(i, 0, n), s)
          zeta_north(i) = zeta_north(i) + TMUL(ztide_r(i, ny + 1, n), c) - TMUL(ztide_i(i, ny + 1, n), s)
          ubar_south(i) = ubar_south(i) + TMUL(utide_r(i, 0, n), c) - TMUL(utide_i(i, 0, n), s)
          ubar_north(i) = ubar_north(i) + TMUL(utide_r(i, ny + 1, n), c) - TMUL(utide_i(i, ny + 1, n), s)
          vbar_south(i) = vbar_south(i) + TMUL(vtide_r(i, 1, n), c) - TMUL(vtide_i(i, 1, n), s)
          vbar_north(i) = vbar_north(i) + TMUL(vtide_r(i, ny + 1, n), c) - TMUL(vtide_i(i, ny + 1, n), s)
        enddo
      endif
      if (pot_tides) then
#ifdef USE_CUDA
        !$cuf kernel do(2) <<<*,*>>>
#endif
        do j = js, je
          do i = is, ie
            ! CPU tides.F90 writes ptide unmasked; prsgrd uses it on land.
            ptide(i, j) = ptide(i, j) + TMUL(ptide_r(i, j, n), c) - TMUL(ptide_i(i, j, n), s)
          enddo
        enddo
      endif
    enddo
    if (want_dump) call sync_arch()
    call dump_tide_north(z0, v0)
  end subroutine set_tides

  subroutine dump_tide_north(z0, v0)
    real(rp), intent(in) :: z0, v0
    integer(ip) :: i, ny
    real(rp) :: omT, c, s, zr, zi, vr, vi

    if (.not. probe_on .and. iic /= ntstart) return
    i = iprb_pt(1)
    ny = decomp%ny
    if (i < decomp%is .or. i > decomp%ie) return
    omT = 0.0_rp; c = 1.0_rp; s = 0.0_rp
    zr = 0.0_rp; zi = 0.0_rp; vr = 0.0_rp; vi = 0.0_rp
    if (ntides >= 1) then
      omT = ftide(1) * (time + 0.5_rp * dt)
      c = cos(omT)
      s = sin(omT)
      if (allocated(ztide_r)) then
        zr = ztide_r(i, ny + 1, 1)
        zi = ztide_i(i, ny + 1, 1)
        vr = vtide_r(i, ny + 1, 1)
        vi = vtide_i(i, ny + 1, 1)
      endif
    endif
    write(*,'(a,i1,a,i8,a,i4,a,14es26.17)') 'diag: tideN P', 1, '  iic=', iic, &
         '  ntides=', ntides, '  time,dt,omT1,cos1,sin1,zRe,zIm,vRe,vIm,dZ,dV,zBry,vBry=', &
         time, dt, omT, c, s, zr, zi, vr, vi, &
         zeta_north(i) - z0, vbar_north(i) - v0, zeta_north(i), vbar_north(i)
    flush(6)
  end subroutine dump_tide_north

  subroutine load2(ncid, vname, n1, n2, rec, dest, gtype)
    integer,          intent(in)    :: ncid, n1, n2, rec
    character(len=*), intent(in)    :: vname, gtype
    real(rp),         intent(inout) :: dest(decomp%is:, decomp%js:)
    real(rp), allocatable :: tmp(:,:)
    integer :: i0, j0
    allocate(tmp(n1, n2))
    call nc_read_2d(ncid, vname, tmp, n1, n2, rec)
    where (is_nan(tmp)) tmp = 0.0_rp
    if (gtype == 'r' .and. n1 == decomp%nx + 2 .and. n2 == decomp%ny + 2) then
      dest(0:decomp%nx+1, 0:decomp%ny+1) = tmp
    elseif (gtype == 'u' .and. n1 == decomp%nx + 1 .and. n2 == decomp%ny + 2) then
      dest(1:decomp%nx+1, 0:decomp%ny+1) = tmp
    elseif (gtype == 'u' .and. n1 == decomp%nx + 1 .and. n2 == decomp%ny) then
      dest(1:decomp%nx+1, 1:decomp%ny) = tmp
    elseif (gtype == 'v' .and. n1 == decomp%nx + 2 .and. n2 == decomp%ny + 1) then
      dest(0:decomp%nx+1, 1:decomp%ny+1) = tmp
    elseif (gtype == 'v' .and. n1 == decomp%nx .and. n2 == decomp%ny + 1) then
      dest(1:decomp%nx, 1:decomp%ny+1) = tmp
    elseif (gtype == 'r' .and. n1 == decomp%nx .and. n2 == decomp%ny) then
      dest(1:decomp%nx, 1:decomp%ny) = tmp
    else
      i0 = 0; j0 = 0
      if (gtype == 'u') i0 = 1
      if (gtype == 'v') j0 = 1
      dest(i0:i0+min(n1, size(dest,1))-1, j0:j0+min(n2, size(dest,2))-1) = &
           tmp(1:min(n1, size(dest,1)), 1:min(n2, size(dest,2)))
    endif
    deallocate(tmp)
  end subroutine load2

  function find_tide_file() result(path)
    character(len=512) :: path
    integer :: i, ncid
    path = ''
    do i = 1, case_nfrc
      call nc_open_read(case_frcfiles(i), ncid)
      if (nc_has_var(ncid, 'omega')) then
        path = case_frcfiles(i)
        call nc_close(ncid)
        return
      endif
      call nc_close(ncid)
    enddo
  end function find_tide_file

  subroutine write_toy_tides(path, nxi, neta, nt)
    character(len=*), intent(in) :: path
    integer,          intent(in) :: nxi, neta, nt
    integer :: ncid, d_xi, d_eta, d_xiu, d_etav, d_t, id
    real(rp), allocatable :: ar(:,:,:), au(:,:,:), av(:,:,:)
    allocate(ar(nxi, neta, nt), au(nxi - 1, neta, nt), av(nxi, neta - 1, nt))
    ar = 0.1_rp
    au = 0.02_rp
    av = 0.0_rp
    call nc_create(trim(path), ncid)
    d_t = nc_def_dim(ncid, 'ntides', nt)
    d_xi = nc_def_dim(ncid, 'xi_rho', nxi)
    d_eta = nc_def_dim(ncid, 'eta_rho', neta)
    d_xiu = nc_def_dim(ncid, 'xi_u', nxi - 1)
    d_etav = nc_def_dim(ncid, 'eta_v', neta - 1)
    id = nc_def_var(ncid, 'omega', (/d_t/))
    id = nc_def_var(ncid, 'ssh_Re', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'ssh_Im', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'pot_Re', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'pot_Im', (/d_xi, d_eta, d_t/))
    id = nc_def_var(ncid, 'u_Re', (/d_xiu, d_eta, d_t/))
    id = nc_def_var(ncid, 'u_Im', (/d_xiu, d_eta, d_t/))
    id = nc_def_var(ncid, 'v_Re', (/d_xi, d_etav, d_t/))
    id = nc_def_var(ncid, 'v_Im', (/d_xi, d_etav, d_t/))
    call nc_enddef(ncid)
    call nc_put_1d(ncid, 'omega', (/ 2.0_rp * 3.141592653589793_rp / 86400.0_rp /))
    call nc_put_3d(ncid, 'ssh_Re', ar)
    ar = 0.0_rp
    call nc_put_3d(ncid, 'ssh_Im', ar)
    ar = 0.01_rp
    call nc_put_3d(ncid, 'pot_Re', ar)
    ar = 0.0_rp
    call nc_put_3d(ncid, 'pot_Im', ar)
    call nc_put_3d(ncid, 'u_Re', au)
    au = 0.0_rp
    call nc_put_3d(ncid, 'u_Im', au)
    call nc_put_3d(ncid, 'v_Re', av)
    call nc_put_3d(ncid, 'v_Im', av)
    call nc_close(ncid)
    deallocate(ar, au, av)
  end subroutine write_toy_tides

end module tide_io
