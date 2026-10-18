module cdr_io
  ! CDR releases (UCLA cdr_frc) on the host: the CDR_FRC_SETTINGS namelist,
  ! the release cells and vertical profiles, the flux records and their time
  ! weights. Parameterized releases with tracer fluxes (cdr_trcflx) only so
  ! far; volume releases (cdr_volume), depth profiles and 3D forcing stop at
  ! start-up (read_cdr_namelist).
  !
  ! Release r (cdr_lon, cdr_lat, cdr_dep, cdr_hsc, cdr_vsc) as the CPU
  ! (find_release_locations, create_cdr_vertical_structure):
  ! - the haversine distance from every interior cell, land at 1e8 m with
  !   cdr_relocate_to_wet_pts; the nearest cell is the first minimum;
  ! - cdr_hsc = 0: the nearest cell with weight 1; otherwise the wet cells
  !   with frac = exp(-(dist/hsc)^2) > 1e-3 (the nearest cell with weight 1
  !   if there are none);
  ! - in the vertical, on the rest-state grid (z_r0, Hz0 with zeta = 0):
  !   cdr_vsc = 0, all at the level nearest to -cdr_dep; otherwise
  !   frac*exp(-((z_r0+dep)/vsc)^2)*Hz0/(vsc*sqrt(pi));
  ! - normalized so the weights of the release sum to 1 over the domain.
  ! The CPU finds the cells and sums the weights across its MPI ranks; here
  ! every rank does it on the global grid in one fixed order, so the
  ! weights are the same for every GPU tiling (the CPU's sum order, and
  ! with it the last bit of the normalization, depends on its tiling).
  !
  ! cdr_trcflx(ncdr, ntracers, time) [mmol/s] has the CPU's tracer axis
  ! (T, S, generic, CDR_LITE, MARBL; passive_mod cpu_tracer_index) and must
  ! carry no temperature or salinity flux (the CPU's check_no_ts_flux). The
  ! records are bracketed and weighted as the CPU's set_frc_data at the
  ! corrector's forcing time (n+1/2), the only one the tracer source uses.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use time_state
  use passive_mod, only: ntp, cpu_tracer_index
  use cdr_src_mod
  use comm_mod, only: comm_rank
  use netcdf
  implicit none
  private
  public :: read_cdr_namelist, init_cdr, set_cdr
  public :: cdr_source

  logical :: cdr_source = .false., cdr_forcing_depth_profiles = .false., cdr_forcing_3d = .false.
  logical :: cdr_forcing_parameterized = .false., cdr_time_interpolation = .false.
  logical :: cdr_relocate_to_wet_pts = .false., cdr_volume = .false.
  character(len=256) :: cdr_file = 'cdr_release.nc'
  integer :: cdr_ncdr_parm = 0, cdr_nz_chd = 0

  ! flux records
  integer :: ncid_cdr = -1, vid_flx = -1, nrec = 0, it1 = 0, it2 = 0, ntr_file = 0, ncdr = 0
  real(rp), allocatable :: rec_t(:)              ! cdr_time (days)
  real(rp), allocatable :: frec(:,:), fpas(:,:)  ! one record (ncdr, ntracers), mapped (ntp, ncdr)
  integer(ip), allocatable :: tmap(:)            ! passive tracer -> CPU tracer index

  real(rp), parameter :: pi = 3.14159265358979323_rp, deg2rad = pi / 180.0_rp   ! the CPU's
  real(rp), parameter :: radius = 6.371e6_rp

contains

  subroutine read_cdr_namelist(unit)
    ! &cdr_frc_settings as the CPU. A namelist without the group has no
    ! releases.
    integer, intent(in) :: unit
    integer :: ios
    character(len=256) :: msg
    namelist /cdr_frc_settings/ cdr_file, cdr_ncdr_parm, cdr_nz_chd, cdr_source, &
         cdr_forcing_depth_profiles, cdr_forcing_3d, cdr_forcing_parameterized, &
         cdr_time_interpolation, cdr_relocate_to_wet_pts, cdr_volume
    cdr_source = .false.
    rewind(unit)
    read(unit, nml=cdr_frc_settings, iostat=ios, iomsg=msg)
    if (ios > 0) then
      write(*,*) 'CDR_FRC_SETTINGS: ', trim(msg)
      error stop 'read_cdr_namelist'
    endif
    if (ios < 0) cdr_source = .false.
    if (.not. cdr_source) return
    if (count([cdr_forcing_parameterized, cdr_forcing_depth_profiles, cdr_forcing_3d]) /= 1) &
      error stop 'read_cdr_namelist: cdr_source needs exactly one of cdr_forcing_parameterized, '// &
                 'cdr_forcing_depth_profiles and cdr_forcing_3d'
    if (.not. cdr_forcing_parameterized) &
      error stop 'read_cdr_namelist: CDR depth-profile and 3D forcing are not on the GPU model yet'
    if (cdr_volume) error stop 'read_cdr_namelist: CDR volume releases (cdr_volume) are not on the GPU model yet'
    if (cdr_ncdr_parm <= 0) error stop 'read_cdr_namelist: cdr_ncdr_parm must be > 0'
  end subroutine read_cdr_namelist

  subroutine init_cdr(grdname)
    ! Release cells and profiles of this tile, the time catalog, and the
    ! device lists (cdr_set_lists). After the passive tracer list is final.
    character(len=*), intent(in) :: grdname
    integer :: ncg, gx, gy, i, j, k, r, nzl, ierr, vid, d(3), nd, ntr_d, ncdr_d, nrec_d
    integer :: imin, jmin, nq, ic, iarg, ntot, q0
    integer(ip) :: itp
    real(rp), allocatable :: lon(:,:), lat(:,:), mask(:,:), hh(:,:), dist(:,:), frac(:,:), fr_all(:,:,:)
    real(rp), allocatable :: rlon(:), rlat(:), rdep(:), rhsc(:), rvsc(:), tsf(:,:,:)
    real(rp), allocatable :: zr0(:), hz0(:), prf(:), w(:,:), wq(:,:)
    integer, allocatable :: ci(:), cj(:), cr(:), nqr(:), cellof(:,:), cnt(:)
    integer(ip), allocatable :: oc_i(:), oc_j(:), oc_col(:), oc_p0(:), oq_rel(:)
    real(rp) :: dmin, vint, arg, gsum, amin
    integer :: nq_r, q, c, li, lj, nloc_c
    if (.not. cdr_source) return
    nzl = int(decomp%nz)
    ncdr = cdr_ncdr_parm

    ! --- the release file: locations, scales, tracer axis, times
    call cdr_check(nf90_open(trim(cdr_file), nf90_nowrite, ncid_cdr), 'open '//trim(cdr_file))
    allocate(rlon(ncdr), rlat(ncdr), rdep(ncdr), rhsc(ncdr), rvsc(ncdr))
    call get_1d('cdr_lon', rlon); call get_1d('cdr_lat', rlat); call get_1d('cdr_dep', rdep)
    call get_1d('cdr_hsc', rhsc); call get_1d('cdr_vsc', rvsc)
    call cdr_check(nf90_inq_varid(ncid_cdr, 'cdr_trcflx', vid_flx), 'var cdr_trcflx')
    call cdr_check(nf90_inquire_variable(ncid_cdr, vid_flx, ndims=nd, dimids=d), 'inquire cdr_trcflx')
    if (nd /= 3) error stop 'init_cdr: cdr_trcflx must have dimensions (ncdr, ntracers, time)'
    call cdr_check(nf90_inquire_dimension(ncid_cdr, d(1), len=ncdr_d), 'dim 1 of cdr_trcflx')
    call cdr_check(nf90_inquire_dimension(ncid_cdr, d(2), len=ntr_d), 'dim 2 of cdr_trcflx')
    call cdr_check(nf90_inquire_dimension(ncid_cdr, d(3), len=nrec_d), 'dim 3 of cdr_trcflx')
    if (ncdr_d < ncdr) error stop 'init_cdr: cdr_trcflx has fewer releases than cdr_ncdr_parm'
    ntr_file = ntr_d; nrec = nrec_d
    if (ntr_file /= int(nt + ntp)) then
      write(*,'(a,i0,a,i0,a)') 'init_cdr: cdr_trcflx has ', ntr_file, ' tracers, the model ', int(nt + ntp), &
           ' (T, S, the passive tracers, in the CPU order)'
      error stop 'init_cdr: cdr_trcflx tracer axis'
    endif
    allocate(rec_t(nrec))
    call cdr_check(nf90_inq_varid(ncid_cdr, 'cdr_time', vid), 'var cdr_time')
    call cdr_check(nf90_get_var(ncid_cdr, vid, rec_t), 'get cdr_time')
    do i = 2, nrec
      if (rec_t(i) < rec_t(i - 1)) error stop 'init_cdr: cdr_time must increase'
    enddo
    ! the CPU's check_no_ts_flux: tracer-flux releases carry no temp/salt
    allocate(tsf(ncdr, 2, nrec))
    call cdr_check(nf90_get_var(ncid_cdr, vid_flx, tsf, start=[1, 1, 1], count=[ncdr, 2, nrec]), 'get cdr_trcflx T/S')
    do r = 1, ncdr
      if (any(tsf(r, :, :) /= 0.0_rp)) then
        write(*,'(a,i0,a)') 'init_cdr: CDR release ', r, ' has a nonzero temp/salt flux in cdr_trcflx '// &
             '(temperature and salinity can only be released with water: cdr_volume)'
        error stop 'init_cdr: temp/salt flux'
      endif
    enddo
    deallocate(tsf)
    allocate(tmap(max(ntp, 1_ip)), frec(ncdr, ntr_file), fpas(max(ntp, 1_ip), ncdr))
    do itp = 1, ntp
      tmap(itp) = cpu_tracer_index(itp, .true.)
    enddo

    ! --- the global grid: lon, lat, mask, h (interior cells 1..gnx, 1..gny
    ! are array indices 2..gnx+1, 2..gny+1)
    gx = int(decomp%gnx) + 2; gy = int(decomp%gny) + 2
    allocate(lon(gx, gy), lat(gx, gy), mask(gx, gy), hh(gx, gy), dist(gx, gy), frac(gx, gy))
    call cdr_check(nf90_open(trim(grdname), nf90_nowrite, ncg), 'open grid '//trim(grdname))
    call get_2d(ncg, 'lon_rho', lon); call get_2d(ncg, 'lat_rho', lat)
    call get_2d(ncg, 'mask_rho', mask); call get_2d(ncg, 'h', hh)
    ierr = nf90_close(ncg)

    ! --- weights frac of every release on the global grid (pass 1)
    allocate(fr_all(gx, gy, ncdr), nqr(ncdr))
    do r = 1, ncdr
      call haversine(lon, lat, rlon(r), rlat(r), dist, gx, gy)
      if (cdr_relocate_to_wet_pts) then
        do j = 2, gy - 1
          do i = 2, gx - 1
            if (mask(i, j) == 0.0_rp) dist(i, j) = 1.0e8_rp
          enddo
        enddo
      endif
      ! first minimum in array order (the CPU's MINLOC within a rank)
      dmin = huge(dmin); imin = 2; jmin = 2
      do j = 2, gy - 1
        do i = 2, gx - 1
          if (dist(i, j) < dmin) then
            dmin = dist(i, j); imin = i; jmin = j
          endif
        enddo
      enddo
      frac = 0.0_rp
      nq_r = 0
      if (rhsc(r) == 0.0_rp) then
        frac(imin, jmin) = 1.0_rp
      else
        do j = 2, gy - 1
          do i = 2, gx - 1
            frac(i, j) = exp(-(dist(i, j) / rhsc(r))**2)
            if (frac(i, j) > 1.0e-3_rp .and. mask(i, j) > 0.0_rp) nq_r = nq_r + 1
          enddo
        enddo
        if (nq_r == 0) then
          frac = 0.0_rp
          frac(imin, jmin) = 1.0_rp
        endif
      endif
      if (.not. cdr_relocate_to_wet_pts .and. frac(imin, jmin) == 1.0_rp .and. mask(imin, jmin) == 0.0_rp) then
        write(*,'(a,i0,a)') 'init_cdr: CDR release ', r, ' is on land'
        error stop 'init_cdr: release on land'
      endif
      if (comm_rank() == 0) write(*,'(a,i0,a,f12.3,a,2i6,a,2f10.4,a,2f10.4)') 'cdr: release ', r, &
           ' nearest cell at ', dmin, ' m, global (i, j)', imin - 1, jmin - 1, &
           ' lon/lat', lon(imin, jmin), lat(imin, jmin), ' intended', rlon(r), rlat(r)
      nqr(r) = count(frac(2:gx-1, 2:gy-1) > 1.0e-3_rp .and. mask(2:gx-1, 2:gy-1) > 0.0_rp)
      fr_all(:, :, r) = frac
    enddo

    ! --- profiles (pass 2): release-major, then j outer, i inner as the
    ! CPU's loop within a rank; then each release normalized to sum 1
    ntot = sum(nqr)
    allocate(ci(ntot), cj(ntot), cr(ntot), w(nzl, ntot))
    allocate(zr0(nzl), hz0(nzl), prf(nzl))
    q = 0
    do r = 1, ncdr
      q0 = q
      do j = 2, gy - 1
        do i = 2, gx - 1
          if (.not. (fr_all(i, j, r) > 1.0e-3_rp .and. mask(i, j) > 0.0_rp)) cycle
          call rest_grid(hh(i, j), nzl, zr0, hz0)
          prf = 0.0_rp
          if (rvsc(r) == 0.0_rp) then
            amin = huge(amin); iarg = 1
            do k = 1, nzl
              if (abs(zr0(k) + rdep(r)) < amin) then
                amin = abs(zr0(k) + rdep(r)); iarg = k
              endif
            enddo
            prf(iarg) = fr_all(i, j, r)
          else
            do k = 1, nzl
              arg = ((zr0(k) + rdep(r)) / rvsc(r))**2
              prf(k) = exp(-arg) * hz0(k)
            enddo
            vint = rvsc(r) * sqrt(pi)
            prf = fr_all(i, j, r) * prf / vint
          endif
          q = q + 1
          ci(q) = i; cj(q) = j; cr(q) = r; w(:, q) = prf
        enddo
      enddo
      ! the release's weights sum to 1 (k outer, cells inner, as the CPU's
      ! SUM over its (cidx, k) section)
      gsum = 0.0_rp
      do k = 1, nzl
        do c = q0 + 1, q
          gsum = gsum + w(k, c)
        enddo
      enddo
      if (gsum /= 0.0_rp) w(:, q0 + 1:q) = w(:, q0 + 1:q) / gsum
    enddo

    ! --- this tile's cells (interior), each with its profiles in release
    ! order: profiles come release-major, so appending per cell keeps it
    allocate(cellof(decomp%nx, decomp%ny), cnt(decomp%nx * decomp%ny + 1))
    cellof = 0; nloc_c = 0; nq = 0
    do c = 1, ntot
      li = ci(c) - 1 - int(decomp%i0) + 1; lj = cj(c) - 1 - int(decomp%j0) + 1
      if (li < 1 .or. li > decomp%nx .or. lj < 1 .or. lj > decomp%ny) cycle
      nq = nq + 1
      if (cellof(li, lj) == 0) then
        nloc_c = nloc_c + 1; cellof(li, lj) = nloc_c
      endif
    enddo
    allocate(oc_i(max(nloc_c, 1)), oc_j(max(nloc_c, 1)), oc_col(max(nloc_c, 1)), oc_p0(nloc_c + 1))
    allocate(oq_rel(max(nq, 1)), wq(nzl, max(nq, 1)))
    cnt = 0
    do c = 1, ntot
      li = ci(c) - 1 - int(decomp%i0) + 1; lj = cj(c) - 1 - int(decomp%j0) + 1
      if (li < 1 .or. li > decomp%nx .or. lj < 1 .or. lj > decomp%ny) cycle
      ic = cellof(li, lj)
      cnt(ic) = cnt(ic) + 1
      oc_i(ic) = li; oc_j(ic) = lj; oc_col(ic) = tpcol_h(li, lj)
    enddo
    oc_p0(1) = 1
    do ic = 1, nloc_c
      oc_p0(ic + 1) = oc_p0(ic) + cnt(ic)
    enddo
    cnt = 0
    do c = 1, ntot
      li = ci(c) - 1 - int(decomp%i0) + 1; lj = cj(c) - 1 - int(decomp%j0) + 1
      if (li < 1 .or. li > decomp%nx .or. lj < 1 .or. lj > decomp%ny) cycle
      ic = cellof(li, lj)
      q = oc_p0(ic) + cnt(ic)
      cnt(ic) = cnt(ic) + 1
      oq_rel(q) = cr(c); wq(:, q) = w(:, c)
    enddo
    call cdr_set_lists(int(nloc_c, ip), oc_i, oc_j, oc_col, oc_p0(1:nloc_c + 1), int(nq, ip), oq_rel, wq, &
                       int(ncdr, ip), ntp)
    write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a)') 'cdr: rank ', comm_rank(), '  releases ', ncdr, '  cells ', nloc_c, &
         '  profiles ', nq, ' (of ', ntot, ' in the domain)'
    deallocate(lon, lat, mask, hh, dist, frac, fr_all, nqr, ci, cj, cr, w, zr0, hz0, prf, wq, cellof, cnt)
    deallocate(oc_i, oc_j, oc_col, oc_p0, oq_rel, rlon, rlat, rdep, rhsc, rvsc)
  end subroutine init_cdr

  subroutine set_cdr()
    ! Flux records and weights at the corrector's forcing time (frc_slot 1,
    ! n+1/2), as the CPU's set_frc_data: the bracket moves on when the upper
    ! record is earlier than the model time; c1 = (t2-t)/(t2-t1),
    ! c2 = (t-t1)/(t2-t1), or the earlier record held without
    ! cdr_time_interpolation.
    real(rp) :: tneed, t1, t2, c1, c2
    integer :: rec, i, old1, old2
    if (.not. cdr_source) return
    tneed = frc_modtime()
    if (nrec >= 2) then
      if (it1 == 0 .or. .not. (rec_t(it2) >= tneed .and. rec_t(it1) <= tneed)) then
        rec = 1
        do i = 1, nrec - 1
          if (rec_t(i) <= tneed .and. rec_t(i + 1) >= tneed) then
            rec = i
            exit
          endif
          if (rec_t(i) <= tneed) rec = i
        enddo
        rec = max(1, min(rec, nrec - 1))
        if (rec /= it1 .or. rec + 1 /= it2) then
          old1 = it1; old2 = it2
          it1 = rec; it2 = rec + 1
          if (it1 /= old1) then
            if (it1 == old2) then
              call cdr_copy_upper()
            else
              call load_record(it1, 1)
            endif
          endif
          if (it2 /= old2) call load_record(it2, 2)
        endif
      endif
      t1 = rec_t(it1); t2 = rec_t(it2)
      if (t1 > tneed + dt * sec2day .or. t2 < tneed - dt * sec2day) then
        write(*,'(a,3es16.8)') 'set_cdr: CDR records do not cover the model time ', t1, t2, tneed
        error stop 'set_cdr: time interpolation error'
      endif
      if (cdr_time_interpolation) then
        c1 = (t2 - tneed) / (t2 - t1)
        c2 = (tneed - t1) / (t2 - t1)
      else
        c1 = 1.0_rp; c2 = 0.0_rp
      endif
    else
      if (it1 == 0) then
        it1 = 1; it2 = 1
        call load_record(1, 1)
        call load_record(1, 2)
      endif
      c1 = 1.0_rp; c2 = 0.0_rp
    endif
    call cdr_set_weights(c1, c2)
  end subroutine set_cdr

  subroutine load_record(idx, slot)
    ! cdr_trcflx(:, :, idx), mapped to (passive tracer, release).
    integer, intent(in) :: idx, slot
    integer(ip) :: itp
    integer :: r
    call cdr_check(nf90_get_var(ncid_cdr, vid_flx, frec, start=[1, 1, idx], count=[ncdr, ntr_file, 1]), &
                   'get cdr_trcflx')
    do r = 1, ncdr
      do itp = 1, ntp
        fpas(itp, r) = frec(r, tmap(itp))
      enddo
    enddo
    call cdr_set_record(slot, fpas)
  end subroutine load_record

  subroutine rest_grid(h, nz, zr0, hz0)
    ! z_r0, Hz0 of one column with zeta = 0, in the CPU's expressions
    ! (set_depth at iic = 0): z = h*(cff + Cs*h)*hinv, hinv = 1/(h+hc).
    real(rp), intent(in) :: h
    integer, intent(in) :: nz
    real(rp), intent(out) :: zr0(nz), hz0(nz)
    real(rp) :: hinv, ds, cff_w, cff_r, zw_prev, zw
    integer :: k
    hinv = 1.0_rp / (h + hc)
    ds = 1.0_rp / real(nz, rp)
    zw_prev = -h
    do k = 1, nz
      cff_w = hc * ds * real(k - nz, rp)
      cff_r = hc * ds * (real(k - nz, rp) - 0.5_rp)
      zw = h * (cff_w + Cs_w(k) * h) * hinv
      zr0(k) = h * (cff_r + Cs_r(k) * h) * hinv
      hz0(k) = zw - zw_prev
      zw_prev = zw
    enddo
  end subroutine rest_grid

  subroutine haversine(lon, lat, lon0, lat0, dist, gx, gy)
    ! The CPU's ll2dist, on the interior cells (the rest stay far away).
    integer, intent(in) :: gx, gy
    real(rp), intent(in) :: lon(gx, gy), lat(gx, gy), lon0, lat0
    real(rp), intent(out) :: dist(gx, gy)
    real(rp) :: dlon2, dlat2, a
    integer :: i, j
    dist = 1.0e6_rp
    do j = 2, gy - 1
      do i = 2, gx - 1
        dlon2 = 0.5_rp * (lon(i, j) - lon0) * deg2rad
        dlat2 = 0.5_rp * (lat(i, j) - lat0) * deg2rad
        a = sin(dlat2)**2 + cos(lat(i, j) * deg2rad) * cos(lat0 * deg2rad) * sin(dlon2)**2
        a = 2 * atan2(sqrt(a), sqrt(1 - a))
        dist(i, j) = radius * a
      enddo
    enddo
  end subroutine haversine

  subroutine get_1d(name, x)
    character(len=*), intent(in) :: name
    real(rp), intent(out) :: x(:)
    integer :: vid
    call cdr_check(nf90_inq_varid(ncid_cdr, name, vid), 'var '//name)
    call cdr_check(nf90_get_var(ncid_cdr, vid, x, start=[1], count=[size(x)]), 'get '//name)
  end subroutine get_1d

  subroutine get_2d(ncid, name, x)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(rp), intent(out) :: x(:,:)
    integer :: vid
    call cdr_check(nf90_inq_varid(ncid, name, vid), 'grid var '//name)
    call cdr_check(nf90_get_var(ncid, vid, x), 'get grid '//name)
  end subroutine get_2d

  subroutine cdr_check(ierr, where)
    integer,          intent(in) :: ierr
    character(len=*), intent(in) :: where
    if (ierr /= nf90_noerr) then
      write(*,'(a,a,a,a)') 'cdr_io: ', trim(where), ': ', trim(nf90_strerror(ierr))
      error stop 'cdr_io: netCDF error'
    endif
  end subroutine cdr_check

end module cdr_io
