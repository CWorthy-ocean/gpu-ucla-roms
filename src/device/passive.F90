#include "mem_attr.inc"
module passive_mod
  ! Passive (anomaly) tracers stored in real(pp) (float32 by default, see
  ! kinds.F90) with all arithmetic in real(rp). They are stepped exactly as
  ! temperature/salinity (LF-AM3 predictor with C4 advection, UP3/U3
  ! corrector, implicit vertical diffusion with the salinity diffusivity,
  ! harmonic lateral diffusion, Orlanski-type open boundaries), with no
  ! surface forcing and a zero external boundary value, as befits CDR
  ! anomaly tracers. The implicit solve factorizes each column once and
  ! back-substitutes every passive right-hand side.
  !
  ! Validation hook: passive_init = 'mirror_salt' initializes tracer 1 to
  ! salinity and drives it with salinity's surface flux, KPP nonlocal
  ! term and boundary data, so its difference from salinity after a run
  ! is the storage-precision error alone.
  use kinds
  use decomp_mod, only: decomp
  use gpu_state
  use formulas
  use time_state
  use passive_column_mod, only: passive_column_run
  implicit none
  private

  integer(ip), public :: ntp = 0
  real(pp), WORK_ATTR public, allocatable :: tp(:,:,:,:,:)   ! (is:ie, js:je, nz, 3, ntp)
  character(len=16), public :: passive_init = 'zero'           ! zero | blob | mirror_salt
  logical, public :: passive_mirror = .false.
  logical, public :: wrt_passive = .true.
  ! MARBL mode: the passives are the MARBL tracers (names from MARBL, read
  ! from the initial file by name, external boundary values from the
  ! boundary data below instead of 0). Set by marbl_gpu before alloc_passive.
  logical, public :: passive_marbl = .false.
  character(len=32), allocatable, public :: passive_names(:)
  real(rp), WORK_ATTR public, allocatable :: tb_west(:,:,:), tb_east(:,:,:)    ! (js:je, nz, ntp)
  real(rp), WORK_ATTR public, allocatable :: tb_south(:,:,:), tb_north(:,:,:)  ! (is:ie, nz, ntp)

  public :: read_passive_namelist
  public :: alloc_passive
  public :: init_passive
  public :: passive_predictor
  public :: passive_corrector
  public :: passive_t3dmix
  public :: passive_t3dbc
  public :: passive_name
  public :: set_passive_marbl

contains

  subroutine read_passive_namelist(unit, n_from_param)
    ! Group passive_tracers is optional; nt_passive from param_settings
    ! is the count unless overridden here.
    integer, intent(in) :: unit
    integer(ip), intent(in) :: n_from_param
    integer :: ios
    integer(ip) :: n_passive
    character(len=16) :: init_mode
    logical :: write_passive
    namelist /passive_tracers/ n_passive, init_mode, write_passive
    n_passive = n_from_param
    init_mode = 'zero'
    write_passive = .true.
    rewind(unit)
    read(unit, nml=passive_tracers, iostat=ios)
    if (ios > 0) error stop 'passive_tracers namelist: read error'
    ntp = max(n_passive, 0_ip)
    passive_init = init_mode
    wrt_passive = write_passive
    passive_mirror = (trim(passive_init) == 'mirror_salt')
    if (passive_mirror .and. ntp < 1) ntp = 1
    if (ntp > 0) write(*,'(a,i0,a,a,a,i0,a)') 'passive tracers: ', ntp, '  init = ', &
         trim(passive_init), '  (', int(storage_size(1.0_pp)), '-bit storage)'
  end subroutine read_passive_namelist

  function passive_name(itp) result(name)
    integer(ip), intent(in) :: itp
    character(len=32) :: name
    if (passive_marbl) then
      name = trim(passive_names(itp))
    else
      write(name, '(a,i2.2)') 'tp', itp
    endif
  end function passive_name

  subroutine set_passive_marbl(names)
    character(len=*), intent(in) :: names(:)
    integer(ip) :: m
    ntp = size(names)
    if (allocated(passive_names)) deallocate(passive_names)
    allocate(passive_names(ntp))
    do m = 1, ntp
      passive_names(m) = trim(names(m))
    enddo
    passive_marbl = .true.
    passive_mirror = .false.
    passive_init = 'zero'
    write(*,'(a,i0,a,i0,a)') 'passive tracers: MARBL mode, ', ntp, ' tracers (', &
         int(storage_size(1.0_pp)), '-bit storage)'
  end subroutine set_passive_marbl

  subroutine alloc_passive()
    integer(ip) :: is, ie, js, je, nz
    if (ntp <= 0) return
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    if (allocated(tp)) deallocate(tp)
    allocate(tp(is:ie, js:je, nz, 3, ntp))
    tp = 0.0_pp
    if (passive_marbl) then
      if (allocated(tb_west)) deallocate(tb_west, tb_east, tb_south, tb_north)
      allocate(tb_west(js:je, nz, ntp), tb_east(js:je, nz, ntp))
      allocate(tb_south(is:ie, nz, ntp), tb_north(is:ie, nz, ntp))
      tb_west = 0.0_rp; tb_east = 0.0_rp; tb_south = 0.0_rp; tb_north = 0.0_rp
    endif
  end subroutine alloc_passive

  subroutine init_passive()
    ! Fill all three time slots (the first LF-AM3 step then behaves as a
    ! forward step for the passives).
    integer(ip) :: i, j, k, n, itp, nx, ny, nz
    real(rp) :: ic, jc, rad, val
    if (ntp <= 0) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz
    select case (trim(passive_init))
    case ('blob')
      ic = 0.5_rp * real(nx, rp); jc = 0.5_rp * real(ny, rp)
      rad = 0.1_rp * real(min(nx, ny), rp)
      do itp = 1, ntp
        do k = 1, nz
          do j = decomp%js, decomp%je
            do i = decomp%is, decomp%ie
              val = exp(-((real(i, rp) - ic)**2 + (real(j, rp) - jc)**2) / rad**2) * &
                    exp(-real(nz - k, rp) / 20.0_rp) * rmask(i, j)
              do n = 1, 3
                tp(i, j, k, n, itp) = real(val, pp)
              enddo
            enddo
          enddo
        enddo
      enddo
    case ('mirror_salt')
      do k = 1, nz
        do j = decomp%js, decomp%je
          do i = decomp%is, decomp%ie
            do n = 1, 3
              tp(i, j, k, n, 1) = real(t(i, j, k, n, isalt), pp)
            enddo
          enddo
        enddo
      enddo
    case default
      tp = 0.0_pp
    end select
    write(*,'(a,a)') 'passive tracers initialized: ', trim(passive_init)
  end subroutine init_passive

  ! ------------------------------------------------------------- faces
  ! Horizontal face fluxes of passive tracer itr from slot ts; the same
  ! arithmetic as advect's c4_fx/c4_fe (predictor) and up3_fx/up3_fe
  ! (corrector), with tp read into real(rp).
#ifdef USE_CUDA
  attributes(device) &
#endif
  function pc4_fx(i, j, k, ts, itr, west, east, nx) result(fx)
    integer(ip), value :: i, j, k, ts, itr, west, east, nx
    real(rp) :: fx, d_l, d_c, d_r, t_l, t_c
    t_c = real(tp(i, j, k, ts, itr), rp)
    t_l = real(tp(i - 1, j, k, ts, itr), rp)
    d_c = apply_mask(t_c - t_l, umask(i, j))
    if (west == 1 .and. i == 1) then
      d_l = d_c
    else
      d_l = apply_mask(t_l - real(tp(i - 2, j, k, ts, itr), rp), umask(i - 1, j))
    endif
    if (east == 1 .and. i == nx + 1) then
      d_r = d_c
    else
      d_r = apply_mask(real(tp(i + 1, j, k, ts, itr), rp) - t_c, umask(i + 1, j))
    endif
    fx = apply_mask(0.5_rp * (t_c + t_l - (d_r - d_l) / 6.0_rp) * FlxU(i, j, k), umask(i, j))
  end function pc4_fx

#ifdef USE_CUDA
  attributes(device) &
#endif
  function pc4_fe(i, j, k, ts, itr, south, north, ny) result(fe)
    integer(ip), value :: i, j, k, ts, itr, south, north, ny
    real(rp) :: fe, d_l, d_c, d_r, t_l, t_c
    t_c = real(tp(i, j, k, ts, itr), rp)
    t_l = real(tp(i, j - 1, k, ts, itr), rp)
    d_c = apply_mask(t_c - t_l, vmask(i, j))
    if (south == 1 .and. j == 1) then
      d_l = d_c
    else
      d_l = apply_mask(t_l - real(tp(i, j - 2, k, ts, itr), rp), vmask(i, j - 1))
    endif
    if (north == 1 .and. j == ny + 1) then
      d_r = d_c
    else
      d_r = apply_mask(real(tp(i, j + 1, k, ts, itr), rp) - t_c, vmask(i, j + 1))
    endif
    fe = apply_mask(0.5_rp * (t_c + t_l - (d_r - d_l) / 6.0_rp) * FlxV(i, j, k), vmask(i, j))
  end function pc4_fe

#ifdef USE_CUDA
  attributes(device) &
#endif
  function pup3_fx(i, j, k, ts, itr, west, east, nx) result(fx)
    integer(ip), value :: i, j, k, ts, itr, west, east, nx
    real(rp) :: fx, d_l, d_c, d_r, curv_l, curv_r, t_l, t_c, face_t, uflux
    t_c = real(tp(i, j, k, ts, itr), rp)
    t_l = real(tp(i - 1, j, k, ts, itr), rp)
    d_c = apply_mask(t_c - t_l, umask(i, j))
    if (west == 1 .and. i == 1) then
      d_l = d_c
    else
      d_l = apply_mask(t_l - real(tp(i - 2, j, k, ts, itr), rp), umask(i - 1, j))
    endif
    if (east == 1 .and. i == nx + 1) then
      d_r = d_c
    else
      d_r = apply_mask(real(tp(i + 1, j, k, ts, itr), rp) - t_c, umask(i + 1, j))
    endif
    curv_l = (d_c - d_l) * umask(i - 1, j) * umask(i, j)
    curv_r = (d_r - d_c) * umask(i, j) * umask(i + 1, j)
    if (rmask(i - 1, j) > 0.5_rp .and. rmask(i, j) > 0.5_rp) then
      face_t = 0.5_rp * (t_l + t_c)
    elseif (rmask(i - 1, j) > 0.5_rp) then
      face_t = t_l
    elseif (rmask(i, j) > 0.5_rp) then
      face_t = t_c
    else
      face_t = 0.0_rp
    endif
    uflux = FlxU(i, j, k)
    fx = apply_mask(face_t * uflux - coef_up3 * ( &
         curv_l * max(uflux, 0.0_rp) + curv_r * min(uflux, 0.0_rp)), umask(i, j))
  end function pup3_fx

#ifdef USE_CUDA
  attributes(device) &
#endif
  function pup3_fe(i, j, k, ts, itr, south, north, ny) result(fe)
    integer(ip), value :: i, j, k, ts, itr, south, north, ny
    real(rp) :: fe, d_l, d_c, d_r, curv_l, curv_r, t_l, t_c, face_t, uflux
    t_c = real(tp(i, j, k, ts, itr), rp)
    t_l = real(tp(i, j - 1, k, ts, itr), rp)
    d_c = apply_mask(t_c - t_l, vmask(i, j))
    if (south == 1 .and. j == 1) then
      d_l = d_c
    else
      d_l = apply_mask(t_l - real(tp(i, j - 2, k, ts, itr), rp), vmask(i, j - 1))
    endif
    if (north == 1 .and. j == ny + 1) then
      d_r = d_c
    else
      d_r = apply_mask(real(tp(i, j + 1, k, ts, itr), rp) - t_c, vmask(i, j + 1))
    endif
    curv_l = (d_c - d_l) * vmask(i, j - 1) * vmask(i, j)
    curv_r = (d_r - d_c) * vmask(i, j) * vmask(i, j + 1)
    if (rmask(i, j - 1) > 0.5_rp .and. rmask(i, j) > 0.5_rp) then
      face_t = 0.5_rp * (t_l + t_c)
    elseif (rmask(i, j - 1) > 0.5_rp) then
      face_t = t_l
    elseif (rmask(i, j) > 0.5_rp) then
      face_t = t_c
    else
      face_t = 0.0_rp
    endif
    uflux = FlxV(i, j, k)
    fe = apply_mask(face_t * uflux - coef_up3 * ( &
         curv_l * max(uflux, 0.0_rp) + curv_r * min(uflux, 0.0_rp)), vmask(i, j))
  end function pup3_fe

  ! ------------------------------------------------------ vertical pass
  ! --------------------------------------------------------- predictor
  subroutine passive_predictor(dtau, cf_stp, cf_bak, n_stp, n_new, indx)
    ! Called from pre_step3d after the T/S predictor: LF-AM3 interpolant
    ! with C4 horizontal fluxes from slot nrhs (= nstp), the Hz-weighted
    ! old field saved into slot indx, C4 vertical fluxes, implicit solve.
    ! The open boundaries of the predictor slot are set afterwards by
    ! passive_t3dbc, called from pre_step3d next to t3dbc(itemp/isalt).
    real(rp), intent(in) :: dtau, cf_stp, cf_bak
    integer(ip), intent(in) :: n_stp, n_new, indx
    integer(ip) :: i, j, k, itr, nx, ny, nz, ntr, ts, ns, nn, nb
    integer(ip) :: west, east, south, north
    real(rp) :: dtpp, hb, hzc, cs, cb, dtp, told
    if (ntp <= 0) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = ntp
    ts = nrhs; ns = n_stp; nn = n_new; nb = indx
    cs = cf_stp; cb = cf_bak; dtp = dtau
    west  = merge(1_ip, 0_ip, decomp%edge_w)
    east  = merge(1_ip, 0_ip, decomp%edge_e)
    south = merge(1_ip, 0_ip, decomp%edge_s)
    north = merge(1_ip, 0_ip, decomp%edge_n)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          dtpp = dtp * pm(i, j) * pn(i, j)
          hb = Hz_bak(i, j, k)
          hzc = Hz(i, j, k)
          do itr = 1, ntr
            told = real(tp(i, j, k, ns, itr), rp)
            tp(i, j, k, nn, itr) = real(hb * (cs * told + cb * real(tp(i, j, k, nb, itr), rp)) - &
                 dtpp * (pc4_fx(i + 1, j, k, ts, itr, west, east, nx) - &
                         pc4_fx(i, j, k, ts, itr, west, east, nx) + &
                         pc4_fe(i, j + 1, k, ts, itr, south, north, ny) - &
                         pc4_fe(i, j, k, ts, itr, south, north, ny)), pp)
            tp(i, j, k, nb, itr) = real(hzc * told, pp)
          enddo
        enddo
      enddo
    enddo
    call passive_column_run(tp, ntp, ts, nn, dtau, 1_ip, .true., passive_mirror)
  end subroutine passive_predictor

  ! --------------------------------------------------------- corrector
  subroutine passive_corrector(dt_in)
    ! Called from step3d_t after the T/S corrector: UP3 horizontal from
    ! slot nrhs (= 3), then U3 vertical + (mirror: salinity forcing) +
    ! implicit solve in one column kernel, open boundaries.
    ! Faces as pup3_fx/pup3_fe (bit-identical), masks/fluxes loaded once
    ! per cell, shared differences reused between adjacent faces.
    real(rp), intent(in) :: dt_in
    integer(ip) :: i, j, k, itr, nx, ny, nz, ntr, ts, nn
    integer(ip) :: west, east, south, north
    real(rp) :: dtpp, dtl
    real(rp) :: um0, um1, um2, um3, vm0, vm1, vm2, vm3, fu1, fu2, fv1, fv2
    real(rp) :: rmw, rmc, rme, rms, rmn, fu1p, fu1m, fu2p, fu2m, fv1p, fv1m, fv2p, fv2m
    real(rp) :: tm2, tm1, t0, tp1, tp2, dcw, dlw, drw, dre, fxw, fxe, fes, fen, face_t
    logical :: wedge, eedge, sedge, nedge
    if (ntp <= 0) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = ntp
    ts = nrhs; nn = nnew; dtl = dt_in
    west  = merge(1_ip, 0_ip, decomp%edge_w)
    east  = merge(1_ip, 0_ip, decomp%edge_e)
    south = merge(1_ip, 0_ip, decomp%edge_s)
    north = merge(1_ip, 0_ip, decomp%edge_n)
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          dtpp = dtl * pm(i, j) * pn(i, j)
          um0 = umask(i - 1, j); um1 = umask(i, j); um2 = umask(i + 1, j); um3 = umask(i + 2, j)
          vm0 = vmask(i, j - 1); vm1 = vmask(i, j); vm2 = vmask(i, j + 1); vm3 = vmask(i, j + 2)
          fu1 = FlxU(i, j, k); fu2 = FlxU(i + 1, j, k)
          fv1 = FlxV(i, j, k); fv2 = FlxV(i, j + 1, k)
          fu1p = max(fu1, 0.0_rp); fu1m = min(fu1, 0.0_rp); fu2p = max(fu2, 0.0_rp); fu2m = min(fu2, 0.0_rp)
          fv1p = max(fv1, 0.0_rp); fv1m = min(fv1, 0.0_rp); fv2p = max(fv2, 0.0_rp); fv2m = min(fv2, 0.0_rp)
          wedge = (west == 1 .and. i == 1)
          eedge = (east == 1 .and. i == nx)
          sedge = (south == 1 .and. j == 1)
          nedge = (north == 1 .and. j == ny)
          rmw = rmask(i - 1, j); rmc = rmask(i, j); rme = rmask(i + 1, j)
          rms = rmask(i, j - 1); rmn = rmask(i, j + 1)
          do itr = 1, ntr
            ! --- x faces i (west: cells i-1,i) and i+1 (east: cells i,i+1)
            t0 = real(tp(i, j, k, ts, itr), rp)
            tm1 = real(tp(i - 1, j, k, ts, itr), rp)
            tp1 = real(tp(i + 1, j, k, ts, itr), rp)
            dcw = apply_mask(t0 - tm1, um1)
            if (wedge) then
              dlw = dcw
            else
              tm2 = real(tp(i - 2, j, k, ts, itr), rp)
              dlw = apply_mask(tm1 - tm2, um0)
            endif
            drw = apply_mask(tp1 - t0, um2)
            if (rmw > 0.5_rp .and. rmc > 0.5_rp) then
              face_t = 0.5_rp * (tm1 + t0)
            elseif (rmw > 0.5_rp) then
              face_t = tm1
            elseif (rmc > 0.5_rp) then
              face_t = t0
            else
              face_t = 0.0_rp
            endif
            fxw = apply_mask(face_t * fu1 - coef_up3 * ( &
                  (dcw - dlw) * um0 * um1 * fu1p + (drw - dcw) * um1 * um2 * fu1m), um1)
            if (eedge) then
              dre = drw
            else
              tp2 = real(tp(i + 2, j, k, ts, itr), rp)
              dre = apply_mask(tp2 - tp1, um3)
            endif
            if (rmc > 0.5_rp .and. rme > 0.5_rp) then
              face_t = 0.5_rp * (t0 + tp1)
            elseif (rmc > 0.5_rp) then
              face_t = t0
            elseif (rme > 0.5_rp) then
              face_t = tp1
            else
              face_t = 0.0_rp
            endif
            fxe = apply_mask(face_t * fu2 - coef_up3 * ( &
                  (drw - dcw) * um1 * um2 * fu2p + (dre - drw) * um2 * um3 * fu2m), um2)
            ! --- y faces j (south) and j+1 (north)
            tm1 = real(tp(i, j - 1, k, ts, itr), rp)
            tp1 = real(tp(i, j + 1, k, ts, itr), rp)
            dcw = apply_mask(t0 - tm1, vm1)
            if (sedge) then
              dlw = dcw
            else
              tm2 = real(tp(i, j - 2, k, ts, itr), rp)
              dlw = apply_mask(tm1 - tm2, vm0)
            endif
            drw = apply_mask(tp1 - t0, vm2)
            if (rms > 0.5_rp .and. rmc > 0.5_rp) then
              face_t = 0.5_rp * (tm1 + t0)
            elseif (rms > 0.5_rp) then
              face_t = tm1
            elseif (rmc > 0.5_rp) then
              face_t = t0
            else
              face_t = 0.0_rp
            endif
            fes = apply_mask(face_t * fv1 - coef_up3 * ( &
                  (dcw - dlw) * vm0 * vm1 * fv1p + (drw - dcw) * vm1 * vm2 * fv1m), vm1)
            if (nedge) then
              dre = drw
            else
              tp2 = real(tp(i, j + 2, k, ts, itr), rp)
              dre = apply_mask(tp2 - tp1, vm3)
            endif
            if (rmc > 0.5_rp .and. rmn > 0.5_rp) then
              face_t = 0.5_rp * (t0 + tp1)
            elseif (rmc > 0.5_rp) then
              face_t = t0
            elseif (rmn > 0.5_rp) then
              face_t = tp1
            else
              face_t = 0.0_rp
            endif
            fen = apply_mask(face_t * fv2 - coef_up3 * ( &
                  (drw - dcw) * vm1 * vm2 * fv2p + (dre - drw) * vm2 * vm3 * fv2m), vm2)
            tp(i, j, k, nn, itr) = real(real(tp(i, j, k, nn, itr), rp) - dtpp * (fxe - fxw + fen - fes), pp)
          enddo
        enddo
      enddo
    enddo
    call passive_column_run(tp, ntp, ts, nn, dtl, 0_ip, .false., passive_mirror)
    call passive_t3dbc()
  end subroutine passive_corrector

  ! ------------------------------------------------ lateral diffusion
  subroutine passive_t3dmix()
    ! Harmonic diffusion along s-surfaces with the salinity diffusivity
    ! (t3dmix for tp; faces computed inline, all tracers in one launch).
    integer(ip) :: i, j, k, itr, nx, ny, nz, ntr, nr, nn, isl
    real(rp) :: dt_loc, fxw, fxe, fes, fen
    if (ntp <= 0) return
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = ntp
    nr = nrhs; nn = nnew; dt_loc = dt
    isl = min(2_ip, int(size(diff2, 3), ip))
#ifdef USE_CUDA
    !$cuf kernel do(3) <<<*,*>>>
#endif
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          do itr = 1, ntr
            fxw = apply_mask(0.25_rp * (diff2(i, j, isl) + diff2(i - 1, j, isl)) * &
                 pmon_u(i, j) * (Hz(i, j, k) + Hz(i - 1, j, k)) * &
                 (real(tp(i, j, k, nr, itr), rp) - real(tp(i - 1, j, k, nr, itr), rp)), umask(i, j))
            fxe = apply_mask(0.25_rp * (diff2(i + 1, j, isl) + diff2(i, j, isl)) * &
                 pmon_u(i + 1, j) * (Hz(i + 1, j, k) + Hz(i, j, k)) * &
                 (real(tp(i + 1, j, k, nr, itr), rp) - real(tp(i, j, k, nr, itr), rp)), umask(i + 1, j))
            fes = apply_mask(0.25_rp * (diff2(i, j, isl) + diff2(i, j - 1, isl)) * &
                 pnom_v(i, j) * (Hz(i, j, k) + Hz(i, j - 1, k)) * &
                 (real(tp(i, j, k, nr, itr), rp) - real(tp(i, j - 1, k, nr, itr), rp)), vmask(i, j))
            fen = apply_mask(0.25_rp * (diff2(i, j + 1, isl) + diff2(i, j, isl)) * &
                 pnom_v(i, j + 1) * (Hz(i, j + 1, k) + Hz(i, j, k)) * &
                 (real(tp(i, j + 1, k, nr, itr), rp) - real(tp(i, j, k, nr, itr), rp)), vmask(i, j + 1))
            tp(i, j, k, nn, itr) = real(real(tp(i, j, k, nn, itr), rp) + dt_loc * pm(i, j) * pn(i, j) * &
                 (fxe - fxw + fen - fes) / max(Hz(i, j, k), 1.0e-30_rp), pp)
          enddo
        enddo
      enddo
    enddo
  end subroutine passive_t3dmix

  ! ------------------------------------------------------ open boundaries
  subroutine passive_t3dbc()
    ! t3dbc for the passives: upstream radiation with tangential upstream
    ! advection, nudged on inflow toward the external value (0, or the
    ! salinity boundary data for the mirror tracer), then corners.
    integer(ip) :: i, j, k, nx, ny, nz, ns, nn, nr, itr, ntr, ilo, ihi, jlo, jhi
    real(rp) :: cx, cy, cext, dtw, cff, text
    real(rp) :: g0c, g0p, g1c, g1p, tn
    logical :: sw, se, nw, ne, mir, mbry
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = ntp
    ns = nstp; nn = nnew; nr = nrhs
    ilo = lbound(tp, 1); ihi = ubound(tp, 1)
    jlo = lbound(tp, 2); jhi = ubound(tp, 2)
    mir = passive_mirror
    mbry = passive_marbl
    if (nnew == 3) then
      dtw = 0.5_rp * dt
    else
      dtw = dt
    endif

    if (decomp%edge_w) then
      if (obc_west) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do itr = 1, ntr
          do k = 1, nz
            do j = 1, ny
              g0c = (real(tp(0, j, k, ns, itr), rp) - real(tp(0, j - 1, k, ns, itr), rp)) * vmask(0, j)
              g0p = (real(tp(0, j + 1, k, ns, itr), rp) - real(tp(0, j, k, ns, itr), rp)) * vmask(0, j + 1)
              g1c = (real(tp(1, j, k, ns, itr), rp) - real(tp(1, j - 1, k, ns, itr), rp)) * vmask(1, j)
              g1p = (real(tp(1, j + 1, k, ns, itr), rp) - real(tp(1, j, k, ns, itr), rp)) * vmask(1, j + 1)
              cx = -dtw * u(1, j, k, nr) * pm(0, j)
              cy = 0.5_rp * dtw * (v(0, j, k, nr) + v(0, j + 1, k, nr)) * pn(0, j)
              if (cx > 0.0_rp) then
                cext = 0.0_rp
              else
                cext = -cx; cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(0, j, k, ns, itr), rp) &
                   - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p) &
                   + cx * (real(tp(1, j, k, ns, itr), rp) - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p)
              text = 0.0_rp
              if (mir .and. itr == 1) text = t_west(j, k, isalt)
              if (mbry) text = tb_west(j, k, itr)
              tp(0, j, k, nn, itr) = real(((1.0_rp - cext) * tn + cext * text) * rmask(0, j), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do itr = 1, ntr
          do k = 1, nz
            do j = jlo, jhi
              tp(0, j, k, nn, itr) = tp(1, j, k, nn, itr)
            enddo
          enddo
        enddo
      endif
    endif

    if (decomp%edge_e) then
      if (obc_east) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do itr = 1, ntr
          do k = 1, nz
            do j = 1, ny
              g0c = (real(tp(nx, j, k, ns, itr), rp) - real(tp(nx, j - 1, k, ns, itr), rp)) * vmask(nx, j)
              g0p = (real(tp(nx, j + 1, k, ns, itr), rp) - real(tp(nx, j, k, ns, itr), rp)) * vmask(nx, j + 1)
              g1c = (real(tp(nx + 1, j, k, ns, itr), rp) - real(tp(nx + 1, j - 1, k, ns, itr), rp)) * vmask(nx + 1, j)
              g1p = (real(tp(nx + 1, j + 1, k, ns, itr), rp) - real(tp(nx + 1, j, k, ns, itr), rp)) * vmask(nx + 1, j + 1)
              cx = dtw * u(nx + 1, j, k, nr) * pm(nx + 1, j)
              cy = 0.5_rp * dtw * (v(nx + 1, j, k, nr) + v(nx + 1, j + 1, k, nr)) * pn(nx + 1, j)
              if (cx > 0.0_rp) then
                cext = 0.0_rp
              else
                cext = -cx; cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(nx + 1, j, k, ns, itr), rp) &
                   - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p) &
                   + cx * (real(tp(nx, j, k, nn, itr), rp) - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p)
              text = 0.0_rp
              if (mir .and. itr == 1) text = t_east(j, k, isalt)
              if (mbry) text = tb_east(j, k, itr)
              tp(nx + 1, j, k, nn, itr) = real(((1.0_rp - cext) * tn + cext * text) * rmask(nx + 1, j), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do itr = 1, ntr
          do k = 1, nz
            do j = jlo, jhi
              tp(nx + 1, j, k, nn, itr) = tp(nx, j, k, nn, itr)
            enddo
          enddo
        enddo
      endif
    endif

    if (decomp%edge_s) then
      if (obc_south) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do itr = 1, ntr
          do k = 1, nz
            do i = 1, nx
              g0c = (real(tp(i, 0, k, ns, itr), rp) - real(tp(i - 1, 0, k, ns, itr), rp)) * umask(i, 0)
              g0p = (real(tp(i + 1, 0, k, ns, itr), rp) - real(tp(i, 0, k, ns, itr), rp)) * umask(i + 1, 0)
              g1c = (real(tp(i, 1, k, ns, itr), rp) - real(tp(i - 1, 1, k, ns, itr), rp)) * umask(i, 1)
              g1p = (real(tp(i + 1, 1, k, ns, itr), rp) - real(tp(i, 1, k, ns, itr), rp)) * umask(i + 1, 1)
              cx = -dtw * v(i, 1, k, nr) * pn(i, 0)
              cy = 0.5_rp * dtw * (u(i, 0, k, nr) + u(i + 1, 0, k, nr)) * pm(i, 0)
              if (cx > 0.0_rp) then
                cext = 0.0_rp
              else
                cext = -cx; cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(i, 0, k, ns, itr), rp) &
                   - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p) &
                   + cx * (real(tp(i, 1, k, ns, itr), rp) - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p)
              text = 0.0_rp
              if (mir .and. itr == 1) text = t_south(i, k, isalt)
              if (mbry) text = tb_south(i, k, itr)
              tp(i, 0, k, nn, itr) = real(((1.0_rp - cext) * tn + cext * text) * rmask(i, 0), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do itr = 1, ntr
          do k = 1, nz
            do i = ilo, ihi
              tp(i, 0, k, nn, itr) = tp(i, 1, k, nn, itr)
            enddo
          enddo
        enddo
      endif
    endif

    if (decomp%edge_n) then
      if (obc_north) then
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do itr = 1, ntr
          do k = 1, nz
            do i = 1, nx
              g0c = (real(tp(i, ny, k, ns, itr), rp) - real(tp(i - 1, ny, k, ns, itr), rp)) * umask(i, ny)
              g0p = (real(tp(i + 1, ny, k, ns, itr), rp) - real(tp(i, ny, k, ns, itr), rp)) * umask(i + 1, ny)
              g1c = (real(tp(i, ny + 1, k, ns, itr), rp) - real(tp(i - 1, ny + 1, k, ns, itr), rp)) * umask(i, ny + 1)
              g1p = (real(tp(i + 1, ny + 1, k, ns, itr), rp) - real(tp(i, ny + 1, k, ns, itr), rp)) * umask(i + 1, ny + 1)
              cx = dtw * v(i, ny + 1, k, nr) * pn(i, ny + 1)
              cy = 0.5_rp * dtw * (u(i, ny + 1, k, nr) + u(i + 1, ny + 1, k, nr)) * pm(i, ny + 1)
              if (cx > 0.0_rp) then
                cext = 0.0_rp
              else
                cext = -cx; cx = 0.0_rp
              endif
              tn = (1.0_rp - cx) * (real(tp(i, ny + 1, k, ns, itr), rp) &
                   - max(cy, 0.0_rp) * g1c - min(cy, 0.0_rp) * g1p) &
                   + cx * (real(tp(i, ny, k, nn, itr), rp) - max(cy, 0.0_rp) * g0c - min(cy, 0.0_rp) * g0p)
              text = 0.0_rp
              if (mir .and. itr == 1) text = t_north(i, k, isalt)
              if (mbry) text = tb_north(i, k, itr)
              tp(i, ny + 1, k, nn, itr) = real(((1.0_rp - cext) * tn + cext * text) * rmask(i, ny + 1), pp)
            enddo
          enddo
        enddo
      else
#ifdef USE_CUDA
        !$cuf kernel do(3) <<<*,*>>>
#endif
        do itr = 1, ntr
          do k = 1, nz
            do i = ilo, ihi
              tp(i, ny + 1, k, nn, itr) = tp(i, ny, k, nn, itr)
            enddo
          enddo
        enddo
      endif
    endif

    sw = decomp%edge_s .and. decomp%edge_w; se = decomp%edge_s .and. decomp%edge_e
    nw = decomp%edge_n .and. decomp%edge_w; ne = decomp%edge_n .and. decomp%edge_e
    if (.not. (sw .or. se .or. nw .or. ne)) return
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do itr = 1, ntr
      do k = 1, nz
        if (sw) then
          cff = rmask(1, 0) + rmask(0, 1)
          if (cff > 0.0_rp) then
            cff = 1.0_rp / cff
            tp(0, 0, k, nn, itr) = real(cff * (rmask(1, 0) * real(tp(1, 0, k, nn, itr), rp) + &
                 rmask(0, 1) * real(tp(0, 1, k, nn, itr), rp)), pp)
          else
            tp(0, 0, k, nn, itr) = 0.0_pp
          endif
        endif
        if (se) then
          cff = rmask(nx, 0) + rmask(nx + 1, 1)
          if (cff > 0.0_rp) then
            cff = 1.0_rp / cff
            tp(nx + 1, 0, k, nn, itr) = real(cff * (rmask(nx, 0) * real(tp(nx, 0, k, nn, itr), rp) + &
                 rmask(nx + 1, 1) * real(tp(nx + 1, 1, k, nn, itr), rp)), pp)
          else
            tp(nx + 1, 0, k, nn, itr) = 0.0_pp
          endif
        endif
        if (nw) then
          cff = rmask(1, ny + 1) + rmask(0, ny)
          if (cff > 0.0_rp) then
            cff = 1.0_rp / cff
            tp(0, ny + 1, k, nn, itr) = real(cff * (rmask(1, ny + 1) * real(tp(1, ny + 1, k, nn, itr), rp) + &
                 rmask(0, ny) * real(tp(0, ny, k, nn, itr), rp)), pp)
          else
            tp(0, ny + 1, k, nn, itr) = 0.0_pp
          endif
        endif
        if (ne) then
          cff = rmask(nx, ny + 1) + rmask(nx + 1, ny)
          if (cff > 0.0_rp) then
            cff = 1.0_rp / cff
            tp(nx + 1, ny + 1, k, nn, itr) = real(cff * (rmask(nx, ny + 1) * real(tp(nx, ny + 1, k, nn, itr), rp) + &
                 rmask(nx + 1, ny) * real(tp(nx + 1, ny, k, nn, itr), rp)), pp)
          else
            tp(nx + 1, ny + 1, k, nn, itr) = 0.0_pp
          endif
        endif
      enddo
    enddo
  end subroutine passive_t3dbc

end module passive_mod
