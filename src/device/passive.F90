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
  implicit none
  private

  integer(ip), public :: ntp = 0
  real(pp), WORK_ATTR public, allocatable :: tp(:,:,:,:,:)   ! (is:ie, js:je, nz, 3, ntp)
  character(len=16), public :: passive_init = 'zero'           ! zero | blob | mirror_salt
  logical, public :: passive_mirror = .false.
  logical, public :: wrt_passive = .true.

  public :: read_passive_namelist
  public :: alloc_passive
  public :: init_passive
  public :: passive_predictor
  public :: passive_corrector
  public :: passive_t3dmix
  public :: passive_t3dbc
  public :: passive_name

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
    character(len=8) :: name
    write(name, '(a,i2.2)') 'tp', itp
  end function passive_name

  subroutine alloc_passive()
    integer(ip) :: is, ie, js, je, nz
    if (ntp <= 0) return
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je; nz = decomp%nz
    if (allocated(tp)) deallocate(tp)
    allocate(tp(is:ie, js:je, nz, 3, ntp))
    tp = 0.0_pp
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
  subroutine passive_vertical(ts, nn, dtp, ctr)
    ! Column kernel with a sliding window (advect's vert_column_fused for tp).
    integer(ip), intent(in) :: ts, nn, ctr
    real(rp), intent(in) :: dtp
    integer(ip) :: i, j, k, itr, nx, ny, nz, ntr
    real(rp) :: dtpp, fz_lo, fz_k, tplus, tminus, rm, w
    real(rp) :: t_km1, t_k, t_kp1, t_kp2, Hz_km1, Hz_k, Hz_kp1, Hz_kp2
    logical :: at_bottom, at_surface
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = ntp
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        dtpp = dtp * pm(i, j) * pn(i, j)
        rm = rmask(i, j)
        do itr = 1, ntr
          fz_lo = 0.0_rp
          t_k = real(tp(i, j, 1, ts, itr), rp)
          t_kp1 = real(tp(i, j, 2, ts, itr), rp)
          Hz_k = Hz(i, j, 1)
          Hz_kp1 = Hz(i, j, 2)
          t_km1 = t_k
          Hz_km1 = Hz_k
          do k = 1, nz - 1
            at_bottom = (k == 1)
            at_surface = (k == nz - 1)
            if (at_surface) then
              t_kp2 = t_kp1
              Hz_kp2 = Hz_kp1
            else
              t_kp2 = real(tp(i, j, k + 2, ts, itr), rp)
              Hz_kp2 = Hz(i, j, k + 2)
            endif
            tplus = thickness_tplus(t_km1, t_k, t_kp1, Hz_km1, Hz_k, Hz_kp1, at_bottom)
            tminus = thickness_tminus(t_km1, t_k, t_kp1, t_kp2, Hz_km1, Hz_k, Hz_kp1, Hz_kp2, &
                                      at_bottom, at_surface)
            w = We(i, j, k)
            if (ctr == 1) then
              fz_k = 0.5_rp * (tplus + tminus) * w
            else
              fz_k = tplus * max(w, 0.0_rp) + tminus * min(w, 0.0_rp)
            endif
            fz_k = apply_mask(fz_k, rm)
            tp(i, j, k, nn, itr) = real(real(tp(i, j, k, nn, itr), rp) - dtpp * (fz_k - fz_lo), pp)
            fz_lo = fz_k
            t_km1 = t_k; t_k = t_kp1; t_kp1 = t_kp2
            Hz_km1 = Hz_k; Hz_k = Hz_kp1; Hz_kp1 = Hz_kp2
          enddo
          tp(i, j, nz, nn, itr) = real(real(tp(i, j, nz, nn, itr), rp) - dtpp * (0.0_rp - fz_lo), pp)
        enddo
      enddo
    enddo
  end subroutine passive_vertical

  ! --------------------------------------------- batched implicit solve
  subroutine passive_implicit(nn, dtau, use_fwd)
    ! Thomas solve of every passive column with the salinity diffusivity.
    ! The matrix (col_fc = 1/diag, col_cf = upper factor, col_wc =
    ! fc + max(wc,0)) is built once per column; each tracer then sweeps
    ! its own right-hand side through col_dc. Same arithmetic as
    ! implicit_tracer_columns. On entry tp(nn) is Hz*tp; on exit tp.
    integer(ip), intent(in) :: nn
    real(rp), intent(in) :: dtau
    logical, intent(in) :: use_fwd
    integer(ip) :: i, j, k, itr, nx, ny, nz, ntr, iAkt
    real(rp) :: cff, dc0, hz_k, hz_kp, fck, wck, cfk, fcm, wcm, cfm, dck, dcm, rm, tt
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = ntp
    iAkt = min(2_ip, int(size(Akt, 4), ip))
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*>>>
#endif
    do j = 1, ny
      do i = 1, nx
        rm = rmask(i, j)
        dc0 = dtau * pm(i, j) * pn(i, j)
        ! --- factorization
        if (use_fwd) then
          hz_k = Hz_fwd(i, j, 1); hz_kp = Hz_fwd(i, j, 2)
        else
          hz_k = Hz(i, j, 1); hz_kp = Hz(i, j, 2)
        endif
        fck = 2.0_rp * dtau * Akt(i, j, 1, iAkt) / (hz_kp + hz_k)
        wck = dc0 * Wi(i, j, 1)
        cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp))
        cfk = cff * (fck - min(wck, 0.0_rp))
        col_fc(i, j, 1) = cff
        col_cf(i, j, 1) = cfk
        col_wc(i, j, 1) = fck + max(wck, 0.0_rp)
        do k = 2, nz - 1
          fcm = fck; wcm = wck; cfm = cfk
          if (use_fwd) then
            hz_k = Hz_fwd(i, j, k); hz_kp = Hz_fwd(i, j, k + 1)
          else
            hz_k = Hz(i, j, k); hz_kp = Hz(i, j, k + 1)
          endif
          fck = 2.0_rp * dtau * Akt(i, j, k, iAkt) / (hz_k + hz_kp)
          wck = dc0 * Wi(i, j, k)
          cff = 1.0_rp / (hz_k + fck + max(wck, 0.0_rp) + &
                          fcm - min(wcm, 0.0_rp) - &
                          cfm * (fcm + max(wcm, 0.0_rp)))
          cfk = cff * (fck - min(wck, 0.0_rp))
          col_fc(i, j, k) = cff
          col_cf(i, j, k) = cfk
          col_wc(i, j, k) = fck + max(wck, 0.0_rp)
        enddo
        if (use_fwd) then
          hz_k = Hz_fwd(i, j, nz)
        else
          hz_k = Hz(i, j, nz)
        endif
        ! top: 1 / (hz + fc(nz-1) - min(wc(nz-1),0) - cf(nz-1)*(fc+max(wc,0))(nz-1))
        col_fc(i, j, nz) = 1.0_rp / (hz_k + fck - min(wck, 0.0_rp) - cfk * (fck + max(wck, 0.0_rp)))
        col_wc(i, j, nz) = fck + max(wck, 0.0_rp)
        ! --- sweeps, one tracer at a time
        do itr = 1, ntr
          dck = col_fc(i, j, 1) * real(tp(i, j, 1, nn, itr), rp)
          col_dc(i, j, 1) = dck
          do k = 2, nz - 1
            dcm = dck
            dck = col_fc(i, j, k) * (real(tp(i, j, k, nn, itr), rp) + dcm * col_wc(i, j, k - 1))
            col_dc(i, j, k) = dck
          enddo
          tt = apply_mask((real(tp(i, j, nz, nn, itr), rp) + dck * col_wc(i, j, nz - 1)) * col_fc(i, j, nz), rm)
          tp(i, j, nz, nn, itr) = real(tt, pp)
          do k = nz - 1, 1, -1
            tt = apply_mask(col_dc(i, j, k) + col_cf(i, j, k) * tt, rm)
            tp(i, j, k, nn, itr) = real(tt, pp)
          enddo
        enddo
      enddo
    enddo
  end subroutine passive_implicit

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
    call passive_vertical(ts, nn, dtau, 1_ip)
    call passive_implicit(nn, dtau, .true.)
  end subroutine passive_predictor

  ! --------------------------------------------------------- corrector
  subroutine passive_corrector(dt_in)
    ! Called from step3d_t after the T/S corrector: UP3 horizontal from
    ! slot nrhs (= 3), U3 vertical, (mirror: salinity forcing), implicit
    ! solve, open boundaries.
    real(rp), intent(in) :: dt_in
    integer(ip) :: i, j, k, itr, nx, ny, nz, ntr, ts, nn
    integer(ip) :: west, east, south, north
    real(rp) :: dtpp, dtl, cff
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
          do itr = 1, ntr
            tp(i, j, k, nn, itr) = real(real(tp(i, j, k, nn, itr), rp) - dtpp * &
                 (pup3_fx(i + 1, j, k, ts, itr, west, east, nx) - &
                  pup3_fx(i, j, k, ts, itr, west, east, nx) + &
                  pup3_fe(i, j + 1, k, ts, itr, south, north, ny) - &
                  pup3_fe(i, j, k, ts, itr, south, north, ny)), pp)
          enddo
        enddo
      enddo
    enddo
    call passive_vertical(ts, nn, dtl, 0_ip)
    if (passive_mirror) then
      ! Salinity's surface flux and KPP nonlocal transport for tracer 1.
#ifdef USE_CUDA
      !$cuf kernel do(2) <<<*,*>>>
#endif
      do j = 1, ny
        do i = 1, nx
          tp(i, j, nz, nn, 1) = real(real(tp(i, j, nz, nn, 1), rp) + dtl * stflx(i, j, isalt), pp)
          do k = nz - 1, 1, -1
            cff = -dtl * ghat(i, j, k) * stflx(i, j, isalt)
            tp(i, j, k + 1, nn, 1) = real(real(tp(i, j, k + 1, nn, 1), rp) - cff, pp)
            tp(i, j, k, nn, 1) = real(real(tp(i, j, k, nn, 1), rp) + cff, pp)
          enddo
        enddo
      enddo
    endif
    call passive_implicit(nn, dtl, .false.)
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
    logical :: sw, se, nw, ne, mir
    nx = decomp%nx; ny = decomp%ny; nz = decomp%nz; ntr = ntp
    ns = nstp; nn = nnew; nr = nrhs
    ilo = lbound(tp, 1); ihi = ubound(tp, 1)
    jlo = lbound(tp, 2); jhi = ubound(tp, 2)
    mir = passive_mirror
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
