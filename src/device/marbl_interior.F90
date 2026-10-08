#include "mem_attr.inc"
module marbl_interior_mod
  ! MARBL interior tendencies on the device, in split form:
  !   K1 carb_kernel     3-D pointwise: in-situ temperature, carbonate system
  !                      (two safeguarded Newton solves per point, pressure
  !                      corrected constants), CO3 saturation values. Writes
  !                      small 3-D arrays used by K2 and as diagnostics.
  !   K2 interior_kernel column kernel, one thread per (i,j), sweeping from
  !                      the surface down: PAR attenuation, the ecosystem
  !                      terms and the sinking-particulate recurrence are
  !                      streamed level by level with only scalar state
  !                      carried between levels (sink_t). Tendencies are
  !                      applied in place to the tracer array.
  ! Expressions follow marbl_interior_tendency_mod one for one. MARBL level
  ! k = 1 is the surface = ROMS level nz. All parameters come from the
  ! initialised MARBL instance (marbl_host_fill_par) through the managed
  ! table pr. Only the configuration MiniPac uses is ported: base biology,
  ! Lig_cnt = 1, one PAR subcolumn, no restoring, no scale-factor forcings,
  ! ciso/abio off.
  use kinds
  use gpu_state, only: nwet, wet_i, wet_j, ntpcol
  use decomp_mod, only: decomp
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  use marbl_host, only: marbl_par_t, MA, MZ, MP, NTRMAX, ND_IT, marbl_host_fill_par
  ! the carbonate helpers (co2_coeffs, ta_fn, drtsafe1, pcorr) are this
  ! module's own copies (#include below), so carb_kernel inlines them
  use marbl_sflux_mod, only: T0_Kelvin, salt_min, dic_min, alk_min, xacc, max_bracket_grow_it, maxit
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
  private

  ! Compile-time PFT counts of the configuration the kernels are built for
  ! (MARBL's CESM2 PFT defaults: sp, diat, diaz; zoo; three prey classes).
  ! Fixed counts let every per-PFT loop unroll and its arrays live in
  ! registers; set_marbl_par stops if the MARBL instance differs.
  integer, parameter :: NAUTO = 3, NZOO = 1, NPREY = 3

  ! Parameter table (set once by set_marbl_par): a host copy and, in the
  ! CUDA build, a constant-memory copy read by the kernels (uniform
  ! across a warp; a managed derived-type module variable gave
  ! illegal-address errors at run time).
  type(marbl_par_t), public, save :: pr
  ! derived flags: the temperature function of autotroph n (tf_same_a) or
  ! zooplankton n (tf_same_z) is the same expression on the same operands
  ! as autotroph 1's, so its value is reused
  type :: marbl_xpar_t
    integer(ip) :: tf_same_a(MA) = 0, tf_same_z(MZ) = 0
    ! 1: prey class p of every grazer is autotroph p alone (no zooplankton
    ! prey), so the grazing needs no gather/scatter over the prey lists
    integer(ip) :: graze_ident = 0
  end type marbl_xpar_t
  type(marbl_xpar_t), save :: px
#ifdef USE_CUDA
  type(marbl_par_t), constant :: prd
  type(marbl_xpar_t), constant :: pxd
#define PR prd
#define PX pxd
#define TRK pp
#else
#define PR pr
#define PX px
#define TRK rp
#endif

  ! K1 outputs (is:ie, js:je, nz): in-situ temperature, carbonate species,
  ! saturation CO3 for calcite and aragonite; pH saved state (base, alt).
  real(rp), WORK_ATTR allocatable, public :: it_temp(:,:,:), it_h2co3(:,:,:), it_co3(:,:,:), it_co3a(:,:,:)
  real(rp), WORK_ATTR allocatable, public :: it_co3sat(:,:,:), it_co3sata(:,:,:)
  real(rp), WORK_ATTR allocatable, public :: it_ph(:,:,:), it_pha(:,:,:)
  ! calcite / aragonite saturation depths (m, positive; MARBL zsatcalc, zsatarag)
  real(rp), WORK_ATTR allocatable, public :: it_zsatc(:,:), it_zsata(:,:)
  ! forcing: surface shortwave (W/m2)
  real(rp), WORK_ATTR allocatable, public :: it_qsw(:,:)
  ! optional (tests): tendencies and debug diagnostics kept per level
  real(rp), WORK_ATTR allocatable, public :: it_tend(:,:,:,:)   ! (i,j,k,m)
  real(rp), WORK_ATTR allocatable, public :: it_diag(:,:,:,:)   ! (i,j,k,ND_IT)
  integer(ip), WORK_ATTR allocatable, public :: it_err(:,:)     ! per column: solver failures

  ! constants (marbl_constants_mod)
  real(rp), parameter :: spd = 86400.0_rp, dps = 1.0_rp / spd, spy = 365.0_rp * spd, yps = 1.0_rp / spy
  real(rp), parameter :: mpercm = 0.01_rp, K_Boltz = 8.617330350e-5_rp, epsC = 1.0e-8_rp, epsTinv = 3.17e-8_rp
  real(rp), parameter :: PAR_threshold = 1.0e-19_rp
  real(rp), parameter :: KFeLig1 = 10.0e13_rp * 1.0e-6_rp
  real(rp), parameter :: c0 = 0.0_rp, c1 = 1.0_rp, c2 = 2.0_rp, c3 = 3.0_rp, c4 = 4.0_rp, c10 = 10.0_rp, p5 = 0.5_rp

  ! recurrence state carried from one level to the next
  type, public :: sink_t
    real(rp) :: par_surf, par_in      ! PAR at the surface and at the interface above the level
    real(rp) :: poc_s, poc_h, pop_s, pop_h, cac_s, cac_h, caca_s, caca_h
    real(rp) :: sio_s, sio_h, dst_s, dst_h, fe_s, fe_h
    real(rp) :: qa_dust_def
  end type sink_t

  public :: alloc_marbl_interior, free_marbl_interior, set_marbl_par
  public :: marbl_carbonate_compute, marbl_interior_compute
  public :: marbl_zsat_compute
#ifndef USE_CUDA
  public :: marbl_interior_column_host
#endif

contains

  subroutine set_marbl_par()
    type(marbl_par_t) :: h
    type(marbl_xpar_t) :: x
    integer(ip) :: n, m, prey, pred, cnt(NTRMAX)
    logical :: ok
    call marbl_host_fill_par(h)
    ! the kernels are compiled for NAUTO/NZOO/NPREY PFTs
    if (h%nauto /= NAUTO .or. h%nzoo /= NZOO .or. h%nprey /= NPREY) then
      write(*,'(a,3i4,a,3i4)') 'set_marbl_par: MARBL has nauto, nzoo, nprey =', h%nauto, h%nzoo, h%nprey, &
           '; the interior kernel is built for', NAUTO, NZOO, NPREY
      error stop 'set_marbl_par: change NAUTO/NZOO/NPREY in marbl_interior.F90'
    endif
    do pred = 1, NZOO
      do prey = 1, NPREY
        ok = h%g_nauto(prey, pred) >= 0 .and. h%g_nauto(prey, pred) <= NAUTO .and. &
             h%g_nzoo(prey, pred) >= 0 .and. h%g_nzoo(prey, pred) <= NZOO
        if (ok) then
          do n = 1, h%g_nauto(prey, pred)
            ok = ok .and. h%g_auto(n, prey, pred) >= 1 .and. h%g_auto(n, prey, pred) <= NAUTO
          enddo
          do n = 1, h%g_nzoo(prey, pred)
            ok = ok .and. h%g_zoo(n, prey, pred) >= 1 .and. h%g_zoo(n, prey, pred) <= NZOO
          enddo
        endif
        if (.not. ok) error stop 'set_marbl_par: grazing prey lists outside NAUTO/NZOO'
      enddo
    enddo
    ! Every tracer must receive exactly one tendency: the kernel updates
    ! only the tracers with a tendency (the old loop also rewrote the
    ! others as tracer + 0*dt) and the host path returns them in place.
    cnt = 0
    call mark(h%po4); call mark(h%no3); call mark(h%sio3); call mark(h%nh4); call mark(h%fe)
    call mark(h%lig); call mark(h%o2); call mark(h%dic); call mark(h%dica); call mark(h%alk)
    call mark(h%alka); call mark(h%doc); call mark(h%don); call mark(h%dop); call mark(h%dopr)
    call mark(h%donr); call mark(h%docr)
    do n = 1, NZOO
      call mark(h%z_c(n))
    enddo
    do n = 1, NAUTO
      call mark(h%a_c(n)); call mark(h%a_chl(n)); call mark(h%a_fe(n))
      if (h%a_p(n) > 0) call mark(h%a_p(n))
      if (h%a_si(n) > 0) call mark(h%a_si(n))
      if (h%a_caco3(n) > 0) call mark(h%a_caco3(n))
    enddo
    do m = 1, h%ntr
      if (cnt(m) /= 1) error stop 'set_marbl_par: a MARBL tracer has no (or two) tendencies'
    enddo
    ok = (NPREY <= NAUTO)
    do pred = 1, NZOO
      do prey = 1, NPREY
        ok = ok .and. h%g_nauto(prey, pred) == 1 .and. h%g_nzoo(prey, pred) == 0
        if (ok) ok = (h%g_auto(1, prey, pred) == prey)
      enddo
    enddo
    if (ok) x%graze_ident = 1
    ! temperature-function reuse: same form; Tref for forms 1 and 2; Ea for form 2
    do n = 1, NAUTO
      if (same_tfunc(h%a_tfunc(n), h%a_Tref(n), h%a_Ea(n))) x%tf_same_a(n) = 1
    enddo
    do n = 1, NZOO
      if (same_tfunc(h%z_tfunc(n), h%z_Tref(n), h%z_Ea(n))) x%tf_same_z(n) = 1
    enddo
    pr = h
    px = x
#ifdef USE_CUDA
    prd = h
    pxd = x
#endif
  contains
    subroutine mark(m)
      integer(ip), intent(in) :: m
      if (m < 1 .or. m > h%ntr) error stop 'set_marbl_par: tracer index out of range'
      cnt(m) = cnt(m) + 1
    end subroutine mark
    logical function same_tfunc(iopt, Tref, Ea)
      integer(ip), intent(in) :: iopt
      real(rp), intent(in) :: Tref, Ea
      same_tfunc = (iopt == h%a_tfunc(1))
      if (iopt == 1 .or. iopt == 2) same_tfunc = same_tfunc .and. (Tref == h%a_Tref(1))
      if (iopt == 2) same_tfunc = same_tfunc .and. (Ea == h%a_Ea(1))
    end function same_tfunc
  end subroutine set_marbl_par

  subroutine alloc_marbl_interior(nz, ntr, keep)
    integer(ip), intent(in) :: nz, ntr
    logical, intent(in) :: keep      ! keep tendencies and diagnostics (tests)
    integer(ip) :: is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call free_marbl_interior()
    allocate(it_temp(is:ie,js:je,nz), it_h2co3(is:ie,js:je,nz), it_co3(is:ie,js:je,nz), it_co3a(is:ie,js:je,nz))
    allocate(it_co3sat(is:ie,js:je,nz), it_co3sata(is:ie,js:je,nz), it_ph(is:ie,js:je,nz), it_pha(is:ie,js:je,nz))
    allocate(it_qsw(is:ie,js:je), it_err(is:ie,js:je), it_zsatc(is:ie,js:je), it_zsata(is:ie,js:je))
    it_temp = c0; it_h2co3 = c0; it_co3 = c0; it_co3a = c0; it_co3sat = c0; it_co3sata = c0
    it_ph = c0; it_pha = c0; it_qsw = c0; it_err = 0; it_zsatc = c0; it_zsata = c0
    if (keep) then
      allocate(it_tend(is:ie,js:je,nz,ntr), it_diag(is:ie,js:je,nz,ND_IT))
      it_tend = c0; it_diag = c0
    endif
  end subroutine alloc_marbl_interior

  subroutine free_marbl_interior()
    if (allocated(it_temp)) deallocate(it_temp, it_h2co3, it_co3, it_co3a, it_co3sat, it_co3sata, it_ph, it_pha)
    if (allocated(it_qsw)) deallocate(it_qsw, it_err, it_zsatc, it_zsata)
    if (allocated(it_tend)) deallocate(it_tend, it_diag)
  end subroutine free_marbl_interior

  !===========================================================================
  ! device helpers
  !===========================================================================

#include "marbl_carb_helpers.inc"

  ! marbl_temperature: adiabatic temperature gradient (degC/bar)
  DEVICE_ATTR
  subroutine adtg(salt, temp, press, g)
    real(rp), intent(in) :: salt, temp, press
    real(rp), intent(out) :: g
    real(rp) :: saltm35
    real(rp), parameter :: bar_to_dbar = 10.0_rp, pk = 1.0e-2_rp
    real(rp), parameter :: a000 = 0.35803e-1_rp * pk, a001 = 0.85258e-2_rp * pk, a002 = -0.68360e-4_rp * pk, &
         a003 = 0.66228e-6_rp * pk, a010 = 0.18932e-2_rp * pk, a011 = -0.42393e-4_rp * pk, &
         a100 = 0.18741e-4_rp * pk * bar_to_dbar, a101 = -0.67795e-6_rp * pk * bar_to_dbar, &
         a102 = 0.87330e-8_rp * pk * bar_to_dbar, a103 = -0.54481e-10_rp * pk * bar_to_dbar, &
         a110 = -0.11351e-6_rp * pk * bar_to_dbar, a111 = 0.27759e-8_rp * pk * bar_to_dbar, &
         a200 = -0.46206e-9_rp * pk * bar_to_dbar * bar_to_dbar, a201 = 0.18676e-10_rp * pk * bar_to_dbar * bar_to_dbar, &
         a202 = -0.21687e-12_rp * pk * bar_to_dbar * bar_to_dbar
    saltm35 = salt - 35.0_rp
    g = a000 + temp * (a001 + temp * (a002 + temp * a003)) &
      + saltm35 * (a010 + temp * a011) &
      + press * (a100 + temp * (a101 + temp * (a102 + temp * a103)) &
                 + saltm35 * (a110 + temp * a111) &
                 + press * (a200 + temp * (a201 + temp * a202)))
  end subroutine adtg

  ! marbl_temperature_potemp(salt, temp, press, ref_press): MARBL calls it
  ! with temp = potential temperature, press = 0, ref_press = pressure to
  ! get the in-situ temperature. Subroutine with by-reference dummies (the
  ! function form was miscompiled by nvfortran 25.9's inliner).
  DEVICE_ATTR
  subroutine potemp_s(salt, temp, press, ref_press, tout)
    real(rp), intent(in) :: salt, temp, press, ref_press
    real(rp), intent(out) :: tout
    real(rp) :: dpress, press_mid, dtemp, temp_arg, qq, g
    real(rp) :: sq2, q2w1, q2w2, q3w1, q3w2, t2w, t3w
    real(rp), parameter :: sixth = c1 / 6.0_rp
    sq2  = sqrt(c2)
    q2w1 = (c2 - sq2); q2w2 = (-c2 + c3 / sq2)
    q3w1 = (c2 + sq2); q3w2 = (-c2 - c3 / sq2)
    t2w  = (c1 - c1 / sq2); t3w = (c1 + c1 / sq2)
    dpress    = ref_press - press
    press_mid = press + p5 * dpress
    call adtg(salt, temp, press, g)
    dtemp     = dpress * g
    temp_arg  = temp + p5 * dtemp
    qq        = dtemp
    call adtg(salt, temp_arg, press_mid, g)
    dtemp     = dpress * g
    temp_arg  = temp_arg + t2w * (dtemp - qq)
    qq        = q2w1 * dtemp + q2w2 * qq
    call adtg(salt, temp_arg, press_mid, g)
    dtemp     = dpress * g
    temp_arg  = temp_arg + t3w * (dtemp - qq)
    qq        = q3w1 * dtemp + q3w2 * qq
    call adtg(salt, temp_arg, ref_press, g)
    dtemp     = dpress * g
    tout = temp_arg + sixth * (dtemp - c2 * qq)
  end subroutine potemp_s

  ! marbl_co2calc_interior (one element): returns ph, H2CO3, HCO3, CO3 in
  ! model concentration units. ierr /= 0: solver failure (outputs zero,
  ! ph unchanged).
  DEVICE_ATTR
  subroutine co2calc_interior1(dic_in, ta_in, pt_in, sit_in, mass_to_vol, &
                               k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, &
                               phlo, phhi, ph, h2co3, hco3, co3, ierr)
    real(rp), intent(in) :: dic_in, ta_in, pt_in, sit_in, mass_to_vol
    real(rp), intent(in) :: k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, phlo, phhi
    real(rp), intent(inout) :: ph
    real(rp), intent(out) :: h2co3, hco3, co3
    integer(ip), intent(out) :: ierr
    real(rp) :: vol_to_mass, dic, ta, pt, sit, x1, x2, htotal, htotal2, denom
    vol_to_mass = c1 / mass_to_vol
    dic = max(dic_in, dic_min) * vol_to_mass
    ta  = max(ta_in, alk_min)  * vol_to_mass
    pt  = max(pt_in, c0)       * vol_to_mass
    sit = max(sit_in, c0)      * vol_to_mass
    x1 = c10 ** (-phhi)
    x2 = c10 ** (-phlo)
    call drtsafe1(k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit, &
                  x1, x2, htotal, ierr)
    h2co3 = c0; hco3 = c0; co3 = c0
    if (ierr /= 0) return
    htotal2 = htotal ** 2
    denom   = c1 / (htotal2 + k1 * htotal + k1 * k2)
    h2co3   = dic * htotal2 * denom
    hco3    = dic * k1 * htotal * denom
    co3     = dic * k1 * k2 * denom
    ph      = -log10(htotal)
    h2co3   = h2co3 * mass_to_vol
    hco3    = hco3 * mass_to_vol
    co3     = co3 * mass_to_vol
  end subroutine co2calc_interior1

  ! marbl_co2calc_co3_sat_vals (one element)
  DEVICE_ATTR
  subroutine co3_sat1(temp, salt, press_bar, mass_to_vol, co3_sat_calc, co3_sat_arag)
    real(rp), intent(in) :: temp, salt, press_bar, mass_to_vol
    real(rp), intent(out) :: co3_sat_calc, co3_sat_arag
    real(rp) :: salt_lim, tk, log10tk, invtk, sqrts, s15, invRtk, arg, K_calc, K_arag, inv_Ca, kfac
    salt_lim = max(salt, salt_min)
    tk       = T0_Kelvin + temp
    log10tk  = log10(tk)
    invtk    = c1 / tk
    invRtk   = (c1 / 83.1451_rp) * invtk
    sqrts    = sqrt(salt_lim)
    s15      = sqrts * salt_lim
    arg = -171.9065_rp - 0.077993_rp * tk + 2839.319_rp * invtk + 71.595_rp * log10tk + &
         (-0.77712_rp + 0.0028426_rp * tk + 178.34_rp * invtk) * sqrts - &
         0.07711_rp * salt_lim + 0.0041249_rp * s15
    arg = log(c10) * arg
    K_calc = exp(arg)
    call pcorr(temp, invRtk, press_bar, -48.76_rp, 0.5304_rp, c0, -11.76_rp, 0.3692_rp, c0, kfac)
    K_calc = K_calc * kfac
    arg = -171.945_rp - 0.077993_rp * tk + 2903.293_rp * invtk + 71.595_rp * log10tk + &
         (-0.068393_rp + 0.0017276_rp * tk + 88.135_rp * invtk) * sqrts - &
         0.10018_rp * salt_lim + 0.0059415_rp * s15
    arg = log(c10) * arg
    K_arag = exp(arg)
    call pcorr(temp, invRtk, press_bar, -45.96_rp, 0.5304_rp, c0, -11.76_rp, 0.3692_rp, c0, kfac)
    K_arag = K_arag * kfac
    inv_Ca = (35.0_rp / 0.01028_rp) / salt_lim
    co3_sat_calc = K_calc * inv_Ca
    co3_sat_arag = K_arag * inv_Ca
    co3_sat_calc = co3_sat_calc * mass_to_vol
    co3_sat_arag = co3_sat_arag * mass_to_vol
  end subroutine co3_sat1

  ! compute_temperature_functional_form (one PFT, one level)
  DEVICE_ATTR
  subroutine tfunc1(iopt, temperature, Tref, Ea, Q_10, f)
    integer(ip), intent(in) :: iopt
    real(rp), intent(in) :: temperature, Tref, Ea, Q_10
    real(rp), intent(out) :: f
    if (iopt == 1) then
      f = Q_10 ** (((temperature + T0_Kelvin) - (Tref + T0_Kelvin)) / c10)
    else if (iopt == 2) then
      f = exp(-Ea * (Tref - temperature) / (K_Boltz * (temperature + T0_Kelvin) * (Tref + T0_Kelvin)))
    else
      f = 0.12_rp * (max(c0, min(temperature, 27.0_rp)) ** 0.4_rp)
    endif
  end subroutine tfunc1

  ! Tendency t of tracer m: applied in place (lapply /= 0: tracer + t*dtm,
  ! the model update) or stored as the tendency itself (lapply = 0, host
  ! reference path); lsave /= 0 also keeps it in it_tend (tests).
  DEVICE_ATTR
  subroutine emit_tend(trv, m, t, dtm, lapply, lsave, i, j, kk)
    real(TRK), intent(inout) :: trv(NTRMAX)
    integer(ip), intent(in) :: m, lapply, lsave, i, j, kk
    real(rp), intent(in) :: t, dtm
    if (lapply /= 0) then
      trv(m) = real(real(trv(m), rp) + t * dtm, TRK)
    else
      trv(m) = real(t, TRK)
    endif
    if (lsave /= 0) it_tend(i, j, kk, m) = t
  end subroutine emit_tend

  !===========================================================================
  ! One level of the interior ecosystem for one column (MARBL level k; k = 1
  ! is the surface). State st is updated for the next level.
  ! trv: the column's tracers at this level (in the kernel the tracer array
  ! itself; tracers are read once at the start, then each tendency is
  ! emitted with emit_tend). PFT counts are the compile-time NAUTO/NZOO/
  ! NPREY and every per-PFT array is indexed by unrolled loop counters only,
  ! so the arrays live in registers; the indirect grazing indices are
  ! resolved by selects in the same accumulation order as MARBL.
  !===========================================================================
  DEVICE_ATTR
  subroutine level_step(k, kmt, dz, dz1, zw, zt, tins, h2co3, co3, co3a, co3sat, dust_flux, &
                        trv, st, dtm, lapply, dg, ldg, lsave, i, j, kk)
    integer(ip), intent(in) :: k, kmt, lapply, ldg, lsave, i, j, kk
    real(rp), intent(in) :: dz, dz1, zw, zt, tins, h2co3, co3, co3a, co3sat, dust_flux, dtm
    real(TRK), intent(inout) :: trv(NTRMAX)
    type(sink_t), intent(inout) :: st
    real(rp), intent(out) :: dg(ND_IT)

    integer(ip) :: m, n, jj, pred, prey, na, imp_n
    ! autotroph locals and derived terms
    real(rp) :: aChl(NAUTO), aC(NAUTO), aP(NAUTO), aFe(NAUTO), aSi(NAUTO), aCaCO3(NAUTO)
    real(rp) :: Tfa(NAUTO), Tfz(NZOO), thetaC(NAUTO), Qp(NAUTO), gQp(NAUTO), Qfe(NAUTO), gQfe(NAUTO)
    real(rp) :: Qsi(NAUTO), gQsi(NAUTO), QCaCO3(NAUTO)
    real(rp) :: Pprime(NAUTO), VNO3(NAUTO), VNH4(NAUTO), VNtot(NAUTO), VFe(NAUTO), f_nut(NAUTO)
    real(rp) :: VPO4(NAUTO), VDOP(NAUTO), VPtot(NAUTO)
    real(rp) :: VSiO3(NAUTO), VCO2(NAUTO), light_lim(NAUTO), PCphoto(NAUTO), photoacc(NAUTO), photoC(NAUTO)
    real(rp) :: NO3_V(NAUTO), NH4_V(NAUTO), PO4_V(NAUTO), DOP_V(NAUTO), photoFe(NAUTO), photoSi(NAUTO)
    real(rp) :: CaCO3_form(NAUTO)
    real(rp) :: Nfix(NAUTO), Nexcrete(NAUTO), auto_loss(NAUTO), auto_agg(NAUTO), auto_loss_poc(NAUTO)
    real(rp) :: auto_loss_doc(NAUTO)
    real(rp) :: auto_loss_dic(NAUTO), auto_graze(NAUTO), auto_graze_zootot(NAUTO), auto_graze_poc(NAUTO)
    real(rp) :: auto_graze_doc(NAUTO), auto_graze_dic(NAUTO), remaining_P_pop(NAUTO), remaining_P_dop(NAUTO)
    real(rp) :: remaining_P_dip(NAUTO)
    ! zooplankton
    real(rp) :: zooC(NZOO), Zprime(NZOO), zoo_loss(NZOO), zoo_loss_bulk(NZOO), zoo_loss_basal(NZOO)
    real(rp) :: zoo_graze(NZOO)
    real(rp) :: zoo_graze_zootot(NZOO), zoo_graze_poc(NZOO), zoo_graze_doc(NZOO), zoo_graze_dic(NZOO)
    real(rp) :: x_graze_zoo(NZOO)
    real(rp) :: f_zoo_detr(NZOO), zoo_loss_poc(NZOO), zoo_loss_doc(NZOO), zoo_loss_dic(NZOO)
    ! scalars
    real(rp) :: totChl, work1, work2, work3, work4, kpardz, wpar, par_k, par_avg
    real(rp) :: f_loss_thres, C_loss_thres, PCmax, lls, pcs, pchl, pacc, picpoc, remaining_P
    real(rp) :: graze_rate, work, auto_sum
    real(rp) :: DOC_prod, DON_prod, DOP_prod, DOC_reminR, DON_reminR, DOP_reminR, DOCr_reminR, DONr_reminR, DOPr_reminR
    real(rp) :: DOC_remin, DON_remin, DOP_remin, DOCr_remin, DONr_remin, DOPr_remin
    real(rp) :: Fefree, FeLig1, sinking_mass, Fe_scavenge_rate, Lig_scavenge_rate, FeLig_scavenge_rate
    real(rp) :: Lig_scavenge, Fe_scavenge, p0, p1, p2
    real(rp) :: POC_prod, POP_prod, cac_prod, sio_prod, fe_prod, DOP_loss_P_bal
    real(rp) :: poc_diss, sio2_diss, caco3_diss, dust_diss, DECAY_Hard, DECAY_HardDust, decay_POC_E, decay_SiO2
    real(rp) :: decay_CaCO3, decay_dust, POC_PROD_avail, new_QA_dust_def, scalelength, o2_scalefactor, dzr
    real(rp) :: cac_so, cac_ho, caca_so, caca_ho, sio_so, sio_ho, dst_so, dst_ho, poc_so, poc_ho, fe_so, fe_ho
    real(rp) :: pop_so, pop_ho
    real(rp) :: cac_remin, caca_remin, sio_remin, poc_remin, pon_remin, dst_remin, fe_remin, pop_remin
    real(rp) :: to_floor, flux_alt, bury_frac, sed_denitrif, other_remin, PON_sed_loss
    real(rp) :: poc_sed, pop_sed, sio_sed, cac_sed, caca_sed, fe_sed, dst_sed, bftt
    real(rp) :: Lig_prod, Lig_photochem, Lig_deg, Lig_loss, rate_per_sec, nitrif, denitrif
    real(rp) :: o2_production, o2_consumption, o2_production_denom, s_zld, s_zgd, s_ald, s_agd, s_zldoc, s_zgdoc
    real(rp) :: cac_rho, sio_rho, dst_rho, dst_gamma, poc_mass, cac_mass, sio_mass, dst_mass, cac_gamma, sio_gamma
    real(rp) :: dst_diss0, Fe_desorb_in
    real(rp) :: O2l, NO3l, NH4l, PO4l, SiO3l, Fel, Ligl, DOCl, DONl, DOPl, DOPrl, DONrl, DOCrl
    ! grazing gather/scatter
    real(rp) :: pp_n, qc_n, gzt, gpoc, gdoc
    ! PFT sums (zero-started, PFT order) and per-PFT terms of later sums
    real(rp) :: s_no3v, s_nh4v, s_po4v, s_dopv, s_photoFe, s_aldoc, s_aldic, s_alpoc, s_aagg
    real(rp) :: s_agdoc, s_rpdop, s_agpoc, s_rppop, s_agdic, s_rpdip
    real(rp) :: s_zlpoc, s_zgpoc, s_zldic, s_zgdic
    real(rp) :: afe(NAUTO), bfe(NAUTO), cfe(NAUTO), xsi(NAUTO), ydic(NAUTO), zalk(NAUTO)
    ! tendencies of the named tracers
    real(rp) :: t_no3, t_nh4, t_fe, t_lig, t_sio3, t_po4, t_doc, t_docr, t_don, t_donr, t_dop, t_dopr
    real(rp) :: t_dic, t_dica, t_alk, t_alka, t_o2, photoC_tot

    !--- setup_local_tracers (max(0,.) of each tracer read)
    PO4l = max(c0, real(trv(PR%po4), rp)); NO3l = max(c0, real(trv(PR%no3), rp))
    SiO3l = max(c0, real(trv(PR%sio3), rp)); NH4l = max(c0, real(trv(PR%nh4), rp))
    Fel = max(c0, real(trv(PR%fe), rp)); Ligl = max(c0, real(trv(PR%lig), rp))
    O2l = max(c0, real(trv(PR%o2), rp)); DOCl = max(c0, real(trv(PR%doc), rp))
    DONl = max(c0, real(trv(PR%don), rp)); DOPl = max(c0, real(trv(PR%dop), rp))
    DOPrl = max(c0, real(trv(PR%dopr), rp)); DONrl = max(c0, real(trv(PR%donr), rp))
    DOCrl = max(c0, real(trv(PR%docr), rp))
    do n = 1, NZOO
      zooC(n) = max(c0, real(trv(PR%z_c(n)), rp))
    enddo
    do n = 1, NAUTO
      aChl(n) = max(c0, real(trv(PR%a_chl(n)), rp))
      aC(n) = max(c0, real(trv(PR%a_c(n)), rp))
      if (PR%a_p(n) > 0) then
        aP(n) = max(c0, real(trv(PR%a_p(n)), rp))
      else
        aP(n) = PR%a_Qp_fixed(n) * aC(n)
      endif
      aFe(n) = max(c0, real(trv(PR%a_fe(n)), rp))
      aSi(n) = c0
      if (PR%a_si(n) > 0) aSi(n) = max(c0, real(trv(PR%a_si(n)), rp))
      aCaCO3(n) = c0
      if (PR%a_caco3(n) > 0) aCaCO3(n) = max(c0, real(trv(PR%a_caco3(n)), rp))
      ! autotroph_zero_consistency_enforce
      if (aChl(n) == c0 .or. aC(n) == c0 .or. aP(n) == c0 .or. aFe(n) == c0 .or. &
          (PR%a_si(n) > 0 .and. aSi(n) == c0)) then
        aChl(n) = c0; aC(n) = c0; aP(n) = c0; aFe(n) = c0; aSi(n) = c0; aCaCO3(n) = c0
      endif
    enddo

    !--- compute_PAR (one subcolumn)
    if (st%par_surf == c0) then
      kpardz = c0; par_k = c0; par_avg = c0
    else
      totChl = c0
      do n = 1, NAUTO
        totChl = totChl + aChl(n)
      enddo
      work1 = max(totChl, 0.02_rp)
      if (work1 < 0.13224_rp) then
        kpardz = (0.0919_rp * PR%len2m) * (work1 ** 0.3536_rp)
      else
        kpardz = (0.1131_rp * PR%len2m) * (work1 ** 0.4562_rp)
      endif
      kpardz = kpardz * dz
      wpar = exp(-kpardz)
      if (st%par_in > c0) then
        par_k = st%par_in * wpar
        if (par_k < PAR_threshold) par_k = c0
      else
        par_k = c0
      endif
      par_avg = st%par_in * (c1 - wpar) / kpardz
    endif

    !--- temperature functions (a PFT whose functional form and parameters
    ! equal autotroph 1's reuses that result: the same expression on the
    ! same operands), elemental ratios, Pprime
    call tfunc1((PR%a_tfunc(1)), tins, (PR%a_Tref(1)), (PR%a_Ea(1)), (PR%Q_10), Tfa(1))
    do n = 2, NAUTO
      if (PX%tf_same_a(n) /= 0) then
        Tfa(n) = Tfa(1)
      else
        call tfunc1((PR%a_tfunc(n)), tins, (PR%a_Tref(n)), (PR%a_Ea(n)), (PR%Q_10), Tfa(n))
      endif
    enddo
    do n = 1, NZOO
      if (PX%tf_same_z(n) /= 0) then
        Tfz(n) = Tfa(1)
      else
        call tfunc1((PR%z_tfunc(n)), tins, (PR%z_Tref(n)), (PR%z_Ea(n)), (PR%Q_10), Tfz(n))
      endif
    enddo
    f_loss_thres = min(max((PR%thres_z2_auto - zt) / (PR%thres_z2_auto - PR%thres_z1_auto), c0), c1)
    ! One loop over the PFTs for the per-PFT terms (MARBL's separate loops
    ! fused: each PFT's terms depend on that PFT only), with the PFT sums
    ! MARBL forms later accumulated here in the same order (from zero).
    s_no3v = c0; s_nh4v = c0; s_po4v = c0; s_dopv = c0; photoC_tot = c0; s_photoFe = c0
    s_aldoc = c0; s_aldic = c0; s_alpoc = c0; s_aagg = c0; o2_production = c0
    !dir$ unroll
    do n = 1, NAUTO
      thetaC(n) = aChl(n) / (aC(n) + epsC)
      if (PR%lvar_ptoc /= 0) then
        Qp(n) = aP(n) / (aC(n) + epsC)
        gQp(n) = min((((PR%PquotaSlope * PO4l) + PR%PquotaIntercept) * 0.001_rp), PR%PquotaMinNP)
      else
        Qp(n) = PR%a_Qp_fixed(n)
        gQp(n) = PR%a_Qp_fixed(n)
      endif
      Qfe(n) = aFe(n) / (aC(n) + epsC)
      gQfe(n) = PR%a_gQfe_0(n)
      if (Fel < PR%gQ_Fe_kFe_thres * PR%a_kFe(n)) then
        gQfe(n) = max((gQfe(n) * Fel / (PR%gQ_Fe_kFe_thres * PR%a_kFe(n))), PR%a_gQfe_min(n))
      endif
      Qsi(n) = c0; gQsi(n) = c0
      if (PR%a_si(n) > 0) then
        Qsi(n) = min(aSi(n) / (aC(n) + epsC), PR%gQsi_max)
        gQsi(n) = PR%gQsi_0
        if ((Fel < PR%gQ_Fe_kFe_thres * PR%a_kFe(n)) .and. (Fel > c0) .and. &
            (SiO3l > (PR%gQ_Si_kSi_thres * PR%a_kSiO3(n)))) then
          gQsi(n) = min((gQsi(n) * PR%gQ_Fe_kFe_thres * PR%a_kFe(n) / Fel), PR%gQsi_max)
        else if (Fel == c0) then
          gQsi(n) = PR%gQsi_max
        endif
        if (SiO3l < (PR%gQ_Si_kSi_thres * PR%a_kSiO3(n))) then
          gQsi(n) = max((gQsi(n) * SiO3l / (PR%gQ_Si_kSi_thres * PR%a_kSiO3(n))), PR%gQsi_min)
        endif
      endif
      QCaCO3(n) = c0
      if (PR%a_caco3(n) > 0) QCaCO3(n) = min(aCaCO3(n) / (aC(n) + epsC), PR%QCaCO3_max)
      if (tins < PR%a_temp_thres(n)) then
        C_loss_thres = f_loss_thres * PR%a_loss_thres2(n)
      else
        C_loss_thres = f_loss_thres * PR%a_loss_thres(n)
      endif
      Pprime(n) = max(aC(n) - C_loss_thres, c0)
      VNO3(n) = (NO3l / PR%a_kNO3(n)) / (c1 + (NO3l / PR%a_kNO3(n)) + (NH4l / PR%a_kNH4(n)))
      VNH4(n) = (NH4l / PR%a_kNH4(n)) / (c1 + (NO3l / PR%a_kNO3(n)) + (NH4l / PR%a_kNH4(n)))
      if (PR%a_nfixer(n) /= 0) then
        VNtot(n) = c1
      else
        VNtot(n) = VNO3(n) + VNH4(n)
      endif
      VFe(n) = Fel / (Fel + PR%a_kFe(n))
      f_nut(n) = min(VNtot(n), VFe(n))
      VPO4(n) = (PO4l / PR%a_kPO4(n)) / (c1 + (PO4l / PR%a_kPO4(n)) + (DOPl / PR%a_kDOP(n)))
      VDOP(n) = (DOPl / PR%a_kDOP(n)) / (c1 + (PO4l / PR%a_kPO4(n)) + (DOPl / PR%a_kDOP(n)))
      VPtot(n) = VPO4(n) + VDOP(n)
      f_nut(n) = min(f_nut(n), VPtot(n))
      VSiO3(n) = c0
      if (PR%a_silic(n) /= 0) then
        VSiO3(n) = SiO3l / (SiO3l + PR%a_kSiO3(n))
        f_nut(n) = min(f_nut(n), VSiO3(n))
      endif
      VCO2(n) = c0
      if (PR%a_climit(n) /= 0) then
        VCO2(n) = h2co3 / (h2co3 + PR%a_kCO2(n))
        f_nut(n) = min(f_nut(n), VCO2(n))
      endif
      if (tins < PR%a_temp_thres(n)) then
        PCmax = c0
      else
        PCmax = PR%a_PCref(n) * f_nut(n) * Tfa(n)
      endif
      light_lim(n) = c0; PCphoto(n) = c0; photoacc(n) = c0; photoC(n) = c0
      if (thetaC(n) > c0) then
        work = PR%a_alphaPI(n) * thetaC(n) * par_avg
        if (work > c0) then
          lls = (c1 - exp(-work / (PCmax + epsTinv)))
          pcs = PCmax * lls
          pchl = PR%a_thetaN_max(n) * pcs / work
          pacc = (pchl * pcs * PR%Q / thetaC(n)) * aChl(n)
          light_lim(n) = light_lim(n) + lls
          PCphoto(n) = PCphoto(n) + pcs
          photoacc(n) = photoacc(n) + pacc
        endif
        photoC(n) = PCphoto(n) * aC(n)
      endif
      if (VNtot(n) > c0) then
        NO3_V(n) = (VNO3(n) / VNtot(n)) * photoC(n) * PR%Q
        NH4_V(n) = (VNH4(n) / VNtot(n)) * photoC(n) * PR%Q
      else
        NO3_V(n) = c0; NH4_V(n) = c0
      endif
      if (VPtot(n) > c0) then
        PO4_V(n) = (VPO4(n) / VPtot(n)) * photoC(n) * gQp(n)
        DOP_V(n) = (VDOP(n) / VPtot(n)) * photoC(n) * gQp(n)
      else
        PO4_V(n) = c0; DOP_V(n) = c0
      endif
      photoFe(n) = photoC(n) * gQfe(n)
      photoSi(n) = c0
      if (PR%a_si(n) > 0) photoSi(n) = photoC(n) * gQsi(n)
      CaCO3_form(n) = c0
      if (PR%a_impcalc(n) /= 0) then
        CaCO3_form(n) = PR%f_prod_sp_CaCO3 * photoC(n)
        CaCO3_form(n) = CaCO3_form(n) * f_nut(n) * f_nut(n)
        if (tins < PR%CaCO3_temp_thres1) then
          CaCO3_form(n) = CaCO3_form(n) * max((tins - PR%CaCO3_temp_thres2), c0) / &
               (PR%CaCO3_temp_thres1 - PR%CaCO3_temp_thres2)
        endif
        if (aC(n) > PR%CaCO3_sp_thres) then
          CaCO3_form(n) = min((CaCO3_form(n) * aC(n) / PR%CaCO3_sp_thres), (PR%f_photosp_CaCO3 * photoC(n)))
        endif
      else if (PR%a_expcalc(n) /= 0) then
        picpoc = min(1.0_rp, max(0.0_rp, 0.0001_rp * tins**3))
        picpoc = max(0.0_rp, -0.0136_rp * h2co3 + picpoc + 0.21_rp)
        picpoc = min(1.0_rp, max(0.0_rp, -0.48_rp * VPO4(n) + picpoc + 0.48_rp))
        CaCO3_form(n) = picpoc * photoC(n)
      endif
      Nfix(n) = c0; Nexcrete(n) = c0
      if (PR%a_nfixer(n) /= 0) then
        work1 = photoC(n) * PR%Q
        Nfix(n) = (work1 * PR%r_Nfix_photo) - NO3_V(n) - NH4_V(n)
        Nexcrete(n) = Nfix(n) + NO3_V(n) + NH4_V(n) - work1
      endif
      auto_loss(n) = PR%a_mort(n) * Pprime(n) * Tfa(n)
      auto_agg(n) = min((PR%a_agg_rate_max(n) * dps) * Pprime(n), PR%a_mort2(n) * Pprime(n)**PR%auto_mort2_exp)
      auto_agg(n) = max((PR%a_agg_rate_min(n) * dps) * Pprime(n), auto_agg(n))
      if (PR%a_impcalc(n) /= 0 .or. PR%a_expcalc(n) /= 0) then
        auto_loss_poc(n) = QCaCO3(n) * auto_loss(n)
      else
        auto_loss_poc(n) = PR%a_loss_poc(n) * auto_loss(n)
      endif
      auto_loss_doc(n) = (c1 - PR%labile_ratio) * (auto_loss(n) - auto_loss_poc(n))
      auto_loss_dic(n) = PR%labile_ratio * (auto_loss(n) - auto_loss_poc(n))
      s_no3v = s_no3v + NO3_V(n)
      s_nh4v = s_nh4v + NH4_V(n)
      s_po4v = s_po4v + PO4_V(n)
      s_dopv = s_dopv + DOP_V(n)
      photoC_tot = photoC_tot + photoC(n)
      s_photoFe = s_photoFe + photoFe(n)
      s_aldoc = s_aldoc + auto_loss_doc(n)
      s_aldic = s_aldic + auto_loss_dic(n)
      s_alpoc = s_alpoc + auto_loss_poc(n)
      s_aagg = s_aagg + auto_agg(n)
      if (PR%a_nfixer(n) == 0) then
        o2_production_denom = NO3_V(n) + NH4_V(n)
        if (o2_production_denom > c0) then
          o2_production = o2_production + photoC(n) &
               * ((NO3_V(n) / (NO3_V(n) + NH4_V(n))) / PR%Red_D_C_O2 &
                  + (NH4_V(n) / (NO3_V(n) + NH4_V(n))) / PR%Remin_D_C_O2)
        endif
      else
        o2_production_denom = NO3_V(n) + NH4_V(n) + Nfix(n)
        if (o2_production_denom > c0) then
          o2_production = o2_production + photoC(n) &
               * ((NO3_V(n) / (NO3_V(n) + NH4_V(n) + Nfix(n))) / PR%Red_D_C_O2 &
                  + (NH4_V(n) / (NO3_V(n) + NH4_V(n) + Nfix(n))) / PR%Remin_D_C_O2 &
                  + (Nfix(n) / (NO3_V(n) + NH4_V(n) + Nfix(n))) / PR%Red_D_C_O2_diaz)
        endif
      endif
    enddo

    !--- compute_Zprime
    f_loss_thres = min(max((PR%thres_z2_zoo - zt) / (PR%thres_z2_zoo - PR%thres_z1_zoo), c0), c1)
    do n = 1, NZOO
      C_loss_thres = f_loss_thres * PR%z_loss_thres(n)
      Zprime(n) = max(zooC(n) - C_loss_thres, c0)
      zoo_loss_basal(n) = PR%z_basal(n) * Zprime(n) * Tfz(n)
      zoo_loss_bulk(n) = (PR%z_mort_0(n) * Zprime(n) + PR%z_mort2_0(n) * Zprime(n)**PR%zoo_mort2_exp) * Tfz(n)
      zoo_loss(n) = zoo_loss_basal(n) + zoo_loss_bulk(n)
    enddo

    !--- compute_grazing. Prey members are gathered and the grazing terms
    ! scattered back with selects over the PFTs (n == index), which adds
    ! exactly the terms MARBL adds, in the same order.
    do n = 1, NAUTO
      auto_graze(n) = c0; auto_graze_zootot(n) = c0; auto_graze_poc(n) = c0; auto_graze_doc(n) = c0
    enddo
    do n = 1, NZOO
      zoo_graze(n) = c0; zoo_graze_zootot(n) = c0; zoo_graze_poc(n) = c0; zoo_graze_doc(n) = c0
      x_graze_zoo(n) = c0
    enddo
    do pred = 1, NZOO
      work3 = c0
      work4 = c0
      if (PX%graze_ident /= 0) then
      ! prey class prey is autotroph prey alone (MARBL's CESM2 defaults):
      ! the same terms as below with the gather/scatter resolved
      do prey = 1, min(NPREY, NAUTO)
        work1 = c0
        work1 = work1 + Pprime(prey)
        graze_rate = c0
        if (PR%g_func(prey, pred) == 1) then
          if (work1 > c0) then
            graze_rate = PR%g_umax(prey, pred) * Tfz(pred) * zooC(pred) &
                       * ( work1 / (work1 + PR%g_grz(prey, pred)) )
          endif
        else if (PR%g_func(prey, pred) == 2) then
          if (work1 > c0) then
            graze_rate = PR%g_umax(prey, pred) * Tfz(pred) * zooC(pred) &
                       * ( work1**2 / (work1**2 + PR%g_grz(prey, pred)**2) )
          endif
        endif
        if (work1 > c0) then
          work2 = (Pprime(prey) / work1) * graze_rate
        else
          work2 = c0
        endif
        auto_graze(prey) = auto_graze(prey) + work2
        auto_graze_zootot(prey) = auto_graze_zootot(prey) + PR%g_zoofrac(prey, pred) * work2
        x_graze_zoo(pred) = x_graze_zoo(pred) + PR%g_zoofrac(prey, pred) * work2
        if (PR%a_impcalc(prey) /= 0) then
          auto_graze_poc(prey) = auto_graze_poc(prey) &
               + work2 * max((PR%caco3_poc_min * QCaCO3(prey)), &
                             min(PR%spc_poc_fac * (Pprime(prey) + 0.6_rp)**1.6_rp, PR%f_graze_sp_poc_lim))
        else
          auto_graze_poc(prey) = auto_graze_poc(prey) + PR%g_poc(prey, pred) * work2
        endif
        auto_graze_doc(prey) = auto_graze_doc(prey) + PR%g_doc(prey, pred) * work2
        work3 = work3 + PR%g_fdetr(prey, pred) * (work2 + epsC * epsTinv)
        work4 = work4 + (work2 + epsC * epsTinv)
      enddo
      else
      do prey = 1, NPREY
        work1 = c0
        do jj = 1, NAUTO
          if (jj <= PR%g_nauto(prey, pred)) then
            na = PR%g_auto(jj, prey, pred)
            pp_n = c0
            do n = 1, NAUTO
              pp_n = merge(Pprime(n), pp_n, n == na)
            enddo
            work1 = work1 + pp_n
          endif
        enddo
        do jj = 1, NZOO
          if (jj <= PR%g_nzoo(prey, pred)) then
            na = PR%g_zoo(jj, prey, pred)
            pp_n = c0
            do n = 1, NZOO
              pp_n = merge(Zprime(n), pp_n, n == na)
            enddo
            work1 = work1 + pp_n
          endif
        enddo
        graze_rate = c0
        if (PR%g_func(prey, pred) == 1) then
          if (work1 > c0) then
            graze_rate = PR%g_umax(prey, pred) * Tfz(pred) * zooC(pred) &
                       * ( work1 / (work1 + PR%g_grz(prey, pred)) )
          endif
        else if (PR%g_func(prey, pred) == 2) then
          if (work1 > c0) then
            graze_rate = PR%g_umax(prey, pred) * Tfz(pred) * zooC(pred) &
                       * ( work1**2 / (work1**2 + PR%g_grz(prey, pred)**2) )
          endif
        endif
        do jj = 1, NAUTO
          if (jj <= PR%g_nauto(prey, pred)) then
            na = PR%g_auto(jj, prey, pred)
            pp_n = c0; qc_n = c0; imp_n = 0
            do n = 1, NAUTO
              pp_n = merge(Pprime(n), pp_n, n == na)
              qc_n = merge(QCaCO3(n), qc_n, n == na)
              imp_n = merge(PR%a_impcalc(n), imp_n, n == na)
            enddo
            if (work1 > c0) then
              work2 = (pp_n / work1) * graze_rate
            else
              work2 = c0
            endif
            gzt = PR%g_zoofrac(prey, pred) * work2
            if (imp_n /= 0) then
              gpoc = work2 * max((PR%caco3_poc_min * qc_n), &
                                 min(PR%spc_poc_fac * (pp_n + 0.6_rp)**1.6_rp, PR%f_graze_sp_poc_lim))
            else
              gpoc = PR%g_poc(prey, pred) * work2
            endif
            gdoc = PR%g_doc(prey, pred) * work2
            do n = 1, NAUTO
              auto_graze(n) = merge(auto_graze(n) + work2, auto_graze(n), n == na)
              auto_graze_zootot(n) = merge(auto_graze_zootot(n) + gzt, auto_graze_zootot(n), n == na)
              auto_graze_poc(n) = merge(auto_graze_poc(n) + gpoc, auto_graze_poc(n), n == na)
              auto_graze_doc(n) = merge(auto_graze_doc(n) + gdoc, auto_graze_doc(n), n == na)
            enddo
            x_graze_zoo(pred) = x_graze_zoo(pred) + gzt
            work3 = work3 + PR%g_fdetr(prey, pred) * (work2 + epsC * epsTinv)
            work4 = work4 + (work2 + epsC * epsTinv)
          endif
        enddo
        do jj = 1, NZOO
          if (jj <= PR%g_nzoo(prey, pred)) then
            na = PR%g_zoo(jj, prey, pred)
            pp_n = c0
            do n = 1, NZOO
              pp_n = merge(Zprime(n), pp_n, n == na)
            enddo
            if (work1 > c0) then
              work2 = (pp_n / work1) * graze_rate
            else
              work2 = c0
            endif
            gzt = PR%g_zoofrac(prey, pred) * work2
            gpoc = PR%g_poc(prey, pred) * work2
            gdoc = PR%g_doc(prey, pred) * work2
            do n = 1, NZOO
              zoo_graze(n) = merge(zoo_graze(n) + work2, zoo_graze(n), n == na)
              zoo_graze_zootot(n) = merge(zoo_graze_zootot(n) + gzt, zoo_graze_zootot(n), n == na)
              zoo_graze_poc(n) = merge(zoo_graze_poc(n) + gpoc, zoo_graze_poc(n), n == na)
              zoo_graze_doc(n) = merge(zoo_graze_doc(n) + gdoc, zoo_graze_doc(n), n == na)
            enddo
            x_graze_zoo(pred) = x_graze_zoo(pred) + gzt
            work3 = work3 + PR%g_fdetr(prey, pred) * (work2 + epsC * epsTinv)
            work4 = work4 + (work2 + epsC * epsTinv)
          endif
        enddo
      enddo
      endif
      f_zoo_detr(pred) = work3 / work4
    enddo

    !--- compute_routing (zooplankton), zooplankton sums of the DOM,
    ! detritus and tendency sections, zooplankton tracer tendencies
    do n = 1, NZOO
      zoo_graze_dic(n) = zoo_graze(n) - (zoo_graze_zootot(n) + zoo_graze_poc(n) + zoo_graze_doc(n))
      zoo_loss_poc(n) = f_zoo_detr(n) * zoo_loss_bulk(n)
      zoo_loss_doc(n) = (c1 - PR%labile_ratio) * (c1 - f_zoo_detr(n)) * zoo_loss_bulk(n)
      zoo_loss_dic(n) = (PR%labile_ratio * (c1 - f_zoo_detr(n)) * zoo_loss_bulk(n)) + zoo_loss_basal(n)
    enddo
    s_zldoc = c0; s_zgdoc = c0; s_zlpoc = c0; s_zgpoc = c0; s_zldic = c0; s_zgdic = c0
    do n = 1, NZOO
      s_zldoc = s_zldoc + zoo_loss_doc(n)
      s_zgdoc = s_zgdoc + zoo_graze_doc(n)
      s_zlpoc = s_zlpoc + zoo_loss_poc(n)
      s_zgpoc = s_zgpoc + zoo_graze_poc(n)
      s_zldic = s_zldic + zoo_loss_dic(n)
      s_zgdic = s_zgdic + zoo_graze_dic(n)
    enddo
    do n = 1, NZOO
      call emit_tend(trv, (PR%z_c(n)), x_graze_zoo(n) - zoo_graze(n) - zoo_loss(n), dtm, lapply, lsave, i, j, kk)
    enddo

    !--- particle class constants (set_surface_particulate_terms)
    poc_mass = 12.01_rp
    cac_mass = 100.09_rp; cac_gamma = PR%CaCO3_gamma; cac_rho = PR%hPOC_CaCO3_ratio * cac_mass / poc_mass
    sio_mass = 60.08_rp;  sio_gamma = PR%SiO2_gamma;  sio_rho = PR%hPOC_SiO2_ratio * sio_mass / poc_mass
    dst_mass = 1.0e9_rp;  dst_gamma = 0.98_rp
    dst_rho = PR%hPOC_dust_ratio * dst_mass / poc_mass
    dst_rho = dst_rho * PR%nmol2mol_prefix * PR%mass2g
    dst_diss0 = 400.0_rp * PR%m2len

    !--- compute_scavenging (Lig_cnt = 1)
    if (Fel > c0) then
      p2 = KFeLig1
      p1 = c1 + KFeLig1 * (Ligl - Fel)
      p0 = -Fel
      Fefree = (-p1 + sqrt(p1**2 - c4 * p2 * p0)) / (c2 * p2)
      FeLig1 = KFeLig1 * Fefree * Ligl / (c1 + KFeLig1 * Fefree)
    else
      Fefree = c0
      FeLig1 = c0
    endif
    sinking_mass = (st%poc_s + st%poc_h) * (3.0_rp * 12.01_rp) &
                 + (st%cac_s + st%cac_h) * cac_mass &
                 + (st%sio_s + st%sio_h) * sio_mass &
                 + (st%dst_s + st%dst_h) * PR%dust_Fe_scav_scale
    Fe_scavenge_rate = PR%Fe_scav_yps * sinking_mass
    Lig_scavenge_rate = PR%Lig_scav_yps * sinking_mass
    FeLig_scavenge_rate = PR%FeLig_scav_yps * sinking_mass
    Lig_scavenge = FeLig1 * Lig_scavenge_rate
    Fe_scavenge = Fefree * Fe_scavenge_rate + FeLig1 * FeLig_scavenge_rate

    !--- compute_routing (autotrophs) and the autotroph tracer tendencies, with
    ! the autotroph sums of the DOM, detritus and tendency sections (from
    ! zero, in PFT order) and the per-PFT terms that MARBL adds to sums
    ! starting from later values (same expressions)
    s_agdoc = c0; s_rpdop = c0; s_agpoc = c0; s_rppop = c0; s_agdic = c0; s_rpdip = c0
    cac_prod = c0; sio_prod = c0
    fe_prod = (s_zlpoc + s_zgpoc) * PR%Qfe_zoo + Fe_scavenge
    !dir$ unroll
    do n = 1, NAUTO
      auto_graze_dic(n) = auto_graze(n) - (auto_graze_zootot(n) + auto_graze_poc(n) + auto_graze_doc(n))
      remaining_P_pop(n) = (auto_graze_poc(n) + auto_loss_poc(n) + auto_agg(n)) * Qp(n)
      remaining_P = (auto_graze(n) + auto_loss(n) + auto_agg(n)) * Qp(n) &
                  - auto_graze_zootot(n) * PR%Qp_zoo - remaining_P_pop(n)
      if (remaining_P < c0) then
        remaining_P_pop(n) = remaining_P_pop(n) + remaining_P
        remaining_P = c0
      endif
      remaining_P_dop(n) = PR%f_toDOP * remaining_P
      remaining_P_dip(n) = (c1 - PR%f_toDOP) * remaining_P
      auto_sum = auto_graze(n) + auto_loss(n) + auto_agg(n)
      call emit_tend(trv, (PR%a_c(n)), photoC(n) - auto_sum, dtm, lapply, lsave, i, j, kk)
      if (PR%a_p(n) > 0) &
        call emit_tend(trv, (PR%a_p(n)), PO4_V(n) + DOP_V(n) - Qp(n) * auto_sum, dtm, lapply, lsave, i, j, kk)
      call emit_tend(trv, (PR%a_chl(n)), photoacc(n) - thetaC(n) * auto_sum, dtm, lapply, lsave, i, j, kk)
      call emit_tend(trv, (PR%a_fe(n)), photoFe(n) - Qfe(n) * auto_sum, dtm, lapply, lsave, i, j, kk)
      if (PR%a_si(n) > 0) &
        call emit_tend(trv, (PR%a_si(n)), photoSi(n) - Qsi(n) * auto_sum, dtm, lapply, lsave, i, j, kk)
      if (PR%a_caco3(n) > 0) &
        call emit_tend(trv, (PR%a_caco3(n)), CaCO3_form(n) - QCaCO3(n) * auto_sum, dtm, lapply, lsave, i, j, kk)
      s_agdoc = s_agdoc + auto_graze_doc(n)
      s_rpdop = s_rpdop + remaining_P_dop(n)
      s_agpoc = s_agpoc + auto_graze_poc(n)
      s_rppop = s_rppop + remaining_P_pop(n)
      s_agdic = s_agdic + auto_graze_dic(n)
      s_rpdip = s_rpdip + remaining_P_dip(n)
      if (PR%a_caco3(n) > 0) then
        cac_prod = cac_prod + ((c1 - PR%f_graze_CaCO3_remin) * auto_graze(n) + auto_loss(n) + auto_agg(n)) * QCaCO3(n)
        ydic(n) = PR%f_graze_CaCO3_remin * auto_graze(n) * QCaCO3(n)
        zalk(n) = c2 * (PR%f_graze_CaCO3_remin * auto_graze(n) * QCaCO3(n) - CaCO3_form(n))
      else
        ydic(n) = c0; zalk(n) = c0
      endif
      if (PR%a_si(n) > 0) then
        sio_prod = sio_prod + Qsi(n) * ((c1 - PR%f_graze_si_remin) * auto_graze(n) + auto_agg(n) &
                   + PR%a_loss_poc(n) * auto_loss(n))
        xsi(n) = Qsi(n) * (PR%f_graze_si_remin * auto_graze(n) + (c1 - PR%a_loss_poc(n)) * auto_loss(n))
      else
        xsi(n) = c0
      endif
      fe_prod = fe_prod + Qfe(n) * (auto_agg(n) + auto_graze_poc(n) + auto_loss_poc(n))
      afe(n) = (Qfe(n) * (auto_loss_dic(n) + auto_graze_dic(n)))
      bfe(n) = auto_graze_zootot(n) * (Qfe(n) - PR%Qfe_zoo)
      cfe(n) = (Qfe(n) * (auto_loss_doc(n) + auto_graze_doc(n)))
    enddo

    !--- compute_dissolved_organic_matter
    DOC_prod = s_zldoc + s_aldoc + s_agdoc + s_zgdoc
    DON_prod = PR%Q * DOC_prod * PR%f_toDON
    DOP_prod = PR%Qp_zoo * (s_zldoc + s_zgdoc) + s_rpdop
    if (par_avg > 1.0_rp) then
      DOC_reminR = PR%DOC_reminR_light; DON_reminR = PR%DON_reminR_light; DOP_reminR = PR%DOP_reminR_light
    else
      DOC_reminR = PR%DOC_reminR_dark; DON_reminR = PR%DON_reminR_dark; DOP_reminR = PR%DOP_reminR_dark
    endif
    DOCr_reminR = PR%DOCr_reminR0; DONr_reminR = PR%DONr_reminR0; DOPr_reminR = PR%DOPr_reminR0
    if (k == 1) then
      if (st%par_surf > 1.0_rp) then
        work = (log(st%par_surf) * 0.4373_rp) * (10._rp * PR%m2len / dz1)
        DOCr_reminR = DOCr_reminR + work * PR%DOMr_reminR_photo
        DONr_reminR = DONr_reminR + work * PR%DOMr_reminR_photo
        DOPr_reminR = DOPr_reminR + work * PR%DOMr_reminR_photo
      endif
    endif
    DOC_remin  = DOCl  * DOC_reminR
    DON_remin  = DONl  * DON_reminR
    DOP_remin  = DOPl  * DOP_reminR
    DOCr_remin = DOCrl * DOCr_reminR
    DONr_remin = DONrl * DONr_reminR
    DOPr_remin = DOPrl * DOPr_reminR

    !--- compute_large_detritus_prod (cac_prod, sio_prod, fe_prod above)
    POC_prod = s_zlpoc + s_agpoc + s_zgpoc + s_aagg + s_alpoc
    POP_prod = PR%Qp_zoo * (s_zlpoc + s_zgpoc) + s_rppop
    if (POP_prod < c0) then
      DOP_loss_P_bal = -POP_prod
      POP_prod = c0
    else
      DOP_loss_P_bal = c0
    endif

    !--- compute_particulate_terms (k <= kmt)
    scalelength = c1
    if (zw < PR%scalelen_z(1)) then
      scalelength = PR%scalelen_vals(1)
    else if (zw >= PR%scalelen_z(4)) then
      scalelength = PR%scalelen_vals(4)
    else
      do n = 2, 4
        if (zw < PR%scalelen_z(n)) then
          scalelength = PR%scalelen_vals(n-1) &
               + (PR%scalelen_vals(n) - PR%scalelen_vals(n-1)) &
               * (zw - PR%scalelen_z(n-1)) / (PR%scalelen_z(n) - PR%scalelen_z(n-1))
          exit
        endif
      enddo
    endif
    DECAY_Hard     = exp(-dz * PR%len2cm / 4.0e6_rp)
    DECAY_HardDust = exp(-dz * PR%len2cm / 1.2e8_rp)
    dzr = c1 / dz
    poc_diss   = PR%POC_diss
    sio2_diss  = PR%SiO2_diss
    caco3_diss = PR%CaCO3_diss
    dust_diss  = dst_diss0
    if (O2l < PR%o2_sf_hi) then
      o2_scalefactor = c1 + (PR%o2_sf_val - c1) * min(c1, (PR%o2_sf_hi - O2l) / (PR%o2_sf_hi - PR%o2_sf_lo))
      poc_diss   = poc_diss   * o2_scalefactor
      sio2_diss  = sio2_diss  * o2_scalefactor
      caco3_diss = caco3_diss * o2_scalefactor
      dust_diss  = dust_diss  * o2_scalefactor
    endif
    poc_diss = scalelength * poc_diss
    sio2_diss = scalelength * sio2_diss
    caco3_diss = scalelength * caco3_diss
    dust_diss = scalelength * dust_diss
    decay_POC_E = exp(-dz / poc_diss)
    decay_SiO2  = exp(-dz / sio2_diss)
    decay_CaCO3 = exp(-dz / caco3_diss)
    decay_dust  = exp(-dz / dust_diss)
    cac_so = st%cac_s * decay_CaCO3 + cac_prod * ((c1 - cac_gamma) * (c1 - decay_CaCO3) * caco3_diss)
    cac_ho = st%cac_h * DECAY_Hard + cac_prod * (cac_gamma * dz)
    caca_so = st%caca_s * decay_CaCO3 + cac_prod * ((c1 - cac_gamma) * (c1 - decay_CaCO3) * caco3_diss)
    caca_ho = st%caca_h * DECAY_Hard + cac_prod * (cac_gamma * dz)
    sio_so = st%sio_s * decay_SiO2 + sio_prod * ((c1 - sio_gamma) * (c1 - decay_SiO2) * sio2_diss)
    sio_ho = st%sio_h * DECAY_Hard + sio_prod * (sio_gamma * dz)
    dst_so = st%dst_s * decay_dust
    dst_ho = st%dst_h * DECAY_HardDust
    POC_PROD_avail = POC_prod - cac_rho * cac_prod - sio_rho * sio_prod
    if (st%qa_dust_def > 0) then
      new_QA_dust_def = st%qa_dust_def * (dst_so + dst_ho) / (st%dst_s + st%dst_h)
    else
      new_QA_dust_def = c0
    endif
    if (new_QA_dust_def > c0 .and. POC_prod > c0) then
      new_QA_dust_def = new_QA_dust_def - POC_PROD_avail * dz
      if (new_QA_dust_def < c0) then
        POC_PROD_avail = -new_QA_dust_def * dzr
        new_QA_dust_def = c0
      else
        POC_PROD_avail = c0
      endif
    endif
    st%qa_dust_def = new_QA_dust_def
    if (st%poc_h == c0 .and. POC_prod == c0) then
      poc_ho = c0
    else
      poc_ho = cac_rho * (cac_so + cac_ho) + sio_rho * (sio_so + sio_ho) + dst_rho * (dst_so + dst_ho) - new_QA_dust_def
      poc_ho = max(poc_ho, c0)
    endif
    poc_so = st%poc_s * decay_POC_E + POC_PROD_avail * ((c1 - decay_POC_E) * poc_diss)
    cac_remin = cac_prod + ((st%cac_s - cac_so) + (st%cac_h - cac_ho)) * dzr
    caca_remin = cac_prod + ((st%caca_s - caca_so) + (st%caca_h - caca_ho)) * dzr
    sio_remin = sio_prod + ((st%sio_s - sio_so) + (st%sio_h - sio_ho)) * dzr
    poc_remin = POC_prod + ((st%poc_s - poc_so) + (st%poc_h - poc_ho)) * dzr
    pon_remin = PR%Q * poc_remin
    dst_remin = ((st%dst_s - dst_so) + (st%dst_h - dst_ho)) * dzr
    if (st%poc_s + st%poc_h == c0) then
      fe_remin = (poc_remin * PR%Red_Fe_C)
    else
      fe_remin = (poc_remin * (st%fe_s + st%fe_h) / (st%poc_s + st%poc_h))
    endif
    fe_remin = fe_remin + (st%fe_s * PR%Fe_desorption_rate0)
    fe_so = st%fe_s + dz * ((c1 - c0) * fe_prod - fe_remin)
    if (fe_so < c0) then
      fe_so = c0
      fe_remin = st%fe_s * dzr + (c1 - c0) * fe_prod
    endif
    fe_remin = fe_remin + dst_remin * PR%dust_to_Fe + (c0 * dzr)   ! fesedflux = 0 in the ROMS driver
    fe_ho = st%fe_h
    pop_remin = c0
    if (st%poc_s + st%poc_h > c0) then
      pop_remin = poc_remin * ((st%pop_s + st%pop_h) / (st%poc_s + st%poc_h))
    else if (POC_prod > c0) then
      pop_remin = poc_remin * (POP_prod / POC_prod)
    endif
    pop_so = st%pop_s + dz * ((c1 - c0) * POP_prod - pop_remin)
    if (pop_so < c0) then
      pop_so = c0
      pop_remin = st%pop_s * dzr + (c1 - c0) * POP_prod
    endif
    pop_ho = st%pop_h
    ! bottom cell: burial, sedimentary denitrification, remin of what hits the floor
    sed_denitrif = c0; other_remin = c0; PON_sed_loss = c0
    poc_sed = c0; pop_sed = c0; sio_sed = c0; cac_sed = c0; caca_sed = c0; fe_sed = c0; dst_sed = c0
    if (k == kmt) then
      bftt = c1 / dz
      to_floor = poc_so + poc_ho
      if (to_floor > c0) then
        flux_alt = to_floor * PR%len2m * spd
        bury_frac = 0.013_rp + 0.53_rp * flux_alt * flux_alt / (7.0_rp + flux_alt)**2
        poc_sed = to_floor * min(PR%POM_bury_frac_max, PR%POC_bury_coeff * bury_frac)
        PON_sed_loss = PR%PON_bury_coeff * PR%Q * poc_sed
        pop_sed = (pop_so + pop_ho) * min(PR%POM_bury_frac_max, PR%POP_bury_coeff * bury_frac)
        sed_denitrif = bftt * PR%sed_denitrif_coeff * to_floor * (0.06_rp + 0.19_rp * 0.99_rp**(O2l - NO3l))
        flux_alt = to_floor * (PR%conc_flux2mmol_m2s * (mpercm**2)) * spy
        other_remin = min(bftt * min(0.1_rp + flux_alt, 0.5_rp) * (to_floor - poc_sed), &
                          bftt * (to_floor - poc_sed) - (sed_denitrif * PR%denitrif_C_N))
        if (O2l < c1) then
          other_remin = bftt * (to_floor - poc_sed) - sed_denitrif * PR%denitrif_C_N
        endif
      endif
      flux_alt = (sio_so + sio_ho) * PR%len2m * spd
      if (flux_alt > c2) then
        bury_frac = 0.2_rp
      else
        bury_frac = 0.04_rp
      endif
      if (PR%bSi_bury_coeff * bury_frac < PR%bSi_bury_frac_max) then
        sio_sed = (sio_so + sio_ho) * PR%bSi_bury_coeff * bury_frac
      else
        sio_sed = (sio_so + sio_ho) * PR%bSi_bury_frac_max
      endif
      if (PR%caco3_bury_iopt == 1) then
        if (zw < PR%caco3_bury_depth) then
          cac_sed = cac_so + cac_ho
          caca_sed = caca_so + caca_ho
        endif
      else
        if (co3 > PR%caco3_bury_omega * co3sat) cac_sed = cac_so + cac_ho
        if (co3a > PR%caco3_bury_omega * co3sat) caca_sed = caca_so + caca_ho
      endif
      if (cac_so + cac_ho > c0) cac_remin = cac_remin + ((cac_so + cac_ho - cac_sed) * bftt)
      if (caca_so + caca_ho > c0) caca_remin = caca_remin + ((caca_so + caca_ho - caca_sed) * bftt)
      if (sio_so + sio_ho > c0) sio_remin = sio_remin + ((sio_so + sio_ho - sio_sed) * bftt)
      if (to_floor > c0) then
        poc_remin = poc_remin + ((to_floor - poc_sed) * bftt)
        pon_remin = pon_remin + ((PR%Q * to_floor - PON_sed_loss) * bftt)
      endif
      if (pop_so + pop_ho > c0) pop_remin = pop_remin + ((pop_so + pop_ho - pop_sed) * bftt)
      fe_sed = fe_so + fe_ho
      dst_sed = dst_so + dst_ho
    endif

    !--- compute_Lig_terms
    Lig_prod = (poc_remin * PR%remin_to_Lig) + (DOC_prod * PR%remin_to_Lig)
    Lig_photochem = c0
    if (k == 1) then
      rate_per_sec = c0
      if (st%par_surf > 1.0_rp) then
        rate_per_sec = (log(st%par_surf) * 0.4373_rp) * (10._rp * PR%m2len / dz1) * ((12.0_rp / 3.0_rp) * yps)
      endif
      Lig_photochem = Ligl * rate_per_sec
    endif
    Lig_deg = poc_remin * PR%Lig_degrade_rate0
    Lig_loss = Lig_scavenge + 0.20_rp * s_photoFe + Lig_photochem + Lig_deg

    !--- compute_nitrif
    nitrif = c0
    if (NH4l /= c0) then
      if (par_k < PR%nitrif_par_lim) then
        nitrif = PR%kappa_nitrif * NH4l
        if (st%par_in > PR%nitrif_par_lim) then
          nitrif = nitrif * log(par_k / PR%nitrif_par_lim) / (-kpardz)
        endif
      endif
    endif

    !--- compute_denitrif
    work = ((PR%o2_min + PR%o2_min_delta) - O2l) / PR%o2_min_delta
    work = min(max(work, c0), c1)
    denitrif = work * ((DOC_remin + DOCr_remin + poc_remin * (c1 - PR%POCremin_refract) &
             - other_remin) / PR%denitrif_C_N - sed_denitrif)
    if (NO3l < ((c10 * spd) * (denitrif + sed_denitrif))) then
      work = NO3l / ((c10 * spd) * (denitrif + sed_denitrif))
      denitrif = denitrif * work
      sed_denitrif = sed_denitrif * work
    endif

    !--- compute_local_tendencies (the PFT tracers were emitted above; the
    ! PFT sums were formed above)
    t_no3 = nitrif - denitrif - sed_denitrif - s_no3v
    t_nh4 = -s_nh4v - nitrif + DON_remin + DONr_remin &
          + PR%Q * (s_zldic + s_zgdic + s_aldic + s_agdic + DOC_prod * (c1 - PR%f_toDON)) &
          + pon_remin * (c1 - PR%PONremin_refract)
    do n = 1, NAUTO
      if (PR%a_nfixer(n) /= 0) t_nh4 = t_nh4 + Nexcrete(n)
    enddo
    t_fe = fe_remin - s_photoFe - Fe_scavenge + PR%Qfe_zoo * (s_zldic + s_zldoc + s_zgdic + s_zgdoc)
    do n = 1, NAUTO
      t_fe = t_fe + afe(n) + bfe(n) + cfe(n)
    enddo
    t_lig = Lig_prod - Lig_loss
    t_sio3 = sio_remin
    do n = 1, NAUTO
      if (PR%a_si(n) > 0) t_sio3 = t_sio3 - photoSi(n) + xsi(n)
    enddo
    t_po4 = DOP_remin + DOPr_remin - s_po4v + (c1 - PR%POPremin_refract) * pop_remin + s_rpdip &
          + PR%Qp_zoo * (s_zldic + s_zgdic)
    t_doc = DOC_prod * (c1 - PR%DOCprod_refract) - DOC_remin
    t_docr = DOC_prod * PR%DOCprod_refract - DOCr_remin + (poc_remin * PR%POCremin_refract)
    t_don = (DON_prod * (c1 - PR%DONprod_refract)) - DON_remin
    t_donr = (DON_prod * PR%DONprod_refract) - DONr_remin + (pon_remin * PR%PONremin_refract)
    t_dop = (DOP_prod * (c1 - PR%DOPprod_refract)) - DOP_remin - s_dopv - DOP_loss_P_bal
    t_dopr = (DOP_prod * PR%DOPprod_refract) - DOPr_remin + (pop_remin * PR%POPremin_refract)
    t_dic = s_aldic + s_agdic - photoC_tot + DOC_remin + poc_remin * (c1 - PR%POCremin_refract) + s_zldic &
          + s_zgdic + cac_remin + DOCr_remin
    do n = 1, NAUTO
      if (PR%a_caco3(n) > 0) t_dic = t_dic + ydic(n) - CaCO3_form(n)
    enddo
    t_dica = t_dic + (caca_remin - cac_remin)
    t_alk = -t_no3 + t_nh4 + c2 * cac_remin
    do n = 1, NAUTO
      if (PR%a_caco3(n) > 0) t_alk = t_alk + zalk(n)
    enddo
    t_alka = t_alk + c2 * (caca_remin - cac_remin)
    o2_consumption = (O2l - PR%o2_min) / PR%o2_min_delta
    o2_consumption = min(max(o2_consumption, c0), c1)
    o2_consumption = o2_consumption * ((poc_remin * (c1 - PR%POCremin_refract) + DOC_remin + DOCr_remin &
                     - (sed_denitrif * PR%denitrif_C_N) - other_remin + s_zldic + s_zgdic + s_aldic + s_agdic) &
                     / PR%Remin_D_C_O2 + (c2 * nitrif))
    t_o2 = o2_production - o2_consumption

    call emit_tend(trv, (PR%no3), t_no3, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%nh4), t_nh4, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%fe), t_fe, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%lig), t_lig, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%sio3), t_sio3, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%po4), t_po4, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%doc), t_doc, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%docr), t_docr, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%don), t_don, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%donr), t_donr, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%dop), t_dop, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%dopr), t_dopr, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%dic), t_dic, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%dica), t_dica, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%alk), t_alk, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%alka), t_alka, dtm, lapply, lsave, i, j, kk)
    call emit_tend(trv, (PR%o2), t_o2, dtm, lapply, lsave, i, j, kk)

    !--- diagnostics (tests; dg(14) is filled by the caller)
    if (ldg /= 0) then
      do m = 1, ND_IT
        dg(m) = c0
      enddo
      dg(1) = par_avg
      dg(2) = nitrif
      dg(3) = denitrif
      dg(4) = photoC_tot                 ! photoC_TOT
      dg(5) = st%poc_s + st%poc_h        ! POC_FLUX_IN
      dg(6) = poc_remin * (c1 - PR%POCremin_refract)
      dg(7) = o2_production
      dg(8) = o2_consumption
      dg(9) = Fe_scavenge
      dg(10) = tins
      dg(11) = co3
      dg(12) = h2co3
      dg(13) = co3sat
      dg(15) = DOC_prod
      dg(16) = POC_prod
    endif

    !--- state for the next level
    st%par_in = par_k
    st%poc_s = poc_so; st%poc_h = poc_ho
    st%pop_s = pop_so; st%pop_h = pop_ho
    st%cac_s = cac_so; st%cac_h = cac_ho
    st%caca_s = caca_so; st%caca_h = caca_ho
    st%sio_s = sio_so; st%sio_h = sio_ho
    st%dst_s = dst_so; st%dst_h = dst_ho
    st%fe_s = fe_so; st%fe_h = fe_ho
  end subroutine level_step

  !===========================================================================
  ! K1: carbonate chemistry at every wet point
  !===========================================================================
#ifdef USE_CUDA
  ! 4 blocks of 128 per SM (128 registers): faster than the uncapped 206
  ! registers / 2 blocks (A100, Pac 4x2: 7.8 -> 6.3 ms). The interior
  ! kernels are fastest at 2 blocks (255 registers; 168 with 3 blocks was
  ! 26% slower) and say so explicitly: nvfortran 25.9 gave the kernels of
  ! this module without launch_bounds the 128-register cap of this one.
  attributes(global) launch_bounds(128, 4) &
  subroutine carb_kernel(ntr, nz, slot, nslot, nts, is, ie, js, je, nx, ny, nwet, wet_i, wet_j, tr, ncl, ts, z_r, mask)
    integer(ip), value :: ntr, nz, slot, nslot, nts, is, ie, js, je, nx, ny
    integer(ip), value :: nwet
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    integer(ip) :: c
    integer(ip), value :: ncl
    real(pp), device, intent(in) :: tr(ntr, 0:ncl, nz, nslot)
    real(rp), device, intent(in) :: ts(is:ie, js:je, nz, nslot, nts)
    real(rp), device, intent(in) :: z_r(is:ie, js:je, nz)
    real(rp), device, intent(in) :: mask(is:ie, js:je)
    integer(ip) :: i, j, kk, ierr, ierr2
    real(rp) :: potemp, salt, press, tins, phlo, phhi, ph, ph0, pha, h2co3, hco3, co3, h2co3a, hco3a, co3a
    real(rp) :: k0, k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, csat, asat
    real(rp) :: t_dic, t_alk, t_po4, t_sio3, t_dica, t_alka
    integer(ip), parameter :: lpress = 1

    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    kk = blockIdx%z
    if (c > nwet .or. kk > nz) return
    i = wet_i(c); j = wet_j(c)
    if (mask(i, j) == c0) return
    potemp = ts(i, j, kk, slot, itemp)
    salt = ts(i, j, kk, slot, isalt)
    press = -z_r(i, j, kk) * 0.1_rp
    call potemp_s(salt, potemp, c0, press, tins)
    it_temp(i, j, kk) = tins
    t_dic = real(tr(PR%dic, c, kk, slot), rp); t_alk = real(tr(PR%alk, c, kk, slot), rp)
    t_po4 = real(tr(PR%po4, c, kk, slot), rp); t_sio3 = real(tr(PR%sio3, c, kk, slot), rp)
    t_dica = real(tr(PR%dica, c, kk, slot), rp); t_alka = real(tr(PR%alka, c, kk, slot), rp)
    ! max(0,.) as in setup_local_tracers
    t_dic = max(c0, t_dic); t_alk = max(c0, t_alk); t_po4 = max(c0, t_po4); t_sio3 = max(c0, t_sio3)
    t_dica = max(c0, t_dica); t_alka = max(c0, t_alka)
    call co2_coeffs(tins, salt, press, lpress, k0, k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft)
    ph = it_ph(i, j, kk)
    if (ph /= c0) then
      phlo = ph - PR%del_ph; phhi = ph + PR%del_ph
    else
      phlo = PR%phlo_3d_init; phhi = PR%phhi_3d_init
    endif
    ph0 = ph
    call co2calc_interior1(t_dic, t_alk, t_po4, t_sio3, (PR%mass_to_vol), &
         k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, phlo, phhi, ph, h2co3, hco3, co3, ierr)
    pha = it_pha(i, j, kk)
    if (t_dica == t_dic .and. t_alka == t_alk .and. pha == ph0) then
      ! identical inputs and bracket: the alternative-CO2 solve would repeat
      ! the base one exactly, so reuse its results (half the kernel's work
      ! when ALT_CO2 runs the same forcing, as in MiniPac)
      pha = ph; h2co3a = h2co3; hco3a = hco3; co3a = co3; ierr2 = ierr
    else
      if (pha /= c0) then
        phlo = pha - PR%del_ph; phhi = pha + PR%del_ph
      else
        phlo = PR%phlo_3d_init; phhi = PR%phhi_3d_init
      endif
      call co2calc_interior1(t_dica, t_alka, t_po4, t_sio3, (PR%mass_to_vol), &
           k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, phlo, phhi, pha, h2co3a, hco3a, co3a, ierr2)
    endif
    call co3_sat1(tins, salt, press, (PR%mass_to_vol), csat, asat)
    it_ph(i, j, kk) = ph
    it_pha(i, j, kk) = pha
    it_h2co3(i, j, kk) = h2co3
    it_co3(i, j, kk) = co3
    it_co3a(i, j, kk) = co3a
    it_co3sat(i, j, kk) = csat
    it_co3sata(i, j, kk) = asat
    if (ierr /= 0 .or. ierr2 /= 0) then
      ierr = atomicAdd(it_err(i, j), 1_ip)
    endif
  end subroutine carb_kernel

  !===========================================================================
  ! K2: interior ecosystem, one thread per column, surface to bottom
  !===========================================================================
  attributes(global) launch_bounds(128, 2) &
  subroutine interior_kernel(ntr, nz, slot, nslot, nts, is, ie, js, je, nx, ny, nwet, wet_i, wet_j, dtm, ldg, &
                             tr, ncl, ts, z_r, z_w, Hz, mask, dust)
    integer(ip), value :: ntr, nz, slot, nslot, nts, is, ie, js, je, nx, ny, ldg
    integer(ip), value :: nwet
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    integer(ip) :: c
    real(rp), value :: dtm
    integer(ip), value :: ncl
    real(pp), device, intent(inout) :: tr(ntr, 0:ncl, nz, nslot)
    real(rp), device, intent(in) :: ts(is:ie, js:je, nz, nslot, nts)
    real(rp), device, intent(in) :: z_r(is:ie, js:je, nz), z_w(is:ie, js:je, 0:nz), Hz(is:ie, js:je, nz)
    real(rp), device, intent(in) :: mask(is:ie, js:je), dust(is:ie, js:je)
    integer(ip) :: i, j, k, kk, m, lapply, l0
    real(rp) :: dz, dz1, zw, zt, tins, h2co3, co3, co3a, co3sat, dust_flux, dst_rho
    real(rp) :: dg(ND_IT)
    type(sink_t) :: st

    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nwet) return
    i = wet_i(c); j = wet_j(c)
    if (mask(i, j) == c0) return
    dust_flux = dust(i, j)
    ! set_surface_particulate_terms + PAR at the surface
    st%par_surf = PR%f_qsw_par * it_qsw(i, j)
    if (st%par_surf < PAR_threshold) st%par_surf = c0
    st%par_in = st%par_surf
    st%poc_s = c0; st%poc_h = c0; st%pop_s = c0; st%pop_h = c0
    st%cac_s = c0; st%cac_h = c0; st%caca_s = c0; st%caca_h = c0
    st%sio_s = c0; st%sio_h = c0; st%fe_s = c0; st%fe_h = c0
    st%dst_s = (c1 - 0.98_rp) * dust_flux
    st%dst_h = 0.98_rp * dust_flux
    dst_rho = PR%hPOC_dust_ratio * 1.0e9_rp / 12.01_rp
    dst_rho = dst_rho * PR%nmol2mol_prefix * PR%mass2g
    st%qa_dust_def = dst_rho * (st%dst_s + st%dst_h)
    dz1 = Hz(i, j, nz)
    lapply = 1; l0 = 0
    do k = 1, nz
      kk = nz - k + 1
      dz = Hz(i, j, kk)
      zw = -z_w(i, j, kk - 1)
      zt = -z_r(i, j, kk)
      tins = it_temp(i, j, kk)
      h2co3 = it_h2co3(i, j, kk); co3 = it_co3(i, j, kk); co3a = it_co3a(i, j, kk); co3sat = it_co3sat(i, j, kk)
      ! the level's tracers are read and updated in place (tr + tend*dtm)
      ! inside level_step; ldg also keeps the tendencies (it_tend)
      call level_step(k, nz, dz, dz1, zw, zt, tins, h2co3, co3, co3a, co3sat, dust_flux, &
                      tr(1, c, kk, slot), st, dtm, lapply, dg, l0, l0, i, j, kk)
    enddo
  end subroutine interior_kernel

  ! the same with the test outputs (it_tend, it_diag)
  attributes(global) launch_bounds(128, 2) &
  subroutine interior_kernel_dbg(ntr, nz, slot, nslot, nts, is, ie, js, je, nx, ny, nwet, wet_i, wet_j, dtm, ldg, &
                             tr, ncl, ts, z_r, z_w, Hz, mask, dust)
    integer(ip), value :: ntr, nz, slot, nslot, nts, is, ie, js, je, nx, ny, ldg
    integer(ip), value :: nwet
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    integer(ip) :: c
    real(rp), value :: dtm
    integer(ip), value :: ncl
    real(pp), device, intent(inout) :: tr(ntr, 0:ncl, nz, nslot)
    real(rp), device, intent(in) :: ts(is:ie, js:je, nz, nslot, nts)
    real(rp), device, intent(in) :: z_r(is:ie, js:je, nz), z_w(is:ie, js:je, 0:nz), Hz(is:ie, js:je, nz)
    real(rp), device, intent(in) :: mask(is:ie, js:je), dust(is:ie, js:je)
    integer(ip) :: i, j, k, kk, m, lapply
    real(rp) :: dz, dz1, zw, zt, tins, h2co3, co3, co3a, co3sat, dust_flux, dst_rho
    real(rp) :: dg(ND_IT)
    type(sink_t) :: st

    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nwet) return
    i = wet_i(c); j = wet_j(c)
    if (mask(i, j) == c0) return
    dust_flux = dust(i, j)
    ! set_surface_particulate_terms + PAR at the surface
    st%par_surf = PR%f_qsw_par * it_qsw(i, j)
    if (st%par_surf < PAR_threshold) st%par_surf = c0
    st%par_in = st%par_surf
    st%poc_s = c0; st%poc_h = c0; st%pop_s = c0; st%pop_h = c0
    st%cac_s = c0; st%cac_h = c0; st%caca_s = c0; st%caca_h = c0
    st%sio_s = c0; st%sio_h = c0; st%fe_s = c0; st%fe_h = c0
    st%dst_s = (c1 - 0.98_rp) * dust_flux
    st%dst_h = 0.98_rp * dust_flux
    dst_rho = PR%hPOC_dust_ratio * 1.0e9_rp / 12.01_rp
    dst_rho = dst_rho * PR%nmol2mol_prefix * PR%mass2g
    st%qa_dust_def = dst_rho * (st%dst_s + st%dst_h)
    dz1 = Hz(i, j, nz)
    lapply = 1
    do k = 1, nz
      kk = nz - k + 1
      dz = Hz(i, j, kk)
      zw = -z_w(i, j, kk - 1)
      zt = -z_r(i, j, kk)
      tins = it_temp(i, j, kk)
      h2co3 = it_h2co3(i, j, kk); co3 = it_co3(i, j, kk); co3a = it_co3a(i, j, kk); co3sat = it_co3sat(i, j, kk)
      ! the level's tracers are read and updated in place (tr + tend*dtm)
      ! inside level_step; ldg also keeps the tendencies (it_tend)
      call level_step(k, nz, dz, dz1, zw, zt, tins, h2co3, co3, co3a, co3sat, dust_flux, &
                      tr(1, c, kk, slot), st, dtm, lapply, dg, ldg, ldg, i, j, kk)
      if (ldg /= 0) then
        dg(14) = it_ph(i, j, kk)
        do m = 1, ND_IT
          it_diag(i, j, kk, m) = dg(m)
        enddo
      endif
    enddo
  end subroutine interior_kernel_dbg
#endif


#ifndef USE_CUDA
  ! Host run of the ported column (same device code compiled for the host):
  ! the quickest check of the port against MARBL on a login node. Inputs as
  ! marbl_host_interior_column (MARBL level 1 = surface).
  subroutine marbl_interior_column_host(nz, dz, zw, zt, potemp, salt, press, qsw, dust, tr, &
                                        ph, ph_alt, tend, diag, ierr)
    integer(ip), intent(in) :: nz
    real(rp), intent(in) :: dz(nz), zw(nz), zt(nz), potemp(nz), salt(nz), press(nz), qsw, dust
    real(rp), intent(in) :: tr(:,:)
    real(rp), intent(inout) :: ph(nz), ph_alt(nz)
    real(rp), intent(out) :: tend(:,:), diag(ND_IT, nz)
    integer, intent(out) :: ierr
    integer(ip) :: k, m, ie1, ie2
    real(rp) :: tins(nz), h2co3(nz), co3(nz), co3a(nz), csat(nz), asat(nz), hco3, h2co3a, hco3a
    real(rp) :: k0, k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, phlo, phhi
    real(rp) :: t_dic, t_alk, t_po4, t_sio3, t_dica, t_alka, dst_rho
    real(rp) :: trk(NTRMAX), dg(ND_IT)
    type(sink_t) :: sk
    integer(ip), parameter :: lpress = 1, ldg = 1, lapply = 0, lsave = 0
    real(rp), parameter :: dtm0 = 0.0_rp
    ierr = 0
    do k = 1, nz
      call potemp_s(salt(k), potemp(k), c0, press(k), tins(k))
      t_dic = max(c0, tr(PR%dic, k)); t_alk = max(c0, tr(PR%alk, k)); t_po4 = max(c0, tr(PR%po4, k))
      t_sio3 = max(c0, tr(PR%sio3, k)); t_dica = max(c0, tr(PR%dica, k)); t_alka = max(c0, tr(PR%alka, k))
      call co2_coeffs(tins(k), salt(k), press(k), lpress, k0, k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft)
      if (ph(k) /= c0) then
        phlo = ph(k) - PR%del_ph; phhi = ph(k) + PR%del_ph
      else
        phlo = PR%phlo_3d_init; phhi = PR%phhi_3d_init
      endif
      call co2calc_interior1(t_dic, t_alk, t_po4, t_sio3, (PR%mass_to_vol), &
           k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, phlo, phhi, ph(k), h2co3(k), hco3, co3(k), ie1)
      if (ph_alt(k) /= c0) then
        phlo = ph_alt(k) - PR%del_ph; phhi = ph_alt(k) + PR%del_ph
      else
        phlo = PR%phlo_3d_init; phhi = PR%phhi_3d_init
      endif
      call co2calc_interior1(t_dica, t_alka, t_po4, t_sio3, (PR%mass_to_vol), &
           k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, phlo, phhi, ph_alt(k), h2co3a, hco3a, co3a(k), ie2)
      call co3_sat1(tins(k), salt(k), press(k), (PR%mass_to_vol), csat(k), asat(k))
      if (ie1 /= 0 .or. ie2 /= 0) ierr = ierr + 1
    enddo
    sk%par_surf = PR%f_qsw_par * qsw
    if (sk%par_surf < PAR_threshold) sk%par_surf = c0
    sk%par_in = sk%par_surf
    sk%poc_s = c0; sk%poc_h = c0; sk%pop_s = c0; sk%pop_h = c0
    sk%cac_s = c0; sk%cac_h = c0; sk%caca_s = c0; sk%caca_h = c0
    sk%sio_s = c0; sk%sio_h = c0; sk%fe_s = c0; sk%fe_h = c0
    sk%dst_s = (c1 - 0.98_rp) * dust
    sk%dst_h = 0.98_rp * dust
    dst_rho = PR%hPOC_dust_ratio * 1.0e9_rp / 12.01_rp
    dst_rho = dst_rho * PR%nmol2mol_prefix * PR%mass2g
    sk%qa_dust_def = dst_rho * (sk%dst_s + sk%dst_h)
    do k = 1, nz
      do m = 1, PR%ntr
        trk(m) = tr(m, k)
      enddo
      ! lapply = 0: level_step returns the tendencies in trk
      call level_step(k, nz, dz(k), dz(1), zw(k), zt(k), tins(k), h2co3(k), co3(k), co3a(k), csat(k), dust, &
                      trk, sk, dtm0, lapply, dg, ldg, lsave, 1_ip, 1_ip, k)
      do m = 1, PR%ntr
        tend(m, k) = trk(m)
      enddo
      dg(14) = ph(k)
      diag(:, k) = dg
    enddo
  end subroutine marbl_interior_column_host
#endif

  ! Host launchers. tr: MARBL tracers (level kk, slot); ts: T/S array.
  subroutine marbl_carbonate_compute(ntr, nz, slot, nslot, tr, nts, ts, z_r, mask)
    integer(ip), intent(in) :: ntr, nz, slot, nslot, nts
    real(pp), WORK_ATTR intent(in) :: tr(ntr, 0:ntpcol, nz, nslot)
    real(rp), WORK_ATTR intent(in) :: ts(decomp%is:decomp%ie, decomp%js:decomp%je, nz, nslot, nts)
    real(rp), WORK_ATTR intent(in) :: z_r(decomp%is:decomp%ie, decomp%js:decomp%je, nz)
    real(rp), WORK_ATTR intent(in) :: mask(decomp%is:decomp%ie, decomp%js:decomp%je)
#ifdef USE_CUDA
    type(dim3) :: grid, block
    ! No explicit cudaLimitStackSize: the runtime sizes the per-thread
    ! stack from the kernels' static requirement, and a large explicit
    ! limit reserves stack x resident threads (64 kB cost 14 GB here).
    block = dim3(128, 1, 1)
    grid = dim3((nwet + 127) / 128, 1, nz)
    call carb_kernel<<<grid, block, 0, work_stream>>>(ntr, nz, slot, nslot, nts, &
         decomp%is, decomp%ie, decomp%js, decomp%je, decomp%nx, decomp%ny, nwet, wet_i, wet_j, tr, ntpcol, ts, z_r, mask)
#else
    error stop 'marbl_carbonate_compute: CUDA build required'
#endif
  end subroutine marbl_carbonate_compute

  subroutine marbl_interior_compute(ntr, nz, slot, nslot, tr, nts, ts, z_r, z_w, Hz, mask, dust, dtm, keep)
    integer(ip), intent(in) :: ntr, nz, slot, nslot, nts
    real(pp), WORK_ATTR intent(inout) :: tr(ntr, 0:ntpcol, nz, nslot)
    real(rp), WORK_ATTR intent(in) :: ts(decomp%is:decomp%ie, decomp%js:decomp%je, nz, nslot, nts)
    real(rp), WORK_ATTR intent(in) :: z_r(decomp%is:decomp%ie, decomp%js:decomp%je, nz)
    real(rp), WORK_ATTR intent(in) :: z_w(decomp%is:decomp%ie, decomp%js:decomp%je, 0:nz)
    real(rp), WORK_ATTR intent(in) :: Hz(decomp%is:decomp%ie, decomp%js:decomp%je, nz)
    real(rp), WORK_ATTR intent(in) :: mask(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(in) :: dust(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), intent(in) :: dtm
    logical, intent(in) :: keep
#ifdef USE_CUDA
    type(dim3) :: grid, block
    integer(ip) :: ldg
    ldg = 0
    if (keep .and. allocated(it_tend)) ldg = 1
    block = dim3(128, 1, 1)
    grid = dim3((nwet + 127) / 128, 1, 1)
    if (ldg /= 0) then
      call interior_kernel_dbg<<<grid, block, 0, work_stream>>>(ntr, nz, slot, nslot, nts, &
           decomp%is, decomp%ie, decomp%js, decomp%je, decomp%nx, decomp%ny, nwet, wet_i, wet_j, dtm, ldg, &
           tr, ntpcol, ts, z_r, z_w, Hz, mask, dust)
    else
      call interior_kernel<<<grid, block, 0, work_stream>>>(ntr, nz, slot, nslot, nts, &
           decomp%is, decomp%ie, decomp%js, decomp%je, decomp%nx, decomp%ny, nwet, wet_i, wet_j, dtm, ldg, &
           tr, ntpcol, ts, z_r, z_w, Hz, mask, dust)
    endif
#else
    error stop 'marbl_interior_compute: CUDA build required'
#endif
  end subroutine marbl_interior_compute

  subroutine marbl_zsat_compute(nz, z_r, z_w, mask)
    ! Calcite and aragonite saturation depths from the K1 carbonate fields,
    ! as marbl_diagnostics_mod::compute_saturation_depth: MARBL level 1 is
    ! the surface (ROMS k = nz), depths positive (zt = -z_r, zw = -z_w),
    ! kmt = nz. Land points keep 0 (the CPU driver writes 0 there too).
    integer(ip), intent(in) :: nz
    real(rp), WORK_ATTR intent(in) :: z_r(decomp%is:decomp%ie, decomp%js:decomp%je, nz)
    real(rp), WORK_ATTR intent(in) :: z_w(decomp%is:decomp%ie, decomp%js:decomp%je, 0:nz)
    real(rp), WORK_ATTR intent(in) :: mask(decomp%is:decomp%ie, decomp%js:decomp%je)
    call zsat_fields(nz, it_co3, it_co3sat, it_co3sata, z_r, z_w, mask, it_zsatc, it_zsata)
  end subroutine marbl_zsat_compute

  subroutine zsat_fields(nz, co3, satc, sata, z_r, z_w, mask, zc, za)
    ! Both depths in one sweep per column (CO3 read once); the sweep stops
    ! at the first undersaturated level of both, which is all it uses.
    integer(ip), intent(in) :: nz
    real(rp), WORK_ATTR intent(in) :: co3(decomp%is:decomp%ie, decomp%js:decomp%je, nz)
    real(rp), WORK_ATTR intent(in) :: satc(decomp%is:decomp%ie, decomp%js:decomp%je, nz)
    real(rp), WORK_ATTR intent(in) :: sata(decomp%is:decomp%ie, decomp%js:decomp%je, nz)
    real(rp), WORK_ATTR intent(in) :: z_r(decomp%is:decomp%ie, decomp%js:decomp%je, nz)
    real(rp), WORK_ATTR intent(in) :: z_w(decomp%is:decomp%ie, decomp%js:decomp%je, 0:nz)
    real(rp), WORK_ATTR intent(in) :: mask(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: zc(decomp%is:decomp%ie, decomp%js:decomp%je)
    real(rp), WORK_ATTR intent(inout) :: za(decomp%is:decomp%ie, decomp%js:decomp%je)
    integer(ip) :: i, j, k, kfc, kfa, nx, ny
    real(rp) :: a, x1, x2, y1, y2, z
    nx = decomp%nx; ny = decomp%ny
#ifdef USE_CUDA
    !$cuf kernel do(2) <<<*,*,0,work_stream>>>
#endif
    do j = 1, ny
      do i = 1, nx
        if (mask(i, j) /= c0) then
          ! kf = first MARBL level (1 = surface) with CO3 - sat <= 0, 0 if none
          kfc = 0; kfa = 0
          do k = 1, nz
            a = co3(i, j, nz + 1 - k) - satc(i, j, nz + 1 - k)
            if (a <= c0 .and. kfc == 0) kfc = k
            a = co3(i, j, nz + 1 - k) - sata(i, j, nz + 1 - k)
            if (a <= c0 .and. kfa == 0) kfa = k
            if (kfc /= 0 .and. kfa /= 0) exit
          enddo
          if (kfc == 0) then
            z = -z_w(i, j, 0)                 ! supersaturated down to the bottom
          else if (kfc == 1) then
            z = c0                            ! undersaturated at the surface
          else
            x1 = -z_r(i, j, nz + 2 - kfc); x2 = -z_r(i, j, nz + 1 - kfc)
            y1 = co3(i, j, nz + 2 - kfc) - satc(i, j, nz + 2 - kfc)
            y2 = co3(i, j, nz + 1 - kfc) - satc(i, j, nz + 1 - kfc)
            if (y2 == c0) then
              z = x2
            else
              z = x1 - (x2 - x1) / (y2 - y1) * y1   ! marbl_utils_linear_root
            endif
          endif
          zc(i, j) = z
          if (kfa == 0) then
            z = -z_w(i, j, 0)
          else if (kfa == 1) then
            z = c0
          else
            x1 = -z_r(i, j, nz + 2 - kfa); x2 = -z_r(i, j, nz + 1 - kfa)
            y1 = co3(i, j, nz + 2 - kfa) - sata(i, j, nz + 2 - kfa)
            y2 = co3(i, j, nz + 1 - kfa) - sata(i, j, nz + 1 - kfa)
            if (y2 == c0) then
              z = x2
            else
              z = x1 - (x2 - x1) / (y2 - y1) * y1
            endif
          endif
          za(i, j) = z
        endif
      enddo
    enddo
  end subroutine zsat_fields

end module marbl_interior_mod
