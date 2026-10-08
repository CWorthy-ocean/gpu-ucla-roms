#include "mem_attr.inc"
module marbl_sflux_mod
  ! MARBL surface fluxes on the device: a transliteration of
  ! marbl_surface_flux_mod + marbl_co2calc_mod (surface branch) +
  ! marbl_schmidt_number_mod + marbl_oxygen + marbl_nhx_surface_emis_mod,
  ! one thread per surface point, reading the model's device arrays
  ! directly. Arithmetic follows MARBL expression by expression so the only
  ! differences from CPU MARBL are transcendental-function and FMA rounding.
  !
  ! Layout: forcing, saved state (pH), fluxes and diagnostics are plain
  ! (i,j[,m]) managed arrays; no per-column marshalling. Device helpers take
  ! and return everything by value (CUF-kernel convention of this code base):
  ! multi-valued results come back in small derived types.
  use kinds
  use gpu_state, only: nwet, wet_i, wet_j, ntpcol
  use decomp_mod, only: decomp
#ifdef USE_CUDA
  use arch_mod, only: work_stream
#endif
  use marbl_host, only: marbl_sf_par_t, MSF_NDIAG, D_PH, D_PCO2, D_DPCO2, D_FGCO2, &
       D_PH_ALT, D_PCO2_ALT, D_FGCO2_ALT, D_PVO2, D_SCHO2, D_O2SAT, D_CO2STAR, &
       D_DCO2STAR, D_PVCO2, D_SCHCO2, D_NHX
#ifdef USE_CUDA
  use cudafor
#endif
  implicit none
#include "cuf_launch.inc"
  private

  ! Surface forcing (model units as the ROMS driver hands them to MARBL):
  ! u10_sqr m2/s2, dust kg/m2/s, iron/nox/nhy mmol/m2/s, xco2 ppm.
  real(rp), WORK_ATTR allocatable, public :: sf_u10sqr(:,:), sf_dust(:,:), sf_iron(:,:), &
                                             sf_nox(:,:), sf_nhy(:,:), sf_xco2(:,:), sf_xco2_alt(:,:)
  ! Saved state: previous surface pH (0 = no previous solution).
  real(rp), WORK_ATTR allocatable, public :: sf_ph(:,:), sf_ph_alt(:,:)
  ! Outputs: surface fluxes (conc units * m/s) per tracer, diagnostics.
  real(rp), WORK_ATTR allocatable, public :: sf_flux(:,:,:)      ! (i,j,m)
  real(rp), WORK_ATTR allocatable, public :: sf_diag(:,:,:)      ! (i,j,MSF_NDIAG)
  integer(ip), WORK_ATTR allocatable, public :: sf_err(:,:)      ! 0 ok, 1 no bracket, 2 no convergence

  ! MARBL constants (marbl_constants_mod, marbl_co2calc_mod)
  real(rp), parameter :: T0_Kelvin = 273.15_rp
  real(rp), parameter :: xacc = 1.0e-10_rp
  integer,  parameter :: max_bracket_grow_it = 3
  integer,  parameter :: maxit = 100
  real(rp), parameter :: salt_min = 0.1_rp
  real(rp), parameter :: dic_min = salt_min / 35.0_rp * 1944.0_rp
  real(rp), parameter :: alk_min = salt_min / 35.0_rp * 2225.0_rp
  real(rp), parameter :: hrps = 1.0_rp / 3600.0_rp
  real(rp), parameter :: vonkar = 0.4_rp

  public :: alloc_marbl_sflux, free_marbl_sflux, marbl_sflux_compute
  public :: co2_coeffs, ta_fn, drtsafe1, pcorr
  public :: T0_Kelvin, salt_min, dic_min, alk_min, xacc, max_bracket_grow_it, maxit

contains

  subroutine alloc_marbl_sflux(ntr)
    integer(ip), intent(in) :: ntr
    integer(ip) :: is, ie, js, je
    is = decomp%is; ie = decomp%ie; js = decomp%js; je = decomp%je
    call free_marbl_sflux()
    allocate(sf_u10sqr(is:ie,js:je), sf_dust(is:ie,js:je), sf_iron(is:ie,js:je), &
             sf_nox(is:ie,js:je), sf_nhy(is:ie,js:je), sf_xco2(is:ie,js:je), sf_xco2_alt(is:ie,js:je))
    allocate(sf_ph(is:ie,js:je), sf_ph_alt(is:ie,js:je))
    allocate(sf_flux(is:ie,js:je,ntr), sf_diag(is:ie,js:je,MSF_NDIAG), sf_err(is:ie,js:je))
    sf_u10sqr = 0.0_rp; sf_dust = 0.0_rp; sf_iron = 0.0_rp; sf_nox = 0.0_rp; sf_nhy = 0.0_rp
    sf_xco2 = 284.7_rp; sf_xco2_alt = 284.7_rp
    sf_ph = 0.0_rp; sf_ph_alt = 0.0_rp
    sf_flux = 0.0_rp; sf_diag = 0.0_rp; sf_err = 0
  end subroutine alloc_marbl_sflux

  subroutine free_marbl_sflux()
    if (allocated(sf_u10sqr)) deallocate(sf_u10sqr, sf_dust, sf_iron, sf_nox, sf_nhy, sf_xco2, sf_xco2_alt)
    if (allocated(sf_ph)) deallocate(sf_ph, sf_ph_alt)
    if (allocated(sf_flux)) deallocate(sf_flux, sf_diag, sf_err)
  end subroutine free_marbl_sflux

  !---------------------------------------------------------------------------
  ! Device helpers (scalar versions of the MARBL routines)
  !---------------------------------------------------------------------------

  ! marbl_schmidt_number_mod: schmidt_co2_surf
  DEVICE_ATTR
  subroutine schmidt_co2(sst_in, s)
    real(rp), intent(in) :: sst_in
    real(rp), intent(out) :: s
    real(rp) :: sst
    real(rp), parameter :: a = 2116.8_rp, b = -136.25_rp, c = 4.7353_rp, d = -0.092307_rp, e = 0.0007555_rp
    sst = max(-2.0_rp, min(40.0_rp, sst_in))
    s = a + sst * (b + sst * (c + sst * (d + sst * e)))
  end subroutine schmidt_co2

  ! marbl_oxygen: schmidt_o2_surf
  DEVICE_ATTR
  subroutine schmidt_o2(sst_in, s)
    real(rp), intent(in) :: sst_in
    real(rp), intent(out) :: s
    real(rp) :: sst
    real(rp), parameter :: a = 1920.4_rp, b = -135.6_rp, c = 5.2122_rp, d = -0.10939_rp, e = 0.00093777_rp
    sst = max(-2.0_rp, min(40.0_rp, sst_in))
    s = a + sst * (b + sst * (c + sst * (d + sst * e)))
  end subroutine schmidt_o2

  ! marbl_schmidt_number_mod: schmidt_nh3_surf
  DEVICE_ATTR
  subroutine schmidt_nh3_surf(sst_in, s)
    real(rp), intent(in) :: sst_in
    real(rp), intent(out) :: s
    real(rp) :: sst
    real(rp), parameter :: a = 1.809018e+03_rp, b = -1.199578e+02_rp, c = 4.369922e+00_rp, &
                           d = -8.824558e-02_rp, e = 7.377934e-04_rp
    sst = max(-2.0_rp, min(40.0_rp, sst_in))
    s = a + sst * (b + sst * (c + sst * (d + sst * e)))
  end subroutine schmidt_nh3_surf

  ! marbl_schmidt_number_mod: schmidt_nh3_air
  DEVICE_ATTR
  subroutine schmidt_nh3_air(sst_in, atmpres, s)
    real(rp), intent(in) :: sst_in, atmpres
    real(rp), intent(out) :: s
    real(rp) :: sst
    real(rp), parameter :: a = 7.537648e-01_rp, b = 4.728550e-04_rp, c = -2.546799e-05_rp, &
                           d = 3.187285e-07_rp, e = -1.985024e-09_rp
    sst = max(-2.0_rp, min(40.0_rp, sst_in))
    s = a + sst * (b + sst * (c + sst * (d + sst * e)))
    s = atmpres * atmpres * s
  end subroutine schmidt_nh3_air

  ! marbl_oxygen: o2sat_surf (1 atm)
  DEVICE_ATTR
  subroutine o2sat_1atm(sst, sss, o)
    real(rp), intent(in) :: sst, sss
    real(rp), intent(out) :: o
    real(rp) :: ts
    real(rp), parameter :: a_0 = 2.00907_rp, a_1 = 3.22014_rp, a_2 = 4.05010_rp, a_3 = 4.94457_rp, &
         a_4 = -2.56847E-1_rp, a_5 = 3.88767_rp, b_0 = -6.24523E-3_rp, b_1 = -7.37614E-3_rp, &
         b_2 = -1.03410E-2_rp, b_3 = -8.17083E-3_rp, c_0 = -4.88682E-7_rp
    ts = log( ((T0_Kelvin + 25.0_rp) - sst) / (T0_Kelvin + sst) )
    o = exp(a_0 + ts*(a_1 + ts*(a_2 + ts*(a_3 + ts*(a_4 + ts*a_5)))) + &
            sss*( (b_0 + ts*(b_1 + ts*(b_2 + ts*b_3))) + sss*c_0 ))
    o = o / 0.0223916_rp
  end subroutine o2sat_1atm

#include "marbl_carb_helpers.inc"

  ! marbl_co2calc_mod: comp_htotal + marbl_co2calc_surface tail (one element).
  ! On a solver failure ierr /= 0 and ph is returned unchanged (MARBL leaves
  ! its outputs untouched and logs an error).
  DEVICE_ATTR
  subroutine co2calc_surface1(dic_in, xco2_in, ta_in, pt_in, sit_in, atmpres, mass_to_vol, &
                              k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, &
                              phlo, phhi, ph_in, oph, oco3, oco2star, odco2star, opco2surf, odpco2, oierr)
    real(rp), intent(in) :: dic_in, xco2_in, ta_in, pt_in, sit_in, atmpres, mass_to_vol, phlo, phhi, ph_in
    real(rp), intent(in) :: k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft
    real(rp), intent(out) :: oph, oco3, oco2star, odco2star, opco2surf, odpco2
    integer(ip), intent(out) :: oierr
    real(rp) :: dic, ta, pt, sit
    real(rp) :: vol_to_mass, x1, x2, htotal, htotal2, denom, xco2, co2starair
    integer(ip) :: ierr
    real(rp), parameter :: c0 = 0.0_rp, c1 = 1.0_rp, c10 = 10.0_rp

    vol_to_mass = c1 / mass_to_vol
    dic = max(dic_in, dic_min) * vol_to_mass
    ta  = max(ta_in, alk_min)  * vol_to_mass
    pt  = max(pt_in, c0)       * vol_to_mass
    sit = max(sit_in, c0)      * vol_to_mass
    x1 = c10 ** (-phhi)
    x2 = c10 ** (-phlo)
    call drtsafe1(k1, k2, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, dic, ta, pt, sit, &
                  x1, x2, htotal, ierr)
    oph = ph_in; oco3 = c0; oco2star = c0; odco2star = c0; opco2surf = c0; odpco2 = c0
    oierr = ierr
    if (ierr /= 0) return

    xco2 = xco2_in * 1e-6_rp
    htotal2    = htotal ** 2
    denom      = c1 / (htotal2 + k1 * htotal + k1 * k2)
    oco3       = dic * k1 * k2 * denom
    oco2star   = dic * htotal2 / (htotal2 + k1 * htotal + k1 * k2)
    co2starair = xco2 * ff * atmpres
    odco2star  = co2starair - oco2star
    oph        = -log10(htotal)
    opco2surf  = oco2star / ff
    odpco2     = opco2surf - xco2 * atmpres
    oco3       = oco3      * mass_to_vol
    oco2star   = oco2star  * mass_to_vol
    odco2star  = odco2star * mass_to_vol
    opco2surf  = opco2surf * 1e6_rp
    odpco2     = odpco2    * 1e6_rp
  end subroutine co2calc_surface1

  ! marbl_nhx_surface_emis_mod (one element)
  DEVICE_ATTR
  subroutine nhx_emis1(nh4, ph, sst, sss, u10_sqr, atmpres, ifrac, len2m, cm2len, m2len, emis)
    real(rp), intent(in) :: nh4, ph, sst, sss, u10_sqr, atmpres, ifrac, len2m, cm2len, m2len
    real(rp), intent(out) :: emis
    real(rp) :: u10_rms_mps, schmidt_nh3, kw_nh3, kg_nh3, ustar_div_u10, ustar
    real(rp) :: pKa, sstK_inv, H_nhx, Hstar_nhx, K
    real(rp), parameter :: c0 = 0.0_rp, c1 = 1.0_rp, c2 = 2.0_rp

    u10_rms_mps = len2m * sqrt(u10_sqr)
    ! kw
    call schmidt_nh3_surf(sst, schmidt_nh3)
    kw_nh3 = hrps * cm2len &
      * u10_rms_mps * (0.061_rp + 0.24_rp * u10_rms_mps) * sqrt(600.0_rp / schmidt_nh3)
    ! kg
    call schmidt_nh3_air(sst, atmpres, schmidt_nh3)
    ustar_div_u10 = sqrt(6.1e-4_rp + 6.3e-5_rp * u10_rms_mps)
    ustar = ustar_div_u10 * u10_rms_mps
    kg_nh3 = 1.0e-3_rp + ustar &
         / (13.3_rp * sqrt(schmidt_nh3) + c1 / ustar_div_u10 - 5.0_rp + log(schmidt_nh3) / (c2 * vonkar))
    kg_nh3 = m2len * kg_nh3
    ! Hstar
    pKa = 10.0423_rp - 0.0315536_rp * sst + 0.003071_rp * sss
    sstK_inv = c1 / (sst + T0_Kelvin)
    H_nhx = (273.15_rp / 17.93_rp) * sstK_inv * exp(9.7_rp - 4092._rp * sstK_inv)
    Hstar_nhx = H_nhx / (c1 + 10.0_rp ** (pKa - ph))
    K = c1 / (c1 / kg_nh3 + Hstar_nhx / kw_nh3)
    emis = (c1 - ifrac) * K * Hstar_nhx * max(nh4, c0)
  end subroutine nhx_emis1

  !---------------------------------------------------------------------------
  ! Surface fluxes for all points, one thread per (i,j). tr: MARBL tracers
  ! (level kk, slot); ts: temperature/salinity array (itemp/isalt at level
  ! kk, slot). Masked points get zero fluxes/diagnostics and unchanged pH.
  ! Written as an explicit global kernel (not a CUF loop) because the
  ! helpers return several values by reference.
  !---------------------------------------------------------------------------
#ifdef USE_CUDA
  attributes(global) &
  subroutine sflux_kernel(ntr, nz, kk, slot, nts, is, ie, js, je, nx, ny, nwet, wet_i, wet_j, &
       ipo4, ino3, isio3, inh4, ife, io2, idic, idica, ialk, ialka, lo2, lco2, lnhx, &
       xkw_coeff, mass_to_vol, dust_po4_fac, dust_sio3_fac, len2m, cm2len, m2len, &
       del_ph, phlo_init, phhi_init, ifrac, atm_pres, tr, ncl, ts, mask)
    ! Parameters come in as scalars: nvfortran 25.9 rejects a by-value
    ! derived type as a global-kernel argument ("unexpected load datatype").
    integer(ip), value :: ntr, nz, kk, slot, nts, is, ie, js, je, nx, ny
    integer(ip), value :: nwet
    integer(ip), device, intent(in) :: wet_i(*), wet_j(*)
    integer(ip) :: c
    integer(ip), value :: ipo4, ino3, isio3, inh4, ife, io2, idic, idica, ialk, ialka, lo2, lco2, lnhx
    real(rp), value :: xkw_coeff, mass_to_vol, dust_po4_fac, dust_sio3_fac, len2m, cm2len, m2len
    real(rp), value :: del_ph, phlo_init, phhi_init, ifrac, atm_pres
    integer(ip), value :: ncl
    real(pp), device, intent(in) :: tr(ntr, 0:ncl, nz, 3)
    real(rp), device, intent(in) :: ts(is:ie, js:je, nz, 3, nts)
    real(rp), device, intent(in) :: mask(is:ie, js:je)
    integer(ip) :: i, j, m, ierr
    real(rp) :: sst, sss, u10sqr, xco2, xco2a, dust, iron, nox, nhy
    real(rp) :: piston, xkw_ice, sch_co2, pv_co2, sch_o2, pv_o2, o2s, flux_o2, flux_co2, flux_alt
    real(rp) :: ph, ph_alt, ph0, phlo, phhi, nhx
    real(rp) :: f_o2, f_dic, f_dica, f_nh4, f_fe, f_po4, f_sio3, f_no3, f_alk, f_alka
    real(rp) :: k0, k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft
    real(rp) :: c_ph, c_co3, c_co2star, c_dco2star, c_pco2surf, c_dpco2
    real(rp) :: t_dic, t_alk, t_po4, t_sio3, t_dica, t_alka, t_nh4
    integer(ip), parameter :: nopress = 0
    real(rp) :: a_ph, a_co3, a_co2star, a_dco2star, a_pco2surf, a_dpco2
    integer(ip) :: c_ierr
    real(rp), parameter :: c0 = 0.0_rp, c1 = 1.0_rp

    c = (blockIdx%x - 1) * blockDim%x + threadIdx%x
    if (c > nwet) return
    i = wet_i(c); j = wet_j(c)
    do m = 1, ntr
      sf_flux(i, j, m) = c0
    enddo
    do m = 1, MSF_NDIAG
      sf_diag(i, j, m) = c0
    enddo
    ierr = 0
    if (mask(i, j) /= c0) then
      sst = ts(i, j, kk, slot, itemp)
      sss = ts(i, j, kk, slot, isalt)
      u10sqr = sf_u10sqr(i, j)
      xco2 = sf_xco2(i, j); xco2a = sf_xco2_alt(i, j)
      dust = sf_dust(i, j); iron = sf_iron(i, j); nox = sf_nox(i, j); nhy = sf_nhy(i, j)
      f_o2 = c0; f_dic = c0; f_dica = c0; f_nh4 = c0; f_fe = c0
      f_po4 = c0; f_sio3 = c0; f_no3 = c0; f_alk = c0; f_alka = c0
      pv_co2 = c0; sch_co2 = c0; pv_o2 = c0; sch_o2 = c0; o2s = c0
      flux_co2 = c0; flux_alt = c0; nhx = c0; piston = c0; xkw_ice = c0
      ph = sf_ph(i, j); ph_alt = sf_ph_alt(i, j)
      c_ph = ph; c_co3 = c0; c_co2star = c0; c_dco2star = c0; c_pco2surf = c0; c_dpco2 = c0; c_ierr = 0
      a_ph = ph_alt; a_co3 = c0; a_co2star = c0; a_dco2star = c0; a_pco2surf = c0; a_dpco2 = c0

      if (lo2 /= 0 .or. lco2 /= 0) then
        piston = xkw_coeff * u10sqr
        xkw_ice = (c1 - ifrac) * piston
      endif
      if (lco2 /= 0) then
        call schmidt_co2(sst, sch_co2)
        pv_co2 = xkw_ice * sqrt(660.0_rp / sch_co2)
      endif
      if (lo2 /= 0) then
        call schmidt_o2(sst, sch_o2)
        call o2sat_1atm(sst, sss, o2s)
        pv_o2 = xkw_ice * sqrt(660.0_rp / sch_o2)
        o2s = atm_pres * o2s
        flux_o2 = pv_o2 * (o2s - real(tr(io2, c, kk, slot), rp))
        f_o2 = f_o2 + flux_o2
      endif
      if (lco2 /= 0) then
        if (ph /= c0) then
          phlo = ph - del_ph
          phhi = ph + del_ph
        else
          phlo = phlo_init
          phhi = phhi_init
        endif
        call co2_coeffs(sst, sss, c0, nopress, k0, k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft)
        t_dic = real(tr(idic, c, kk, slot), rp); t_alk = real(tr(ialk, c, kk, slot), rp)
        t_po4 = real(tr(ipo4, c, kk, slot), rp); t_sio3 = real(tr(isio3, c, kk, slot), rp)
        t_dica = real(tr(idica, c, kk, slot), rp); t_alka = real(tr(ialka, c, kk, slot), rp)
        ph0 = ph
        call co2calc_surface1(t_dic, xco2, t_alk, t_po4, t_sio3, atm_pres, mass_to_vol, &
             k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, phlo, phhi, ph, &
             c_ph, c_co3, c_co2star, c_dco2star, c_pco2surf, c_dpco2, c_ierr)
        ierr = c_ierr
        if (ierr == 0) then
          ph = c_ph
          flux_co2 = pv_co2 * c_dco2star
          if (t_dica == t_dic .and. t_alka == t_alk .and. xco2a == xco2 .and. ph_alt == ph0) then
            ! identical inputs and bracket: the alternative-CO2 solve would
            ! repeat the base one exactly, so reuse its results
            a_ph = c_ph; a_co3 = c_co3; a_co2star = c_co2star; a_dco2star = c_dco2star
            a_pco2surf = c_pco2surf; a_dpco2 = c_dpco2
            ph_alt = a_ph
            flux_alt = pv_co2 * a_dco2star
          else
            if (ph_alt /= c0) then
              phlo = ph_alt - del_ph
              phhi = ph_alt + del_ph
            else
              phlo = phlo_init
              phhi = phhi_init
            endif
            call co2calc_surface1(t_dica, xco2a, t_alka, t_po4, t_sio3, atm_pres, mass_to_vol, &
                 k1, k2, ff, kw, kb, ks, kf, k1p, k2p, k3p, ksi, bt, st, ft, phlo, phhi, ph_alt, &
                 a_ph, a_co3, a_co2star, a_dco2star, a_pco2surf, a_dpco2, c_ierr)
            ierr = c_ierr
            if (ierr == 0) then
              ph_alt = a_ph
              flux_alt = pv_co2 * a_dco2star
            endif
          endif
        endif
        f_dic = f_dic + flux_co2
        f_dica = f_dica + flux_alt
      endif
      if (lnhx /= 0) then
        t_nh4 = real(tr(inh4, c, kk, slot), rp)
        call nhx_emis1(t_nh4, ph, sst, sss, u10sqr, atm_pres, ifrac, len2m, cm2len, m2len, nhx)
        f_nh4 = f_nh4 - nhx
      endif
      f_fe = f_fe + iron
      f_po4  = f_po4  + dust_po4_fac  * dust
      f_sio3 = f_sio3 + dust_sio3_fac * dust
      f_no3 = f_no3 + nox
      f_nh4 = f_nh4 + nhy
      f_alk  = f_alk  + f_nh4 - f_no3
      f_alka = f_alka + f_nh4 - f_no3

      sf_flux(i, j, io2) = f_o2
      sf_flux(i, j, idic) = f_dic
      sf_flux(i, j, idica) = f_dica
      sf_flux(i, j, inh4) = f_nh4
      sf_flux(i, j, ife) = f_fe
      sf_flux(i, j, ipo4) = f_po4
      sf_flux(i, j, isio3) = f_sio3
      sf_flux(i, j, ino3) = f_no3
      sf_flux(i, j, ialk) = f_alk
      sf_flux(i, j, ialka) = f_alka
      sf_ph(i, j) = ph
      sf_ph_alt(i, j) = ph_alt

      sf_diag(i, j, D_PH) = ph
      sf_diag(i, j, D_PCO2) = c_pco2surf
      sf_diag(i, j, D_DPCO2) = c_dpco2
      sf_diag(i, j, D_FGCO2) = flux_co2
      sf_diag(i, j, D_PH_ALT) = ph_alt
      sf_diag(i, j, D_PCO2_ALT) = a_pco2surf
      sf_diag(i, j, D_FGCO2_ALT) = flux_alt
      sf_diag(i, j, D_PVO2) = pv_o2
      sf_diag(i, j, D_SCHO2) = sch_o2
      sf_diag(i, j, D_O2SAT) = o2s
      sf_diag(i, j, D_CO2STAR) = c_co2star
      sf_diag(i, j, D_DCO2STAR) = c_dco2star
      sf_diag(i, j, D_PVCO2) = pv_co2
      sf_diag(i, j, D_SCHCO2) = sch_co2
      sf_diag(i, j, D_NHX) = nhx
    endif
    sf_err(i, j) = ierr
  end subroutine sflux_kernel
#endif

  subroutine marbl_sflux_compute(p, ntr, nz, kk, slot, tr, nts, ts, mask)
    type(marbl_sf_par_t), intent(in) :: p
    integer(ip), intent(in) :: ntr, nz, kk, slot, nts
    real(pp), WORK_ATTR intent(in) :: tr(ntr, 0:ntpcol, nz, 3)
    real(rp), WORK_ATTR intent(in) :: ts(decomp%is:decomp%ie, decomp%js:decomp%je, nz, 3, nts)
    real(rp), WORK_ATTR intent(in) :: mask(decomp%is:decomp%ie, decomp%js:decomp%je)
#ifdef USE_CUDA
    type(dim3) :: grid, block
    block = dim3(128, 1, 1)
    grid = dim3((nwet + 127) / 128, 1, 1)
    call sflux_kernel<<<grid, block, 0, work_stream>>>(ntr, nz, kk, slot, nts, &
         decomp%is, decomp%ie, decomp%js, decomp%je, decomp%nx, decomp%ny, nwet, wet_i, wet_j, &
         p%po4, p%no3, p%sio3, p%nh4, p%fe, p%o2, p%dic, p%dic_alt, p%alk, p%alk_alt, &
         p%flux_o2, p%flux_co2, p%nhx_emis, &
         p%xkw_coeff, p%mass_to_vol, p%dust_po4_fac, p%dust_sio3_fac, p%len2m, p%cm2len, p%m2len, &
         p%del_ph, p%phlo_init, p%phhi_init, p%ifrac, p%atm_pres, tr, ntpcol, ts, mask)
#else
    error stop 'marbl_sflux_compute: CUDA build required'
#endif
  end subroutine marbl_sflux_compute

end module marbl_sflux_mod
