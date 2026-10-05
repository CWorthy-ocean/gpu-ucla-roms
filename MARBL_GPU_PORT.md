# MARBL on the GPU model: port notes (2026-10-02)

Status: surface-flux and interior paths ported to CUDA Fortran, validated
column by column against unchanged CPU MARBL, integrated into the model
(MiniPac, 32 tracers, hourly MARBL) and validated over a day against CPU
ROMS-MARBL with the noise-floor method (all tracers within the CPU model's
own perturbation sensitivity).

## Layout

| Piece | File | Notes |
|---|---|---|
| MARBL sources | `$(MARBL_SRC)/*.F90`, built into `build_gpu/marbl/*.o` | unchanged upstream code, compiled with nvfortran as host code; `marbl_interior_tendency_mod.F90` at -O0 (nvfortran -O2 miscompiles its WHERE constructs) |
| Host instance | `src/host/marbl_host.F90` | `put_setting` from `marbl_in`, `init(nz, mks)`; exports tracer names/indices, forcing indices, the flat parameter table `marbl_par_t`; per-column reference calls (`marbl_host_sflux_column`, `marbl_host_interior_column`) for tests |
| Surface fluxes | `src/device/marbl_sflux.F90` | one thread per point: gas exchange (Schmidt, O2 saturation), carbonate solver (coefficients incl. optional pressure correction, safeguarded Newton `drtsafe1`), NHx emission, dust/iron/NOx/NHy deposition; arrays `sf_*` (forcing, pH saved state, fluxes, 15 diagnostics) |
| Interior | `src/device/marbl_interior.F90` | K1 `carb_kernel` (3-D pointwise: in-situ T, two carbonate solves, CO3 saturation) and K2 `interior_kernel` (one thread per column, surface to bottom, PAR + full ecosystem + sinking recurrence streamed level by level with scalar state `sink_t`); tendencies applied in place to slot nnew; parameters in the static device table `prd` |
| Model driver | `src/host/marbl_step.F90` (`marbl_gpu`) | namelist, setup, forcing, the per-call sequence (CPU order: fluxes applied to the top layer, then interior), restart/history hooks; history carries the surface set (PH, pCO2SURF, DpCO2, FG_CO2, FG_ALT_CO2, PH_ALT_CO2, pCO2SURF_ALT_CO2), the saturation depths zsatcalc/zsatarag (`marbl_zsat_compute`, MARBL's `compute_saturation_depth` on the K1 fields) and the 3-D CO3, CO3_ALT_CO2, co3_sat_calc, co3_sat_arag (float32, CPU names and units) |
| Tracers | `src/device/passive.F90` MARBL mode | the 32 MARBL tracers are the passive tracers (names from MARBL, IC by name, boundary data `tb_*` from the bgc boundary file); build with `-DPASSIVE_FP64` for double storage |
| Forcing | `src/host/frc_io.F90`, `src/host/bry_io.F90` | climatology cycling (`cycle_length`, same bracket as the CPU `find_new_record_cycle`), bgc surface series (dust, iron, nox, nhy, xco2_air, xco2_air_alt) and per-tracer boundary series |

Namelist: `nt_bgc = 32` turns MARBL on; `&marbl_biogeochemistry_settings`
(`marbl_config_file`, `marbl_timestep`); the bgc files go in `frcfiles`.
Example: `cases/GPU/namelist_roms_gpu_marbl.nml`, job `submit_roms_gpu_marbl.sh`.

## Validation

Column tests (`tests/test_marbl_sflux.F90`, `tests/test_marbl_interior.F90`,
jobs `cases/GPU/submit_test_marbl_*.sh`) feed identical MiniPac columns from
the UP3 restart to CPU MARBL and to the kernels:

- surface flux, 4096 columns, two pH-bracket scenarios: deposition terms,
  Schmidt numbers, piston velocities, NHx emission exact; pH 4e-15
  relative; CO2 fluxes 1e-9 relative worst case (round-off of the air-sea
  difference near equilibrium).
- interior, 2304 full columns (230,400 levels): all 32 tendencies within
  4e-16 relative, pH within 4e-15, PAR/nitrification within 2e-15; no
  solver failures. The host-compiled copy of the same source is bit-exact
  against MARBL.
- MARBL off: day run restart bit-identical and history identical to the
  pre-port reference.

Timing (A100, warm, 25,600 columns = 15% of MiniPac): carbonate 59 ms,
interior 93 ms per call, i.e. about 1 s per hourly call for the full
domain before any tuning (255 registers, 3.3 kB stack per thread in K2).

## Day run against CPU ROMS-MARBL (MiniPac, 2010-03-30, 32 tracers, hourly MARBL)

Cases: GPU `cases/GPU/namelist_roms_gpu_marbl.nml`; CPU reference and
perturbed twin (initial T + 1e-6 degC pattern) in `cases/CPU_marbl`
(same dycore sources as the matched CPU case, MARBL/NOX/NHY/PCO2AIR
enabled). Evaluation `scratchpad/marbl_noise_floor_eval.py`: RMS over wet
points of (GPU - ref) and (twin - ref), ratio = GPU/twin.

Result (first full run): every one of the 32 tracers stays inside the CPU
model's own perturbation sensitivity at hours 1, 6, 12 and 24; the worst
ratio over the day is 0.90 (diatC at hour 24), most are 0.3-0.6, and the
surface pH and pCO2 ratios are 0.35 and 0.32. 24 MARBL calls, no
carbonate-solver failures, 27.5 GB device memory. The GPU-CPU difference
grows from ~1e-8 of the tracer scale at hour 1 (round-off) to the
twin's level by day end, as it does for the physics. FG_CO2 at hour 1 had
a relative difference of ~1e-5 before the twin's signal had grown (ratio
33): the bgc forcing was interpolated to the n+3/2 boundary time instead
of the CPU's n+1/2 set_forces time; fixed in `marbl_gpu_step`. After the
fix the hour-1 FG_CO2 difference is 2e-17 (round-off) and its ratio at
hours 6 and 24 is 0.00 and 0.31; the worst tracer ratio stays 0.90.

Three-day run (2026-10-05, same setup, 6-hourly output; table in
`MARBL_NOISE_FLOOR_3D_2026-10-05.txt`): hour 24 reproduces the one-day
numbers exactly (both models are deterministic across runs). The worst
ratio over all 32 tracers and the surface diagnostics is 0.90 at hour 24,
0.97 at hour 48 and 1.04 at hour 72 (spCaCO3); in the double-precision
day-3 restarts every field, including temperature (0.66), salinity (0.98)
and the saved 3-D pH (0.84), is at or below the twin's level. The drift of
the ratios toward 1 is the expected saturation: by day 3 both the GPU-CPU
difference and the twin's 1e-6 degC perturbation have grown to the model's
own divergence level, so the two are statistically indistinguishable.
72 hourly MARBL calls, no solver failures, 27.5 GB.

Cost: a step with 32 FP64 tracers takes 0.42 s (0.15 s for the dycore
with T/S only), and the step that includes the hourly MARBL call 0.72 s,
so MARBL itself costs about 0.3 s per hour of simulation.

## Toolchain lessons (nvfortran 25.9)

- A by-value derived type as a global-kernel argument does not compile;
  parameters are scalar `value` args or a module-level `device` struct
  (a `managed` struct gave illegal-address errors at run time).
- Device subroutines must take every dummy by reference; mixing `value`
  inputs with by-reference outputs returned undefined inputs, and even
  pure device functions with `value` args (NHx emission) returned NaN in
  the FP64 model build although they passed the float32 column test.
- The inliner miscompiled callers of a device/host *function* with
  `value` args (`potemp_fn`): wrong pH saved state and a zero diagnostic
  with correct tendencies. Rewritten as a subroutine. Use
  `-Minline=except:<name>` to bisect such cases.
- MARBL's own interior module segfaults at -O2 (built at -O0 for the
  host reference).
- Never set `cudaLimitStackSize` large: the reservation is stack size x
  resident threads (64 kB cost 14 GB of device memory and slowed every
  kernel). The runtime sizes the stack from each kernel's requirement.
- The I/O server caps variables per record (`MAXV` in `io_server.F90`,
  now 96) and the staging capacity counts the MARBL 2-D diagnostics.

## Open items

- Restart precision: with `-DPASSIVE_FP64` the tracers are written to the
  restart in double; the history stays float32.
- History diagnostics match the CPU `_bgc_dia` set used for MiniPac
  (surface set + zsatcalc/zsatarag + 3-D CO3, CO3_ALT_CO2, co3_sat_calc,
  co3_sat_arag; added 2026-10-05). Over the three-day run they are at the
  noise floor like the tracers: CO3 ratio 0.10 (h6) to 0.72 (h72),
  co3_sat 0.82 worst, zsat 0.07 worst. The 3-D pH is in the restart only
  (`MARBL_PH_3D`), as in the CPU model. Other MARBL diagnostics
  (production, remineralisation, fluxes per PFT) are not written.
- Performance: K2 register pressure, carbonate solver cost, the 128
  per-step boundary fills for the tracers (could move to the hourly call).
- Not ported (not used by MiniPac): ciso, abio_dic, Lig_cnt = 2, restoring,
  O2/remin scale-factor forcings, ladjust_bury_coeff, PAR subcolumns > 1.
