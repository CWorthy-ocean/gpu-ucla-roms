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

## Passive-tracer transport performance (2026-10-05)

Profile of the 32-FP64-tracer step before this work (nsys, steady state,
per step): passive kernels 268 ms (predictor horizontal 41, vertical 40,
implicit 38; corrector horizontal 42, vertical 42, implicit 38; t3dmix 27),
dycore + T/S 153 ms, MARBL 118 ms on the hourly step. All passive kernels
are fused over the 32 tracers already; each ran at ~1.2 ms per tracer,
20-70% of memory bandwidth, at 25% occupancy (128 registers).

Changes (all bit-identical to the previous output in the FP64 build,
verified on the 3-day run `minipac_gpu_marbl3dv3` vs `3dv2`):

- `src/device/passive_column.F90` (new module): vertical advection and
  the implicit solve fused into one column kernel, one thread per
  (block of NTB = 2 tracers, column); the Thomas intermediate is kept in
  place in `tp(nn)`, the column factors come from a separate small
  factorization kernel (`passive_factor`, 1 ms). NTB = 2 (50 ms per call)
  beat 1 (55), 4 (59) and 8 (81): occupancy matters more than reuse.
  Replaces `passive_vertical` + `passive_implicit` (78 ms per call).
- Corrector horizontal kernel: masks and fluxes loaded once per cell and
  the shared face differences reused (42 -> 38 ms). The same change made
  the C4 predictor slower (41 -> 45 ms) and was reverted there.
- Net: passive kernels 268 -> 209 ms per step; mean step 0.500 -> 0.441 s
  on the 3-day MiniPac run (pre_step3d 154 -> 128 ms, step3d_t bracket
  201 -> 167 ms).

Not done / next candidates: float32 storage for the passives would halve
their traffic (a precision decision for the carbonate tracers); the C4/UP3
horizontal kernels (~1.3 ms per tracer) and t3dmix (27 ms) are at the
same per-tracer cost as the T/S path; MARBL's K2 (255 registers) costs
~100 ms per hourly call; the per-step passive boundary fills are small
(kernels < 1 ms, `set_bry` 1.7 ms) and not worth moving.

Second round (2026-10-05, evening), all checked bit-identical on the 3-day
run unless noted:

- Exit hang fixed: with `wrt_file_his = .false.` the I/O rank was never
  told to quit (`io_finalize` returned early when nothing had been staged)
  and the run hung after its last step.
- Alternative-CO2 carbonate solve skipped when its inputs and bracket equal
  the base solve's (true wherever ALT_CO2 runs the same forcing, as in
  MiniPac): carbonate kernel 18.5 -> 13.7 ms per call, surface flux
  0.53 -> 0.37 ms. Interior kernel per-thread arrays sized to the 32
  tracers in use (NTRMAX 64 -> 32): 96.9 -> 92.7 ms. Hourly MARBL call
  117 -> 107 ms.
- Tried and reverted: fusing the horizontal C4/UP3 step into the passive
  column kernel (bit-identical, but 165-180 ms per call against 50 + 40 ms
  for the separate kernels; a per-column kernel cannot hide the latency of
  the eight neighbour loads per level that the 3-D kernel hides with 100x
  more threads). Source kept as `passive*.F90.hfuse_slow`.
- Measured and declined: float32 storage for the passives
  (build without `-DPASSIVE_FP64`): passive kernels 210 -> 206 ms per
  step, mean step 0.441 -> 0.429 s, device memory 27.5 -> 19.2 GB. The
  transport kernels are not bandwidth-bound; their cost is instruction
  issue and latency. Science of the float32 run (`minipac_gpu_marbl3df32`,
  `MARBL_NOISE_FLOOR_3D_2026-10-05.txt`): all 32 tracers stay inside the
  CPU noise floor (worst ratio 1.04, as FP64; float32 vs FP64 GPU
  differences 1e-7 to 3e-6 relative at hour 72), but the saturation-depth
  diagnostics break: zsatcalc/zsatarag RMS differences 2,000-4,000 times
  the twin's, because the root of CO3 - CO3_sat at depth flips level under
  float32 rounding of DIC/ALK. FP64 storage is kept (decision 2026-10-05).
- Not done: splitting the interior kernel into light / pointwise
  physiology / particle-recurrence kernels. The level routine is a strict
  per-level chain (PAR feeds the physiology, Fe scavenging depends on the
  particle state and feeds it back, remineralisation feeds denitrification,
  ligands, O2 and the tendencies), so a split needs three kernels and ~40
  intermediate 3-D fields for an estimated ~25 ms per hourly call (< 2% of
  a step), with loss of bit-identity.

Third round (2026-10-05, late): tracer-innermost passive layout. `tp` is
now `(ntp, is:ie, js:je, nz, 3)`, so the tracers of a point are one
contiguous 256-byte line. The transport kernels (`passive_column.F90`)
are one thread per (tracer, cell) for the C4 predictor, the UP3 corrector
and the lateral diffusion, and one thread per (tracer, column) for the
fused vertical advection + implicit solve; consecutive threads hold
consecutive tracers, so every tracer load of a warp is one coalesced
line, and per-thread state is one tracer (80-91 registers, no stack).
The per-element arithmetic is unchanged and the 3-day run is
bit-identical to the previous layout (history and restart). The boundary
kernels loop over the tracer innermost; the MARBL kernels take the
tracer index first; history staging gathers one tracer from the
interleaved slab; the restart write and the initial read use array
sections. Result on the 4-step profile: passive kernels 209 -> 179 ms
per step (horizontal 2 x 40 -> 2 x 40, column 2 x 50 -> 2 x 36, t3dmix
27 -> 23), step 367 -> 351 ms; 3-day mean step 0.441 -> 0.402 s. The
column tests (`tests/test_marbl_*`) were re-mapped to the new layout and
pass with the same accuracy as before.

Tried and rejected on top of it: shared-memory tile kernels for the
horizontal steps and the lateral diffusion (a 4 x 4-cell block computes
each face once into shared memory, 2.5 face evaluations per cell instead
of 4, bit-identical by construction). Slower: horizontal 2 x 40 -> 2 x 55
ms, diffusion 23 -> 31 ms; the shared-memory round trip, the block
synchronisation and the half-empty second face round cost more than the
repeated face arithmetic. Source kept as `passive_column.F90.tile_slow`.
With one tracer per thread the horizontal kernels did not change at all
(2 x 40 ms before and after the layout change), so their cost is the
per-cell face arithmetic itself (four C4 faces with a division each, or
the UP3 faces), which bit-identity forbids reducing algebraically.

Dycore round (2026-10-06): kernel fusion, all bit-identical on the 3-day
run. Tracer lateral diffusion: the two flux temporaries and the divergence
kernel became one kernel per tracer with the faces inline (5.1 -> 3.3 ms
per step). Horizontal pressure gradient: per direction, the face
differences, the two edge-copy kernels, the harmonic averages and the
update became one kernel, with the edge copies folded into index clamps
(14.8 -> 6.1 ms). Horizontal momentum right-hand side: the Coriolis/
curvature term, the four flux families (curvature temporaries with edge
copies, then faces) and the divergence became one kernel per component
(15.0 -> 9.8 ms; 21 launches per call -> 1). The same per-point expressions
in the same accumulation order make these bit-identical; the price is that
each face is evaluated by both neighbouring cells, which here costs less
than writing and re-reading the temporaries. Dycore 152.7 -> 134.6 ms per
step, 3-day mean step 0.402 -> 0.389 s. Not pursued: raising the occupancy
of the column kernels above 128 registers (vertical momentum advection,
KPP, momentum corrector) by block-size caps; at most ~2 ms at stake.

Where the remaining time goes (per plain step, 0.351 s on the GPU
timeline after the layout change; the numbers below are from before it): passive kernels 206-210 ms, dycore + T/S 153 ms spread over
~1,700 launches with no kernel above 6 ms, GPU idle 4 ms; the hourly MARBL
call adds 107 ms; a history record adds 70-100 ms.

Pitfalls met: nvfortran inlined the explicit `<<<>>>` launch into
`passive_predictor` and the host segfaulted in `cuLaunchKernel`
(compute-sanitizer clean); keeping the kernel + launcher in their own
module fixed it. CUF kernel bodies cannot hold local arrays (per-thread
tracer blocks need an explicit kernel). Never run two makes concurrently
in `build_gpu`.

## Multi-GPU tiling (2026-10-06)

MiniPac on four A100s of one node: `np_xi = np_eta = 2` compute ranks,
one GPU each (`select_device(rank)` before `init_arch`), plus the I/O
server on the last world rank (`mpiexec -n 5 -ppn 5`, qsub
`select=1:ncpus=64:mpiprocs=5:ngpus=4`). Output stays in single files:
every tile reads its window of the global grid, forcing, boundary and
tide files (`decomp%i0/j0` is the file index of local point 0, the same
pair for every stagger), and writes its owned slab at the matching global
offset, either through the server (history) or rank by rank into one
NetCDF file (restart). The 4-step and the 3-day 2x2 runs are bit-identical
to the 1x1 runs in history and restart (`tools/gpu_bitwise_pair.py 4r 4q`,
`3dv3 3dq`), and the 1x1 run is unchanged against the committed reference.

Seam bugs that had to be fixed (none of them visible on one tile, which
is why the earlier single-rank validation never caught them):

1. Static grid fields and every field read from the initial/restart file
   were read over the tile window only; their two halo points toward a
   neighbor were never filled (`fill_static_halos`, `fill_loaded_halos`
   in case_io.F90).
2. The u/v exchange never carried the seam face. Convention: the halo
   exchange uses the rho-cell range for every stagger, so the seam face
   u(nx+1)/v(ny+1) is the east/north tile's point 1 and arrives on the
   west/south tile as a halo (plus one more point). uv_rhs and prsgrd
   compute the face on both sides, step3d_uv and rufrc only on the
   east/north side; the exchange makes them consistent. The old sends
   were off by one for the staggered direction (also for periodic wrap).
3. The fast barotropic loop exchanged zeta only. ubar/vbar are now
   exchanged every fast step after the OBC, zeta again after its OBC, and
   the step2d_mid kernel was split so Dnew and the PGF state at the halo
   points use the exchanged zeta(knew) (the east/north tile's u(1)/v(1)
   update reads them).
4. Loop ranges that were the CPU single-domain ranges: the boundary-column
   accumulation of DU/DV_avg1 (`jstrV:jend` = 2:ny only at a physical
   south edge) and the bulk_coare current-feedback loops (stop at nx/ny
   only at a physical east/north edge).
5. Writers wrote each tile's full local range including halos; the last
   arrival won, which showed up only in arrays that are never exchanged
   (DU/DV_avg*, MARBL pH state, MARBL diagnostics). `set_write_bounds`
   now writes owned points only and each slab carries its file start
   shift (rho axes +ir0/+jr0, u/v axes 0) through `io_stage_*(…, di, dj)`
   and the restart `put_*` helpers.

How they were found: per-step history runs of both cases
(`output_period_his = 900`, cases `4r1`/`4q1`) and
`scratchpad/seam_map.py <rec>`, which counts differing columns per field
by signed offset from the seams (file i = 257, j = 217 for MiniPac). The
first record that differs, the fields in it and the offset pattern
(e.g. hbls differing at W nx-1, nx and E 1 only; vbar zero at the
boundary column on the seam row) pointed at each bug directly.

Exchange cost: the original delivery (`comm_exchange2/3`: managed send
and receive buffers read and written on the host with allocate+reshape
and a cudaDeviceSynchronize per exchange) costs ~19 ms per 3-D exchange
and ~1.2 ms per 2-D one; with 60 fast steps the 2x2 step took 7.5 s
against 0.69 s on one GPU. `deliver_dev` (halo.F90) now packs on the
device, copies the packed buffers with cudaMemcpy into pinned host
buffers, exchanges them with `comm_sendrecv_raw` (contiguous, no
allocation) and copies back before the unpack kernels; pure copies, so
bit-identity is unaffected: 7.3 -> 1.43 s per step. The rest was two
kernels (rho_eos column kernel, the two fused prsgrd kernels) that the
CUF runtime launched with a single block on tiles only (161 / 117 / 85 ms
instead of 1.5 / 1.8 / 1.2 ms; the strided loop still covered the tile, so
results were unaffected). Cause: loop scalars passed by reference to the
formulas device functions (`-Minfo=accel`: "implicit copyin",
"Reference argument passing prevents parallelization", launch reported as
`<<< (1,1), (32,4) >>>`); the binary holds two variants of such a kernel
and the runtime picked the single-block one on tiles. Parenthesizing the
scalar call arguments (`jackett_rho1((Tt), (Ts))`) passes temporaries,
restores a runtime grid and is bit-identical. 2x2 four-step last step
(with the hourly MARBL call): 0.536 s against 0.580 s on one GPU; the
3-day mean step: 0.151 s on 2x2 tiles against 0.389 s on one GPU (2.6x
on 4 GPUs; 288 steps in 43.5 s of step time, history every 6 h), with the
3-day 2x2 run bit-identical to the 1x1 run. The remaining gap to 4x is
per-exchange latency: ~1800 exchanges per step (the fast loop alone does
60 x 4 fills), each with four synchronous staging copies and two
MPI_Sendrecv calls; GPU busy time is ~0.27 s of a 0.8 s MARBL step.
GPU-aware MPI on the device buffers (`comm_sendrecv_dev`, used when the
job sets MPICH_GPU_SUPPORT_ENABLED=1 and the binary is linked with
`-L/opt/cray/pe/mpich/8.1.32/gtl/lib -lmpi_gtl_cuda`; `deliver_dev` falls
back to pinned staging otherwise) and one 3-D exchange for the smoothed
Richardson number in lmd (was nz-1 2-D exchanges) bring the 3-day mean
step to 0.144 s (2.7x on 4 GPUs; 9.7 min of walltime for the 3-day case),
still bit-identical in history and restart. The per-rank step totals are
now unequal (28 to 41.5 s over 288 steps): the slowest tile sets the
pace, so load balance (wet columns per tile) and the remaining ~1800
exchanges per step are the next levers, followed by multi-node runs and
the CPU-vs-GPU benchmark at equal node counts.

Case files (cases/GPU): `namelist_roms_gpu_marbl4q.nml` /
`submit_roms_gpu_marbl4q.sh` (4 steps, restart after 4 steps),
`…4q1` (history every step), `…3dq` (3 days, 6-hourly history, 50 min
walltime with the old exchange); 1x1 counterparts `4r`, `4r1`, `3dv3`.

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
