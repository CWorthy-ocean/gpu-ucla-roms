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

## Multi-GPU round 2: balance, exchange count, output path (2026-10-06)

Measured with the corrected per-step metric (the `cumul` of a
`step: baroclinic step` line is that step's wall time; the earlier "0.144 s"
figure in this file's first multi-GPU section was the largest single step
divided by 288 and is wrong). Non-output steps of the 3-day MiniPac case:

| configuration | non-output step | output steps / 3 d | 3-day walltime |
|---|---|---|---|
| 1 GPU | 0.424 s | 11 s | 3:59 |
| 2x2, commit 57690cc | 0.209 s | 190 s | 10:28 |
| 2x2 + mask-balanced cuts (lever 1) | 0.212 s | 183 s | 10:28 |
| 2x2 + server assembles records | 0.215 s | 6 s | 3:39 |
| 2x2 + fused fast-loop tail, non-blocking exchange | 0.192 s | 6 s | 3:35 |

- Lever 1, `src/shared/tile_balance.F90` + `&tile_settings`: tensor-product
  cuts from the land mask (cost = wet columns + land_weight x land cells,
  2-D prefix sums, quantile seeds, alternating exact min-max partition with
  the 2*halo+2 width floor; `uniform` and `explicit` modes; cuts and
  per-tile costs logged as `tiles:` lines). On MiniPac the cost ratio goes
  1.144 -> 1.091 but the step does not move: the tiled step is bound by the
  ~1400 exchanges per step (GPU busy 0.22 s of a 0.80 s MARBL step), not by
  the slowest tile's work. Keep it; it costs nothing and helps land-heavy
  domains once the exchange overhead is down.
- Output path: the server received one tile's record, wrote it (a partial-
  chunk rewrite), then received the next tile's, so the other tiles waited
  behind every write (10-25 s per history step). `write_assembled` now
  collects all tiles' parts of a record and writes each variable once at
  full extent: history steps 183 s -> 6 s per 3 days. The restart is still
  written tile by tile (24 s for 4 tiles); routing it through the server
  in double precision is the next output item.
- Lever 2 so far: one 3-D exchange for the smoothed Richardson number
  (lmd), two redundant passive fills dropped, the fast-loop tail doing one
  fused ubar/vbar exchange with zeta exchanged only on the last fast step
  (`step2d_tail`; the driver uses `step2d_with_halo`, not `step2d_fused`),
  and non-blocking exchanges (both directions in flight, one wait) in
  `comm_sendrecv_raw/dev`. 0.215 -> 0.192 s. The remaining idle time is the
  per-exchange latency; the next step is batching several fields per
  exchange at the heavy call sites (rho_eos, prsgrd, lmd, pre_step3d).
- Mask-aware tiling beyond this (the CPU MPI_MASKING idea: more subdomains
  than ranks, drop land-only subdomains; and wet-column compaction of the
  column kernels) is discussed in the resume notes.

## Multi-GPU round 3: exchange batching, wet columns, restart, init, two nodes (2026-10-06)

Every run below is bit-identical to the single-GPU reference over 4 steps
and over 3 days (history and restart). Non-output step = mean of the
`step: baroclinic step` cumul values over the 3-day run.

| configuration | non-output step | init | restart write | 3-day walltime |
|---|---|---|---|---|
| 1 GPU (reference) | 0.424 s | 97 s | 24 s | 3:59 |
| 2x2, round 2 | 0.192 s | 93 s | 24 s (rank-ordered) | 3:35 |
| 2x2 + A: group exchanges | 0.169 s | | | 3:40 |
| 2x2 + B: wet-column kernels | 0.155 s | | | 3:37 |
| 2x2 + C: restart via server | 0.169 s (run-to-run noise 0.155-0.169) | | 10.5 s on the server, overlapped | 3:30 |
| 4x2 on two nodes (D) | 0.126 s | 76 s | | 3:06 |
| final main binary (round 3), 1 GPU | 0.424 s | 67 s | 10 s on the server | |
| final, 2x2 | 0.171 s | 35 s | 10 s on the server | 2:27 |
| final, 4x2 on two nodes | 0.124 s | 22 s | 10 s on the server | 2:11 |

Pac CPU reference (`cases/CPU_Pac`, 512 ranks on 4 CPU nodes, daily
history + MARBL diagnostics): 3.55 s per step over the last 280 steps,
init 226 s, walltime 25:55, 365 GB of output (rst 118 GB).

- A, `halo_mod` group exchanges (`hg_begin / hg_add(id, slot, itr) / hg_end`):
  the consecutive fills at each call site (rho_eos, prsgrd, geom, lmd,
  pre_step3d, step3d_t/uv, visc3d, step2d, advect, the driver) become one
  EW and one NS round over packed multi-field buffers (ids F_HZ..F_ZETA
  resolved by a dispatcher; wrk3a/wrk3b keep their nz-from-level-0 quirk).
- B, wet columns: `build_wet_list()` (gpu_state) after `setup_metrics`
  gives `nwet, wet_i, wet_j`; the MARBL carb/interior/sflux kernels and the
  passive horiz/t3dmix/column kernels run over wet columns only (1-D grids,
  block 128). KPP is untouched because hbls carries nonzero land values.
  Land values of tracers/zeta/temp are exact zeros in the restart, so the
  skipped stores change nothing.
- C, restart through the I/O server in double precision: `io_stage8_*`
  stage real(8) windows through pinned/device buffers, the record header
  carries a precision flag (hdr(7)), the server receives MPI_REAL8 into
  real(8) tile buffers and `write_assembled` writes each variable once with
  the explicit-shape double putters `nc_put_2d/3d_rec_d`. Compute ranks no
  longer wait (24 s -> 0). Bug found on the way: the server's variable
  names were 16 characters (`NAMELEN`), truncating
  `MARBL_PH_SURF_ALT_CO2`; now 32 with a length check in `add_var`.
- D, two nodes: PBS `select=1:ncpus=64:mpiprocs=4:ngpus=4+1:ncpus=64:mpiprocs=5:ngpus=4`
  and `mpiexec -n 9` (no -ppn) place ranks 0-3 on node 1 and 4-8 on node 2,
  so the server (last rank) shares node 2 and every compute rank has its own
  GPU (`select_device` = rank mod device count). 0.169 -> 0.126 s per step
  (1.34x for 2x the GPUs: still exchange-latency bound). The history output
  steps stall ~4 s each: one server rank writes a 3.8 GB record in 6.5 s
  while the eight tiles compute the next 24 steps in 3 s, so the run is now
  bound by the single writer (13 records + restart = 95 s of writing per 3
  days on both 2x2 and 4x2). Several writer ranks or parallel NetCDF is the
  next output item.
- Initialization (97 -> 36 s on 2x2, bit-identical on 1x1, 2x2 and 4x2):
  - forcing file handles cached for the whole init (`frc_file_ncid`,
    `close_frc_cache`): the series binders opened every file three times
    per variable (174 series x 22 files) - 42 + 23 s -> 20 + 2 s;
  - time axes prefetched once in parallel (`prefetch_time_axes`: files
    assigned to ranks by size, all local reads first, then broadcasts;
    `cached_times` in the binders): a classic-format record axis costs one
    seek per record, 4348 records for `uwnd` - 20 s -> 5-9 s. A first
    version interleaved read and broadcast per file and serialized the
    reads (no gain): keep reads and broadcasts in separate loops;
  - `src/host/par_read.F90`: collective reader for the grid and the initial
    records. Phase 1 queues every window (`pr_store_2d/3d/3d_pt` with the
    same sequence on all ranks), `pr_flush` reads whole variables
    contiguously (one per rank per batch, windows allgathered) and
    redistributes the windows over MPI, phase 2 stores them on the device
    through a staging buffer and small kernels (`stage_to_device`). The
    tile-window reads straight from the contiguous file were the cost
    (0.29 s per tracer per tile on MiniPac, 1.5 s on Pac; 0.01 GB/s for an
    8x8 Pac tile): 25 s -> 19 s on MiniPac (bound by the 4-stripe file,
    reads 7-11 s per rank at ~0.43 GB/s) and the host-side stores into
    managed memory 8 s -> 1 s. Pac's restart has 24 stripes.
- SFLX_CORR (sea-surface salinity restoring, `roms_bugfix bulk_frc.F90`)
  ported for the Pac case: `&surf_frc_settings sflx_corr = .true.` and
  `&sss_correction dsssdt` (cm/day, stored as m/s), series `sss`/`sss_time`
  through the usual bulk interpolation, kernel at the end of
  `calc_bulk_coare`: `stflx(isalt) = -dSSSdt (S(nz, nrhs) - sss) rmask`
  (bulk forcing only, as the CPU). Default off keeps MiniPac bit-identical.
  Smoke test (4 steps, 2x2): mean surface salt change 2.7e-4 PSU, hbls
  responds everywhere, isolated KPP-depth outliers up to 0.08 PSU.
  Validation against the CPU comes with the Pac reference.
- Pac CPU reference: `cases/CPU_Pac` (repo sources at HEAD, CPU_marbl's
  cppdefs plus SFLX_CORR, CDR off, 32x16 tiles on 4 nodes, exact restart
  2010-02-08, daily history and MARBL diagnostics, restart at day 3). The
  CPU_marbl local source overrides (KPP, probe headers) must not be copied:
  they need a removed probe module, and they equal the repo files when
  LMD_KPP and LMD_BKPP are both defined.

## Pac on the GPU (2026-10-06): first run and CPU comparison

Case: `cases/GPU/namelist_roms_gpu_pac3d8.nml` + `submit_roms_gpu_pac3d8.sh`
(1856x960x100, 32 MARBL tracers, hourly MARBL, SFLX_CORR on, exact restart
2010-02-08, 4x2 wet-balanced tiles on 2 nodes = 8 GPUs + server rank);
`namelist_roms_gpu_pac8.nml` is the 4-step test. CPU reference:
`cases/CPU_Pac` (512 ranks on 4 CPU nodes).

| | GPU, 8 GPUs on 2 nodes | CPU, 512 cores on 4 nodes |
|---|---|---|
| mean step (3 days, with output) | 1.15 s | 3.55 s |
| initialization | 103 s (restart read 60 s) | 226 s |
| 3-day walltime | 9:54 - 11:03 | 25:55 |
| GPU memory per tile | 30 GB of 40 | |
| tile cost imbalance | 1.11 | |

- Correctness: `tools/gpu_cpu_day_cmp.py` (RMS of gpu-cpu over wet points
  per field, ratio to the CPU field's RMS). Day-3 ratios on Pac vs the
  MiniPac GPU/CPU pair: temp 1.3e-4 vs 1.1e-4, salt 3.2e-5 vs 2.4e-5, zeta
  4.7e-5 vs 2.2e-5, DIC 1.6e-5 vs 2.2e-5, PH 3.5e-6 vs 3.6e-6, O2 1.7e-4 vs
  1.6e-4, phytoplankton carbon 1.0-1.3e-2 vs 3-6e-3: the Pac run sits at the
  same noise-floor level as MiniPac.
- The first Pac run had temp RMS 0.17 C at day 3 (ratio 1.3e-2, 100x
  MiniPac), confined to sea-surface temperatures at -1.8 C: the CPU code
  always applies `SEA_ICE_NOFLUX` (set_global_definitions.h): where
  SST <= -1.8 C it zeroes stflx(itemp) and srflx. Ported into the bulk kernel
  (`calc_bulk_coare`); MiniPac stays bit-identical (never that cold). The
  CDR_LITE k_gas and DAILYPAR variants of the rule are not in the GPU model
  (those options are off).
- Step time is bimodal on Pac (p10 0.45 s, p50 0.90 s, p90 2.3 s, max 8.7 s):
  step3d_t (0.43 s mean, p10 0.14 / p90 1.22) and pre_step3d (0.30 mean,
  p50 0.15 / p80 0.67) vary 5-10x from step to step, which they do not on
  MiniPac tiles (flat 0.042 / 0.034 s). The pure dycore steps (p10) are 2.9x a
  MiniPac 2x2 step for 3.7x the cells, as expected; the rest looks like
  managed-memory migration (30 GB resident of 40, the MARBL surface host
  loop touching tp pages, 2 x 11.5 GB of tracers per tile). Profile a few Pac
  steps with nsys (UM traffic vs kernel time) before anything else.
- Output: the server writes a 30 GB history record in 50-64 s and the 55 GB
  restart in 90 s; three daily records plus the restart are ~4 minutes of a
  10-minute run, so the single writer is now the second item after the
  step-time variability.

## Pac optimization round (2026-10-06 night)

- The bimodal 8-GPU step was memory pressure, not kernels: the wet-balanced
  cuts gave rank 4 a 310k-cell tile at 39.4 of 39.5 GB (`gpumem:` report
  printed by `wall_stamp` at 'init done' / 'restart staged'; ~0.12 GB per
  1000 tile cells including halos), unified memory paged in a 16-step
  cycle and every rank waited at the exchanges. On 16 GPUs (18 GB tiles)
  the step was flat at 0.46 s. Fix: `&tile_settings tile_max_cells_ratio`
  (default 1.2): the land weight is raised in x1.5 steps until the largest
  tile area is within that ratio of the mean (Pac: 0.844, max tile 238k
  cells, 31.7 GB, cost imbalance 1.016).

| Pac 3 days | 8 GPUs before | 8 GPUs, cap (final) | 16 GPUs, cap (final) | CPU, 512 cores |
|---|---|---|---|---|
| step p10 / p50 / p90 | 0.45 / 0.90 / 2.3 s | 0.42 / 0.53 / 0.62 s | 0.25 / 0.31 / 0.38 s | 3.55 s mean |
| stepping per rank | 330 s | 166-172 s | 119 s | 1020 s |
| walltime | 9:54 | 7:14 - 8:42 | 6:36 | 25:55 |

With four writer ranks (2026-10-07, "Next levers"): 8 GPUs 5:16 walltime at
0.51 s per step (p50).

Final-binary verification (edge-row exchange + cap): MiniPac 4r/4ref,
4q/4r, 8q/4r, 3dq/3dv3, 3d8q/3dv3 bit-identical; Pac 8-tile and 16-tile
3-day histories bit-identical to each other; Pac 8-tile vs the CPU
reference at day 3: zeta 1.7e-5 (was 4.7e-5 before the seam fix), temp
1.3e-4, salt 3.8e-5, DIC 1.7e-5, PH 3.8e-6, O2 1.8e-4, diatC 1.1e-2 - the
MiniPac noise-floor levels.

- nsys on rank 0 (12 steps, `tools/nsys_timeline.py` on the sqlite export):
  steady fast steps are kernel-bound (0.47 s of kernels per step); passive
  tracer kernels (horiz 62 ms, column 48, pre-advection 26, t3dmix 18 ms
  per step) are a third and run near memory bandwidth with 32 FP64
  tracers; MARBL interior 74 ms per hour; exchange pack/unpack 42 ms over
  ~350 launches; per-step memsets 17 ms (step3d_uv2); UM traffic during
  steady steps ~170 MB/step. The output path dominates the fixed cost:
  one server rank writes a 30 GB history record in 50-64 s and the 55 GB
  restart in 90 s (4 of the 7 minutes of the 3-day run); compute ranks
  stall ~20 s at a history step when both staging slots are busy.
- Seam bug 6, visible only on Pac: `interior_bounds` exchanged rows 1..ny,
  so the physical boundary rows (j = 0 south, ny+1 north) were never
  exchanged east-west while the boundary routines (u2dbc, v2dbc, u3dbc,
  v3dbc, t3dbc radiation) use tangential neighbours on them. 8- and
  16-tile Pac runs diverged from step 1 at the south edge, exactly at the
  cuts (per-step history pair `pac8s`/`pac16s`, offsets from the cuts).
  MiniPac never showed it: its seams meet masked edge points. Fix: j1 = 0
  on a south edge, j2 = ny+1 on a north edge (the north-south round already
  packs columns i1-h..i2+h). MiniPac 2x2 stays bit-identical; the Pac
  8- and 16-tile per-step runs agree on every completed record.
- I/O server slots: with output faster than the writer (per-step 30 GB
  records on Pac) the server evicted slot 1 rather than the oldest record
  and dropped tiles still assembling at quit (records 2 and 4 of 5 had
  holes). Now `NSLOT = 4`, the oldest record is evicted with a WARNING,
  and incomplete records are flushed at quit: all five per-step records
  complete, MiniPac 2x2 bit-identical. A writer slower than the output
  cadence still costs partial-chunk writes; several writer ranks or
  parallel NetCDF remain the real fix.

## Next levers (2026-10-07): writer ranks, exchange batching, initialization

Build now links the parallel NetCDF module (`module load netcdf-mpi` AFTER
`cray-mpich`; `LIBS` adds `-L$(nc-config --prefix)/lib64 -lnetcdf`); the
serial paths are unchanged. All results bit-identical to the references.

- Several writer ranks (`ROMS_GPU_IO_SERVERS=n`, default 1; the last n
  world ranks; `comm_writers()`). Each compute rank sends every record
  header to every writer and variable iv's slab to writer mod(iv-1, n);
  each writer assembles and writes its own variables. Parallel HDF5 only
  extends a record variable collectively, so `write_assembled` first does
  a collective round (ocean_time, then one element per variable at
  x = writer index + 1, then a barrier) and the owners put their variables
  independently in parallel. Files are opened lazily when the first record
  for them is written, so that writers reach the collective open in record
  order (an eager open at CMD_OPEN deadlocked against another writer's
  record write). Restart (double) variables are now chunked one record per
  chunk (`nc_chunk_one_record`); with default chunking the independent
  writes had to allocate chunks and the sync failed with "HDF error".
  Slot overrun with several writers is an error (collective order) and an
  incomplete record at quit is dropped with a warning.
  MiniPac 2x2, 2 writers: history record 6.1 -> 3.7 s, restart 10.5 -> 5.5 s
  per writer. Pac 8 GPUs, 4 writers (ROMS_GPU_IO_SERVERS=4, node 2 with
  mpiprocs=8): 30 GB history record 60 -> 20 s per writer, 55 GB restart
  90 -> 31 s, 3-day walltime 7:14-8:42 -> 5:16, history bit-identical to
  the single-writer run. A deadlock found on the way: compute ranks sent
  record headers with blocking MPI_Send; a writer in the collective round
  does not receive, the compute rank blocks on that header, and another
  writer waits for that rank's data (posted after the headers). Now the
  data Isends are posted first and the headers are Isends too (buffers in
  the staging set, `nreq = nvars + up to 5 per writer`).
  A single process writes at 0.6 GB/s whatever the striping (24 stripes
  tested), so writers are the lever, not Lustre layout.
  Writer-count rule (no manual setting): the model takes the compute rank
  count from the namelist (np_xi x np_eta, scanned in `init_comm`) and every
  extra launched rank becomes a writer (`ROMS_GPU_IO_SERVERS` overrides).
  `tools/gpu_layout.py <namelist>` sizes the history record from the grid and
  the write flags and picks the fewest writers that write one record in
  about 15 s at 0.6 GB/s each (1..8), printing the PBS select line and the
  rank count; `cases/submit_gpu.sh <submit_script> [NWRITERS=n] [EXE=...]`
  submits with them (the submit scripts launch `mpiexec -n ${NRANKS:-...}`).
  MiniPac -> 1 writer, Pac at daily output -> 4.
- Exchange batching: `hg_end` resolves every registered field to a device
  address (`c_devloc`, kept in `hg_ptr`), and `hg_run_phase` packs/unpacks
  up to 4 fields with the same level range per launch (`hg_slab4`: device
  pointers from `c_f_pointer`, CONTIGUOUS so the explicit-shape device
  dummies get the base address). Default on; `ROMS_GPU_HG_BATCH=0` restores
  one launch per field, `=n` sets the group size. Lesson: the kernels must
  run on `work_stream` (the CUF_DO macros); on the default stream they
  raced the model kernels and filled the halos with garbage, with 1 field
  per launch as well. Gain: 2x2 0.170 -> 0.168 s per step, 4x2 two-node
  0.125 -> 0.123 s (launch count 354 -> ~120 per step; the exchange is
  latency-bound, not launch-bound).
- Initialization: the double-precision restart staging is pageable and
  allocated at the first restart write (two pinned 12 GB sets cost ~10 s
  per rank at init on Pac: output init 15.7 -> 6.2 s; the restart stage
  itself now takes 3 s). Overlapping the whole-variable reads with the
  window redistribution (`pr_flush` posts all receives, keeps one batch of
  sends in flight) changed nothing: the reads are the limit (~2.4 GB/s
  aggregate with 8 streams, 3.6 with 16). A collective MPI-IO read mode
  (`ROMS_GPU_PAR_READ=mpiio`, `flush_mpiio`) exists but needs pnetcdf for
  classic-format files, which netcdf-mpi lacks, so it falls back.

## Restart read through collective MPI-IO (2026-10-07): measured, not adopted

The question was whether every rank reading its own windows collectively
(parallel HDF5 through netcdf-mpi, `ROMS_GPU_PAR_READ=mpiio`) beats the
whole-variable reader (each rank reads whole variables and redistributes
the windows). Collective reads need NetCDF-4 files, so the CPU restarts
were converted once with `nccopy -k 4` (one record per chunk, 24 Lustre
stripes, 118 GB Pac file in 5 minutes) and the GPU-written restart was
used as is. All runs 4 steps, "ini read" segment of the init stamps, 8
GPUs for Pac, 4 for MiniPac; every pair restart bit-identical.

| restart file | whole-variable reads | collective MPI-IO reads |
|---|---|---|
| Pac, CPU restart, 87 variables, classic -> converted NetCDF-4 | 51.5 s | 86.7 s |
| Pac, GPU-written NetCDF-4 restart, 53 variables | 52.8 s | 38.7 s |
| MiniPac, CPU restart, classic -> converted NetCDF-4 | 13.7 s | 23.5 s |

Collective reads win only on the GPU-written file and lose elsewhere, so
whole-variable reads stay the default and `mpiio` stays opt-in. Two
observations for later: HDF5 whole-variable reads run at about half the
rate of classic-format reads (0.64 vs 0.33 s per 1.4 GB variable), so the
GPU-written restart reads slower than the CPU one; and the collective read
is a strided window read (one short row per (j, k)) that HDF5/ROMIO
aggregate poorly. Tile-friendly chunking of the restart (e.g. nz x 128 x
128 chunks, independent of the tiling) and MPI-IO hints are the untested
knobs; neither is geometry-free enough to adopt blind.

## Pipeline hygiene (2026-10-07): what a steady Pac step is made of

Fresh nsys profile of rank 0 (8 GPUs, 4 writers, batching on;
`tools/nsys_step_breakdown.py <report.sqlite> [step]`), steady step:

| item | per step |
|---|---|
| step wall time | 0.42-0.45 s |
| GPU busy | 0.30 s (71%) |
| kernel launches | 3161, of which 2666 under 20 us (~25 ms of GPU time) |
| GPU idle at the all-tracer exchange (`halo_fill_3d_tp`, 5 per step) | 62 ms |
| GPU idle at the group exchanges | 13-35 ms |
| memsets | none |
| end-of-step diagnostics (`check_mode_coupling`, `step_maxima`) | ~2 ms |

So hygiene is worth 2-3% at most; the all-tracer exchange (five ~50 MB
GPU-direct exchanges per step with nothing queued on the GPU meanwhile) is
the largest remaining lever at ~15% of the step (exchange overlap).

Changes (bit-identical by construction):
- `step_mark`: the 19 per-phase stamps (each a cudaDeviceSynchronize + print
  + flush) are off by default; only the `baroclinic step` stamp remains
  (its cumul is the step time the tools read). `ROMS_GPU_STEP_MARKS=1`
  restores them.
- `set_tides`: one kernel per target set (west/east, south/north, potential)
  with the constituent loop inside, host cos/sin in `tide_c/tide_s`
  (device cos/sin would differ in the last bit); the per-element sum runs
  in the original constituent order with the same rounded products.
  92 launches -> 6 per step.
- Redundant all-tracer exchanges removed (3 of the 5 `halo_fill_3d_tp` per
  step). Slot rotation: predictor reads nstp, writes 3; corrector reads 3,
  writes 3-nstp; the next step's nstp is that slot. So (1) the exchange of
  nstp at the predictor start repeats the previous step's post-t3dmix
  exchange (nothing writes tp in between; the load exchanges slots 1 and
  2); (2) the exchange of slot 3 right after the predictor kernels is
  repeated by pre_step3d after passive_t3dbc, and nothing reads the slot-3
  halo in between (passive_t3dbc reads slot nstp at tangential neighbours
  and writes slot nn; its closed-boundary copy writes halo rows of column
  0 that the following NS exchange overwrites either way); (3) the
  exchange of nnew after the corrector feeds only pointwise code (MARBL
  columns, t3dmix's stencil is slot 3) before passive_t3dmix exchanges it.
  Left: pre_step3d (slot 3 after t3dbc) and passive_t3dmix (nnew).
  The T/S exchanges have the same pattern (per-tracer after the predictor,
  then the group after t3dbc; after the corrector, then after t3dmix) but
  carry 2 fields instead of 32; left for the exchange-overlap round.

Results (every pair bit-identical: MiniPac 2x2 vs 1x1 4r history+restart;
Pac 8 GPUs 4-step restart vs pac8; Pac 3-day history vs pac3d8w4m):

| run | MiniPac non-output step | Pac 3-day: p50 step, per-rank stepping, walltime |
|---|---|---|
| main (b6f7621 + writers) | 0.1224 s | 0.511 s, 142.2 s, 5:29 |
| stamps off + fused tides | 0.1216 s | 0.510 s, 141.9 s, 5:03 |
| + 3 tracer exchanges removed | 0.1168 s | 0.488 s, 136.7 s, 5:04 |

Net 4.5% per step on Pac, 4.6% on MiniPac; the walltime difference between
the first two rows is init/queue noise (init 86 vs 83-93 s).

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
