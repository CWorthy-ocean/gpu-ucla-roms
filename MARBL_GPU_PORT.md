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

## Exchange overlap (2026-10-07): non-blocking tracer exchange, fewer exchanges, and what the step is really bound by

- The two remaining all-tracer exchanges per step are non-blocking in three
  pieces (`halo_tp_start` packs and posts the east-west round on device
  buffers through GPU-direct MPI with its own tags 321-324; `halo_tp_mid`
  waits, unpacks, packs and posts the north-south round; `halo_tp_finish`
  waits and unpacks). The slot-3 exchange is posted at the end of
  pre_step3d, advanced after visc3d and finished in passive_corrector; the
  end-of-step exchange is posted after passive_t3dmix, advanced after the
  next step's lmd_vmix and finished in passive_predictor
  (`passive_exchange_finish` before finalize). Between post and finish the
  slot's interior is unchanged and its halo unread (see the hygiene
  section for the slot-rotation argument). Without GPU-direct MPI the
  start does the blocking exchange. Debug knobs: `ROMS_GPU_TP_OVERLAP`
  (bitmask, 1 = slot-3 window, 2 = end-of-step window, default 3),
  `ROMS_GPU_TP_CHECK=1` (redo each overlapped exchange blockingly and
  count differing halo values), `ROMS_GPU_TP_DEVSYNC=1`.
- Eleven redundant single-field 3-D exchanges per step removed by the same
  argument: u and v (nrhs) and temp (nrhs) at the step start (exchanged by
  step3d_uv2's and rho_eos(nnew)'s groups after the last writes), the
  per-tracer exchanges after the T/S predictor (the group after t3dbc
  follows) and after the corrector (MARBL and t3dmix read nnew at the
  point; rho_eos(nnew) exchanges it), and t3dmix's exchanges of nrhs
  before and nnew after (slot 3 unchanged since the groups; nnew exchanged
  by rho_eos).
- Results, all bit-identical (MiniPac 4-step and 3-day vs 4r/3dv3, Pac
  4-step restart vs pac8, Pac 3-day history+restart vs pac3d8w4m):
  MiniPac non-output step 0.1168 -> 0.1168 (overlap) -> 0.1128 s (fewer
  exchanges); Pac 3-day p50 0.488 -> 0.494 (overlap) -> 0.478 s (fewer exchanges; per-rank stepping 136.7 -> 132.8 s).
  On rank 0 of the Pac profile the overlap removed the 62 ms of idle at the
  tracer exchanges, but the 3-day step did not move: an all-rank profile
  (`tools/nsys_rank_balance.py`, `nsys_allranks.sh`) shows GPU busy per
  step from 0.198 s (rank 7, 28k wet of 232k cells) to 0.340 s (rank 2,
  all wet); mean 0.29 s. The step is paced by the wettest tile, and the
  other ranks' exchange "waits" were waiting for it. The busy times fit
  cost = wet + 0.47 x land per cell, but the memory cap
  (`tile_max_cells_ratio` 1.2, ~0.12 GB per 1000 cells, 3 FP64 slots of 32
  tracers) forced the balancer to land weight 0.844, i.e. near-equal
  areas. So at 8 GPUs the remaining lever is balance, not latency: either
  memory per cell must drop (wet-column storage of the passive tracers,
  which would let the cap be set on wet cells) or the run needs 16 GPUs,
  where the cap is loose enough for the 0.47 weight. Consequence for the
  wet-column dycore item: it makes land cheaper still, which at the 1.2
  cap mostly speeds up tiles that already wait.
- Rebalancing alone does not help: a 3-day run with the fitted land weight
  (0.47) and the cap at 1.3 (largest tile 282k cells, model max/mean cost
  1.07, 30 GB on the sampled node) gave p50 0.487 s vs 0.478 s, still
  bit-identical. Reason: the ~30 exchanges per step synchronize neighbours
  phase by phase, so the step is the sum over phases of the slowest rank
  in each phase. The passive phases are paced by the wettest tile and the
  dycore phases (full-tile kernels) by the largest tile; a cost-balanced
  cut moves work between the two without shortening the sum. Phase times
  with stamps on (4-step Pac, median over ranks): pre_step3d 0.113 s,
  step3d_t 0.122 s (the passive kernels), step2d fast loop 0.039 s,
  corrector forces 0.028, step3d_uv1+visc 0.024, lmd_vmix 0.020, prsgrd
  0.013, the rest under 0.01; sum 0.39 s. So the fast loop (wider-halo
  item) is 8-10% of the step, and making every phase scale with wet cells
  (wet-column dycore + wet-column storage of the passive tracers so the
  memory cap can be set on wet cells) is what would let a wet-balanced cut
  balance all phases at once (busy 0.34 -> ~0.29 s on the pacing rank).
- A false alarm cost half a day: two runs diverged mid-run at last-digit
  level. The known-good blocking binary diverged at the same step on the
  same node (deg0072); 20+ runs on other nodes were bit-identical. Check
  `exec_host` before chasing a race.

## Wet-column dycore, compressed passive storage, memory-aware tiles, fast-loop exchange (2026-10-08)

Starting point (all-rank Pac profile, 8 GPUs): the passive kernels cost
nothing on land, every dycore kernel cost the same per land cell as per wet
cell (~76 ms per 1e5 cells per step), and the memory cap forced near-equal
tile areas, so the all-wet tile paced every phase.

- **Dry-column skip** (`gpu_state`: `build_dry_map`, `colact`, `dry_lim`,
  groups G_GEOM..G_2D). "Deep land" = land with no wet point within 3
  cells (judged inside the tile arrays; columns whose neighbourhood leaves
  the arrays count as active). Every stored field is constant there
  (history and restart: all zero on deep land at every record, the
  barotropic forcing and transports zero on every land face; hbls/hbbl
  vary only on coastal land), so the ~55 dycore kernels (depths, transports,
  omega, EOS, KPP, pressure gradient, momentum advection and viscosity,
  T/S advection, implicit solves, the momentum predictor/corrector, the
  fast loop) skip deep land after the first two steps of a run (which
  establish the values in every time slot). Test in the kernels:
  `if (lim > 0) then; if (colact(i, j) == 0) cycle; end if` (no mask load
  on tiles without deep land). `ROMS_GPU_SKIP_DRY` = bitmask of groups
  (default all), `ROMS_GPU_DRY_RADIUS` (default 3). Land cost 76 -> 40 ms
  per 1e5 land cells (the rest: coastal band, exchanges).
- **Compressed passive storage** (`tpcol`, `tpnb`, `ntpcol`): tp(ntr,
  0:ntpcol, nz, 3) over stored columns only: the interior wet columns
  first in wet-list order (so MARBL and the passive kernels use their wet
  index directly), then the other wet points and the physical boundary
  rows; column 0 is a shared zero column for every other point (what land
  held before). Stencil neighbours: x neighbours arithmetically
  (ci +- d inside the tile row: exact whenever the value is used, since the
  faces are masked with apply_mask otherwise), y neighbours and halo
  neighbours from a per-column table (tpnb, 8 ints). Exchange
  pack/unpack, restart load, output gather and init go through tpcol.
  Memory: rank 7 (12% wet) 27.2 -> 12.4 GB. Kernel cost: horiz kernel
  82 -> 85 ms on the all-wet rank (a first version with map lookups was
  107 ms, the table alone 95), t3dmix 24 -> 26.5 ms (table; arithmetic x
  was slower there, 33.6).
- **Memory-aware balance**: the 'wet' balance raises the land weight until
  the estimated memory of every tile (0.532*nz kB per cell + 3*ntp*nz*8 B
  per wet cell, fitted on Pac within ~2 GB) is below 92% of the smallest
  device memory (`tile_max_mem_gb`, `tile_mem_cell_kb`, `tile_mem_wet_kb`;
  `tile_max_cells_ratio` is now off by default). Default land weight 0.10
  (measured best).
- **Fast loop: one exchange per substep** instead of two: zeta_new is formed
  on the first halo ring of MPI seams (DUon/DVom one face further there)
  with the neighbour's arithmetic, so the mid-substep zeta exchange is gone
  and zeta(knew) travels with ubar/vbar in one exchange
  (`halo_fill_zuv_n`). The rufrc/rvfrc exchange of every substep is now
  done on the first substep only (the only one that changes them).
- **Layout**: tensor-product 4x2 cuts cannot balance Pac better than
  max/mean cost ~1.08-1.14; 8x1 strips reach 1.002 and drop every
  north-south (inter-node) exchange round. With strips the fast loop
  spends 46 ms per step (14 ms kernels) instead of 111 ms.

Pac 3-day, 8 GPUs + 4 writers, all bit-identical to the reference
(history and restart; different cuts give identical results):

| configuration | median step | mean step |
|---|---|---|
| start of this round (4x2, forced land weight 0.84) | 0.478 s | 0.462 s |
| + dry skip (same cuts) | 0.475 s | 0.459 s |
| + compressed storage, 4x2 rebalanced (0.40) | 0.448 s | 0.429 s |
| + one exchange per substep, 4x2 (0.25) | 0.449 s | 0.428 s |
| 8x1 strips, land weight 0.25 | 0.416 s | 0.418 s |
| 8x1 strips, land weight 0.10 | 0.404 s | 0.404 s |

Final verification with the rebuilt main binary (all bit-identical):
MiniPac 1x1 4-step vs 4ref; 2x2 4-step vs 4r; 4x2 two-node 4-step vs 4r;
2x2, 4x2 and 4x1 3-day vs 3dv3 (median step 0.133, 0.087, 0.137 s; 2x2
was 0.144 before this round, so strips do not help MiniPac on one node);
Pac 4x2 4-step vs pac8; Pac 8x1 3-day (default land weight 0.10) vs
pac3d8w4m, median 0.404 s, peak GPU memory 27.3 GB.

Not done: a 4-cell halo to exchange every second fast step. With a 2-cell
halo one exchange per substep is the minimum (ubar(knew) at a face depends
on fields 2 cells away within a substep); a 4-cell halo means widening every
array while keeping the 3-D exchanges at width 2. Remaining fast-loop idle
is ~32 ms per step, of which perhaps 10-15 ms is the exchange rounds this
would halve (~2-3%).

## Mixing skip, exchange audit, and where a Pac step goes (2026-10-08)

Lateral mixing only near nonzero coefficients: Pac has visc2 = 0 and
diff2 nonzero only in the sponge band. `build_mix_map` (gpu_state) marks
the cells within 2 (`mixact`) and 3 (`mixact3`, visc3d stress points) of
a nonzero visc2_r/visc2_p/diff2. A neighbourhood that leaves the array
counts as active. `mixc` lists the active wet columns for the passive
t3dmix launch. t3dmix and visc3d test the mask after the dry test.
`ROMS_GPU_SKIP_MIX=0` turns the skip off. Active wet cells per Pac strip:
0-12%. Pac 3-day mean step 0.404 -> 0.381 s, bit-identical.

Exchange audit: `ROMS_GPU_HG_AUDIT=1` makes `hg_end` copy every field's
slab before the exchange and count the halo values the exchange changed,
keyed by the exchange's position in the step (`hg audit rank R seq S`
lines at the end). Positions differ on the first run step, so read the
table from the second step on. Over MiniPac 2x2/4x2 3-day and Pac
4x2/strips 24-step runs, 16 group exchanges per step never changed a
value, 2 changed only on the first step, and 3 groups carried
never-changing fields. Each removal below also has a reason in a comment
at the call site:

- set_depth: z_w, z_r, Hz are formed over the whole array from zeta(knew),
  whose halo is current at every call: no exchange.
- rho_eos(tidx, stage): stage 1 (predictor), 2 (corrector), 3 (end of
  step). T is exchanged only at stage 3, plus stage 1 on the first run
  step. The geometry is never exchanged here. The results (rho1, qp1,
  rho, bvf, rhoA, rhoS) are not exchanged at stage 1 except on the first
  run step (`is_first_run_step`), because they equal the halo stage 3 of
  the previous step left.
- lmd_vmix: only the surface forcing (sustr, svstr, stflx t/s, srflx).
- Removed outright: the opening exchanges of prsgrd, pre_step3d,
  set_bottom_drag, visc3d and step3d_t.
- step3d_uv1: only u, v(nnew).
- tracer_halos: first run step only.

Pac 3-day mean step 0.381 -> 0.348 s, walltime 4:57 -> 4:40 (both items).
Bit-identical: MiniPac 1x1/2x2 4-step, 2x2/4x2 3-day; Pac 4x2 4-step;
strips 3-day history and restart.

Where the time goes now. Do not read idle time from an all-rank nsys
profile. It showed ~80 ms of idle per step, including 20 ms gaps at the
same wall time on all 8 ranks at random fast substeps: profiler stalls on
one rank, spread by the exchange chain. Without nsys the step times
repeat with a 4-step (hourly) cycle:

| step in the cycle | time (s) | extra |
|---|---|---|
| 1, 2 | 0.270 | - |
| 3 (MARBL call) | 0.360 | MARBL interior 73 ms + carbonate 11 ms of GPU |
| 4 (forcing read) | 0.433 | ~150 ms in the corrector's set_forces (host) |

Clean steps take 0.267-0.272 s against 0.248-0.261 s of GPU busy on the
slowest rank. So overlapping the 3-D exchanges and a wide fast-loop halo
can recover at most ~10 ms per step together. The wide halo would need
halo 3 (not 4) plus two time levels exchanged every second substep,
because AB3 reads the level formed by the unexchanged substep. It would
also need the OBC and volume sources applied in the seam halos at
physical edges. The phase table comes from `ROMS_GPU_STEP_MARKS=1`
(lines come in groups of 8, one per rank).

The forcing read (`read_catalog_rec`) opens the file, reads one record
of the tile window, closes, and copies to the device. When a series
advances by one record it rereads both brackets (`advance_series`),
although the new lower bracket is the old upper one.

A GPU-aware MPI benchmark (8 ranks on 2 nodes, chain exchange of device
buffers):

| setting | 52 MB message |
|---|---|
| default (CUDA IPC) | 3-4 ms (13-18 GB/s per direction, inter-node included) |
| MPICH_GPU_IPC_ENABLED=0 | 36 ms |
| MPICH_ASYNC_PROGRESS=1 | 2x slower |

The all-tracer exchange (~50 MB per side on the strips) costs ~4 ms of
transfer.

## Forcing reads while stepping (2026-10-08)

The hourly surface forcing cost ~150 ms per Pac hour (every 4th step,
inside the corrector's set_forces).

- The Pac files are CDF5 classic: 33 GB per month, 7 float32 record
  variables of 962x1858 (7 MB per variable and hour).
- Every rank read a strided window of every variable. For the strips
  that spans nearly every row, so each rank pulled each whole slab.
- When a series moved on by one record, `advance_series` reread both
  brackets.
- The file was opened and closed per record.
- A cold 7 MB slab read on Lustre takes 35-180 ms; a warm one ~14 ms.

`ROMS_GPU_FRC_BATCH` (bitmask, default 7; 0 = the former path):

1. Slot reuse: the new lower bracket is the old upper one, so slot 2 is
   copied into slot 1 on the device (`copy_upper_to_lower`, a D2D
   cudaMemcpy) instead of reading that record again.
2. One read per record: `set_forces` and `set_bgc_forces` advance every
   series first with the reads deferred (`frc_batch_begin`).
   `frc_batch_flush` then reads each needed record whole, once, on rank
   mod(r-1, nranks), through a handle kept open for the run
   (`frc_run_ncid`), and sends every tile its window. Files with time as
   the first Fortran dimension keep the window reads.
3. Read-ahead: `frc_prefetch`, called right after step3d_t, reads the
   record after each series' upper bracket.
   - It runs while the passive-corrector kernels are queued.
     marbl_gpu_step drains the device on some calls, so the call sits
     before it.
   - Each rank reads at most one record per call. The windows go out
     with non-blocking sends, one fixed tag per series.
   - The flush that needs the record only waits for its messages. A
     prefetched record that turns out not to be the needed one is dropped
     and read as in 2.
   - `frc_prefetch_drain` completes leftovers before finalize.

The bytes that land in a tile are the window read's (float32 -> double
is exact), so results are unchanged.

`ROMS_GPU_FRC_TIMING=1` prints every flush and prefetch. Pac hourly flush:
60-115 ms with 1+2 (`records 7 ... whole T`), 4 ms with 1+2+3 (`prefetched
7`). The prefetch reads cost the owner 15-70 ms of host time, mostly
hidden behind the queued kernels.

| Pac strips 3-day | mean step | 4-step cycle (s) | walltime |
|---|---|---|---|
| before | 0.348 s | 0.283 0.272 0.383 0.455 | 4:40 |
| slot reuse + one read per record | 0.321 s | 0.287 0.272 0.398 0.327 | 4:36 |
| + read-ahead (final) | 0.308 s | 0.291 0.273 0.378 0.290 | 4:33 |

Bit-identical for every build:

- MiniPac 1x1 and 2x2 4-step.
- MiniPac 2x2 and 4x2 3-day. These cross into April; one flush there
  needed 12 records with only 4 prefetched, which exercised the fallback.
- Pac 4x2 4-step.
- Pac strips 3-day history and restart.

What is left of a Pac hour beyond the clean steps: the MARBL call (~105 ms,
of which ~85 ms MARBL kernels) and ~35 ms around the forcing step.

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
