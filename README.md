# roms_gpu

CUDA Fortran ocean core with UCLA-ROMS numerics. New tree — not an `#ifdef` of `roms_bugfix`.

**v1 scope:** one rank ↔ one GPU ↔ one tile (CPU backend works without a GPU). Halo `deliver` is NBR_SELF on one rank and host-staged MPI when a neighbor rank is real. Loaded cases stay 1×1 until tiled NetCDF I/O exists. Working fields are a single allocation (managed under CUDA). BGC later. `nt` is already a parameter.

## Numerics (frozen)

- SM09 s-coordinate, Shchepetkin PGF
- LF-AM3 predictor / corrector
- Split-explicit FB AB3-AM4 + `weight(1:2,1:nfast)`
- Thickness-aware C4 predictor / U3 corrector
- UPSTREAM_TS horizontal (`UPSTREAM_TS_LAND_CURV` always on)
- LMD94 interior Ri + KPP/BKPP/nonlocal (SMOOTH_RIG, INT_AT_RHO_POINTS, SMOOTH_HBL, Jwt=1)
- no NHMG, no isoneutral, no splines, no LMD_DDMIX / LMD_CONVEC

## Layout

```
src/shared    kinds, decomp, arch (CPU/GPU), formulas (host+device), time indices / weights
src/device    gpu_state (one working copy), halo_fill, set_depth / set_HUV / set_HUV1 / omega, step2d, prsgrd, advect, LMD
src/host      time loop (nfast on the host), namelist + NetCDF via host_buf, main
tests         formulas, halo, set_depth, step2d, prsgrd, advect, rho_eos, lmd, io, dyn_core
```

One rank ↔ one GPU ↔ one tile. Arrays are always local `nx,ny` plus halo. Neighbors are `NBR_SELF` (periodic wrap on this rank), `NBR_NONE` (physical edge), or a real MPI rank. `deliver` host-stages the pack buffers and `MPI_Sendrecv`s them. CUDA graphs stay off when `comm_size > 1`. Tiled NetCDF I/O is not wired: a loaded case on more than one rank stops.

## What is real vs stub

| Real | Stub (`not_implemented`) |
|---|---|
| decomp, gpu_state, halo EW+NS+corners | |
| SM09 `set_depth`, `set_HUV`, `set_HUV1`, `omega` | |
| fused `step2d` (iif=1 PGF, rufrc strip, FB correction) | rivers (sparse hook `apply_volume_sources` is empty) |
| Shchepetkin `prsgrd` (SPLIT_EOS) | |
| horiz UP3 + vert C4/U3 fluxes (`FX`, `FE`, `Fz`) | |
| SPLIT_EOS `rho_eos`, LF-AM3 `pre_step3d`, `step3d_t` | |
| `step3d_uv1/2`, `visc3d` (UV_VIS2), `t3dmix` (TS_DIF2) | |
| LMD `lmd_vmix` (interior Ri + KPP/BKPP/nonlocal) | |
| namelist + grid/init + flux/bry NetCDF (`load`/`run`) | rivers |
| history + restart write (`BASIC_OUTPUT_SETTINGS`) | averages, monthly restarts, true `w` |
| Flather M2 + Chapman zeta + Orlanski M3/T | Specified OBC, sponge-tune |
| COARE 3.0 bulk fluxes; SSH/UV/POT tides | analytical tides, QCORR/SFLX_CORR |
| FB weights, thickness-aware faces, flux mismatch, VAR_RHO_2D PGF | |

## Build

```
cd /glade/derecho/scratch/bachman/roms_gpu
make              # gfortran tests + ./build/roms_gpu
make run          # geom-only ideal box
make gpu          # nvfortran -cuda (Derecho GPU nodes)
```

`./build/roms_gpu geom` runs set_depth / set_HUV / omega.
`./build/roms_gpu baro` runs the fused barotropic loop on the ideal box.
`./build/roms_gpu prsgrd` runs set_depth + Shchepetkin PGF.
`./build/roms_gpu advect` runs horizontal UP3 + vertical C4/U3 fluxes.
`./build/roms_gpu step` runs one baroclinic step of the dynamical core (LMD on).
`./build/roms_gpu lmd` runs `set_depth` + `rho_eos` + `lmd_vmix` on the rest box.
`./build/roms_gpu load path/to/namelist.nml` reads namelist + grid + init NetCDF (MiniPac layout: `xi_rho = LLm+2`).
`./build/roms_gpu run path/to/namelist.nml` loads then takes `ntimes` baroclinic steps (flux or COARE bulk + Flather/Orlanski OBC + tides if those files are in `frcfiles`).
History/restart: `&basic_output_settings` / `&ts_output_settings` (same keys as the CPU case). Files are `{output_root_name}_his.nc` and `{output_root_name}_rst.nc` (new file every `nrpf_his` / 2 records). A restart is loadable as `inifile`. No averages, no monthly calendar, no true `w`.
Smoke: `./build/roms_gpu run build/standalone.nml` (flux). Bulk+tides: `./build/roms_gpu run build/standalone_bulk.nml`.

Needs NetCDF-Fortran (`nf-config` on the PATH). Parallel HDF5 builds (conda npl) also need HDF5 and MPI on the link line; the Makefile adds them when those libs sit next to NetCDF. Paths in the namelist are relative to the namelist file. MiniPac's default `inifile` is a restart under `output_up3/`; `input_data/PACMED12km_v2ni_initial_conditions.nc` is the smaller IC.

Halo pack/unpack and the interiors of `set_depth`, `rho_eos`, `prsgrd`, LMD (`swr_frac`, Ri, nu/mult/store), and `step2d` launch the same way: `!$cuf kernel do` over `(i,j)` on a single work stream. NBR_SELF deliver is a device copy; a real neighbor rank uses host-staged `MPI_Sendrecv`. `nfast` walks on the host and, on one rank, launches a CUDA graph of `{step2d, halo_fill}`. Kernels read a managed `fast` pack for time indices and FB weights. SMOOTH_RIG (per level) and the KPP column search stay on the host.

## Next

1. Tiled NetCDF I/O (so MiniPac can use `np_xi × np_eta`); rivers; BGC

