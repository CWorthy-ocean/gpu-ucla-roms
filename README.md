# roms_gpu

A GPU version of the UCLA ROMS ocean model coupled to the MARBL
biogeochemistry library, written in CUDA Fortran for NVIDIA GPUs. The time
step follows CPU UCLA ROMS routine for routine, and the model reads the same
namelists and input files.

- **On the GPU:** the whole baroclinic and barotropic time step (dynamics,
  KPP mixing, tracer advection and mixing, tides, bulk surface forcing,
  rivers, open boundaries), MARBL's surface fluxes and interior ecosystem, and generic
  passive tracers. Model fields stay in GPU memory for the whole run.
- **Many GPUs:** one MPI rank per GPU, tiles balanced by ocean cells and
  memory, halo exchanges GPU to GPU with a CUDA-aware MPI. Separate writer
  ranks write history and restart files while the GPUs keep stepping.
  Restart files hold two records, so restarts are exact.
- **Checked:** step by step, GPU minus CPU is at most one float32 unit in
  every physics field and MARBL tracer (Iceland0 with and without rivers,
  MiniPac). After 3 days it is within the CPU model's own sensitivity to a
  1e-6 degC initial temperature perturbation, apart from local events (one
  column, open-boundary inflow switches) that do not recur from a common
  state. Every optimization was checked to reproduce the previous version
  bit for bit.
- **Speed (Derecho):** Pacific domain with MARBL, 2 nodes of each kind: 21x
  faster per time step than CPU ROMS-MARBL (8 A100s vs 256 AMD cores).
- **Not yet on the GPU:** pipes (point sources), CDR tracer surface
  fluxes and air-sea gas exchange, output for nested child grids, closed
  basins (one known boundary bug), averages and monthly restarts. A
  namelist that switches on a feature the GPU lacks stops the run at
  start-up with a list of the switches.

## Getting started

### Requirements

- NVIDIA GPUs and the NVIDIA HPC SDK (`nvfortran`; developed with 25.9).
- MPI with Fortran bindings. A CUDA-aware MPI is optional: it lets halo
  exchanges go straight from GPU to GPU; otherwise they go through pinned
  host memory.
- NetCDF-C and NetCDF-Fortran built with parallel I/O (parallel HDF5);
  `nf-config` and `nc-config` on the `PATH`.
- Python 3 for `tools/gpu_layout.py` (standard library only); `numpy` and
  `netCDF4` for the comparison tools.

### Get the code

MARBL is a git submodule:

```
git clone --recurse-submodules <repository-url> roms_gpu
cd roms_gpu
# or, in an existing clone:  git submodule update --init
```

### Build

`make gpu` compiles MARBL's Fortran from the submodule together with the
model and writes `build_gpu/roms_gpu_cuda`. The general form:

```
make gpu FC_GPU=nvfortran \
  FFLAGS_GPU="-O2 -g -cuda -gpu=cc80 -DUSE_CUDA -DPASSIVE_FP64 -Mpreprocess -Mfree -Mnofma -Isrc/shared $(nf-config --fflags)" \
  FFLAGS_MPI="-DUSE_MPI -I<MPI include directory>" \
  LIBS="$(nf-config --flibs) -lnetcdf <MPI Fortran libraries> <rpath flags>"
```

- `-gpu=ccXX` selects the GPU architecture: `cc80` for A100, `cc90` for
  H100 and H200, `cc100` for B200.
- `-DPASSIVE_FP64` stores the MARBL and passive tracers in double precision,
  the validated configuration. Single precision saves memory but moves the
  carbonate saturation depths far outside the CPU noise floor.
- `-Mnofma` turns off fused multiply-add, for reproducible rounding; it
  costs no measurable speed.
- Run one `make` at a time. After changing flags, start from a clean
  `build_gpu/` (`make clean`): the objects do not depend on the flags.

On NCAR's Derecho (Cray MPICH, which needs `libmpi_gtl_cuda` for GPU-to-GPU
halo exchange):

```
module --force purge
module load ncarenv nvhpc cuda cray-mpich netcdf-mpi
NFF=$(nf-config --fflags); NCP=$(nc-config --prefix)
GTL=/opt/cray/pe/mpich/8.1.32/gtl/lib
make gpu FC_GPU=nvfortran \
  FFLAGS_GPU="-O2 -g -cuda -gpu=cc80 -DUSE_CUDA -DPASSIVE_FP64 -Mpreprocess -Mfree -Mnofma -Isrc/shared ${NFF}" \
  FFLAGS_MPI="-DUSE_MPI -I${CRAY_MPICH_DIR}/include" \
  LIBS="$(nf-config --flibs) -L${NCP}/lib64 -lnetcdf -L${CRAY_MPICH_DIR}/lib -lmpifort -lmpi -L${GTL} -lmpi_gtl_cuda \
        -Wl,-rpath,$(nf-config --prefix)/lib -Wl,-rpath,${CRAY_MPICH_DIR}/lib -Wl,-rpath,${NCP}/lib64 -Wl,-rpath,${GTL}"
```

### Run

```
mpiexec -n <ranks> build_gpu/roms_gpu_cuda run namelist.nml
```

- **Ranks:** the first `np_xi x np_eta` ranks (from the namelist) each drive
  one GPU; every extra rank is a writer. `python3 tools/gpu_layout.py
  namelist.nml` prints the suggested numbers (compute ranks, writers, total,
  and a PBS select line); `tools/submit_gpu.sh <job script>` does the same
  and submits a PBS job with them.
- **GPUs:** compute rank r uses GPU (r mod the number of visible GPUs), so
  place ranks node by node and make all of a node's GPUs visible to every
  rank. `tools/numa_bind.sh` (launch with `mpiexec --cpu-bind none ...
  tools/numa_bind.sh <exe> run <namelist>`) pins each GPU rank to the CPU
  cores next to its GPU.
- **Environment:** set `MPICH_GPU_SUPPORT_ENABLED=1` to use GPU-to-GPU halo
  exchange (CUDA-aware MPI required); leave it unset otherwise.
- **Namelist:** the same groups and keys as UCLA ROMS, plus `np_xi` and
  `np_eta` (GPU tiles), `nt_bgc` (32 with MARBL) and `nt_passive`. Paths are
  relative to the namelist file.

`examples/minipac/` is a complete small case (namelist, MARBL settings, PBS
script): 3 days of MiniPac with MARBL on one 4-GPU node. The input data are
not included; its README lists the files.

### Check the results

- **Same GPU and compiler as a reference run:** expect bit-for-bit
  agreement. `tools/cmp_nc.py A.nc B.nc [--last]` compares two files variable
  by variable.
- **New hardware or compiler:** compare with CPU ROMS against the CPU model's
  own noise floor: run CPU ROMS twice, once with the initial temperature
  perturbed by 1e-6 degC, and use `tools/marbl_noise_floor_eval.py` (MARBL
  tracers and diagnostics) or `tools/noise_floor_eval.py` (physics). GPU
  minus CPU should be no larger than perturbed minus unperturbed CPU.
- **Tools:** scripts that need file locations have a settings block at the
  top; most can also be set with environment variables (`ROMS_GPU_CASES`,
  `ROMS_GPU_OUTPUT`, `ROMS_GPU_GRID`, `ROMS_GPU_PREFIX`).
- **Tests:** `tests/test_marbl_sflux.F90` and `tests/test_marbl_interior.F90`
  check the GPU MARBL kernels column by column against unchanged MARBL
  (`make build_gpu/test_marbl_sflux build_gpu/test_marbl_interior` with the
  same flags as `make gpu`). The other tests exercise individual routines on
  the CPU backend (`make tests`, gfortran); they date from the single-GPU
  development phase and have not been kept up to date.

## Layout

```
src/shared    kinds, decomposition, MPI layer, formulas (host + device), timers
src/device    GPU state and kernels: halos, geometry, step2d, prsgrd, advection,
              KPP, step3d, tracers, passive tracers, bulk forcing, MARBL kernels
src/host      main, time loop (step_driver), namelist and NetCDF I/O, forcing
              read-ahead, output writer ranks (io_server), MARBL driver
tests         unit tests and the MARBL column tests
tools         launcher, rank layout, NUMA binding, comparison and profiling scripts
examples      a small runnable case (MiniPac with MARBL)
MARBL         MARBL submodule
```

## Numerics

- SM09 s-coordinate, Shchepetkin PGF
- LF-AM3 predictor / corrector
- Split-explicit FB AB3-AM4 + `weight(1:2,1:nfast)`
- Thickness-aware C4 predictor / U3 corrector
- UPSTREAM_TS horizontal (`UPSTREAM_TS_LAND_CURV` always on)
- LMD94 interior Ri + KPP/BKPP/nonlocal (SMOOTH_RIG, INT_AT_RHO_POINTS, SMOOTH_HBL, Jwt=1)
- no NHMG, no isoneutral, no splines, no LMD_DDMIX / LMD_CONVEC

## MARBL

MARBL (git submodule `MARBL`, CWorthy-ocean/MARBL @ 6e6b2f7 = marbl0.45.0,
the version the CPU model uses) runs on the GPU: MARBL's own sources are
compiled as host code for settings and initialisation
(`src/host/marbl_host.F90`), and the surface flux and interior tendency paths
are CUDA Fortran kernels (`src/device/marbl_sflux.F90`,
`src/device/marbl_interior.F90`), driven from `src/host/marbl_step.F90`. The
32 MARBL tracers are the passive tracers in MARBL mode. Design, validation
and lessons: `MARBL_GPU_PORT.md`.

## Development builds

`make` builds the CPU backend (`gfortran`, `./build/roms_gpu`) and the unit
tests, from the single-GPU development phase; `./build/roms_gpu geom | baro |
prsgrd | advect | step | lmd` run single components on an ideal box.

## License

roms_gpu is derived from UCLA ROMS (https://github.com/CWorthy-ocean/ucla-roms)
and is distributed under the GNU General Public License, version 3 (see
`LICENSE`). MARBL is distributed under its own license
(`MARBL/LICENSE.txt`).
