"""Bit-identity of two GPU MARBL runs: every variable of the last history record and of the restart.
usage: python gpu_bitwise_pair.py <tagA> <tagB>   (files <OUTPUT>/<PREFIX><tag>_{his,rst}.nc)
The last record of each file is compared (restart files may hold one or two records)."""
import os, sys, netCDF4 as nc, numpy as np
# ---- settings: edit here, or set the environment variable ----
OUTPUT = os.environ.get("ROMS_GPU_OUTPUT", "/path/to/output")     # directory with the runs' output files
PREFIX = os.environ.get("ROMS_GPU_PREFIX", "minipac_gpu_marbl")  # file name before <tag>_{his,rst}.nc
B = os.path.join(OUTPUT, PREFIX)
a, b = sys.argv[1], sys.argv[2]
bad = 0
for kind in ("his", "rst"):
    da = nc.Dataset(f"{B}{a}_{kind}.nc"); db = nc.Dataset(f"{B}{b}_{kind}.nc")
    rec = len(da.dimensions["ocean_time"]) - 1
    recb = len(db.dimensions["ocean_time"]) - 1
    n = 0
    for v in da.variables:
        if v not in db.variables: print(kind, v, "missing in", b); bad += 1; continue
        x = da[v]; y = db[v]
        xa = np.asarray(x[rec] if x.ndim > 0 and x.dimensions[0] == "ocean_time" else x[:])
        ya = np.asarray(y[recb] if y.ndim > 0 and y.dimensions[0] == "ocean_time" else y[:])
        if not np.array_equal(xa, ya, equal_nan=True):
            d = np.nanmax(np.abs(xa.astype(np.float64) - ya.astype(np.float64)))
            print(f"{kind} {v}: DIFFERS, max abs {d:.3e}"); bad += 1
        n += 1
    print(f"{kind}: {n} variables compared at record {rec}")
print("RESULT:", "bit-identical" if bad == 0 else f"{bad} differences")
