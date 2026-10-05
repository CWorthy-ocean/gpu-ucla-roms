"""Compare a float32 GPU history file against a double reference:
equal iff float32(ref) == new bitwise, record by record."""
import sys
import numpy as np
from netCDF4 import Dataset
ref, new = Dataset(sys.argv[1]), Dataset(sys.argv[2])
nrec = min(len(ref.dimensions['ocean_time']), len(new.dimensions['ocean_time']))
print(f"records: ref {len(ref.dimensions['ocean_time'])} new {len(new.dimensions['ocean_time'])} compared {nrec}")
ok = True
for v in ref.variables:
    if v not in new.variables or 'ocean_time' not in ref.variables[v].dimensions:
        print(f"  {v:12s} missing in new" if v not in new.variables else ""); ok = ok and (v in new.variables); continue
    nv = new.variables[v]
    first, maxd = None, 0.0
    for r in range(nrec):
        x = np.asarray(ref.variables[v][r]); y = np.asarray(nv[r])
        if nv.dtype == np.float32:
            x = x.astype(np.float32)
        ne = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        if ne.any():
            if first is None: first = r
            maxd = max(maxd, float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
    tag = f"{nv.dtype}"
    if first is None:
        print(f"  {v:12s} [{tag}] identical to float32(ref)" if nv.dtype == np.float32 else f"  {v:12s} [{tag}] bit-identical")
    else:
        ok = False
        print(f"  {v:12s} [{tag}] DIFFERS from record {first}, max |d| = {maxd:.3e}")
print("ALL MATCH" if ok else "MISMATCH")
