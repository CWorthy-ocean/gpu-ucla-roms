"""Bitwise comparison of two GPU day history files, variable by variable."""
import sys
import numpy as np
from netCDF4 import Dataset

a, b = Dataset(sys.argv[1]), Dataset(sys.argv[2])
nrec = min(len(a.dimensions['ocean_time']), len(b.dimensions['ocean_time']))
print(f"records compared: {nrec}")
same_all = True
for v in a.variables:
    if v not in b.variables or 'ocean_time' not in a.variables[v].dimensions:
        continue
    first_diff, maxd = None, 0.0
    for r in range(nrec):
        x = np.asarray(a.variables[v][r]); y = np.asarray(b.variables[v][r])
        ne = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        if ne.any():
            if first_diff is None: first_diff = r
            maxd = max(maxd, float(np.nanmax(np.abs(x - y))))
    if first_diff is None:
        print(f"  {v:12s} bit-identical")
    else:
        same_all = False
        print(f"  {v:12s} DIFFERS from record {first_diff}, max |d| = {maxd:.3e}")
print("ALL BIT-IDENTICAL" if same_all else "NOT bit-identical")
