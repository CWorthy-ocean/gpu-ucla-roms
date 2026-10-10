#!/usr/bin/env python3
"""Bit-for-bit comparison of two NetCDF files.

usage: python cmp_nc.py A B [--last] [label]
  default: every record of every record variable (same record count needed)
  --last : the last record of A against the last record of B
Variables only in one file are listed and skipped. Prints max |A-B| per
differing variable."""
import sys
import numpy as np
import netCDF4 as nc
args = [a for a in sys.argv[1:] if not a.startswith('--')]
last = '--last' in sys.argv
A, B = nc.Dataset(args[0]), nc.Dataset(args[1])
label = args[2] if len(args) > 2 else ''
ta = 'ocean_time' if 'ocean_time' in A.dimensions else 'time'
na, nb = len(A.dimensions[ta]), len(B.dimensions[ta])
only = sorted(set(A.variables) ^ set(B.variables))
bad, n = [], 0
for v in sorted(set(A.variables) & set(B.variables)):
    x, y = A[v], B[v]
    rec = x.ndim > 0 and x.dimensions[-1 if False else 0] == ta
    if rec:
        if last:
            xa, ya = np.asarray(x[na - 1]), np.asarray(y[nb - 1])
        else:
            if na != nb:
                bad.append('%s(records %d vs %d)' % (v, na, nb)); continue
            xa, ya = np.asarray(x[:]), np.asarray(y[:])
    else:
        xa, ya = np.asarray(x[:]), np.asarray(y[:])
    n += 1
    if xa.shape != ya.shape:
        bad.append('%s(shape)' % v); continue
    eq = (xa == ya) | (np.isnan(xa) & np.isnan(ya)) if xa.dtype.kind == 'f' else (xa == ya)
    if not np.all(eq):
        d = np.nanmax(np.abs(xa.astype(float) - ya.astype(float)))
        bad.append('%s(max diff %.3e)' % (v, d))
print('%s %s: %d variables, records %d vs %d%s, only in one file: %s' % (
    label, 'last record' if last else 'all records', n, na, nb, '', ', '.join(only) if only else 'none'))
print('  RESULT:', 'bit-identical' if not bad else 'DIFFER: ' + ' '.join(bad[:12]))
