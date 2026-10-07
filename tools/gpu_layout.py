"""Rank layout for a roms_gpu case: compute ranks from np_xi x np_eta, writer
ranks from the history record size, and the PBS select line to launch it.

usage: python gpu_layout.py <namelist> [--gpus-per-node 4] [--target-seconds 15]
                                       [--rate-gbps 0.6] [--max-writers 8]
Prints:  NTILE NWRITERS NRANKS SELECT   (one line, space separated)

Rule: one writer streams about 0.6 GB/s (measured, Derecho Lustre); choose
the smallest number of writers that writes one history record within
--target-seconds, clamped to 1..--max-writers. The writers are the extra
ranks on the last node; the model derives their count from the namelist.
"""
import math
import re
import sys

args = sys.argv[1:]
opts = {'--gpus-per-node': 4, '--target-seconds': 15.0, '--rate-gbps': 0.6, '--max-writers': 8}
path = None
while args:
    a = args.pop(0)
    if a in opts:
        opts[a] = type(opts[a])(args.pop(0))
    else:
        path = a
if path is None:
    sys.exit(__doc__)
txt = open(path).read()
txt = re.sub(r'!.*', '', txt)


def val(key, default=None):
    m = re.search(r'(?<![\w])%s\s*=\s*([^,\n/]+)' % re.escape(key), txt)
    if not m:
        return default
    v = m.group(1).strip().strip("'\"")
    if v.lower() in ('.true.', 't', 'true'):
        return True
    if v.lower() in ('.false.', 'f', 'false'):
        return False
    try:
        return float(v) if '.' in v or 'e' in v.lower() else int(v)
    except ValueError:
        return v


npx = int(val('np_xi', 1)); npy = int(val('np_eta', 1))
llm = int(val('llm', 0)); mmm = int(val('mmm', 0)); nz = int(val('nz', 0))
nt_bgc = int(val('nt_bgc', 0))
cells = (llm + 2) * (mmm + 2)
n2d = sum(1 for k in ('wrt_z', 'wrt_ub', 'wrt_vb', 'wrt_hbls', 'wrt_hbbl') if val(k, False))
n3d = sum(1 for k in ('wrt_u', 'wrt_v', 'wrt_temp', 'wrt_salt', 'wrt_akv', 'wrt_akt', 'wrt_r') if val(k, False))
if val('write_passive', True) and nt_bgc > 0:
    n3d += nt_bgc
if nt_bgc > 0:
    n2d += 8      # MARBL surface diagnostics (PH, pCO2, FG_CO2, zsat, ...)
    n3d += 4      # MARBL 3-D carbonate diagnostics
record_gb = 4.0 * cells * (n3d * nz + n2d) / 1e9
nw = 1
if val('wrt_file_his', True):
    nw = max(1, min(opts['--max-writers'], math.ceil(record_gb / (opts['--rate-gbps'] * opts['--target-seconds']))))
ntile = npx * npy
g = opts['--gpus-per-node']
nodes = max(1, math.ceil(ntile / g))
if nodes == 1:
    select = '1:ncpus=64:mpiprocs=%d:ngpus=%d' % (ntile + nw, g)
else:
    last = ntile - g * (nodes - 1)
    select = '%d:ncpus=64:mpiprocs=%d:ngpus=%d+1:ncpus=64:mpiprocs=%d:ngpus=%d' % (nodes - 1, g, g, last + nw, g)
print('%d %d %d %s' % (ntile, nw, ntile + nw, select))
print('# history record ~%.1f GB -> %d writer(s) for <= %.0f s per record at %.1f GB/s each' %
      (record_gb, nw, opts['--target-seconds'], opts['--rate-gbps']), file=sys.stderr)
