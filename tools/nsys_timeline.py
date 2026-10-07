"""Time-binned view of an nsys sqlite export: kernel busy time, memcpy bytes
by kind (unified-memory migrations included), unified-memory page faults.

usage: python nsys_timeline.py <report.sqlite> [bin_seconds] [t0_seconds] [t1_seconds]
Prints one line per bin (relative to the first kernel), the whole-run
kernel totals by name (top 25) and the memcpy totals by kind.
"""
import sqlite3
import sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
binw = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
t0s = float(sys.argv[3]) if len(sys.argv) > 3 else None
t1s = float(sys.argv[4]) if len(sys.argv) > 4 else None

names = dict(cur.execute("select id, value from StringIds"))
kinds = {}
try:
    kinds = {r[0]: r[2] for r in cur.execute("select id, name, label from ENUM_CUDA_MEMCPY_OPER")}
except Exception:
    pass

k = cur.execute("select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
if not k:
    sys.exit('no kernels')
T0 = k[0][0]
nb = int((k[-1][1] - T0) / 1e9 / binw) + 2
kbusy = [0.0] * nb
kcount = [0] * nb
ktot = defaultdict(float)
for s, e, nm in k:
    ktot[names.get(nm, str(nm))] += (e - s) / 1e9
    b0 = int((s - T0) / 1e9 / binw)
    b1 = int((e - T0) / 1e9 / binw)
    for b in range(b0, b1 + 1):
        lo = max(s, T0 + b * binw * 1e9)
        hi = min(e, T0 + (b + 1) * binw * 1e9)
        if hi > lo:
            kbusy[b] += (hi - lo) / 1e9
    kcount[b0] += 1

mbytes = defaultdict(lambda: [0.0] * nb)
mtot = defaultdict(float)
for s, e, by, ck in cur.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
    b = int((s - T0) / 1e9 / binw)
    if 0 <= b < nb:
        lab = kinds.get(ck, str(ck))
        mbytes[lab][b] += by
        mtot[lab] += by

gpf = [0] * nb
try:
    for s, n in cur.execute("select start, numberOfPageFaults from CUDA_UM_GPU_PAGE_FAULT_EVENTS"):
        b = int((s - T0) / 1e9 / binw)
        if 0 <= b < nb:
            gpf[b] += n
except Exception:
    pass
cpf = [0] * nb
try:
    for (s,) in cur.execute("select start from CUDA_UM_CPU_PAGE_FAULT_EVENTS"):
        b = int((s - T0) / 1e9 / binw)
        if 0 <= b < nb:
            cpf[b] += 1
except Exception:
    pass

labs = sorted(mbytes.keys())
print('t[s]    kernel_busy  nkern  gpu_faults  cpu_faults  ' + '  '.join('%14s' % l[:14] for l in labs))
for b in range(nb):
    t = b * binw
    if t0s is not None and t < t0s:
        continue
    if t1s is not None and t > t1s:
        break
    print('%6.1f  %10.3f  %6d  %10d  %10d  ' % (t, kbusy[b], kcount[b], gpf[b], cpf[b]) +
          '  '.join('%12.1f MB' % (mbytes[l][b] / 1e6) for l in labs))
print('\nkernel totals (s), top 25:')
for nm, v in sorted(ktot.items(), key=lambda x: -x[1])[:25]:
    print('  %9.3f  %s' % (v, nm[:110]))
print('\nmemcpy totals by kind (GB):')
for l in labs:
    print('  %8.2f  %s' % (mtot[l] / 1e9, l))
