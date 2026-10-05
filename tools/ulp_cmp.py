"""Bitwise check of CPU vs GPU probe dumps printed with es26.17 (18 digits,
enough to round-trip a double). For each probe line type and variable, report
how many values are bit-identical and the largest difference, in ULPs of the
variable's scale (its largest |value| in the group, so values near zero do not
inflate the count) and in absolute terms.

usage: python ulp_cmp.py CPU_LOG GPU_LOG [iic ...] [--only TEXT ...]
--only keeps probe types containing any TEXT and lists all their variables.
CPU probes are read from the CPU log plus probe_cpu.r*.txt next to it.
"""
import collections
import glob
import os
import re
import sys

import numpy as np

args = sys.argv[3:]
only = args[args.index("--only") + 1:] if "--only" in args else []
args = args[:args.index("--only")] if "--only" in args else args
cpu_log, gpu_log = sys.argv[1], sys.argv[2]
steps = {int(s) for s in args} or {13477}
NAMES = re.compile(r"(\S+(?:,\S+)+)=\s")


def parse(paths):
    out = collections.defaultdict(list)
    for p in paths:
        with open(p, errors="replace") as fh:
            for line in fh:
                if "diag:" not in line:
                    continue
                line = line[line.index("diag:"):]
                m = list(NAMES.finditer(line))
                im = re.search(r"iic=\s*(\d+)", line)
                if not m or not im or int(im.group(1)) not in steps:
                    continue
                m = m[-1]
                head = " ".join(line[:m.start()].split())
                names = m.group(1).split(",")
                vals = []
                for tok in line[m.end():].split():
                    try:
                        v = float(tok)
                    except ValueError:
                        break
                    # significant digits printed in the mantissa
                    mant = tok.lower().split("e")[0].lstrip("+-").replace(".", "")
                    vals.append((v, len(mant.lstrip("0")) or len(mant)))
                out[head].append((names, vals))
    return out


cdir = os.path.dirname(os.path.abspath(cpu_log))
cpu = parse([cpu_log] + sorted(glob.glob(os.path.join(cdir, "probe_cpu.r*.txt"))))
gpu = parse([gpu_log])

# group (cpu, gpu, head) pairs by probe type (numbers stripped) and variable
pairs = collections.defaultdict(list)
skipped = collections.Counter()
for head in sorted(set(cpu) & set(gpu)):
    kind = " ".join(re.sub(r"[-+]?\d+", "", head).replace("=", "").split())
    for (names, a), (gnames, b) in zip(cpu[head], gpu[head]):
        if names != gnames:          # same header, different variables: skip
            skipped["names"] += 1
            continue
        for name, (x, nx), (y, ny) in zip(names, a, b):
            if min(nx, ny) < 17:     # too few digits printed to resolve ULPs
                skipped["digits"] += 1
                continue
            pairs[(kind, name)].append((x, y, head))

rows = []
for key, pr in pairs.items():
    x = np.array([p[0] for p in pr])
    y = np.array([p[1] for p in pr])
    d = np.abs(x - y)
    scale = max(np.max(np.abs(x)), np.max(np.abs(y)))
    ulp = d / np.spacing(scale) if scale > 0 else d
    i = int(np.argmax(d))
    rows.append((key, len(pr), np.mean(x == y), np.mean(ulp <= 4), ulp.max(),
                 d.max(), scale, pr[i]))
rows.sort(key=lambda r: -r[4])
if only:
    rows = [r for r in rows if any(t in r[0][0] for t in only)]

print(f"steps {sorted(steps)}; line types CPU {len(cpu)} GPU {len(gpu)} "
      f"common {len(set(cpu) & set(gpu))}")
print(f"skipped: {skipped['names']} lines with different variable lists, "
      f"{skipped['digits']} values printed with < 17 digits")
print(f"{len(rows)} probe/variable groups; ULPs relative to each group's largest |value|")
ulps = np.array([r[4] for r in rows])
for lim in (0, 4, 64, 1024, 2**20):
    print(f"  groups with max diff <= {lim:>7} ULP: {np.sum(ulps <= lim):5d} / {len(rows)}")
print()
print(f"{'probe / variable':54} {'n':>5} {'ident':>6} {'<=4ULP':>7} {'maxULP':>9} "
      f"{'max|d|':>9} {'scale':>9}")
for key, n, same, ok, mu, md, sc, _ in (rows if only else rows[:40]):
    print(f"{(key[0] + ' / ' + key[1])[:54]:54} {n:5d} {same:6.0%} {ok:7.0%} {mu:9.3g} "
          f"{md:9.2e} {sc:9.2e}")
print()
print("largest differences:")
for key, n, same, ok, mu, md, sc, (x, y, head) in rows[:10]:
    print(f"  {key[1]} ({mu:.3g} ULP of scale {sc:.3g}): {head}")
    print(f"      CPU {x!r}  GPU {y!r}")
