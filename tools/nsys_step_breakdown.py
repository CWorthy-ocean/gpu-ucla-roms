"""Per-step breakdown of an nsys sqlite export of one rank: kernel busy
time, launch count, idle gaps (attributed to the kernel that follows them),
runtime API counts and tiny-kernel families.

usage: python nsys_step_breakdown.py <report.sqlite> [step_index]
Steps are delimited by the first 'step3d_t' kernel of each baroclinic step.
Without step_index: one line per step. With it: the detail of that step
(negative counts from the end, e.g. -2 = second-to-last)."""
import sqlite3, sys, re, collections

db = sqlite3.connect(sys.argv[1]); cur = db.cursor()
names = dict(cur.execute("select id, value from StringIds"))
k = [(s, e, names.get(n, str(n))) for s, e, n in
     cur.execute("select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
rt = [(s, e, names.get(n, str(n))) for s, e, n in
      cur.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
fam = lambda n: re.sub(r'_\d+_gpu(__\d+)?$', '', n)
marks = [s for s, e, n in k if 'step3d_t' in n.lower()]
st = []
for m in marks:
    if not st or m - st[-1] > 0.2e9: st.append(m)
if len(st) < 2: sys.exit('fewer than two steps in the profile')

def window(i):
    return st[i], st[i + 1]

def summary(i):
    ta, tb = window(i); W = (tb - ta) / 1e9
    kw = [x for x in k if ta <= x[0] < tb]; kb = sum(e - s for s, e, n in kw) / 1e9
    gaps = collections.Counter(); last = ta; ng = 0; gt = 0.0
    for s, e, n in kw:
        if s - last > 0.5e6: gaps[fam(n)] += (s - last) / 1e6; ng += 1; gt += (s - last) / 1e6
        last = max(last, e)
    return W, kb, kw, gaps, ng, gt

if len(sys.argv) < 3:
    for i in range(len(st) - 1):
        W, kb, kw, gaps, ng, gt = summary(i)
        top = ", ".join("%s %.0f" % (f.replace('halo_mod_', '')[:22], t) for f, t in gaps.most_common(4))
        print("step %2d: %.3f s busy %.3f (%2.0f%%) launches %4d  gaps>0.5ms %3d = %5.0f ms  [%s]"
              % (i, W, kb, 100 * kb / W, len(kw), ng, gt, top))
    sys.exit(0)

i = int(sys.argv[2])
if i < 0: i += len(st) - 1
W, kb, kw, gaps, ng, gt = summary(i)
ta, tb = window(i)
print("step %d: %.3f s, kernel busy %.3f s (%.0f%%), launches %d, idle gaps >0.5 ms: %d = %.0f ms"
      % (i, W, kb, 100 * kb / W, len(kw), ng, gt))
print("idle before (ms):")
for f, t in gaps.most_common(12): print("   %-45s %7.1f" % (f[:45], t))
api = collections.Counter(); apit = collections.Counter()
for s, e, n in rt:
    if ta <= s < tb: api[n] += 1; apit[n] += (e - s) / 1e9
print("runtime API:")
for n, c in api.most_common(8): print("   %-36s %6d calls %8.3f s" % (n[:36], c, apit[n]))
kt = collections.Counter(); kc = collections.Counter(); tiny = collections.Counter()
for s, e, n in kw:
    f = fam(n); kt[f] += (e - s) / 1e6; kc[f] += 1
    if e - s < 20e3: tiny[f] += 1
print("kernel families by time (ms), count, tiny(<20us) count:")
for f, t in kt.most_common(25): print("   %-50s %7.1f %5d %5d" % (f[:50], t, kc[f], tiny[f]))
print("tiny kernels total: %d of %d launches" % (sum(tiny.values()), len(kw)))
