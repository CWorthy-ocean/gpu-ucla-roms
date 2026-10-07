"""Per-rank GPU busy time per step from per-rank nsys sqlite exports.
usage: python nsys_rank_balance.py <base> <nranks>   (files <base>_r<k>.sqlite)
Steps are delimited by the first 'step3d_t' kernel of each step; for each
steady step the kernel busy time of every rank is printed, so the slowest
tile (which sets the pace) and the imbalance are visible."""
import sqlite3, sys
base, nr = sys.argv[1], int(sys.argv[2])
busy = {}; steps = {}
for r in range(nr):
    db = sqlite3.connect("%s_r%d.sqlite" % (base, r)); cur = db.cursor()
    names = dict(cur.execute("select id, value from StringIds"))
    k = [(s, e, names.get(n, str(n))) for s, e, n in
         cur.execute("select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
    marks = [s for s, e, n in k if 'step3d_t' in n.lower()]
    st = []
    for m in marks:
        if not st or m - st[-1] > 0.2e9: st.append(m)
    b = []
    for i in range(len(st) - 1):
        ta, tb = st[i], st[i + 1]
        b.append((sum(e - s for s, e, n in k if ta <= s < tb) / 1e9, (tb - ta) / 1e9))
    busy[r] = b
n = min(len(busy[r]) for r in busy)
print("step   " + "".join("  r%d busy" % r for r in range(nr)) + "   step(r0)  max/min")
for i in range(n):
    row = [busy[r][i][0] for r in range(nr)]
    print("%4d   " % i + "".join("%8.3f" % x for x in row) + "   %7.3f   %5.2f" % (busy[0][i][1], max(row) / min(row)))
import statistics
med = [statistics.median(busy[r][i][0] for i in range(n)) for r in range(nr)]
print("median busy per rank:", " ".join("%.3f" % m for m in med), " slowest/fastest %.2f" % (max(med) / min(med)))
