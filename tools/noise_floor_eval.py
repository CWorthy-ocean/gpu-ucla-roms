"""Noise-floor test of the GPU port, after noise_floor_cmp.py.

For each hour, RMS differences from the CPU reference day run of:
  pert : CPU twin with initial T perturbed by PERT_AMP (default 1e-6 degC),
         i.e. the CPU model's own sensitivity (the noise floor)
  gpu  : GPU day run
and the ratio gpu/pert. A ratio near or below 1 means the GPU-CPU difference
is no larger than what a tiny perturbation produces in the CPU model itself.
"""
import sys

import netCDF4 as nc
import numpy as np

B = "/glade/derecho/scratch/bachman/roms_marbl_setup_assistant/cases/"
grd = nc.Dataset(B + "MiniPac/input_data/PACMED12km_v2ni_grid.nc")
S = (slice(1, -1), slice(1, -1))
h = grd["h"][:][S]
mr = grd["mask_rho"][:][S] > 0.5
shelf = mr & (h < 200)
deep = mr & (h > 2000)


def cpu(root):
    f1 = nc.Dataset(B + f"CPU/output/{root}_his.20100330000000.nc")
    f2 = nc.Dataset(B + f"CPU/output/{root}_his.20100331000000.nc")
    return lambda v, n: (f1[v][n] if n < 24 else f2[v][0])


ref = cpu("minipac_cpu_day")
pert = cpu("minipac_cpu_day_pert")
gd = nc.Dataset(B + "GPU/output/minipac_gpu_day_his.nc")
gpu = lambda v, n: gd[v][n]


def rms(a, m):
    return np.sqrt(np.mean(a[..., m] ** 2))


def stats(A, n):
    t0 = np.asarray(ref("temp", n), "f8")[(slice(None),) + S]
    t1 = np.asarray(A("temp", n), "f8")[(slice(None),) + S]
    dT = t1 - t0
    s0 = np.asarray(ref("salt", n), "f8")[(slice(None),) + S]
    s1 = np.asarray(A("salt", n), "f8")[(slice(None),) + S]
    u0 = np.asarray(ref("u", n), "f8")[:, 1:-1, 1:-1]
    u1 = np.asarray(A("u", n), "f8")[:, 1:-1, 1:-1]
    z0 = np.asarray(ref("zeta", n), "f8")[S]
    z1 = np.asarray(A("zeta", n), "f8")[S]
    mu = np.isfinite(u0) & np.isfinite(u1) & (u0 != 0)
    return np.array([rms(dT[-1], mr), rms(dT, mr), rms(dT, shelf), rms(dT, deep),
                     rms(s1 - s0, mr), np.sqrt(np.mean((u1 - u0)[mu] ** 2)),
                     1e3 * rms(z1 - z0, mr), np.abs(dT[:, mr]).max()])


cols = ["Tsfc", "T3d", "Tshelf", "Tdeep", "S3d", "u3d", "zeta_mm", "max|dT|"]
print(f"records: GPU {len(gd['ocean_time'])}")
print("hr  run       " + "  ".join(f"{c:>9}" for c in cols))
hours = [int(x) for x in sys.argv[1:]] or list(range(25))
for n in hours:
    p, g = stats(pert, n), stats(gpu, n)
    print(f"{n:2d}  pert      " + "  ".join(f"{x:9.2e}" for x in p))
    print(f"{n:2d}  gpu       " + "  ".join(f"{x:9.2e}" for x in g))
    with np.errstate(divide="ignore", invalid="ignore"):
        print(f"{n:2d}  gpu/pert  " + "  ".join(f"{x:9.2f}" for x in g / p), flush=True)
