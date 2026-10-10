"""Day-3 restart (double precision) noise-floor comparison: GPU vs CPU ref,
with the CPU perturbed twin as the floor. Last time record of the CPU
restarts (EXACT_RESTART writes two)."""
import os
import sys
import netCDF4 as nc
import numpy as np
TAG = sys.argv[1] if len(sys.argv) > 1 else "3d"   # GPU run tag (minipac_gpu_marbl<TAG>_rst.nc)
# ---- settings: edit here, or set the environment variable ----
CASES = os.environ.get("ROMS_GPU_CASES", "/path/to/cases")             # root of the case directories
GRID = os.environ.get("ROMS_GPU_GRID", os.path.join(CASES, "MiniPac/input_data/PACMED12km_v2ni_grid.nc"))
GPU_OUT = os.path.join(CASES, "GPU/output/minipac_gpu_marbl")          # GPU file prefix (before <tag>)
CPU_OUT = os.path.join(CASES, "CPU_marbl/output/minipac_cpu_marbl")    # CPU file prefix (before <tag>)
CPU_REF_RST = CPU_OUT + "3d_rst.20100402000000.nc"         # CPU reference restart at the comparison time
CPU_PERT_RST = CPU_OUT + "3d_pert_rst.20100402000000.nc"   # CPU perturbed-twin restart
grd = nc.Dataset(GRID)
wet = (grd["mask_rho"][:] > 0.5)[1:-1, 1:-1]
g = nc.Dataset(GPU_OUT + TAG + "_rst.nc")
r = nc.Dataset(CPU_REF_RST)
p = nc.Dataset(CPU_PERT_RST)
print("times gpu", float(g["ocean_time"][-1]), "cpu", np.asarray(r["ocean_time"][:]) if "ocean_time" in r.variables else [v for v in r.variables if "time" in v])

def rms(d):
    d = d[..., wet] if d.ndim == 3 else d[wet]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d * d))) if d.size else float("nan")

VARS = ["zeta", "temp", "salt", "PO4", "NO3", "SiO3", "NH4", "Fe", "Lig", "O2", "DIC", "DIC_ALT_CO2", "ALK", "ALK_ALT_CO2",
        "DOC", "DON", "DOP", "DOPr", "DONr", "DOCr", "zooC", "spChl", "spC", "spP", "spFe", "spCaCO3",
        "diatChl", "diatC", "diatP", "diatFe", "diatSi", "diazChl", "diazC", "diazP", "diazFe",
        "MARBL_PH_SURF", "MARBL_PH_SURF_ALT_CO2", "MARBL_PH_3D", "MARBL_PH_3D_ALT_CO2"]
print(f"{'var':22s} {'scale':>11s} {'rms gpu-ref':>12s} {'rms pert-ref':>13s} {'ratio':>8s}")
worst = 0.0
for v in VARS:
    if v not in g.variables or v not in r.variables:
        print(f"{v:22s} missing (gpu {v in g.variables}, cpu {v in r.variables})"); continue
    gg = np.asarray(g[v][-1], dtype=np.float64)[..., 1:-1, 1:-1]
    rr = np.asarray(r[v][-1], dtype=np.float64)[..., 1:-1, 1:-1]
    pp = np.asarray(p[v][-1], dtype=np.float64)[..., 1:-1, 1:-1]
    if gg.shape != rr.shape:
        print(f"{v:22s} shape mismatch {gg.shape} {rr.shape}"); continue
    dg, dp = rms(gg - rr), rms(pp - rr)
    ratio = dg / dp if dp > 0 else float("inf")
    worst = max(worst, ratio)
    print(f"{v:22s} {rms(rr):11.4e} {dg:12.4e} {dp:13.4e} {ratio:8.3f}")
print("worst ratio", f"{worst:.3f}")
