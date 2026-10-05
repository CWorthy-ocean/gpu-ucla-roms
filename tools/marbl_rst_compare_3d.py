"""Day-3 restart (double precision) noise-floor comparison: GPU vs CPU ref,
with the CPU perturbed twin as the floor. Last time record of the CPU
restarts (EXACT_RESTART writes two)."""
import netCDF4 as nc
import numpy as np
B = "/glade/derecho/scratch/bachman/roms_marbl_setup_assistant/cases/"
grd = nc.Dataset(B + "MiniPac/input_data/PACMED12km_v2ni_grid.nc")
wet = (grd["mask_rho"][:] > 0.5)[1:-1, 1:-1]
g = nc.Dataset(B + "GPU/output/minipac_gpu_marbl3d_rst.nc")
r = nc.Dataset(B + "CPU_marbl/output/minipac_cpu_marbl3d_rst.20100402000000.nc")
p = nc.Dataset(B + "CPU_marbl/output/minipac_cpu_marbl3d_pert_rst.20100402000000.nc")
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
