"""Stage e: per-year RH-binned sums of every profile and surface variable, per FT energy box.

Port of Paper26_Cleaned/Scripts/e_RH_bins.py. Same output file and layout (sum_VAR, n_events,
box bounds, lat_min / lat_max attrs), so years still combine by summing. Changes:
- E01: latitude band from config/analysis.yaml (−50…65°; legacy ±55°).
- E12: boxes from config/analysis.yaml `boxes` (values unchanged from the legacy CASE_SPEC).
- Mask from tpl_mll.masks.analysis_mask (band, land > 0.9, terrain < 0.5 km): the one shared mask (E10).
As before: event hour (lead 0, selected by value), RH_2m in 5 % bins, RH ≥ 1 falls outside the
last bin, profile sums use the stored levels as they are (terrain cut keeps below-ground levels rare).

Reads:  Profiles/profiles_{E}_b{BUF}_{YEAR}.nc, Profiles/derived/derived_{E}_b{BUF}_{YEAR}.nc (E = T, M)
Writes: rh_binned/rh_binned_b{BUF}_{YEAR}.nc
Usage:  python pipeline/e.py YEAR [--buffer 6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpl_mll.config import load                        # noqa: E402
from tpl_mll.masks import analysis_mask                 # noqa: E402

LEAD = 0
RH_BIN_EDGES = np.round(np.arange(0.0, 1.0 + 0.05, 0.05), 6)
PROFILE_VARS = ["T", "Q", "Z", "q_sat", "MSE", "MSE_sat"]
SURFACE_VARS_PROF = ["T2m", "q2m", "SP", "MSE2m"]
SURFACE_VARS_DER = ["RH_2m", "RH_LPL", "LPL_pressure", "T_LPL", "Q_LPL", "Z_LPL", "MSE_LPL", "MSE_sat_LPL",
                    "T_LCL_2m", "p_LCL_2m", "z_LCL_2m", "T_LCL_LPL", "p_LCL_LPL", "z_LCL_LPL"]
SURFACE_VARS = SURFACE_VARS_PROF + SURFACE_VARS_DER


def accumulate(year, buffer):
    paths = load("paths")
    work = Path(paths["work"])
    boxes = load("analysis")["boxes"]
    cases = list(boxes)
    nb = len(RH_BIN_EDGES) - 1
    sums_p, sums_s, n_evt, levels = None, None, np.zeros((len(cases), nb), dtype=np.int64), None
    for et in ("T", "M"):
        mine = [c for c in cases if c.startswith(et + "_")]
        if not mine:
            continue
        with xr.open_dataset(work / paths["outputs"]["profiles"] / f"profiles_{et}_b{buffer:02d}_{year}.nc") as p, \
             xr.open_dataset(work / paths["outputs"]["derived"] / f"derived_{et}_b{buffer:02d}_{year}.nc") as d:
            for tag, ds in (("profiles", p), ("derived", d)):
                if LEAD not in ds["lead"].values:
                    raise ValueError(f"lead {LEAD} not in {tag} file for {et} {year}")
            prof = p[PROFILE_VARS + SURFACE_VARS_PROF].isel(year=0).sel(lead=LEAD).load()
            der = d[SURFACE_VARS_DER].isel(year=0).sel(lead=LEAD).load()
        lat, lon = prof["latitude"].values, prof["longitude"].values
        if levels is None:
            levels = prof["level"].values
            sums_p = {v: np.zeros((len(cases), nb, len(levels))) for v in PROFILE_VARS}
            sums_s = {v: np.zeros((len(cases), nb)) for v in SURFACE_VARS}
        mask = analysis_mask(lat, lon)
        grid = lambda da: da.transpose("latitude", "longitude", ...).values
        rh = grid(der["RH_2m"])
        h500, hs500 = grid(prof["MSE"].sel(level=500)), grid(prof["MSE_sat"].sel(level=500))
        for case in mine:
            box, c = boxes[case], cases.index(case)
            base = (mask & np.isfinite(rh) & (h500 > box["h"][0]) & (h500 < box["h"][1])
                    & (hs500 > box["h_star"][0]) & (hs500 < box["h_star"][1]))
            k = np.digitize(rh[base], RH_BIN_EDGES) - 1
            ok = (k >= 0) & (k < nb)
            k = k[ok]
            print(f"  {case}: {int(ok.sum())} events", flush=True)
            np.add.at(n_evt[c], k, 1)
            for v in PROFILE_VARS:
                np.add.at(sums_p[v][c], k, grid(prof[v])[base][ok])
            for v in SURFACE_VARS:
                src = prof if v in SURFACE_VARS_PROF else der
                np.add.at(sums_s[v][c], k, grid(src[v])[base][ok])
    return cases, sums_p, sums_s, n_evt, levels


def run(year, buffer):
    cfg = load("analysis")
    cases, sums_p, sums_s, n_evt, levels = accumulate(year, buffer)
    boxes = cfg["boxes"]
    lo, hi = RH_BIN_EDGES[:-1].astype("float32"), RH_BIN_EDGES[1:].astype("float32")
    data = {f"sum_{v}": (("case", "rh_bin", "level"), s) for v, s in sums_p.items()}
    data.update({f"sum_{v}": (("case", "rh_bin"), s) for v, s in sums_s.items()})
    data["n_events"] = (("case", "rh_bin"), n_evt)
    for name, key, j in (("box_mse500_lo", "h", 0), ("box_mse500_hi", "h", 1),
                         ("box_mses500_lo", "h_star", 0), ("box_mses500_hi", "h_star", 1)):
        data[name] = ("case", np.array([boxes[c][key][j] for c in cases], dtype="float32"))
    band = cfg["lat_band"]
    ds = xr.Dataset(data, coords={"case": cases,
                                  "rh_bin": [f"R{int(round(a * 100))}_{int(round(b * 100))}" for a, b in zip(lo, hi)],
                                  "rh_lo": ("rh_bin", lo), "rh_hi": ("rh_bin", hi),
                                  "rh_mid": ("rh_bin", (lo + hi) / 2), "level": levels},
                    attrs={"description": f"Per-year RH-binned sums (all variables) for year {year}",
                           "year": int(year), "buffer_hours": int(buffer), "lead": LEAD,
                           "lat_min": float(band[0]), "lat_max": float(band[1]),
                           "mask": f"land > {cfg['mask']['lsm_min']}, z_s < {cfg['mask']['zs_max_km']} km",
                           "note": "Stitch across years: sum sum_VAR and n_events over years, then divide. "
                                   "Boxes from config/analysis.yaml (E12); band from config (E01)."})
    out = Path(load("paths")["work"]) / load("paths")["outputs"]["rh_binned"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rh_binned_b{buffer:02d}_{year}.nc"
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--buffer", type=int, default=6)
    a = ap.parse_args()
    run(a.year, a.buffer)
