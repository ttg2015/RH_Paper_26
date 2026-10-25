"""Stage g: figure data. Reduce one year × event of profiles + derived into small, additive arrays.

Everything written is a count or a sum on fine bins, so years combine by summation and
figures can coarsen bins, pick sub-bands and switch area weighting without rerunning this.

  δ_FT    = h_LPL − h*500        δ_S = h_2m − h_LPL        δ_total = h_2m − h*500
  RH_2m   from stage d (q / q_sat)
  zLCL_2m LCL of 2 m air above ground [m]; legacy d files store it above sea level (E07)
          and are corrected here on read.

Reads:  Profiles/profiles_{E}_b{BUF}_{YEAR}.nc (c2), Profiles/derived/derived_{E}_b{BUF}_{YEAR}.nc (d)
Writes: FigData/figdata_{E}_b{BUF}_{YEAR}.nc
Bins:   config/figdata.yaml
Usage:  python pipeline/g.py YEAR EVENT [--buffer 6]
Design: pipeline-map/Planned/Figures & FigData.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpl_mll.config import load                                  # noqa: E402
from tpl_mll.corrections import z_lcl_above_ground_legacy        # noqa: E402
from tpl_mll.masks import analysis_mask, surface_height_km       # noqa: E402

UNITS = {"delta_FT": "kJ kg-1", "delta_S": "kJ kg-1", "delta_total": "kJ kg-1",
         "RH_2m": "1", "zLCL_2m": "m above ground"}
FEEDS = {"h1": "Fig - Excess Histograms", "h2_rh_dFT": "Fig - Excess vs RH",
         "h2_rh_dS": "Excess vs RH (surface layer)", "h2_zlcl_dFT": "Fig - LCL-Binned Decomposition",
         "map": "Fig - Excess Maps"}


def edges(spec):
    start, stop, step = spec
    return np.round(np.arange(start, stop + step / 2, step), 6)


def fields(year, event, buffer, lead):
    """All quantities on the full (lat, lon) grid at the event hour."""
    paths = load("paths")
    work = Path(paths["work"])
    out = paths["outputs"]
    with xr.open_dataset(work / out["profiles"] / f"profiles_{event}_b{buffer:02d}_{year}.nc") as p:
        prof = p[["MSE2m", "MSE_sat"]].sel(lead=lead).isel(year=0).load()
    with xr.open_dataset(work / out["derived"] / f"derived_{event}_b{buffer:02d}_{year}.nc") as d:
        der = d[["MSE_LPL", "RH_2m", "z_LCL_2m"]].sel(lead=lead).isel(year=0).load()
    grid = lambda da: da.transpose("latitude", "longitude").values.astype("float64")
    lat, lon = prof["latitude"].values, prof["longitude"].values
    h_lpl = grid(der["MSE_LPL"])
    h_2m = grid(prof["MSE2m"])
    hs500 = grid(prof["MSE_sat"].sel(level=500))
    z = grid(der["z_LCL_2m"])
    if der["z_LCL_2m"].attrs.get("reference") != "above_ground":      # legacy d (E07)
        z = z_lcl_above_ground_legacy(z, surface_height_km(lat, lon) * 1000.0)
    f = {"delta_FT": h_lpl - hs500, "delta_S": h_2m - h_lpl, "delta_total": h_2m - hs500,
         "RH_2m": grid(der["RH_2m"]), "zLCL_2m": z}
    return f, lat, lon


def run(year, event, buffer):
    cfg = load("figdata")
    acfg = load("analysis")
    f, lat, lon = fields(year, event, buffer, cfg["lead"])
    mask = analysis_mask(lat, lon)
    sub = {"all": acfg["lat_band"], **acfg["lat_subbands"]}
    lat2 = np.broadcast_to(lat[:, None], mask.shape)
    weight = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)
    E = {k: edges(v) for k, v in cfg["bins"].items()}

    ds = xr.Dataset(coords={"subband": list(sub)})
    for k, e in E.items():
        ds[f"edges_{k}"] = (f"edge_{k}", e)
    sel = {}
    for name, (lo, hi) in sub.items():
        sel[name] = mask & (lat2 >= lo) & (lat2 <= hi)

    for q in cfg["hist1d"]:
        for wname, wsuffix in ((None, ""), ("cos", "_w")):
            rows = []
            for name in sub:
                m = sel[name] & np.isfinite(f[q])
                rows.append(np.histogram(f[q][m], E["delta"],
                                         weights=weight[m] if wname else None)[0])
            ds[f"h1_{q}{wsuffix}"] = (("subband", "bin_delta"), np.array(rows, dtype="float64"))
            ds[f"h1_{q}{wsuffix}"].attrs.update(units="count" if not wname else "cos(lat)-weighted count",
                                                quantity=q, quantity_units=UNITS[q], edges="edges_delta",
                                                feeds=FEEDS["h1"])

    for name2, (qx, bx, qy, by) in cfg["hist2d"].items():
        for wname, wsuffix in ((None, ""), ("cos", "_w")):
            rows = []
            for name in sub:
                m = sel[name] & np.isfinite(f[qx]) & np.isfinite(f[qy])
                rows.append(np.histogram2d(f[qx][m], f[qy][m], [E[bx], E[by]],
                                           weights=weight[m] if wname else None)[0])
            ds[f"{name2}{wsuffix}"] = (("subband", f"bin_{bx}", f"bin_{by}"), np.array(rows))
            ds[f"{name2}{wsuffix}"].attrs.update(
                units="count" if not wname else "cos(lat)-weighted count", x=qx, y=qy,
                x_units=UNITS[qx], y_units=UNITS[qy], x_edges=f"edges_{bx}", y_edges=f"edges_{by}",
                feeds=FEEDS[name2])

    # Maps: every cell's value (one event per cell per year), mask applied at plot time
    ds = ds.assign_coords(latitude=lat, longitude=lon)
    ds["in_mask"] = (("latitude", "longitude"), mask.astype("int8"))
    for q in cfg["maps"]:
        ds[f"map_{q}"] = (("latitude", "longitude"), f[q].astype("float32"))
        ds[f"map_{q}"].attrs.update(units=UNITS[q], feeds=FEEDS["map"],
                                    note="one value per cell per year; sum over years and divide by the count of finite values")

    n = {name: int(sel[name].sum()) for name in sub}
    ds.attrs.update(year=year, event_type=event, buffer_hours=buffer, lead=cfg["lead"],
                    lat_band=list(acfg["lat_band"]), mask_lsm_min=acfg["mask"]["lsm_min"],
                    mask_zs_max_km=acfg["mask"]["zs_max_km"],
                    definitions="delta_FT = MSE_LPL - MSE_sat(500); delta_S = MSE2m - MSE_LPL; "
                                "delta_total = MSE2m - MSE_sat(500)",
                    zlcl_reference="above ground (legacy d corrected on read, E07)",
                    n_events=str(n))
    out = Path(load("paths")["work"]) / load("paths")["outputs"]["figdata"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"figdata_{event}_b{buffer:02d}_{year}.nc"
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"{year} {event}: events per sub-band {n}", flush=True)
    for q in cfg["hist1d"]:
        v = f[q][sel["all"] & np.isfinite(f[q])]
        print(f"  {q}: mean {v.mean():.2f}  median {np.median(v):.2f}  >0 {np.mean(v > 0):.0%}", flush=True)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("event", choices=["T", "M"])
    ap.add_argument("--buffer", type=int, default=6)
    a = ap.parse_args()
    run(a.year, a.event, a.buffer)
