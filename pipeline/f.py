"""Stage f: plausibility of the entraining plume (Ztop distribution).

For one year and event type, at the event hour (lead 0), for every land cell in the
analysis mask: lift a plume from the LPL for each ε in the sweep, find its top, and record
whether the top lands in the config window (400–600 hPa). Also the ε window that puts each
event's top in the window, and the excess split at the LCL (LPL air, E08).

Reads:  Profiles/profiles_{E}_b{BUF}_{YEAR}.nc (c2), Profiles/derived/derived_{E}_b{BUF}_{YEAR}.nc (d)
Writes: Ztop/ztop_{E}_b{BUF}_{YEAR}.nc

Usage:  python pipeline/f.py YEAR EVENT [--buffer 6] [--chunk 40000]
Design: pipeline-map/Planned/Stage f - Ztop Distribution.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpl_mll.config import load                      # noqa: E402
from tpl_mll import ztop                              # noqa: E402
from tpl_mll.masks import analysis_mask               # noqa: E402

PER_EPS = ["p_top_first", "p_top", "z_top", "category", "multi_layer", "non_monotonic",
           "E_total", "E_below", "E_above"]
UNITS = {"p_top_first": "hPa", "p_top": "hPa", "z_top": "km above sea level", "category": "1",
         "multi_layer": "1", "non_monotonic": "1", "E_total": "kJ kg-1", "E_below": "kJ kg-1",
         "E_above": "kJ kg-1", "eps_top_lo": "km-1", "eps_top_hi": "km-1"}
DESCRIPTION = {
    "p_top": "highest downward crossing of plume h and environment h* (EL of the top layer)",
    "p_top_first": "first downward crossing (EL of the lowest buoyant layer)",
    "multi_layer": "more than one buoyant layer: p_top_first and p_top differ",
    "non_monotonic": "environment h exceeds plume h below the top (entrainment raises the plume)",
    "E_total": "eps * integral of (h_plume - h_env) dz from the LPL to the top",
    "E_below": "part of E_total below the LCL of LPL air",
    "E_above": "part of E_total above the LCL",
    "eps_top_lo": "eps putting the top at the upper window edge; NaN = too shallow even at eps=0, inf = beyond 3/km",
    "eps_top_hi": "eps putting the top at the lower window edge (larger eps, lower top)",
}


def events(year, event, buffer):
    """Masked events at lead 0 as flat (n, ...) arrays plus their (i, j) grid positions."""
    paths = load("paths")
    work = Path(paths["work"])
    prof = xr.open_dataset(work / paths["outputs"]["profiles"] / f"profiles_{event}_b{buffer:02d}_{year}.nc")
    der = xr.open_dataset(work / paths["outputs"]["derived"] / f"derived_{event}_b{buffer:02d}_{year}.nc")
    prof = prof.sel(lead=0).isel(year=0)
    der = der.sel(lead=0).isel(year=0)
    lat, lon = prof["latitude"].values, prof["longitude"].values
    assert np.all(np.diff(prof["level"].values) > 0), "levels must ascend (TOA → surface)"
    mask = analysis_mask(lat, lon) & np.isfinite(der["MSE_LPL"].values)
    i, j = np.nonzero(mask)
    take = lambda da: da.transpose("latitude", "longitude", ...).values[i, j]
    data = dict(levels=prof["level"].values.astype(float), h=take(prof["MSE"]),
                h_sat=take(prof["MSE_sat"]), phi=take(prof["Z"]), h0=take(der["MSE_LPL"]),
                p0=take(der["LPL_pressure"]), p_lcl=take(der["p_LCL_LPL"]))
    return data, (i, j), lat, lon


def run(year, event, buffer, chunk):
    cfg = load("analysis")
    eps_list = np.array(cfg["epsilon"]["sweep_km"], dtype=float)
    window = tuple(cfg["ztop_window_hPa"])
    data, (i, j), lat, lon = events(year, event, buffer)
    n = len(i)
    print(f"{year} {event}: {n} events, eps {list(eps_list)}, window {window}", flush=True)

    res = {k: np.full((len(eps_list), n), np.nan) for k in PER_EPS}
    win = {"eps_top_lo": np.full(n, np.nan), "eps_top_hi": np.full(n, np.nan)}
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        cols = ztop.Columns(data["levels"], data["h"][sl], data["h_sat"][sl], data["phi"][sl],
                            data["h0"][sl], data["p0"][sl])
        out = ztop.sweep(cols, eps_list, data["p_lcl"][sl], window)
        for k in PER_EPS:
            res[k][:, sl] = out[k]
        win["eps_top_lo"][sl] = ztop.eps_for_top(cols, window[0])
        win["eps_top_hi"][sl] = ztop.eps_for_top(cols, window[1])
        print(f"  {sl.stop}/{n}", flush=True)

    def grid(flat, lead_shape=()):
        g = np.full(lead_shape + (len(lat), len(lon)), np.nan, dtype="float32")
        g[..., i, j] = flat
        return g

    ds = xr.Dataset(coords={"epsilon": ("epsilon", eps_list, {"units": "km-1"}),
                            "latitude": lat, "longitude": lon})
    for k in PER_EPS:
        ds[k] = (("epsilon", "latitude", "longitude"), grid(res[k], (len(eps_list),)))
    for k, v in win.items():
        ds[k] = (("latitude", "longitude"), grid(v))
    for k in ds.data_vars:
        ds[k].attrs["units"] = UNITS[k]
        if k in DESCRIPTION:
            ds[k].attrs["description"] = DESCRIPTION[k]
    ds["category"].attrs.update(flag_values=list(ztop.CATEGORIES),
                                flag_meanings=" ".join(v.replace(" ", "_") for v in ztop.CATEGORIES.values()))
    ds.attrs.update(year=year, event_type=event, buffer_hours=buffer, lead=0,
                    window_hPa=list(window), lat_band=list(cfg["lat_band"]),
                    plume_start="LPL with MSE_LPL (stage d)", lcl="p_LCL_LPL (LPL air)")

    out = Path(load("paths")["work"]) / load("paths")["outputs"]["ztop"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ztop_{event}_b{buffer:02d}_{year}.nc"
    ds.to_netcdf(path, encoding={k: {"zlib": True, "complevel": 4} for k in ds.data_vars})
    cat = res["category"]
    for e, row in zip(eps_list, cat):
        frac = {name: np.mean(row == code) for code, name in ztop.CATEGORIES.items() if code}
        print(f"  eps {e:.2f}: " + "  ".join(f"{k} {v:.0%}" for k, v in frac.items()), flush=True)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("event", choices=["T", "M"])
    ap.add_argument("--buffer", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=40000)
    a = ap.parse_args()
    run(a.year, a.event, a.buffer, a.chunk)
