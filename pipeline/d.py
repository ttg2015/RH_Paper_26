"""Stage d: derived diagnostics for one stitched profiles file (year × event type).

Port of Paper26_Cleaned/Scripts/d_Add_Derived.py. Same output file, variables and units, so
stages e, f and g read it unchanged. Changes from the legacy script:
- E07: z_LCL_2m and z_LCL_LPL are heights ABOVE GROUND. The LPL parcel starts from the LPL's own
  height (Z_LPL / g), not from the surface; legacy stored z_s + Δz for both (above sea level).
  Both variables carry attrs reference = "above_ground" (stage g then skips its correction).
- Thermodynamics from src/tpl_mll/thermo.py (canonical constants; Romps c_vv = 1418, legacy 1318:
  p_LCL moves by < 0.3 hPa).
- E08 needs no change here: the LPL LCL already uses T, RH and p at the LPL.
RH stays q / q_sat as before (stored unclipped; clipped to (0, 1] only for the LCL).

Reads:  Profiles/profiles_{E}_b{BUF}_{YEAR}.nc, ERA5 invariant geopotential
Writes: Profiles/derived/derived_{E}_b{BUF}_{YEAR}.nc
Usage:  python pipeline/d.py YEAR EVENT [--buffer 6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpl_mll.config import load                        # noqa: E402
from tpl_mll import thermo                              # noqa: E402
from tpl_mll.masks import surface_height_km             # noqa: E402

G = 9.8
UNITS = {"LPL_pressure": "hPa", "T_LPL": "K", "Q_LPL": "kg kg-1", "Z_LPL": "kJ kg-1",
         "MSE_LPL": "kJ kg-1", "MSE_sat_LPL": "kJ kg-1", "RH_2m": "1", "RH_LPL": "1",
         "T_LCL_2m": "K", "p_LCL_2m": "hPa", "z_LCL_2m": "m", "T_LCL_LPL": "K", "p_LCL_LPL": "hPa",
         "z_LCL_LPL": "m"}


def lowest_pressure_level(ds):
    """Deepest level with p <= SP per (year, lead, lat, lon); 1000 hPa where SP > 1000 hPa."""
    level = ds["level"].values
    assert np.all(np.diff(level) > 0), "level must be ascending"
    sp_hpa = ds["SP"].values / 100.0
    idx = np.clip(np.searchsorted(level, sp_hpa, side="right") - 1, 0, len(level) - 1)
    at = lambda v: np.take_along_axis(ds[v].values, idx[..., None], axis=-1)[..., 0]
    return {"LPL_pressure": level[idx], "T_LPL": at("T"), "Q_LPL": at("Q"), "Z_LPL": at("Z"),
            "MSE_LPL": at("MSE"), "MSE_sat_LPL": at("MSE_sat")}


def lcl(T, RH, p_pa, z_base_agl):
    """(T_LCL [K], p_LCL [hPa], z_LCL above ground [m]) for parcels at height z_base_agl."""
    return (thermo.T_lcl(T, RH), thermo.p_lcl(T, RH, p_pa) / 100.0, thermo.z_lcl(T, RH, z_base_agl))


def derive(ds, zs_m):
    """All derived variables as a dict of arrays shaped like SP (year, lead, lat, lon)."""
    out = lowest_pressure_level(ds)
    T2m, q2m, sp = ds["T2m"].values, ds["q2m"].values, ds["SP"].values
    lpl_pa = out["LPL_pressure"] * 100.0
    out["RH_2m"] = thermo.rh(q2m, T2m, sp)
    out["RH_LPL"] = thermo.rh(out["Q_LPL"], out["T_LPL"], lpl_pa)
    zs = np.broadcast_to(zs_m, T2m.shape)
    # 2 m parcel starts at the ground; LPL parcel starts at the LPL's height above ground (E07)
    lpl_agl = np.maximum(out["Z_LPL"] / G * 1000.0 - zs, 0.0)
    out["T_LCL_2m"], out["p_LCL_2m"], out["z_LCL_2m"] = lcl(T2m, out["RH_2m"], sp, 0.0)
    out["T_LCL_LPL"], out["p_LCL_LPL"], out["z_LCL_LPL"] = lcl(out["T_LPL"], out["RH_LPL"], lpl_pa, lpl_agl)
    return out


def run(year, event, buffer):
    paths = load("paths")
    work = Path(paths["work"])
    src = work / paths["outputs"]["profiles"] / f"profiles_{event}_b{buffer:02d}_{year}.nc"
    out_dir = work / paths["outputs"]["derived"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"derived_{event}_b{buffer:02d}_{year}.nc"
    print(f"Reading {src}", flush=True)
    with xr.open_dataset(src) as p:
        ds = p[["T", "Q", "Z", "MSE", "MSE_sat", "T2m", "q2m", "SP"]].load()
    zs_m = surface_height_km(ds["latitude"].values, ds["longitude"].values) * 1000.0
    print(f"  z_surface range [{np.nanmin(zs_m):.0f}, {np.nanmax(zs_m):.0f}] m", flush=True)
    d = derive(ds, zs_m)
    dims = ("year", "lead", "latitude", "longitude")
    out = xr.Dataset({k: (dims, np.asarray(v, dtype="float32")) for k, v in d.items()},
                     coords={c: ds[c].values for c in dims},
                     attrs={"description": f"Derived diagnostics for {event} buffer={buffer} year={year}",
                            "source": src.name, "event_type": event, "year": int(year),
                            "buffer_hours": int(buffer),
                            "note": "RH = q/q_sat. LCL via Romps (2017), c_vv = 1418. "
                                    "z_LCL_* are heights above ground (E07 fixed); the LPL parcel starts "
                                    "at the LPL's height."})
    for v, u in UNITS.items():
        out[v].attrs["units"] = u
    for v in ("z_LCL_2m", "z_LCL_LPL"):
        out[v].attrs["reference"] = "above_ground"
    out.to_netcdf(out_path, encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
    print(f"  z_LCL_2m median {np.nanmedian(d['z_LCL_2m']):.0f} m, z_LCL_LPL median "
          f"{np.nanmedian(d['z_LCL_LPL']):.0f} m (above ground)", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("event", choices=["T", "M"])
    ap.add_argument("--buffer", type=int, default=6)
    a = ap.parse_args()
    run(a.year, a.event, a.buffer)
