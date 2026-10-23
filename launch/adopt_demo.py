"""Toy test cases for `launch adopt`: one good file and one deliberately bad file per check.

    python -m launch adopt --demo

Builds a tiny fake GLADE tree (5° grid, two years), runs adopt on it and writes the
report to runs/adopt_demo/. Each planted fault should be caught by exactly the check
named next to it below, which makes this both a test and a demo for advisors.
"""
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr

from tpl_mll.config import load
from . import adopt

LAT = np.arange(-50.0, 65.1, 5.0)
LON = np.arange(0.0, 360.0, 5.0)

# (stage, label) -> what was planted and which check should catch it
PLANTED = {
    ("b", "T2m 1981"): ("values saved in °C instead of K", "range"),
    ("d", "T 1980"): ("legacy z_LCL above sea level", "e07"),
    ("d", "T 1981"): ("job killed mid-write: file is garbage bytes", "opens"),
    ("d", "M 1980"): ("variable z_LCL_LPL never written", "vars"),
    ("d", "M 1981"): ("job never ran", "exists"),
    ("e", "1980"): ("made with the old ±55° band", "band"),
    ("b", "MSE2m 1981"): ("log has no '=== DONE' (job crashed after writing)", "log"),
}


def _terrain(rng):
    lon2, lat2 = np.meshgrid(LON, LAT)
    hills = 1500 * np.exp(-((lat2 - 35) ** 2 + (lon2 - 90) ** 2) / 400)
    return (hills + 200 * rng.random(lat2.shape)).astype("float32")


def build(root, rng):
    work, logs = root / "work", root / "logs"
    out = load("paths")["outputs"]
    for p in [*out.values()]:
        (work / p).mkdir(parents=True, exist_ok=True)
    logs.mkdir()
    zs = _terrain(rng)
    inv = root / "invariant_z.nc"
    xr.Dataset({"Z": (("latitude", "longitude"), zs[::-1] * 9.8)},
               coords={"latitude": LAT[::-1], "longitude": LON}).to_netcdf(inv)
    shape = (len(LAT), len(LON))
    grid = {"latitude": LAT, "longitude": LON}

    def log(name, done=True):
        (logs / f"{name}.log").write_text("=== START\n" + ("=== DONE\n" if done else "Killed\n"))

    # b: annual maxima
    for event, var, base in (("T", "T2m", 305.0), ("M", "MSE2m", 340.0)):
        for year in (1980, 1981):
            val = base + 5 * rng.standard_normal(shape)
            if (event, year) == ("T", 1981):
                val = val - 273.15
            ds = xr.Dataset({"event_time": (("latitude", "longitude"), np.zeros(shape)),
                             "event_value": (("latitude", "longitude"), val.astype("float32")),
                             "event_doy": (("latitude", "longitude"), np.full(shape, 200)),
                             "event_hour": (("latitude", "longitude"), np.full(shape, 14))},
                            coords=grid)
            ds.to_netcdf(work / out[f"annmax_{var}"] / f"AnnMax_{var}_{year}.nc")
            log(f"maxevent_b06_{year}_{event}", done=(event, year) != ("M", 1981))

    # d: derived files
    for event in ("T", "M"):
        for year in (1980, 1981):
            path = work / out["derived"] / f"derived_{event}_b06_{year}.nc"
            log(f"derived_{event}_b06_{year}")
            if (event, year) == ("M", 1981):
                continue
            if (event, year) == ("T", 1981):
                path.write_bytes(b"\x89HDF\r\n" + bytes(200))
                continue
            dz = (1000 + 600 * rng.random(shape)).astype("float32")
            data = {v: (("year", "lead", "latitude", "longitude"),
                        (300 + rng.random((1, 1) + shape)).astype("float32"))
                    for v in adopt.DERIVED_VARS}
            data["z_LCL_2m"] = (data["z_LCL_2m"][0], (zs + dz)[None, None])
            data["z_LCL_LPL"] = (data["z_LCL_LPL"][0], (zs + dz)[None, None])
            if (event, year) == ("M", 1980):
                data.pop("z_LCL_LPL")
            xr.Dataset(data, coords={"year": [year], "lead": [0], **grid}).to_netcdf(path)

    # e: rh_binned
    band = load("analysis")["lat_band"]
    for year, (lo, hi) in ((1980, (-55.0, 55.0)), (1981, tuple(band))):
        xr.Dataset({"n_events": (("case", "rh_bin"), rng.integers(0, 500, (4, 20)))},
                   attrs={"lat_min": lo, "lat_max": hi}).to_netcdf(
            work / out["rh_binned"] / f"rh_binned_b06_{year}.nc")
        log(f"rh_bins_b06_{year}")
    return work, logs, inv


def run(out_dir):
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmp:
        work, logs, inv = build(Path(tmp), rng)
        report = adopt.run(["b", "d", "e"], (1980, 1981), ["T", "M"], out_dir, work=work,
                           log_dir=logs, script_dir=Path(tmp) / "no_scripts", zs_file=inv,
                           sample=None)
    _append_cases(report)
    return report


def _append_cases(report):
    """Add the 'planted fault → caught by' table to the top of the demo report."""
    rows = {}
    import csv
    with open(report.with_suffix(".csv")) as f:
        for r in csv.DictReader(f):
            rows[(r["stage"], r["file"])] = r
    L = ["## Demo: planted faults", "",
         "Toy files on a 5° grid. Each row is a fault planted on purpose and the check that "
         "should catch it.", "",
         "| Stage | File | Planted fault | Expected check | Caught? | Verdict |", "|---|---|---|---|---|---|"]
    for (stage, label), (what, cid) in PLANTED.items():
        r = rows.get((stage, label), {})
        caught = r.get(cid, "").split(":")[0] in (adopt.FAIL, adopt.WARN)
        L.append(f"| {stage} | {label} | {what} | {adopt.CHECKS[cid][0]} | "
                 f"{'✅ yes' if caught else '❌ no'} | {r.get('verdict', '')} |")
    text = report.read_text().split("\n", 2)
    report.write_text("\n".join([text[0], text[1], "", *L, "", text[2]]))
