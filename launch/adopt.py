"""Read-only check of outputs that already exist, before anything is rebuilt.

For every expected output file of a stage it runs a list of checks and gives the file a
verdict (adopt / adopt-with-correction / adopt-with-note / rebuild / broken / missing).
Nothing outside the report folder is written. Each check carries a plain-language
explanation, and the report includes a small evidence figure per check where one helps.
"""
import calendar
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from tpl_mll.config import load
from .registry import STAGES, expand

# --- What each check is for (shown in the report) --------------------------------
CHECKS = {
    "exists": ("File exists",
               "The job was supposed to write this file. If it is missing, the job never ran "
               "or failed before writing."),
    "size": ("Size close to the other files of its stage",
             "Files from the same stage should be roughly the same size. One much smaller "
             "than the median (< 50%) usually means a job was killed while writing."),
    "opens": ("Opens as netCDF",
              "A truncated or corrupted file cannot be opened at all."),
    "vars": ("Expected variables present",
             "The file contains the variables the next stage reads."),
    "nan": ("Not all NaN",
            "A sample of every variable is read. A variable that is entirely NaN means the "
            "job wrote an empty result (e.g. a mask that selected nothing)."),
    "log": ("Job log says DONE",
            "Every PBS job prints '=== DONE' as its last line. A log without it means the "
            "job crashed, even if a file was written."),
    "lineage": ("Made after the script's last edit",
                "If the file is older than the last change to the script that makes it, it "
                "may come from an earlier version of the code."),
    "range": ("Values in a physical range",
              "Annual-max 2 m temperature should be 220–340 K and 2 m MSE 250–420 kJ/kg. "
              "Values outside mean a units or selection error."),
    "lat_order": ("Latitude runs south → north",
                  "The pipeline stores profiles S→N while ERA5 pressure-level files are N→S. "
                  "A flipped file would put every event at the wrong latitude."),
    "e07": ("LCL height reference (E07)",
            "Legacy z_LCL is stored above sea level (z_s + Δz) instead of above ground. If "
            "z_LCL_2m − z_s is never negative and tracks the terrain, the file has the E07 "
            "offset, which corrections.z_lcl_above_ground_legacy removes without a rerun."),
    "band": ("Latitude band matches the config (E01)",
             "rh_binned files record the band they used. The project band is now set in "
             "config/analysis.yaml; a file made with another band has to be rebuilt."),
    "recompute": ("Recompute matches",
                  "Rerun the new code on a sample and diff against the old file. "
                  "Not built yet: comes once each stage is ported to pipeline/."),
}

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"
ICON = {PASS: "✅", WARN: "⚠️", FAIL: "❌", SKIP: "·"}

PROFILE_VARS = ["T", "Q", "Z", "MSE", "MSE_sat", "T2m", "q2m", "SP", "MSE2m"]
DERIVED_VARS = ["LPL_pressure", "T_LPL", "Q_LPL", "Z_LPL", "MSE_LPL", "MSE_sat_LPL",
                "RH_2m", "RH_LPL", "T_LCL_2m", "p_LCL_2m", "z_LCL_2m",
                "T_LCL_LPL", "p_LCL_LPL", "z_LCL_LPL"]
ANNMAX = {"T": ("annmax_T2m", "T2m", (220, 340)), "M": ("annmax_MSE2m", "MSE2m", (250, 420))}
SURFACE = [("T2m", "T2m"), ("Q2m", "q2m"), ("SP", "SP"), ("MSE2m", "MSE2m")]
LEGACY_TAG = {"a": "surface", "b": "maxevent", "c0": "mask_build", "c1": "sweep",
              "c2": "stitch", "d": "derived", "e": "rh_bins"}
SCRIPT = {"a": "a_Surface_Fields.py", "b": "b_Ann_Max_Finder.py", "c0": "c0_build_mask.py",
          "c1": "c1_sweep_month.py", "c2": "c2_stitch_year.py", "d": "d_Add_Derived.py",
          "e": "e_RH_bins.py"}


@dataclass
class Result:
    stage: str
    label: str
    path: Path
    log: Path = None
    checks: dict = field(default_factory=dict)   # check id -> (status, detail)
    verdict: str = ""

    def add(self, cid, status, detail=""):
        self.checks[cid] = (status, detail)


# --- Expected files and logs for each job ----------------------------------------
def _legacy_log(job, log_dir):
    """Log name exactly as the old !_Launcher wrote it (_tag in cell [2])."""
    s = LEGACY_TAG[job.stage.key]
    if job.event is not None and job.stage.key != "b":
        s += f"_{job.event}"
    s += f"_b{job.buffer:02d}"
    if job.year is not None:
        s += f"_{job.year}"
    if job.month is not None:
        s += f"_{job.month:02d}"
    if job.stage.key == "b":
        s += f"_{job.event}"
    return Path(log_dir) / f"{s}.log"


def expected_files(job, work, years):
    """[(label, path, required_vars)] for one job."""
    out = load("paths")["outputs"]
    d = lambda key: Path(work) / out[key]
    k, y, m, e, b = job.stage.key, job.year, job.month, job.event, job.buffer
    if k == "a":
        days = calendar.monthrange(y, m)[1]
        return [(f"{var} {y}-{m:02d}-{dd:02d}", d(key) / f"{var}_{y}_{m:02d}_{dd:02d}.nc", [])
                for key, var in SURFACE for dd in range(1, days + 1)]
    if k == "b":
        key, var, _ = ANNMAX[e]
        return [(f"{var} {y}", d(key) / f"AnnMax_{var}_{y}.nc",
                 ["event_time", "event_value", "event_doy", "event_hour"])]
    if k == "c0":
        return [(f"{e} {yy}-{mm:02d}", d("masks") / f"mask_{e}_b{b:02d}_{yy}_{mm:02d}.nc", [])
                for yy in range(years[0], years[1] + 1) for mm in range(1, 13)]
    if k == "c1":
        return [(f"{e} {y}-{m:02d}",
                 d("profiles_intermediate") / f"profiles_{e}_b{b:02d}_{y}_{m:02d}.nc",
                 PROFILE_VARS)]
    if k == "c2":
        return [(f"{e} {y}", d("profiles") / f"profiles_{e}_b{b:02d}_{y}.nc", PROFILE_VARS)]
    if k == "d":
        return [(f"{e} {y}", d("derived") / f"derived_{e}_b{b:02d}_{y}.nc", DERIVED_VARS)]
    if k == "e":
        return [(f"{y}", d("rh_binned") / f"rh_binned_b{b:02d}_{y}.nc", ["n_events"])]
    return []


# --- Individual checks ------------------------------------------------------------
def _sample(da, cap=200_000):
    """Strided sample of a DataArray with at most ~cap elements."""
    step = max(1, int((da.size / cap) ** (1 / max(da.ndim, 1))) + 1) if da.size > cap else 1
    return da.isel({dim: slice(None, None, step) for dim in da.dims}).values


def _zs(zs_file, lat, lon):
    """Surface height [m] from the ERA5 invariant geopotential, on the file's lat/lon."""
    with xr.open_dataset(zs_file) as inv:
        z = inv[list(inv.data_vars)[0]]
        if "time" in z.dims:
            z = z.isel(time=0)
        return (z.sel(latitude=lat, longitude=lon, method="nearest") / 9.8).values


def deep_checks(r, ds, req, ctx, evidence):
    """Checks that need the file open. Adds evidence for the figures."""
    missing = [v for v in req if v not in ds.variables]
    r.add("vars", FAIL if missing else PASS,
          f"missing {', '.join(missing)}" if missing else f"{len(req) or len(ds.data_vars)} variables")
    empty = [v for v in (req or list(ds.data_vars))
             if v in ds and np.issubdtype(ds[v].dtype, np.number) and np.isnan(_sample(ds[v])).all()]
    r.add("nan", FAIL if empty else PASS, f"all NaN: {', '.join(empty)}" if empty else "")

    k = r.stage
    if k == "b":
        lo, hi = ANNMAX[ctx["event"]][2]
        v = _sample(ds["event_value"])
        v = v[np.isfinite(v)]
        bad = np.mean((v < lo) | (v > hi)) if v.size else 1.0
        r.add("range", FAIL if bad > 0.01 else PASS, f"{bad:.1%} outside {lo}–{hi}")
        evidence.setdefault("range", {}).setdefault(ctx["event"], []).append(v)
    if k in ("c2", "d") and "latitude" in ds.coords:
        lat = ds["latitude"].values
        r.add("lat_order", PASS if lat[0] < lat[-1] else FAIL, f"{lat[0]:.2f} → {lat[-1]:.2f}")
    if k == "d" and "z_LCL_2m" in ds and ctx.get("zs_file"):
        zs = _zs(ctx["zs_file"], ds["latitude"], ds["longitude"])
        z = ds["z_LCL_2m"]
        z = z.isel({d: 0 for d in z.dims if d not in ("latitude", "longitude")}).values
        ok = np.isfinite(z) & np.isfinite(zs)
        dz = (z - zs)[ok]
        asl = dz.size and np.mean(dz >= -1.0) > 0.999 and np.corrcoef(z[ok], zs[ok])[0, 1] > 0.3
        r.add("e07", WARN if asl else PASS,
              "stored above sea level → apply correction" if asl else "already above ground")
        evidence.setdefault("e07", []).append((zs[ok], z[ok]))
    if k == "e":
        band = load("analysis")["lat_band"]
        got = (ds.attrs.get("lat_min"), ds.attrs.get("lat_max"))
        same = got[0] is not None and np.allclose(got, band)
        r.add("band", PASS if same else FAIL, f"file {got}, config {tuple(band)}")


def check_file(r, req, ctx, deep, sizes, evidence):
    if not r.path.exists():
        r.add("exists", FAIL, "not found")
        return
    r.add("exists", PASS)
    sizes.setdefault(r.stage, []).append((r.label, r.path.stat().st_size, r))
    if r.log is not None:
        if not r.log.exists():
            r.add("log", WARN, "no log found")
        else:
            r.add("log", PASS if "=== DONE" in r.log.read_text(errors="ignore") else FAIL,
                  r.log.name)
    script = ctx.get("script")
    if script is not None and script.exists():
        older = r.path.stat().st_mtime < script.stat().st_mtime
        r.add("lineage", WARN if older else PASS,
              f"older than {script.name}" if older else "")
    if not deep:
        return
    try:
        with xr.open_dataset(r.path) as ds:
            r.add("opens", PASS)
            deep_checks(r, ds, req, ctx, evidence)
    except Exception as exc:          # any read failure is the finding
        r.add("opens", FAIL, type(exc).__name__)
    r.add("recompute", SKIP, "not built yet")


def size_checks(sizes):
    for stage, rows in sizes.items():
        med = np.median([s for _, s, _ in rows])
        for _, s, r in rows:
            r.add("size", WARN if s < 0.5 * med else PASS, f"{s/1e6:.1f} MB (median {med/1e6:.1f})")


def verdict(r):
    st = {c: s for c, (s, _) in r.checks.items()}
    if st.get("exists") == FAIL:
        return "missing"
    if FAIL in (st.get(c) for c in ("opens", "vars", "nan", "log", "range", "lat_order")):
        return "broken"
    if st.get("band") == FAIL:
        return "rebuild"
    if st.get("e07") == WARN:
        return "adopt-with-correction"
    if WARN in st.values():
        return "adopt-with-note"
    return "adopt"


# --- Driver -----------------------------------------------------------------------
def run(stage_keys, years, events, out_dir, work=None, log_dir=None, script_dir=None,
        zs_file=None, sample=50, seed=0, apply=False, git="nogit"):
    """Check every expected file; open a random sample of `sample` per stage (None = all).

    With apply=True (needs sample=None, so every file was opened) the files whose verdict
    starts with "adopt" get provenance attrs, so `launch run` treats them as done.
    """
    if apply and sample is not None:
        raise ValueError("--apply needs --sample all: every file must be opened before stamping")
    paths = load("paths")
    work = Path(work or paths["work"])
    log_dir = Path(log_dir or paths["log_dir"])
    script_dir = Path(script_dir or paths["legacy_scripts"])
    if zs_file is None:
        zs_file = Path(paths["era5"]["root"]) / paths["era5"]["invariant_z"]
    rng = random.Random(seed)
    results, sizes, evidence = [], {}, {}
    for key in stage_keys:
        if key not in SCRIPT:
            continue
        items = []
        for job in expand([key], years, events):
            for label, path, req in expected_files(job, work, years):
                log = _legacy_log(job, log_dir) if key != "a" or label.startswith("T2m") else None
                items.append((Result(key, label, path, log), req, job))
        deep = set(range(len(items))) if sample is None else \
            set(rng.sample(range(len(items)), min(sample, len(items))))
        for i, (r, req, job) in enumerate(items):
            ctx = {"event": job.event, "zs_file": zs_file, "script": script_dir / SCRIPT[key]}
            check_file(r, req, ctx, i in deep, sizes, evidence)
            results.append(r)
    size_checks(sizes)
    for r in results:
        r.verdict = verdict(r)
    stamped = 0
    if apply:
        from .provenance import stamp
        for r in results:
            if r.verdict.startswith("adopt"):
                note = "E07: z_LCL stored above sea level" if r.verdict == "adopt-with-correction" else ""
                stamp(r.path, r.stage, git, how=r.verdict, note=note)
                stamped += 1
    from .adopt_report import write_report
    report = write_report(results, sizes, evidence, Path(out_dir))
    if apply:
        with open(report, "a") as f:
            f.write(f"\n## Applied\nStamped {stamped} files as adopted.\n")
    return report
