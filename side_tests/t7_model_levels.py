"""Task 7 (PARKED, not in run_side_tests.pbs, never run yet). Model-level vs pressure-level ERA5
for ~200 events of ONE year: do h_LPL, h*500 and δ_a change when the profile comes from the
137 model levels instead of the 37 pressure levels?

Reads   AnnMax/{T2m,MSE2m}/ (event hour per cell, stage b), Profiles/ and Profiles/derived/ (the
        pressure-level values), ERA5 model levels under paths.yaml era5.model_levels (GRIB)
Writes  <out>/t7_{E}_{Y}.npz and <out>/t7_report.md

Step 1, cheap and safe (login node is fine, reads directory listings and one file header):
    python side_tests/t7_model_levels.py inspect 2000
  prints the stage b file layout and the model-level files for one day, so the reader below can
  be pointed at the right variable and file names. Nothing is computed.
Step 2, only after step 1 looks right (run inside a PBS job, not on the login node):
    python side_tests/t7_model_levels.py run 2000 T --n 100
    python side_tests/t7_model_levels.py run 2000 M --n 100
    python side_tests/t7_model_levels.py report

Method: per event, read t, q (model levels), lnsp and the surface geopotential at the event hour;
half-level pressures p = a + b·sp from the GRIB 'pv' coefficients; geopotential by integrating
the hypsometric equation upward with T_v (the IFS recipe); MSE and h* on full levels; then
interpolate to the 37 pressure levels linearly in ln p and recompute h_LPL, h*500, δ_a with the
pipeline's LPL rule. Unweighted. Needs eccodes (python -c "import eccodes").
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tpl_mll import thermo                          # noqa: E402
from tpl_mll.config import load                     # noqa: E402

RD, RV, G0 = 287.06, 461.5, 9.80665


def inspect(year):
    import xarray as xr
    paths = load("paths")
    work = Path(paths["work"])
    for key in ("annmax_T2m", "annmax_MSE2m"):
        d = work / paths["outputs"][key]
        files = sorted(d.glob(f"*{year}*"))
        print(f"\n{key}: {d}  ({len(files)} files for {year})")
        if files:
            with xr.open_dataset(files[0]) as ds:
                print(ds)
    ml = Path(paths["era5"]["model_levels"])
    month = ml / f"{year}07"
    print(f"\nmodel levels: {month}")
    for p in sorted(month.glob("*"))[:30]:
        print("  ", p.name)
    print("\nNext: set ANNMAX_TIME_VAR and the file patterns in read_column() from the output above.")


# ---- filled in after `inspect` (left unset on purpose so nothing runs on a guess) -----------
ANNMAX_TIME_VAR = None                 # name of the event-hour variable in the stage b files
ML_PATTERN = None                      # e.g. "e5.oper.an.ml.0_5_0_0_0_{var}.*.{yyyymmddhh0}_*.grb"


def half_level_pressure(pv, sp):
    """pv: GRIB pv array (a coefficients then b coefficients, 138 each); sp: Pa → (138,) Pa."""
    n = len(pv) // 2
    return np.asarray(pv[:n]) + np.asarray(pv[n:]) * sp


def column_mse(T, q, ph, zs_geo):
    """Full-level MSE, saturated MSE (kJ/kg) and pressure (hPa) for one column.
    T, q on 137 full levels (top → bottom); ph on 138 half levels (Pa); zs_geo surface Φ (m²/s²)."""
    tv = T * (1 + (RV / RD - 1) * q)
    phi_half = np.empty(len(ph))
    phi_half[-1] = zs_geo
    phi_full = np.empty(len(T))
    for k in range(len(T) - 1, -1, -1):                     # integrate upward from the surface
        if k == 0:
            dlog, alpha = np.log(2.0), np.log(2.0)
        else:
            dlog = np.log(ph[k + 1] / ph[k])
            alpha = 1 - ph[k] / (ph[k + 1] - ph[k]) * dlog
        phi_full[k] = phi_half[k + 1] + RD * tv[k] * alpha
        phi_half[k] = phi_half[k + 1] + RD * tv[k] * dlog
    p_full = 0.5 * (ph[1:] + ph[:-1])
    z_kj = phi_full / 1000.0                                # kJ/kg, as the pipeline stores Z
    h = thermo.mse(T, q, z_kj)
    hs = thermo.mse_sat(T, p_full, z_kj)
    return h, hs, p_full / 100.0


def read_column(year, event, i, j):
    """Model-level t, q, pv, sp, surface Φ for one event (lat index i, lon index j)."""
    if ANNMAX_TIME_VAR is None or ML_PATTERN is None:
        raise SystemExit("Task 7 is parked: run `inspect` first and fill ANNMAX_TIME_VAR / ML_PATTERN.")
    raise NotImplementedError("fill in from the inspect output (GRIB reader via eccodes)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect")
    s.add_argument("year", type=int)
    r = sub.add_parser("run")
    r.add_argument("year", type=int)
    r.add_argument("event", choices=["T", "M"])
    r.add_argument("--n", type=int, default=100)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "inspect":
        inspect(a.year)
    else:
        read_column(getattr(a, "year", 0), getattr(a, "event", "T"), 0, 0)


if __name__ == "__main__":
    main()
