"""Surface jump: how big is h_2m − h(lowest model level) at the extremes, and is it temperature or moisture?

Prompted by Fig. 1 (2026-11-10): at 38.75°N 116.25°E the 1986 T extreme has h_2m 11 kJ/kg above the lowest
model level (~10 m), 90 % of it moisture (q 5.5 → 9.5 g/kg in 9 m), while q is nearly uniform in the
model's lowest 150 m. ERA5 analyses its 2-m fields separately from the upper air (screen-level
analysis), so the 2-m values need not match the model column. This measures how common and how large
that is, because δ_S = h_2m − h_ML rests on the 2-m fields.

It also does parked Task 7 (t7_model_levels.py) for free: h_LPL and h*500 from the model levels against
the pressure-level product, for the same events.

Sample: N_PER_BIN events per 10 % surface-RH bin and per event type, drawn at random over the years and
the analysis mask (land, 50°S–65°N, terrain < 0.5 km), unweighted. Per event:
    pipeline (0.25°, lead 0): h_2m, T2m, q2m, SP, RH_2m, h_LPL and the LPL, p_LCL, h*500 (pressure levels)
    model levels (RDA d633006, nearest N320 point, the bottom NLEV levels): T, q, Φ (IFS half-level
        scheme with T_v), h and h*; the lowest three levels; h_LPL and h*500 interpolated in ln p
Measures: jump = h_2m − h(lowest level) = cp ΔT + Lv Δq + ΔΦ; δ_S two ways (h_2m − h_LPL, the paper's,
and h(lowest level) − h_LPL); a log-law check, the 2 m → z1 change divided by what the model's own
z1 → z3 change implies under a neutral log profile, reported as the part of cp ΔT and Lv Δq beyond it
(0 = consistent with the model's own surface layer; strong daytime instability allows ~1.6 times the
log-law value, so a few tenths of a kJ/kg either way is noise).

    python side_tests/surface_jump.py plan   [--years 1980-2019] [--n-per-bin 30] [--out DIR]   (login node ok)
    python side_tests/surface_jump.py sample --limit 5 --procs 1                                 (timing trial)
    python side_tests/surface_jump.py sample [--procs 8] [--out DIR]                            (PBS job)
    python side_tests/surface_jump.py report [--out DIR]                                         (anywhere)
Writes <out>/surface_jump_plan.csv, <out>/surface_jump.csv (rows appended; reruns skip done rows),
<out>/surface_jump_report.md and <out>/surface_jump.png.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "figures"))

N_PER_BIN = 30
RH_EDGES = np.round(np.arange(0.0, 1.0001, 0.1), 2)
POOL_PER_YEAR = 600                 # random mask cells per year and event type before stratifying
NLEV = 60                           # bottom model levels read (reaches above 400 hPa on L137)
SEED = 20261110
PLAN_COLS = ["id", "event", "year", "lat", "lon", "time", "rh", "h_2m", "T2m", "q2m", "p_sfc", "h_LPL_pl",
             "p_lpl", "p_lcl", "hs500_pl"]
OUT_COLS = PLAN_COLS + ["lat_ml", "lon_ml", "z1", "T1", "q1", "h1", "z2", "T2", "q2", "z3", "T3", "q3",
                        "phi1", "phi_s", "h_LPL_ml", "hs500_ml"]


def default_out():
    from tpl_mll.config import load
    return Path(load("paths")["work"]) / "side_tests" / "surface_jump"


# --- plan: which events ------------------------------------------------------------------------------
def year_pool(year, event, rng, n=POOL_PER_YEAR):
    """Random in-mask cells of one year with their pipeline values (lead 0) and event hour."""
    import xarray as xr
    import fig_plume_case as P
    from tpl_mll.config import load
    from tpl_mll.masks import analysis_mask
    prof, der = P._open(year, event)
    lat, lon = prof["latitude"].values, prof["longitude"].values
    mask = analysis_mask(lat, lon) & np.isfinite(P._grid(der["MSE_LPL"]))
    i, j = np.nonzero(mask)
    pick = rng.choice(i.size, size=min(n, i.size), replace=False)
    i, j = i[pick], j[pick]
    paths = load("paths")
    key, var = ("annmax_T2m", "T2m") if event == "T" else ("annmax_MSE2m", "MSE2m")
    with xr.open_dataset(Path(paths["work"]) / paths["outputs"][key] / f"AnnMax_{var}_{year}.nc") as am:
        et = am["event_time"].sel(latitude=xr.DataArray(lat[i]), longitude=xr.DataArray(lon[j]),
                                  method="nearest").values
    k500 = int(np.argmin(abs(prof["level"].values - 500.0)))
    f = {"rh": der["RH_2m"], "h_2m": prof["MSE2m"], "T2m": prof["T2m"], "q2m": prof["q2m"], "p_sfc": prof["SP"],
         "h_LPL_pl": der["MSE_LPL"], "p_lpl": der["LPL_pressure"], "p_lcl": der["p_LCL_LPL"],
         "hs500_pl": prof["MSE_sat"].isel(level=k500)}
    v = {k: P._grid(da)[i, j].astype(float) for k, da in f.items()}      # each field read once
    v["p_sfc"] = v["p_sfc"] / 100.0
    rows = []
    for n_ in range(i.size):
        r = {"event": event, "year": year, "lat": float(lat[i[n_]]), "lon": float(lon[j[n_]]),
             "time": str(np.datetime64(et[n_], "h"))}
        r.update({k: float(a_[n_]) for k, a_ in v.items()})
        rows.append(r)
    return rows


def stratify(rows, n_per_bin, rng):
    """n_per_bin rows per 10 % RH bin (fewer where the pool has fewer)."""
    rh = np.array([r["rh"] for r in rows])
    out = []
    for lo, hi in zip(RH_EDGES[:-1], RH_EDGES[1:]):
        idx = np.nonzero((rh >= lo) & (rh < hi if hi < 1 else rh <= 1.5))[0]
        out += [rows[k] for k in rng.choice(idx, size=min(n_per_bin, idx.size), replace=False)]
    return out


def plan(years, n_per_bin, out):
    rng = np.random.default_rng(SEED)
    chosen = []
    for e in ("T", "M"):
        pool = []
        for y in range(years[0], years[1] + 1):
            pool += year_pool(y, e, rng)
            print(f"  pool {e} {y}: {len(pool)}", flush=True)
        chosen += stratify(pool, n_per_bin, rng)
    for k, r in enumerate(chosen):
        r["id"] = k
    write_csv(Path(out) / "surface_jump_plan.csv", chosen, PLAN_COLS)
    print(f"Wrote {Path(out) / 'surface_jump_plan.csv'} ({len(chosen)} events)")


# --- sample: the model levels ------------------------------------------------------------------------
def ml_bottom(t, lat, lon, nlev=NLEV, ml_dir=None, invariant=None):
    """Bottom nlev model levels at the N320 point nearest (lat, lon), ordered top → bottom."""
    import xarray as xr
    import fig_example_profiles as E
    from tpl_mll import thermo
    paths, it = E.ml_paths(t, ml_dir or E.ML_DIR)
    sel = dict(latitude=lat, longitude=lon % 360.0, method="nearest")
    with xr.open_dataset(paths["T"]) as dt, xr.open_dataset(paths["Q"]) as dq, \
            xr.open_dataset(paths["SP"]) as ds, xr.open_dataset(invariant or E.invariant_path()) as dz:
        t_ = dt.sel(**sel).isel(time=it)
        L = t_.sizes["level"]
        bottom_last = float(t_["b_model"].values[-1]) > float(t_["b_model"].values[0])
        lev = slice(L - nlev, L) if bottom_last else slice(0, nlev)
        half = slice(L - nlev, L + 1) if bottom_last else slice(0, nlev + 1)
        T = t_["T"].isel(level=lev).values.astype(float)
        q = dq.sel(**sel).isel(time=it)["Q"].isel(level=lev).values.astype(float)
        a, b = t_["a_model"].isel(level=lev).values, t_["b_model"].isel(level=lev).values
        ah, bh = t_["a_half"].isel(half_level=half).values, t_["b_half"].isel(half_level=half).values
        sp = float(ds.sel(**sel).isel(time=it)["SP"])
        zs = dz.sel(**sel)["Z"]
        phi_s = float(zs.isel(time=0) if "time" in zs.dims else zs)
        lat_ml, lon_ml = float(t_["latitude"]), float(t_["longitude"])
    if not bottom_last:
        T, q, a, b, ah, bh = T[::-1], q[::-1], a[::-1], b[::-1], ah[::-1], bh[::-1]
    p, ph = a + b * sp, ah + bh * sp
    phi = E.geopotential(T, q, ph, phi_s)
    h, hs = thermo.mse(T, q, phi / 1000.0), thermo.mse_sat(T, p, phi / 1000.0)
    return dict(p=p / 100.0, T=T, q=q, phi=phi, h=h, hs=hs, phi_s=phi_s, lat_ml=lat_ml, lon_ml=lon_ml)


def interp_lnp(p, y, p0, below_hpa=15.0):
    """y at p0, linear in ln p; an LPL just below the lowest model level takes that level's value."""
    o = np.argsort(p)
    if p0 > p.max():
        return float(y[o][-1]) if p0 - p.max() <= below_hpa else np.nan
    if p0 < p.min():
        return np.nan
    return float(np.interp(np.log(p0), np.log(p[o]), y[o]))


def measure(row, ml_dir=None, invariant=None):
    c = ml_bottom(np.datetime64(row["time"]), float(row["lat"]), float(row["lon"]), ml_dir=ml_dir,
                  invariant=invariant)
    z = (c["phi"] - c["phi_s"]) / 9.80665
    out = dict(row)
    out.update(lat_ml=c["lat_ml"], lon_ml=c["lon_ml"], phi1=c["phi"][-1], phi_s=c["phi_s"],
               h_LPL_ml=interp_lnp(c["p"], c["h"], float(row["p_lpl"])), hs500_ml=interp_lnp(c["p"], c["hs"], 500.0))
    for k, s in ((1, -1), (2, -2), (3, -3)):
        out.update({f"z{k}": z[s], f"T{k}": c["T"][s], f"q{k}": c["q"][s]})
    out["h1"] = c["h"][-1]
    return out


def _measure_safe(args):
    row, ml_dir, invariant = args
    try:
        return measure(row, ml_dir, invariant), None
    except Exception as exc:                                  # noqa: BLE001  report and keep going
        return row, f"{type(exc).__name__}: {exc}"


def sample(out, procs=8, ml_dir=None, invariant=None, limit=None):
    out = Path(out)
    rows = read_csv(out / "surface_jump_plan.csv")
    done_path = out / "surface_jump.csv"
    done = {int(float(r["id"])) for r in read_csv(done_path)} if done_path.exists() else set()
    todo = [r for r in rows if int(float(r["id"])) not in done][:limit]
    print(f"{len(todo)} of {len(rows)} events to read", flush=True)
    new_file = not done_path.exists()
    with open(done_path, "a", newline="") as f:
        w = csv.DictWriter(f, OUT_COLS)
        if new_file:
            w.writeheader()
        jobs = [(r, ml_dir, invariant) for r in todo]
        if procs > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(procs) as ex:
                results = ex.map(_measure_safe, jobs, chunksize=4)
                for k, (res, err) in enumerate(results):
                    _write(w, f, res, err, k, len(jobs))
        else:
            for k, job in enumerate(jobs):
                _write(w, f, *_measure_safe(job), k, len(jobs))


def _write(w, f, res, err, k, n):
    if err:
        print(f"  FAILED id {res['id']} ({res['event']} {res['time']}): {err}", flush=True)
        return
    w.writerow({c: res.get(c, "") for c in OUT_COLS})
    f.flush()
    if k % 25 == 0:
        print(f"  {k + 1}/{n}", flush=True)


# --- report ------------------------------------------------------------------------------------------
def derived(rows):
    """Arrays of everything the report uses."""
    from tpl_mll import thermo
    f = lambda k: np.array([float(r[k]) for r in rows])
    d = {k: f(k) for k in ("rh", "h_2m", "h1", "T2m", "T1", "q2m", "q1", "T3", "q3", "z1", "z3", "phi1", "phi_s",
                           "h_LPL_pl", "h_LPL_ml", "hs500_pl", "hs500_ml")}
    d["event"] = np.array([r["event"] for r in rows])
    d["jump"] = d["h_2m"] - d["h1"]
    d["jump_T"] = thermo.cp * (d["T2m"] - d["T1"])
    d["jump_q"] = thermo.Lv * (d["q2m"] - d["q1"])
    d["jump_phi"] = (d["phi_s"] + 2.0 * 9.80665 - d["phi1"]) / 1000.0
    d["dS_paper"] = d["h_2m"] - d["h_LPL_pl"]
    d["dS_ml"] = d["h1"] - d["h_LPL_pl"]
    law = np.log(d["z1"] / 2.0) / np.log(d["z3"] / d["z1"])           # neutral log profile
    d["beyond_q"] = thermo.Lv * ((d["q2m"] - d["q1"]) - (d["q1"] - d["q3"]) * law)
    d["beyond_T"] = thermo.cp * ((d["T2m"] - d["T1"]) - (d["T1"] - d["T3"]) * law)
    return d


def q3(x):
    x = x[np.isfinite(x)]
    return np.percentile(x, [25, 50, 75]) if x.size else np.full(3, np.nan)


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import style
    out = Path(out)
    d = derived(read_csv(out / "surface_jump.csv"))
    cen = 0.5 * (RH_EDGES[1:] + RH_EDGES[:-1])
    lines = ["# Surface jump: h_2m − h(lowest model level)", "",
             "Unweighted, stratified by 10 % RH bins. Values are median [25th, 75th percentile], kJ/kg unless noted.", ""]
    style.apply()
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for e in ("T", "M"):
        m = d["event"] == e
        col = style.EVENT_COLOR[e]
        lines += [f"## {style.EVENT_SHORT[e].replace('$', '')} extremes ({m.sum()} events)", "",
                  "| RH | n | jump | cp ΔT | Lv Δq | δ_S paper | δ_S from lowest level | Lv Δq beyond log law | cp ΔT beyond log law |",
                  "|---|---|---|---|---|---|---|---|---|"]
        med = {k: [] for k in ("jump", "jump_T", "jump_q", "dS_paper", "dS_ml", "beyond_q")}
        for lo, hi in zip(RH_EDGES[:-1], RH_EDGES[1:]):
            b = m & (d["rh"] >= lo) & (d["rh"] < (hi if hi < 1 else 9))
            cells = []
            for k in ("jump", "jump_T", "jump_q", "dS_paper", "dS_ml", "beyond_q", "beyond_T"):
                p = q3(d[k][b])
                if k in med:
                    med[k].append(p)
                cells.append(f"{p[1]:.1f} [{p[0]:.1f}, {p[2]:.1f}]")
            lines.append(f"| {100 * lo:.0f}–{100 * hi:.0f}% | {b.sum()} | " + " | ".join(cells) + " |")
        P = {k: np.array(v) for k, v in med.items()}
        axs[0].plot(cen, P["jump"][:, 1], color=col, lw=2, marker="o", ms=3.5, label=f"{style.EVENT_SHORT[e]}: total")
        axs[0].fill_between(cen, P["jump"][:, 0], P["jump"][:, 2], color=col, alpha=0.15, lw=0)
        axs[0].plot(cen, P["jump_q"][:, 1], color=col, lw=1.2, marker="^", ms=3, label=r"$L_v\Delta q$ part")
        axs[0].plot(cen, P["jump_T"][:, 1], color=col, lw=1.2, marker="s", ms=3, alpha=0.7, label=r"$c_p\Delta T$ part")
        axs[1].plot(cen, P["dS_paper"][:, 1], color=col, lw=2, marker="o", ms=3.5, label=f"{style.EVENT_SHORT[e]}: $h_{{2m}}-h_{{ML}}$")
        axs[1].plot(cen, P["dS_ml"][:, 1], color=col, lw=1.2, marker="v", ms=3, label="lowest level $- h_{ML}$")
        axs[2].plot(cen, P["beyond_q"][:, 1], color=col, lw=2, marker="o", ms=3.5, label=style.EVENT_SHORT[e])
        axs[2].fill_between(cen, P["beyond_q"][:, 0], P["beyond_q"][:, 2], color=col, alpha=0.15, lw=0)
        lines.append("")
    axs[2].axhline(0.0, color="0.5", lw=0.8)
    titles = ("2-m minus lowest model level", r"$\delta_S$ two ways", r"2-m $L_v\Delta q$ beyond the log law")
    ylab = ("jump " + style.KJ, r"$\delta_S$ " + style.KJ, style.KJ)
    for k, ax in enumerate(axs):
        style.paper_axes(ax)
        style.rh_percent(ax)
        ax.set(title=titles[k], ylabel=ylab[k])
        ax.legend(fontsize=6.5, frameon=False)
        style.panel_letter(ax, k)
    fig.tight_layout()
    fig.savefig(out / "surface_jump.png", dpi=200)
    # parked Task 7 numbers
    lines += ["## Task 7: model levels vs pressure levels (same events)", ""]
    for name, a, b in (("h_LPL", "h_LPL_ml", "h_LPL_pl"), ("h*500", "hs500_ml", "hs500_pl")):
        for e in ("T", "M"):
            m = d["event"] == e
            p = q3(d[a][m] - d[b][m])
            lines.append(f"- {name} model − pressure levels, {e}: {p[1]:+.2f} [{p[0]:+.2f}, {p[2]:+.2f}]")
    (out / "surface_jump_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out / 'surface_jump_report.md'} and {out / 'surface_jump.png'}")


# --- csv ---------------------------------------------------------------------------------------------
def write_csv(path, rows, cols):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["plan", "sample", "report"])
    ap.add_argument("--years", default="1980-2019")
    ap.add_argument("--n-per-bin", type=int, default=N_PER_BIN)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, help="read only this many events (a timing trial; reruns resume)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else default_out()
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "plan":
        y0, y1 = (int(v) for v in a.years.split("-"))
        plan((y0, y1), a.n_per_bin, out)
    elif a.step == "sample":
        sample(out, a.procs, limit=a.limit)
    else:
        report(out)


if __name__ == "__main__":
    main()
