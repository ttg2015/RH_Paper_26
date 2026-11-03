"""Checks asked for by the 2026-11-02 review of the draft. Read-only on the pipeline.

Part A, sensitivity of the Section 4 and 6-7 numbers (figure mask: band + land, all terrain).
  Variants of the mixed-layer energy and the free-tropospheric reference:
    lpl   h_ML = MSE at the lowest pressure level (the paper's choice)
    avg   h_ML = pressure-weighted mean MSE from the surface (2 m value) up to the LCL of LPL air
    h*_a at 400, 500 or 600 hPa; "low" keeps only grid points below 0.5 km
  For each: cos(lat)-weighted means and shares of δ, δ_S, δ_a (as Section 4) and unweighted
  RH-binned sums of δ_a (RH0 and the Section 7 fit). Also the height of the LPL above ground.
Part B, null test for the plume pass check (stage f analysis mask: low terrain).
  Each event's environment is paired with h_ML from (cell) a random event in the same 2 kJ/kg
  cell of (h500, h*500), and (global) a random event anywhere. Category counts per ε for the real
  pairing and both nulls, as in stage f. If the pass share barely drops, the pass test says little.

    python side_tests/review_tests.py run YEAR EVENT [--out DIR] [--no-null]
    python side_tests/review_tests.py report [--out DIR]

Reads   Profiles/profiles_*, Profiles/derived/derived_*. Writes <out>/review_{E}_{Y}.npz, review_report.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "pipeline")]
from tpl_mll.config import load                               # noqa: E402
from tpl_mll import ztop                                      # noqa: E402
from tpl_mll.masks import analysis_mask, surface_height_km    # noqa: E402
import g                                                      # noqa: E402

BUFFER = 6
RH_EDGES = np.round(np.arange(0.0, 1.0501, 0.01), 6)
Z0_EDGES = np.arange(0.0, 1010.0, 10.0)                        # m
VARIANTS = [("lpl", 500, "all"), ("lpl", 500, "low"), ("lpl", 400, "all"), ("lpl", 600, "all"),
            ("avg", 500, "all"), ("avg", 500, "low")]
STATS = ["w", "d", "dS", "da", "d>0", "dS>0", "da>0", "da<0"]
NULLS = ["real", "cell", "global"]
CELL = 2.0                                                     # kJ/kg, null-test cell in (h500, h*500)
ZS_LOW = 0.5                                                   # km
WINDOW_FIT = (0.30, 0.60)
NCODE = len(ztop.CATEGORIES)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def default_out():
    return Path(load("paths")["work"]) / "side_tests" / "review"


def name(v):
    return f"{v[0]}{v[1]}_{v[2]}"


# ---------------------------------------------------------------- Part A
def surface_to_lcl_mean(levels, mse, sp, h2m, plcl, n_fine=40):
    """Pressure-weighted mean MSE from the surface (h2m at sp) up to plcl, one event per row.
    mse (n, L) on ascending levels with NaN below ground. Falls back to h2m when the LCL is at
    or below the surface."""
    out = np.array(h2m, dtype=float)
    for k in range(len(sp)):
        if not (np.isfinite(plcl[k]) and np.isfinite(sp[k]) and sp[k] - plcl[k] > 1.0):
            continue
        ok = np.isfinite(mse[k]) & (levels < sp[k])
        p = np.append(levels[ok], sp[k])                       # ascending, surface last
        h = np.append(mse[k][ok], h2m[k])
        pp = np.linspace(plcl[k], sp[k], n_fine)
        hh = np.interp(pp, p, h)
        out[k] = _trapz(hh, pp) / (sp[k] - plcl[k])
    return out


def part_a(year, event):
    cfg = load("figdata")
    f, lat, lon, levels, profiles = g.fields(year, event, BUFFER, cfg["lead"])
    mask = g.figure_mask(lat, lon, cfg) & np.isfinite(f["h_LPL"]) & np.isfinite(f["h_2m"]) \
        & np.isfinite(f["RH_2m"])
    zs = surface_height_km(lat, lon)
    weight = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)
    i, j = np.nonzero(mask)
    with xr.open_dataset(Path(load("paths")["work"]) / load("paths")["outputs"]["profiles"]
                         / f"profiles_{event}_b{BUFFER:02d}_{year}.nc") as p:
        p = p["MSE_sat"].sel(lead=cfg["lead"]).isel(year=0)
        hs = {lev: p.sel(level=lev).transpose("latitude", "longitude").values.astype(float)[i, j]
              for lev in (400, 500, 600)}
    with xr.open_dataset(Path(load("paths")["work"]) / load("paths")["outputs"]["derived"]
                         / f"derived_{event}_b{BUFFER:02d}_{year}.nc") as d:
        zlpl = d["Z_LPL"].sel(lead=cfg["lead"]).isel(year=0).transpose("latitude", "longitude").values[i, j]
    h2m, rh, w, low = f["h_2m"][i, j], f["RH_2m"][i, j], weight[i, j], zs[i, j] < ZS_LOW
    hml = {"lpl": f["h_LPL"][i, j],
           "avg": surface_to_lcl_mean(levels, profiles["MSE"][i, j], f["SP_hPa"][i, j], h2m, f["p_LCL_LPL"][i, j])}
    z0 = zlpl / 9.8 * 1000.0 - zs[i, j] * 1000.0               # LPL height above ground, m
    out = {"z0_all": np.histogram(z0, Z0_EDGES)[0], "z0_low": np.histogram(z0[low], Z0_EDGES)[0]}
    k = np.digitize(rh, RH_EDGES) - 1
    for v in VARIANTS:
        hm, ha = hml[v[0]], hs[v[1]]
        sel = np.isfinite(hm) & np.isfinite(ha) & (low if v[2] == "low" else True)
        d, dS, da = h2m - ha, h2m - hm, hm - ha
        ww = w * sel
        out[f"W_{name(v)}"] = np.array([np.nansum(ww * x) for x in
                                        (np.ones_like(d), d, dS, da, d > 0, dS > 0, da > 0, da < 0)])
        use = sel & (k >= 0) & (k < len(RH_EDGES) - 1)
        out[f"B_{name(v)}"] = np.stack([np.bincount(k[use], minlength=len(RH_EDGES) - 1),
                                        np.bincount(k[use], weights=da[use], minlength=len(RH_EDGES) - 1),
                                        np.bincount(k[use], weights=dS[use], minlength=len(RH_EDGES) - 1)], 1)
    print(f"  part A: {len(i)} events", flush=True)
    return out


# ---------------------------------------------------------------- Part B
def donors(cell, rng):
    """Index of a random event in the same cell for each event (may be itself in small cells)."""
    out = np.arange(len(cell))
    order = np.argsort(cell, kind="stable")
    bounds = np.flatnonzero(np.diff(cell[order])) + 1
    for grp in np.split(order, bounds):
        out[grp] = rng.permutation(grp)
    return out


def part_b(year, event, chunk=20000):
    cfg = load("analysis")
    eps_list = np.array([0.0] + list(cfg["epsilon"]["sweep_km"]), dtype=float)
    window = tuple(cfg["ztop_window_hPa"])
    paths = load("paths")
    work = Path(paths["work"])
    with xr.open_dataset(work / paths["outputs"]["profiles"] / f"profiles_{event}_b{BUFFER:02d}_{year}.nc") as p, \
            xr.open_dataset(work / paths["outputs"]["derived"] / f"derived_{event}_b{BUFFER:02d}_{year}.nc") as d:
        p, d = p.sel(lead=0).isel(year=0), d.sel(lead=0).isel(year=0)
        lat, lon = p["latitude"].values, p["longitude"].values
        mask = analysis_mask(lat, lon) & np.isfinite(d["MSE_LPL"].values)
        i, j = np.nonzero(mask)
        take = lambda da: da.transpose("latitude", "longitude", ...).values[i, j].astype(float)
        levels = p["level"].values.astype(float)
        h, hs, phi = take(p["MSE"]), take(p["MSE_sat"]), take(p["Z"])
        h0, p0 = take(d["MSE_LPL"]), take(d["LPL_pressure"])
    k500 = int(np.argmin(abs(levels - 500)))
    cx, cy = np.floor(h[:, k500] / CELL), np.floor(hs[:, k500] / CELL)
    cell = np.nan_to_num((cx - np.nanmin(cx)) * 10000 + (cy - np.nanmin(cy)), nan=-1.0)
    rng = np.random.default_rng(year * 10 + (event == "M"))
    pick = {"real": np.arange(len(i)), "cell": donors(cell, rng), "global": rng.permutation(len(i))}
    out = {}
    for nm, src in pick.items():
        counts = np.zeros((len(eps_list), NCODE))
        for s in range(0, len(i), chunk):
            sl = slice(s, min(s + chunk, len(i)))
            cols = ztop.Columns(levels, h[sl], hs[sl], phi[sl], h0[src[sl]], p0[sl])
            buoyant0 = cols.top(0.0)[4]
            for e, eps in enumerate(eps_list):
                _, _, p_high, _, buoyant, at_top = cols.top(eps)
                cat = ztop.categorise(p_high, buoyant, at_top, window, buoyant0)
                counts[e] += np.bincount(cat, minlength=NCODE)[:NCODE]
        out[f"N_{nm}"] = counts
        out[f"self_{nm}"] = np.array([np.mean(src == np.arange(len(i)))])
    out["eps"] = eps_list
    print(f"  part B: {len(i)} events", flush=True)
    return out


def run(year, event, out_dir, null=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{year} {event}", flush=True)
    res = part_a(year, event)
    if null:
        res.update(part_b(year, event))
    np.savez(out_dir / f"review_{event}_{year}.npz", **res)


# ---------------------------------------------------------------- report
def zero_crossing(x, m):
    s = np.flatnonzero(np.isfinite(m[:-1]) & np.isfinite(m[1:]) & (m[:-1] < 0) & (m[1:] >= 0))
    if not len(s):
        return np.nan
    a = s[0]
    return x[a] - m[a] * (x[a + 1] - x[a]) / (m[a + 1] - m[a])


def binned(B, coarsen=2, min_events=20):
    nb = (B.shape[0] // coarsen) * coarsen
    C = B[:nb].reshape(-1, coarsen, B.shape[1]).sum(1)
    e = RH_EDGES[::coarsen][: C.shape[0] + 1]
    x = 0.5 * (e[1:] + e[:-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = C[:, 1] / C[:, 0]
    m[C[:, 0] < min_events] = np.nan
    return x, m, C[:, 0], C[:, 2] / np.where(C[:, 0] > 0, C[:, 0], np.nan)


def fit_k(x, m, n, rh0):
    """K in δ_a = K ln(RH/RH0) over the fit window (positive convention, as the draft)."""
    ok = np.isfinite(m) & (x >= WINDOW_FIT[0]) & (x <= WINDOW_FIT[1])
    if ok.sum() < 3 or not np.isfinite(rh0):
        return np.nan
    gg = np.log(x[ok] / rh0)
    return (n[ok] * gg * m[ok]).sum() / (n[ok] * gg * gg).sum()


def pct(h, edges, q):
    c = np.cumsum(h) / max(h.sum(), 1)
    return float(np.interp(q / 100.0, c, edges[1:]))


def report(out_dir):
    files = sorted(out_dir.glob("review_*_*.npz"))
    acc = {}
    for path in files:
        ev = path.stem.split("_")[1]
        with np.load(path) as z:
            for k in z.files:
                if k == "eps":
                    acc["eps"] = z[k]
                    continue
                acc.setdefault(ev, {}).setdefault(k, 0)
                acc[ev][k] = acc[ev][k] + z[k]
    evs = [e for e in ("T", "M") if e in acc]
    years = sorted({p.stem.split("_")[2] for p in files})
    L = [f"# Review checks · {len(years)} years ({', '.join(years)})", "",
         "Part A: figure mask (band + land); means and shares cos(lat) weighted as in Section 4; "
         "RH0 and K unweighted, pooled T + M, RH bins 0.02, K for δ_a = K ln(RH/RH0) over "
         f"{WINDOW_FIT[0]:.2f}–{WINDOW_FIT[1]:.2f}.", ""]

    L += ["## A1 · Height of the lowest pressure level above ground (z0)", "",
          "| events | terrain | median | p5 | share < 30 m | share < 50 m | share < 100 m |", "|---|---|---|---|---|---|---|"]
    for ev in evs:
        for t in ("all", "low"):
            hz = acc[ev][f"z0_{t}"]
            share = lambda zc: 100 * hz[: int(zc / 10)].sum() / max(hz.sum(), 1)
            L.append(f"| {ev} | {t} | {pct(hz, Z0_EDGES, 50):.0f} m | {pct(hz, Z0_EDGES, 5):.0f} m | "
                     f"{share(30):.1f}% | {share(50):.1f}% | {share(100):.1f}% |")
    L += ["", "Below about 30–50 m the LPL is likely still in the surface layer.", ""]

    L += ["## A2 · Section 4 means and shares by variant (cos(lat) weighted)", "",
          "| variant | events | mean δ | mean δ_S | mean δ_a | δ > 0 | δ_S > 0 | δ_a > 0 | δ_a < 0 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for ev in evs:
            W = acc[ev][f"W_{name(v)}"]
            L.append(f"| {name(v)} | {ev} | {W[1] / W[0]:.2f} | {W[2] / W[0]:.2f} | {W[3] / W[0]:.2f} | "
                     f"{100 * W[4] / W[0]:.0f}% | {100 * W[5] / W[0]:.0f}% | {100 * W[6] / W[0]:.0f}% | "
                     f"{100 * W[7] / W[0]:.0f}% |")
    L += ["", "lpl500_all is the paper's Section 4 (draft: δ 3.9 / 16.7, δ_S 5.8 / 10.2, δ_a −2.0 / +6.5).", ""]

    L += ["## A3 · RH0 and the Section 7 slope by variant (unweighted)", "",
          "| variant | RH0 (pooled) | K pooled | K T | K M | mean δ_S at RH0 (pooled) |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        B = {ev: acc[ev][f"B_{name(v)}"] for ev in evs}
        x, m, n, mS = binned(sum(B.values()))
        r0 = zero_crossing(x, m)
        ks = []
        for ev in evs:
            xe, me, ne, _ = binned(B[ev])
            ks.append(fit_k(xe, me, ne, r0))
        dS0 = np.interp(r0, x[np.isfinite(mS)], mS[np.isfinite(mS)]) if np.isfinite(r0) else np.nan
        L.append(f"| {name(v)} | {r0:.3f} | {fit_k(x, m, n, r0):.2f} | " + " | ".join(f"{k:.2f}" for k in ks)
                 + f" | {dS0:.2f} |")
    L += ["", "lpl500_all should reproduce RH0 = 0.316 and K = 10.5 (small differences: RH bins 0.02 here).", ""]

    if "N_real" in acc.get(evs[0], {}):
        eps = acc["eps"]
        L += ["## B · Null test for the plume pass check (low terrain, unweighted)", "",
              f"cell: h_ML from a random event in the same {CELL:g} kJ/kg cell of (h500, h*500); "
              "global: from a random event anywhere. Shares of unstable events (buoyant at ε = 0 "
              "for that pairing), as in Figure ztop_fractions.", ""]
        for ev in evs:
            L += [f"### {ev}", "", "| ε | real: pass | cell: pass | global: pass | real: diluted | cell: diluted | "
                  "global: diluted | unstable share real / cell / global |", "|---|---|---|---|---|---|---|---|"]
            sh = {}
            for nm in NULLS:
                N = acc[ev][f"N_{nm}"]
                unst = N[:, [1, 2, 3, 5]].sum(1)
                sh[nm] = (N[:, 1] / unst, N[:, 5] / unst, unst / N[:, 1:].sum(1))
            for e in range(1, len(eps)):
                L.append(f"| {eps[e]:.2f} | " + " | ".join(f"{100 * sh[nm][0][e]:.0f}%" for nm in NULLS) + " | "
                         + " | ".join(f"{100 * sh[nm][1][e]:.0f}%" for nm in NULLS) + " | "
                         + " / ".join(f"{100 * sh[nm][2][0]:.0f}%" for nm in NULLS) + " |")
            n_null = sum(1 for p in files if p.stem.split("_")[1] == ev and "N_real" in np.load(p).files)
            selfp = acc[ev]["self_cell"][0] / max(n_null, 1)
            L += ["", f"Self-pairs in the cell null: {100 * selfp:.1f}% (events alone in their cell keep their own h_ML).", ""]
    text = "\n".join(L) + "\n"
    (out_dir / "review_report.md").write_text(text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("year", type=int)
    r.add_argument("event", choices=["T", "M"])
    r.add_argument("--out", type=Path)
    r.add_argument("--no-null", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    out = a.out or default_out()
    if a.cmd == "run":
        run(a.year, a.event, out, null=not a.no_null)
    else:
        report(out)


if __name__ == "__main__":
    main()
