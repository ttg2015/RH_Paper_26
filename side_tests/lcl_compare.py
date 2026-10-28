"""Task 1. LCL of 2 m air vs LCL of LPL air: how different are they, and does the choice move the paper's numbers?

Read-only on the pipeline. It imports pipeline/g.py (fields, figure_mask, convective) and the
stage d files, and writes small additive .npz summaries under --out. Nothing in the repo changes.

  python lcl_compare.py run YEAR EVENT [--repo PATH] [--out DIR] [--buffer 6]
  python lcl_compare.py report DIR          # combine every lcl_{E}_{YEAR}.npz, print + write lcl_report.md

All statistics are unweighted (conditional samples, not global area statistics).
Tests, per event type:
  1 direct difference  dz = z_LCL,LPL - z_LCL,2m (both above ground), also the difference of the
                       LCL heights above each parcel's own level; by RH and by LPL height
  2 consistency        LCL below its own start level, LPL below ground, p and z ordering disagree
  3 split sensitivity  g.convective() with p_LCL of LPL air (current) vs p_LCL of 2 m air, same events,
                       including the events the `plcl <= p0` guard drops
  4 model check        z_LCL against ln(RH_2m) for both LCLs (the section-7 fit is in 2 m RH)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

G = 9.8
DZ_EDGES = np.arange(-1000.0, 1000.1, 10.0)                  # m, 10 m bins, ends absorb the tails
DB_EDGES = np.arange(-10.0, 10.01, 0.1)                       # kJ/kg change in delta_below
RH_EDGES = np.linspace(0.0, 1.0, 11)                          # 0.1 wide
RH_FINE = np.linspace(0.0, 1.0, 21)                           # 0.05 wide, for the fit
LPL_EDGES = np.array([0.0, 50, 100, 150, 200, 300, 400, 600, np.inf])      # LPL height above ground, m
ZL_EDGES = np.array([0.0, 500, 1000, 1500, 2000, 2500, 3000, 4000, np.inf])  # z_LCL,LPL, m
P_TOL = 0.5          # hPa
ORDER_MIN_DZ = 20.0  # m, below this the p and z orderings are not required to agree


def _idx(x, edges):
    """Bin index with the last edge open; -1 outside."""
    i = np.digitize(x, edges) - 1
    return np.where((i < 0) | (i >= len(edges) - 1), -1, i)


def _hist(x, edges):
    return np.histogram(np.clip(x, edges[0] + 1e-6, edges[-1] - 1e-6), edges)[0].astype("float64")


def _rows(x, key, key_edges, edges):
    """Histogram of x for each bin of `key`: (n_key_bins, n_x_bins)."""
    k = _idx(key, key_edges)
    return np.array([_hist(x[k == b], edges) for b in range(len(key_edges) - 1)])


def _binsum(key, key_edges, *vals):
    """Per key bin: count and the sum of each val."""
    k = _idx(key, key_edges)
    nb = len(key_edges) - 1
    ok = k >= 0
    out = [np.bincount(k[ok], minlength=nb).astype("float64")]
    for v in vals:
        out.append(np.bincount(k[ok], weights=v[ok], minlength=nb))
    return out


def summarise(f, x, mask, levels, raw, eps, convective):
    """Additive summary arrays for one year x event.

    f          g.fields() dict: delta_FT, zLCL_2m, zLCL_LPL, RH_2m, p_LCL_LPL, p_LPL, h_LPL, SP_hPa
    x          extras: p_LCL_2m, RH_LPL, lpl_agl (m)
    convective g.convective, called once per LCL choice; both calls see the same events
    """
    out = {}
    z2, zL, lpl = f["zLCL_2m"], f["zLCL_LPL"], x["lpl_agl"]
    ok = mask & np.isfinite(z2) & np.isfinite(zL) & np.isfinite(f["RH_2m"])
    dz = (zL - z2)[ok]
    own = dz - lpl[ok]                                          # LCL above own level: LPL minus 2 m
    out["n_valid"] = np.array([ok.sum()], float)
    out["h_dz"] = _hist(dz, DZ_EDGES)
    out["h_own"] = _hist(own, DZ_EDGES)
    out["h_dz_rh"] = _rows(dz, f["RH_2m"][ok], RH_EDGES, DZ_EDGES)
    out["h_dz_lpl"] = _rows(dz, lpl[ok], LPL_EDGES, DZ_EDGES)
    out["h_own_rh"] = _rows(own, f["RH_2m"][ok], RH_EDGES, DZ_EDGES)

    # 2 consistency
    p2, pL = x["p_LCL_2m"][ok], f["p_LCL_LPL"][ok]
    cons = {
        "lcl_below_own_start": np.sum(zL[ok] < lpl[ok] - 1.0),
        "lpl_above_600m": np.sum(lpl[ok] > 600.0),
        "lpl_below_ground": np.sum(f["p_LPL"][ok] > f["SP_hPa"][ok] + P_TOL),
        "p_lcl_2m_below_ground": np.sum(p2 > f["SP_hPa"][ok] + P_TOL),
        "p_lcl_lpl_below_lpl": np.sum(pL > f["p_LPL"][ok] + P_TOL),
        "p_z_order_disagree": np.sum(((dz > ORDER_MIN_DZ) & (pL > p2 + P_TOL))
                                     | ((dz < -ORDER_MIN_DZ) & (pL < p2 - P_TOL))),
        "rh_lpl_gt_rh_2m": np.sum(x["RH_LPL"][ok] > f["RH_2m"][ok]),
        "nonfinite_p_lcl": np.sum(~np.isfinite(p2) | ~np.isfinite(pL)),
    }
    for k, v in cons.items():
        out[f"c_{k}"] = np.array([v], float)

    # 3 split sensitivity: unstable (delta_FT > 0) events, plume from the LPL, same eps
    unst = mask & (f["delta_FT"] > 0) & np.isfinite(f["RH_2m"]) & np.isfinite(zL)
    dB_L, _ = convective(f, levels, raw, mask, eps)
    f2 = dict(f)
    f2["p_LCL_LPL"] = x["p_LCL_2m"]                                 # the only key convective() reads for the LCL
    dB_2, _ = convective(f2, levels, raw, mask, eps)
    dFT = f["delta_FT"]
    vL, v2 = np.isfinite(dB_L) & unst, np.isfinite(dB_2) & unst
    both = vL & v2
    out["s_unstable"] = np.array([unst.sum()], float)
    out["s_valid_lpl"] = np.array([vL.sum()], float)
    out["s_valid_2m"] = np.array([v2.sum()], float)
    out["s_both"] = np.array([both.sum()], float)
    out["s_drop_lpl_only"] = np.array([(vL & ~v2).sum()], float)   # kept with LPL LCL, dropped with 2 m LCL
    out["s_drop_2m_only"] = np.array([(v2 & ~vL).sum()], float)
    d = (dB_2 - dB_L)[both]
    out["h_dB_diff"] = _hist(d, DB_EDGES)
    out["s_sumabs_diff"] = np.array([np.abs(d).sum()], float)
    for name, key, edges in (("rh", f["RH_2m"], RH_EDGES), ("zl", zL, ZL_EDGES)):
        n, sL, sA = _binsum(key[both], edges, dB_L[both], (dFT - dB_L)[both])
        _, s2, s2A = _binsum(key[both], edges, dB_2[both], (dFT - dB_2)[both])
        _, sT = _binsum(key[both], edges, dFT[both])
        out[f"b_{name}_n"], out[f"b_{name}_dB_lpl"], out[f"b_{name}_dB_2m"] = n, sL, s2
        out[f"b_{name}_dA_lpl"], out[f"b_{name}_dA_2m"], out[f"b_{name}_dFT"] = sA, s2A, sT

    # 4 model check: z_LCL against ln RH, bins of 0.05 in the RH that goes with each parcel
    n2, s2z, q2z = _binsum(f["RH_2m"][ok], RH_FINE, z2[ok], z2[ok] ** 2)
    nL, sLz, qLz = _binsum(x["RH_LPL"][ok], RH_FINE, zL[ok], zL[ok] ** 2)
    nX, sXz, qXz = _binsum(f["RH_2m"][ok], RH_FINE, zL[ok], zL[ok] ** 2)       # z_LCL,LPL against RH_2m
    for k, v in dict(fit_n_2m=n2, fit_z_2m=s2z, fit_zz_2m=q2z, fit_n_lpl=nL, fit_z_lpl=sLz, fit_zz_lpl=qLz,
                     fit_n_x=nX, fit_z_x=sXz, fit_zz_x=qXz).items():
        out[k] = v
    return out


# ---------------------------------------------------------------------------------------------
def _load_g(repo):
    repo = Path(repo).resolve()
    for sub in ("src", "pipeline"):
        sys.path.insert(0, str(repo / sub))
    import g                                               # noqa: E402  pipeline/g.py
    return g


def run(year, event, repo, out_dir, buffer=6):
    import xarray as xr
    g = _load_g(repo)
    cfg = g.load("figdata")
    f, lat, lon, levels, profiles = g.fields(year, event, buffer, cfg["lead"])
    mask = g.figure_mask(lat, lon, cfg)
    paths = g.load("paths")
    src = Path(paths["work"]) / paths["outputs"]["derived"] / f"derived_{event}_b{buffer:02d}_{year}.nc"
    need = ["p_LCL_2m", "RH_LPL", "Z_LPL"]
    with xr.open_dataset(src) as d:
        missing = [v for v in need if v not in d]
        if missing:
            raise SystemExit(f"{src.name} lacks {missing}: rerun stage d (it writes them)")
        der = d[need].sel(lead=cfg["lead"]).isel(year=0).load()
    grid = lambda da: da.transpose("latitude", "longitude").values.astype("float64")
    zs_m = g.surface_height_km(lat, lon) * 1000.0
    x = {"p_LCL_2m": grid(der["p_LCL_2m"]), "RH_LPL": grid(der["RH_LPL"]),
         "lpl_agl": np.maximum(grid(der["Z_LPL"]) * 1000.0 / G - zs_m, 0.0)}   # Z_LPL is kJ/kg, as in d.py
    res = summarise(f, x, mask, levels, profiles["raw"], cfg["convective"]["eps"], g.convective)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lcl_{event}_{year}.npz"
    np.savez_compressed(path, **res)
    print(f"{path}  n_valid={int(res['n_valid'][0]):,}  unstable={int(res['s_unstable'][0]):,}  "
          f"dropped by 2 m LCL guard={int(res['s_drop_lpl_only'][0]):,}", flush=True)


# ---------------------------------------------------------------------------------------------
def pct(hist, edges, q):
    """Percentile (0-100) of a histogram, linear within the bin."""
    c = np.cumsum(hist)
    if c[-1] == 0:
        return np.nan
    t = q / 100.0 * c[-1]
    i = int(np.searchsorted(c, t))
    lo = c[i - 1] if i else 0.0
    return edges[i] + (t - lo) / max(hist[i], 1e-12) * (edges[i + 1] - edges[i])


def share_abs_gt(hist, edges, thr):
    centres = 0.5 * (edges[1:] + edges[:-1])
    return hist[np.abs(centres) > thr].sum() / max(hist.sum(), 1.0)


def fit(n, s, ss, thr_n=100, window=(0.05, 0.6)):
    """z = a - C ln(RH) on bin means. Returns C [km], R^2, bins used."""
    c = 0.5 * (RH_FINE[1:] + RH_FINE[:-1])
    use = (n >= thr_n) & (c >= window[0]) & (c <= window[1])
    if use.sum() < 3:
        return np.nan, np.nan, int(use.sum())
    xx, yy = np.log(c[use]), s[use] / n[use]
    b, a = np.polyfit(xx, yy, 1)
    r2 = 1 - np.sum((yy - (a + b * xx)) ** 2) / np.sum((yy - yy.mean()) ** 2)
    return -b / 1000.0, r2, int(use.sum())


def report(in_dir, quiet=False):
    in_dir = Path(in_dir)
    files = sorted(in_dir.glob("lcl_*_*.npz"))
    if not files:
        raise SystemExit(f"no lcl_*.npz in {in_dir}")
    groups = {}
    for p in files:
        _, ev, yr = p.stem.split("_")
        z = np.load(p)
        tot = groups.setdefault(ev, {"years": []})
        tot["years"].append(int(yr))
        for k in z.files:
            tot[k] = tot.get(k, 0) + z[k]
    groups["T+M"] = {"years": []}
    for ev in ("T", "M"):
        for k, v in groups.get(ev, {}).items():
            if k != "years":
                groups["T+M"][k] = groups["T+M"].get(k, 0) + v
    L = ["# LCL of LPL air vs 2 m air: report", "", f"Files: {len(files)}. Unweighted. "
         "Years: " + ", ".join(f"{e}: {sorted(g['years'])}" for e, g in groups.items() if g['years']), ""]
    verdict = {}
    for ev, t in groups.items():
        n = int(t["n_valid"][0])
        L += [f"## {ev}  (n = {n:,} events)", ""]
        L += ["### 1 Difference dz = z_LCL,LPL - z_LCL,2m [m, above ground]", "",
              "| quantity | p5 | p25 | median | p75 | p95 | share \\|dz\\|>50 m | share \\|dz\\|>100 m |", "|---|---|---|---|---|---|---|---|"]
        for lab, h in (("dz (heights above ground)", t["h_dz"]), ("dz above each parcel's own level", t["h_own"])):
            L.append(f"| {lab} | " + " | ".join(f"{pct(h, DZ_EDGES, q):.0f}" for q in (5, 25, 50, 75, 95))
                     + f" | {100 * share_abs_gt(h, DZ_EDGES, 50):.1f}% | {100 * share_abs_gt(h, DZ_EDGES, 100):.1f}% |")
        mabs = pct(np.add(t["h_dz"][len(t["h_dz"]) // 2:], t["h_dz"][:len(t["h_dz"]) // 2][::-1]),
                   DZ_EDGES[len(DZ_EDGES) // 2:], 50)
        L += ["", f"Median |dz| = {mabs:.0f} m.", "", "Median dz by surface RH (2 m) and by LPL height above ground:", "",
              "| RH bin | n | median dz | | LPL height [m] | n | median dz |", "|---|---|---|---|---|---|---|"]
        for i in range(len(RH_EDGES) - 1):
            j = i if i < len(LPL_EDGES) - 1 else None
            a = f"| {RH_EDGES[i]:.1f}-{RH_EDGES[i + 1]:.1f} | {int(t['h_dz_rh'][i].sum()):,} | {pct(t['h_dz_rh'][i], DZ_EDGES, 50):.0f} |"
            b = (f" | {LPL_EDGES[j]:.0f}-{LPL_EDGES[j + 1]:.0f} | {int(t['h_dz_lpl'][j].sum()):,} | "
                 f"{pct(t['h_dz_lpl'][j], DZ_EDGES, 50):.0f} |") if j is not None else " | | | |"
            L.append(a + b)
        L += ["", "### 2 Consistency (counts, share of valid events)", "", "| check | count | share |", "|---|---|---|"]
        for k in sorted(k for k in t if k.startswith("c_")):
            L.append(f"| {k[2:]} | {int(t[k][0]):,} | {100 * t[k][0] / max(n, 1):.3f}% |")
        L += ["", "Expected: all near zero except `rh_lpl_gt_rh_2m` (the LPL is cooler and usually moister in RH).", ""]

        nb = max(t["s_both"][0], 1)
        L += ["### 3 Split sensitivity (eps from figdata.yaml, unstable = delta_FT > 0)", "",
              f"- unstable events: {int(t['s_unstable'][0]):,}; with a split using the LPL LCL: {int(t['s_valid_lpl'][0]):,}; "
              f"using the 2 m LCL: {int(t['s_valid_2m'][0]):,}",
              f"- kept with the LPL LCL but dropped with the 2 m LCL (guard `plcl <= p0`): {int(t['s_drop_lpl_only'][0]):,}; "
              f"the reverse: {int(t['s_drop_2m_only'][0]):,}",
              f"- on the {int(nb):,} events valid in both: mean |change in delta_below| = {t['s_sumabs_diff'][0] / nb:.3f} kJ/kg, "
              f"p95 of the change = {pct(t['h_dB_diff'], DB_EDGES, 95):.2f}, p5 = {pct(t['h_dB_diff'], DB_EDGES, 5):.2f}", ""]
        worst, nbins_used = 0.0, 0
        for name, edges, lab in (("rh", RH_EDGES, "RH_2m"), ("zl", ZL_EDGES, "z_LCL,LPL [m]")):
            L += [f"| {lab} | n | delta_FT | delta_below LPL | delta_below 2m | delta_above LPL | delta_above 2m | max shift |",
                  "|---|---|---|---|---|---|---|---|"]
            for i in range(len(edges) - 1):
                nn = t[f"b_{name}_n"][i]
                if nn < 1:
                    continue
                m = {k: t[f"b_{name}_{k}"][i] / nn for k in ("dB_lpl", "dB_2m", "dA_lpl", "dA_2m", "dFT")}
                shift = max(abs(m["dB_2m"] - m["dB_lpl"]), abs(m["dA_2m"] - m["dA_lpl"]))
                if nn >= 500:
                    worst, nbins_used = max(worst, shift), nbins_used + 1
                hi = "inf" if np.isinf(edges[i + 1]) else f"{edges[i + 1]:g}"
                L.append(f"| {edges[i]:g}-{hi} | {int(nn):,} | {m['dFT']:.2f} | {m['dB_lpl']:.2f} | {m['dB_2m']:.2f} | "
                         f"{m['dA_lpl']:.2f} | {m['dA_2m']:.2f} | {shift:.2f} |")
            L.append("")
        L += ["### 4 Model check: z_LCL = a - C ln(RH), bins of 0.05, RH 0.05-0.6", "",
              "| pair | C [km] | R^2 | bins |", "|---|---|---|---|"]
        for lab, k in (("z_LCL,2m vs RH_2m", "2m"), ("z_LCL,LPL vs RH_LPL", "lpl"), ("z_LCL,LPL vs RH_2m", "x")):
            C, r2, nbins = fit(t[f"fit_n_{k}"], t[f"fit_z_{k}"], t[f"fit_zz_{k}"])
            L.append(f"| {lab} | {C:.2f} | {r2:.3f} | {nbins} |")
        L.append("")
        verdict[ev] = (mabs, worst, nbins_used)

    L += ["## Decision rule (agreed in METHODS-PLAN.md)", "",
          "If median |dz| < 100 m and no bin with n >= 500 shifts by 0.2 kJ/kg or more, the choice does not matter: "
          "state the LPL LCL in Methods and say the 2 m one gives the same.", ""]
    for ev, (mabs, worst, nused) in verdict.items():
        ok = mabs < 100 and worst < 0.2
        word = "INSUFFICIENT DATA (no bin with n >= 500)" if nused == 0 else (
            "EQUIVALENT" if ok else "DIFFERENT, state both or choose with care")
        L.append(f"- {ev}: median |dz| = {mabs:.0f} m, largest binned shift = {worst:.2f} kJ/kg over {nused} bins -> {word}")
    text = "\n".join(L) + "\n"
    (in_dir / "lcl_report.md").write_text(text)
    if not quiet:
        print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("year", type=int)
    r.add_argument("event", choices=["T", "M"])
    r.add_argument("--repo", default=os.environ.get("PAPER26_REPO", str(Path(__file__).resolve().parents[1])))
    r.add_argument("--out", default="lcl_out")
    r.add_argument("--buffer", type=int, default=6)
    c = sub.add_parser("report")
    c.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.year, a.event, a.repo, a.out, a.buffer)
    else:
        report(a.dir)


if __name__ == "__main__":
    main()
