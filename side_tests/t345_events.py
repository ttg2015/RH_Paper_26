"""Tasks 3, 4, 5. Event-level checks of stage f against stage d/g definitions, η and closure,
and the per-box cascade slope Γ_h. Read-only on the pipeline.

Reads   <work>/Profiles/profiles_{E}_b06_{Y}.nc, Profiles/derived/derived_{E}_b06_{Y}.nc,
        Ztop/ztop_{E}_b06_{Y}.nc (Tasks 3, 4 only), ERA5 invariants for the mask
Writes  <out>/t345_{E}_{Y}.npz (small additive summaries) and <out>/t345_report.md

    python side_tests/t345_events.py run YEAR EVENT [--out DIR]
    python side_tests/t345_events.py report [--out DIR]

Events: analysis mask of stage f (band, land > 0.9, z_s < 0.5 km), lead 0. Everything is
unweighted (conditional statistics). Task 3 needs ztop files written with the stable /
diluted-out codes (4 / 5); older files are reported as such and Task 3 is skipped for them.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tpl_mll.config import load                     # noqa: E402
from tpl_mll.corrections import z_lcl_above_ground_legacy   # noqa: E402
from tpl_mll.masks import analysis_mask, surface_height_km   # noqa: E402

D_EDGES = np.arange(-40.0, 60.01, 0.25)             # δ_a histograms
ETA_EDGES = np.arange(-20.0, 20.01, 0.05)
RES_EDGES = np.concatenate([np.arange(0.0, 5.0, 0.01), [5.0, 1e9]])
UNI_EDGES = np.arange(-10.0, 10.01, 0.05)
RNG_EDGES = np.arange(0.0, 20.01, 0.05)
RH10 = np.linspace(0.0, 1.0, 11)
RH5 = np.linspace(0.0, 1.0, 21)
BAND = [400.0, 450.0, 500.0, 550.0, 600.0]
N_WORST = 20


def _hist(x, e):
    return np.histogram(np.clip(x[np.isfinite(x)], e[0], e[-1] - 1e-9), e)[0].astype(float)


def interp_lnp(levels, field, p):
    """field (n, L) on ascending levels → value at p (n,), linear in ln p."""
    lnl, lnp = np.log(levels), np.log(p)
    i = np.clip(np.searchsorted(lnl, lnp) - 1, 0, len(levels) - 2)
    w = (lnp - lnl[i]) / (lnl[i + 1] - lnl[i])
    r = np.arange(len(p))
    return field[r, i] * (1 - w) + field[r, i + 1] * w


def load_events(year, event):
    paths = load("paths")
    work = Path(paths["work"])
    out = paths["outputs"]
    with xr.open_dataset(work / out["profiles"] / f"profiles_{event}_b06_{year}.nc") as p:
        prof = p[["MSE", "MSE_sat"]].sel(lead=0).isel(year=0).load()
    with xr.open_dataset(work / out["derived"] / f"derived_{event}_b06_{year}.nc") as d:
        der = d[["MSE_LPL", "RH_2m", "z_LCL_2m", "z_LCL_LPL", "Z_LPL"]].sel(lead=0).isel(year=0).load()
    lat, lon = prof["latitude"].values, prof["longitude"].values
    levels = prof["level"].values.astype(float)
    mask = analysis_mask(lat, lon) & np.isfinite(der["MSE_LPL"].values)
    i, j = np.nonzero(mask)
    take = lambda da: da.transpose("latitude", "longitude", ...).values[i, j].astype("float64")
    hs = take(prof["MSE_sat"])
    k500 = int(np.nonzero(levels == 500)[0][0])
    zl, z2 = take(der["z_LCL_LPL"]), take(der["z_LCL_2m"])
    if der["z_LCL_2m"].attrs.get("reference") != "above_ground":   # legacy stage d (E07), as in g.fields
        zs_m = (surface_height_km(lat, lon) * 1000.0)[i, j]
        z2 = z_lcl_above_ground_legacy(z2, zs_m)
        zl = z_lcl_above_ground_legacy(zl, zs_m, take(der["Z_LPL"]) / 9.8 * 1000.0)
    ev = dict(h_lpl=take(der["MSE_LPL"]), rh=take(der["RH_2m"]), zl=zl, z2=z2, hs=hs,
              h500=take(prof["MSE"])[:, k500], hs500=hs[:, k500])
    ev["da"] = ev["h_lpl"] - ev["hs500"]
    return ev, (i, j), lat, lon, levels, mask


def task3(ev, z, s, out):
    cat = z["category"].values                                    # (eps, n)
    out["t3_codes_new"] = np.array([float("diluted" in str(z["category"].attrs.get("flag_meanings", "")))])
    if not out["t3_codes_new"][0]:
        return
    da = ev["da"]
    unst = cat[0] != 4
    pos = da > 0
    out["t3_2x2"] = np.array([[np.sum(unst & pos), np.sum(unst & ~pos)], [np.sum(~unst & pos), np.sum(~unst & ~pos)]], float)
    out["t3_hist_da_unstable_neg"] = _hist(da[unst & ~pos], D_EDGES)
    out["t3_hist_da_stable_pos"] = _hist(da[~unst & pos], D_EDGES)
    cls = np.where(np.isnan(da), 2, np.where(pos, 0, 1))          # δ_a > 0, ≤ 0, NaN
    tab = np.zeros((3, cat.shape[0], 6))
    for k in range(cat.shape[0]):
        c = np.nan_to_num(cat[k], nan=0).astype(int)
        for q in range(3):
            tab[q, k] = np.bincount(c[cls == q], minlength=6)[:6]
    out["t3_by_class"] = tab
    out["t3_f_totals"] = np.array([np.bincount(np.nan_to_num(cat[k], nan=0).astype(int), minlength=6)[:6]
                                   for k in range(cat.shape[0])], float)
    k2 = int(np.argmin(abs(z["epsilon"].values - 0.2)))
    rb = np.clip(np.digitize(ev["rh"], RH10) - 1, 0, 9)
    t = np.zeros((10, 6))
    m = pos & np.isfinite(ev["rh"])
    np.add.at(t, (rb[m], np.nan_to_num(cat[k2][m], nan=0).astype(int)), 1)
    out["t3_rh_eps02"] = t


def task4(ev, z, levels, ij, out):
    eps = z["epsilon"].values
    p_top, cat, E = z["p_top"].values, z["category"].values, z["E_total"].values
    ne = len(eps)
    out["t4_eta_all"], out["t4_eta_pass"] = np.zeros((ne, len(ETA_EDGES) - 1)), np.zeros((ne, len(ETA_EDGES) - 1))
    out["t4_sums"] = np.zeros((ne, 2, 3))                         # (all, pass) × (n, Ση, Σ|η|)
    out["t4_res"] = np.zeros((ne, len(RES_EDGES) - 1))
    worst = []
    for k in range(ne):
        top = np.isfinite(p_top[k]) & np.isfinite(E[k]) & np.isfinite(ev["da"])
        eta = np.full(len(top), np.nan)
        eta[top] = interp_lnp(levels, ev["hs"][top], p_top[k][top]) - ev["hs500"][top]
        res = E[k] - (ev["da"] - eta)
        ps = top & (cat[k] == 1)
        for g, m in ((0, top), (1, ps)):
            out["t4_sums"][k, g] = [m.sum(), eta[m].sum(), np.abs(eta[m]).sum()]
        out["t4_eta_all"][k], out["t4_eta_pass"][k] = _hist(eta[top], ETA_EDGES), _hist(eta[ps], ETA_EDGES)
        out["t4_res"][k] = _hist(np.abs(res[top]), RES_EDGES)
        for n in np.argsort(-np.abs(np.nan_to_num(res)))[:N_WORST]:
            if top[n]:
                worst.append([eps[k], ij[0][n], ij[1][n], res[n], p_top[k][n], E[k][n], ev["da"][n], eta[n]])
    w = np.array(worst) if worst else np.zeros((0, 8))
    out["t4_worst"] = w[np.argsort(-np.abs(w[:, 3]))][:N_WORST] if len(w) else w


def uniformity(ev, levels, out):
    kb = [int(np.nonzero(levels == p)[0][0]) for p in BAND]
    band = ev["hs"][:, kb]
    out["t4_band_mean_minus_500"] = _hist(band.mean(1) - ev["hs500"], UNI_EDGES)
    out["t4_band_range"] = _hist(band.max(1) - band.min(1), RNG_EDGES)


def task5(ev, event, out):
    boxes = {k: v for k, v in load("figdata")["cascade"]["boxes"].items() if k.startswith(event + "_")}
    out["t5_boxes"] = np.array(list(boxes), dtype="U16")
    acc = np.zeros((len(boxes), len(RH5) - 1, 6))                 # n, Σh_LPL, Σz_LPL, Σz_2m, Σh*500, Σδ_a
    rb = np.digitize(ev["rh"], RH5) - 1
    for b, box in enumerate(boxes.values()):
        m = ((ev["hs500"] > box["h_star"][0]) & (ev["hs500"] < box["h_star"][1]) & (ev["h500"] > box["h"][0])
             & (ev["h500"] < box["h"][1]) & (rb >= 0) & (rb < len(RH5) - 1)
             & np.isfinite(ev["zl"]) & np.isfinite(ev["z2"]))
        for c, v in enumerate((np.ones(m.sum()), ev["h_lpl"][m], ev["zl"][m], ev["z2"][m], ev["hs500"][m], ev["da"][m])):
            np.add.at(acc[b, :, c], rb[m], v)
    out["t5_acc"] = acc


def run(year, event, out_dir):
    ev, ij, lat, lon, levels, mask = load_events(year, event)
    out = {"n_events": np.array([float(len(ev["da"]))])}
    paths = load("paths")
    zp = Path(paths["work"]) / paths["outputs"]["ztop"] / f"ztop_{event}_b06_{year}.nc"
    uniformity(ev, levels, out)
    task5(ev, event, out)
    if zp.exists():
        z = xr.load_dataset(zp)
        zz = xr.Dataset({k: (("epsilon", "n"), z[k].transpose("epsilon", "latitude", "longitude").values[:, ij[0], ij[1]])
                         for k in ("p_top", "category", "E_total")}, coords={"epsilon": z["epsilon"].values})
        zz["category"].attrs = z["category"].attrs
        out["t3_f_mask_mismatch"] = np.array([float(np.sum(np.isnan(zz["category"].values[0])))])
        task3(ev, zz, None, out)
        task4(ev, zz, levels, ij, out)
        out["eps"] = z["epsilon"].values
        out["t4_worst_latlon"] = np.c_[lat[out["t4_worst"][:, 1].astype(int)], lon[out["t4_worst"][:, 2].astype(int)]] \
            if len(out["t4_worst"]) else np.zeros((0, 2))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"t345_{event}_{year}.npz", **out)
    print(f"{year} {event}: {len(ev['da']):,} events, ztop {'yes' if zp.exists() else 'MISSING'}, "
          f"new codes {bool(out.get('t3_codes_new', [0])[0])}", flush=True)


# ------------------------------------------------------------------------------------- report
def pct(h, e, q):
    c = np.cumsum(h)
    if c[-1] == 0:
        return np.nan
    t = q / 100 * c[-1]
    i = min(int(np.searchsorted(c, t)), len(h) - 1)
    lo = c[i - 1] if i else 0.0
    return e[i] + (t - lo) / max(h[i], 1e-12) * (e[i + 1] - e[i])


def slope(acc, col, window, nmin=30):
    """Least-squares slope of bin-mean h_LPL on bin-mean z (km) over RH bins in window; local slopes."""
    c = 0.5 * (RH5[1:] + RH5[:-1])
    n = acc[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        h, z = acc[:, 1] / n, acc[:, col] / n / 1000.0
    use = (n >= nmin) & (c > window[0]) & (c < window[1])
    g = np.polyfit(z[use], h[use], 1)[0] if use.sum() >= 3 else np.nan
    local = np.diff(h) / np.diff(z)
    local[(n[:-1] < nmin) | (n[1:] < nmin)] = np.nan
    return g, local


def report(out_dir):
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("t345_*_*.npz"))
    if not files:
        raise SystemExit(f"no t345_*.npz in {out_dir}")
    L = ["# Tasks 3–5 · event-level checks", "",
         "Events: stage f analysis mask (band, land > 0.9, z_s < 0.5 km), lead 0. All numbers unweighted.", ""]
    by = {}
    for p in files:
        _, e, y = p.stem.split("_")
        by.setdefault(e, {})[int(y)] = dict(np.load(p))
    L.append("Files: " + "; ".join(f"{e}: {sorted(v)}" for e, v in by.items()))
    L.append("")
    for e, yrs in by.items():
        S = lambda k: sum(d[k] for d in yrs.values() if k in d)
        have3 = [y for y, d in yrs.items() if "t3_2x2" in d]
        no_codes = [y for y, d in yrs.items() if "t3_codes_new" in d and not d["t3_codes_new"][0]]
        no_f = [y for y, d in yrs.items() if "eps" not in d]
        L += [f"## {e} events · {int(S('n_events')[0]):,} events in {len(yrs)} years", ""]
        if no_f:
            L.append(f"- no ztop file for {no_f}: Tasks 3–4 skipped there")
        if no_codes:
            L.append(f"- **ztop files for {no_codes} predate the stable / diluted-out codes: rerun stage f; Task 3 skipped there**")
        mm = sum(d.get("t3_f_mask_mismatch", np.zeros(1))[0] for d in yrs.values())
        L += [f"- events in this mask without a stage f category: {int(mm):,} (expect 0)", ""]
        eps = next((d["eps"] for d in yrs.values() if "eps" in d), None)
        names = {1: "pass", 2: "too shallow", 3: "too deep", 4: "stable", 5: "diluted out"}
        if have3:
            t = sum(yrs[y]["t3_2x2"] for y in have3)
            L += [f"### Task 3a · unstable (buoyant at ε = 0) vs δ_a > 0 · years {have3}", "",
                  "| | δ_a > 0 | δ_a ≤ 0 |", "|---|---|---|",
                  f"| unstable | {int(t[0, 0]):,} | {int(t[0, 1]):,} |", f"| stable | {int(t[1, 0]):,} | {int(t[1, 1]):,} |", "",
                  "Stable with δ_a > 0 should be ~0 (h_LPL > h*500 makes the undiluted plume buoyant at 500 hPa).", ""]
            for lab, k in (("unstable, δ_a ≤ 0", "t3_hist_da_unstable_neg"), ("stable, δ_a > 0", "t3_hist_da_stable_pos")):
                h = sum(yrs[y][k] for y in have3)
                if h.sum():
                    L.append(f"- {lab}: δ_a median {pct(h, D_EDGES, 50):.2f}, p5 {pct(h, D_EDGES, 5):.2f}, "
                             f"p95 {pct(h, D_EDGES, 95):.2f} kJ/kg (n = {int(h.sum()):,})")
            tab = sum(yrs[y]["t3_by_class"] for y in have3)
            tot = sum(yrs[y]["t3_f_totals"] for y in have3)
            ok = np.allclose(tab.sum(0), tot)
            L += ["", f"### Task 3b · category shares per ε (rows add to 1) · validation vs stage f totals: "
                      f"{'PASS' if ok else 'FAIL'}", ""]
            for q, lab in ((0, "δ_a > 0"), (1, "δ_a ≤ 0 (control)")):
                L += [f"**{lab}**", "", "| ε | n | " + " | ".join(names.values()) + " |", "|---" * 7 + "|"]
                for k, ee in enumerate(eps):
                    r = tab[q, k, 1:]
                    L.append(f"| {ee:.2f} | {int(r.sum()):,} | " + " | ".join(f"{100 * x / max(r.sum(), 1):.1f}%" for x in r) + " |")
                L.append("")
            if tab[2].sum():
                L.append(f"- events with NaN δ_a: {int(tab[2, 0].sum()):,}")
            t = sum(yrs[y]["t3_rh_eps02"] for y in have3)
            L += ["### Task 3c · δ_a > 0 by RH at ε = 0.2", "", "| RH | n | " + " | ".join(names.values()) + " |", "|---" * 7 + "|"]
            for i in range(10):
                r = t[i, 1:]
                if r.sum() >= 20:
                    L.append(f"| {RH10[i]:.1f}–{RH10[i + 1]:.1f} | {int(r.sum()):,} | "
                             + " | ".join(f"{100 * x / r.sum():.1f}%" for x in r) + " |")
            L.append("")
        if eps is not None:
            have4 = [y for y, d in yrs.items() if "t4_sums" in d]
            s4 = sum(yrs[y]["t4_sums"] for y in have4)
            L += [f"### Task 4 · η = h*(p_top) − h*500 and closure E_total − (δ_a − η) · years {have4}", "",
                  "| ε | plumes with a top | mean η | mean \\|η\\| | p5 | p95 | pass: n | pass mean η | pass mean \\|η\\| | pass p5 | pass p95 | median \\|res\\| | p95 \\|res\\| | p99 \\|res\\| |",
                  "|---" * 14 + "|"]
            worst95 = 0.0
            for k, ee in enumerate(eps):
                ha = sum(yrs[y]["t4_eta_all"][k] for y in have4)
                hp = sum(yrs[y]["t4_eta_pass"][k] for y in have4)
                hr = sum(yrs[y]["t4_res"][k] for y in have4)
                a, p_ = s4[k, 0], s4[k, 1]
                r95 = pct(hr, RES_EDGES, 95)
                worst95 = max(worst95, r95 if np.isfinite(r95) else 0)
                L.append(f"| {ee:.2f} | {int(a[0]):,} | {a[1] / max(a[0], 1):.2f} | {a[2] / max(a[0], 1):.2f} | "
                         f"{pct(ha, ETA_EDGES, 5):.2f} | {pct(ha, ETA_EDGES, 95):.2f} | {int(p_[0]):,} | "
                         f"{p_[1] / max(p_[0], 1):.2f} | {p_[2] / max(p_[0], 1):.2f} | {pct(hp, ETA_EDGES, 5):.2f} | "
                         f"{pct(hp, ETA_EDGES, 95):.2f} | {pct(hr, RES_EDGES, 50):.3f} | {r95:.3f} | {pct(hr, RES_EDGES, 99):.3f} |")
            L += ["", f"Closure criterion p95 |residual| ≤ 0.5 kJ/kg at every ε: **{'PASS' if worst95 <= 0.5 else 'FAIL'}** "
                      f"(largest p95 {worst95:.3f}). The integral stops at the last fine-grid point below the top "
                      "(160 log-p points, ~100 m) and η uses profile levels, so residuals of a few tenths are expected.", "",
                  "Worst cases (all years):", "", "| year | ε | lat | lon | residual | p_top | E_total | δ_a | η |", "|---" * 9 + "|"]
            rows = []
            for y in have4:
                d = yrs[y]
                for w, ll in zip(d["t4_worst"], d["t4_worst_latlon"]):
                    rows.append((abs(w[3]), y, w, ll))
            for _, y, w, ll in sorted(rows, key=lambda r: -r[0])[:10]:
                L.append(f"| {y} | {w[0]:.2f} | {ll[0]:.2f} | {ll[1]:.2f} | {w[3]:.2f} | {w[4]:.0f} | {w[5]:.2f} | {w[6]:.2f} | {w[7]:.2f} |")
            L.append("")
        hu, hrng = S("t4_band_mean_minus_500"), S("t4_band_range")
        L += ["### Task 4 · h* uniformity over 400–600 hPa (levels 400, 450, 500, 550, 600), all events", "",
              f"- band mean of h* minus h*500: median {pct(hu, UNI_EDGES, 50):.2f}, p5 {pct(hu, UNI_EDGES, 5):.2f}, "
              f"p95 {pct(hu, UNI_EDGES, 95):.2f} kJ/kg",
              f"- range of h* across the band: median {pct(hrng, RNG_EDGES, 50):.2f}, p95 {pct(hrng, RNG_EDGES, 95):.2f}, "
              f"p99 {pct(hrng, RNG_EDGES, 99):.2f} kJ/kg", ""]

        # Task 5
        boxes = next(iter(yrs.values()))["t5_boxes"]
        c = 0.5 * (RH5[1:] + RH5[:-1])
        ylist = sorted(yrs)
        L += ["### Task 5 · per-box cascade slope Γ_h (kJ/kg per km of z_LCL) · unweighted, 5 % RH bins, z_s < 0.5 km", ""]
        for b, name in enumerate(boxes):
            accs = {y: yrs[y]["t5_acc"][b] for y in ylist}
            tot = sum(accs.values())
            L += [f"**{name}** · {int(tot[:, 0].sum()):,} events", "",
                  "| z_LCL | window | Γ_h | jackknife ± | |", "|---|---|---|---|---|"]
            for col, zlab in ((2, "LPL air"), (3, "2 m air")):
                for win in ((0.30, 0.60), (0.30, 0.70)):
                    g, _ = slope(tot, col, win)
                    jk = [slope(tot - accs[y], col, win)[0] for y in ylist] if len(ylist) > 2 else []
                    jk = np.array(jk)
                    se = np.sqrt((len(jk) - 1) / len(jk) * np.nansum((jk - np.nanmean(jk)) ** 2)) if len(jk) else np.nan
                    L.append(f"| {zlab} | {win[0]:.2f}–{win[1]:.2f} | {g:.2f} | {se:.2f} | {len(ylist)} years |")
            _, locL = slope(tot, 2, (0, 1))
            _, loc2 = slope(tot, 3, (0, 1))
            n = tot[:, 0]
            with np.errstate(invalid="ignore", divide="ignore"):
                hs, da = tot[:, 4] / n, tot[:, 5] / n
            L += ["", "| RH | n | local Γ_h (LPL) | local Γ_h (2 m) | mean h*500 | mean δ_a |", "|---|---|---|---|---|---|"]
            for i in range(len(c)):
                if n[i] >= 30:
                    lL = f"{locL[i]:.2f}" if i < len(locL) and np.isfinite(locL[i]) else ""
                    l2 = f"{loc2[i]:.2f}" if i < len(loc2) and np.isfinite(loc2[i]) else ""
                    L.append(f"| {RH5[i]:.2f}–{RH5[i + 1]:.2f} | {int(n[i]):,} | {lL} | {l2} | {hs[i]:.2f} | {da[i]:.2f} |")
            L.append("")
    facts = REPO / "runs" / "figs" / "facts.json"
    if facts.exists():
        import json
        f = json.loads(facts.read_text())
        ent = next((v["facts"] for k, v in f.get("figures", {}).items() if "entrainment" in k.lower()), None)
        L += ["### Pooled fit for comparison (figures/fig_excess_vs_rh_entrainment.py, facts.json)", "",
              "```", json.dumps(ent, indent=1, ensure_ascii=False) if ent else "not found in facts.json", "```", ""]
    else:
        L += ["Pooled fit: run `python figures/make_all.py` to get facts.json, then rerun this report.", ""]
    text = "\n".join(L) + "\n"
    (out_dir / "t345_report.md").write_text(text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("year", type=int)
    r.add_argument("event", choices=["T", "M"])
    for p in (r, sub.add_parser("report")):
        p.add_argument("--out")
    a = ap.parse_args(argv)
    out = a.out or str(Path(load("paths")["work"]) / "side_tests")
    run(a.year, a.event, out) if a.cmd == "run" else report(out)


if __name__ == "__main__":
    main()
