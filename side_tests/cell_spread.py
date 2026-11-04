"""How tightly the free troposphere pins h_ML, and where each RH bin's events sit. Read-only.

    python side_tests/cell_spread.py [--years 1980-2019] [--step 1]

Stage f analysis mask (low terrain, land, band), unweighted, as in the plume null test.
  1  Spread of h_ML (= MSE at the LPL) inside the null test's 2 kJ/kg cells of (h500, h*500),
     against its spread over all events. If the within-cell spread is small, the cell null
     swaps nearly identical mixed layers and says little; if large, the pass test is insensitive
     to h_ML.
  2  Per 0.05 RH bin and event type: mean |latitude|, tropical share (|lat| < 23.5), mean h*500,
     mean h_ML. Shows whether the pooled fall of h*500 with RH in wet-bulb extremes is geography.

Reads Profiles/profiles_*, Profiles/derived/derived_* (one level). Writes <work>/side_tests/cell_spread.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src")]
from tpl_mll.config import load                               # noqa: E402
from tpl_mll.masks import analysis_mask                       # noqa: E402

BUFFER, CELL = 6, 2.0
RH_EDGES = np.round(np.arange(0.0, 1.0001, 0.05), 3)
MIN_N = 200


def one(year, event):
    paths = load("paths")
    work = Path(paths["work"])
    with xr.open_dataset(work / paths["outputs"]["profiles"] / f"profiles_{event}_b{BUFFER:02d}_{year}.nc") as p, \
            xr.open_dataset(work / paths["outputs"]["derived"] / f"derived_{event}_b{BUFFER:02d}_{year}.nc") as d:
        p, d = p.sel(lead=0).isel(year=0), d.sel(lead=0).isel(year=0)
        lat, lon = p["latitude"].values, p["longitude"].values
        grid = lambda da: da.transpose("latitude", "longitude").values.astype(float)
        h500, hs500 = grid(p["MSE"].sel(level=500)), grid(p["MSE_sat"].sel(level=500))
        hml, rh = grid(d["MSE_LPL"]), grid(d["RH_2m"])
    ok = analysis_mask(lat, lon) & np.isfinite(hml) & np.isfinite(h500) & np.isfinite(hs500) & np.isfinite(rh)
    i, j = np.nonzero(ok)
    return {"cx": np.floor(h500[i, j] / CELL).astype(int), "cy": np.floor(hs500[i, j] / CELL).astype(int),
            "hml": hml[i, j], "hs": hs500[i, j], "rh": rh[i, j], "alat": np.abs(lat[i])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1980-2019")
    ap.add_argument("--step", type=int, default=1, help="every n-th year (4 = the null-test years)")
    a = ap.parse_args(argv)
    y0, y1 = (int(x) for x in a.years.split("-"))
    years = list(range(y0, y1 + 1, a.step))
    L = [f"# Cell spread and RH geography · {len(years)} years ({y0}–{y1}, step {a.step}) · low terrain, unweighted", ""]
    sec1 = ["## 1 · Spread of h_ML inside the null-test cells (2 kJ/kg in h500 and h*500)", "",
            "Cells are per year, as in the null test. Only cells with ≥ 2 events count.", "",
            "| events | n | std h_ML, all | std h_ML, within cell | within / all | share of variance explained by the cell |",
            "|---|---|---|---|---|---|"]
    sec2 = ["## 2 · Where each RH bin's events sit", "",
            "| events | RH | n | mean abs lat | tropical share | mean h*500 | mean h_ML |", "|---|---|---|---|---|---|---|"]
    for ev in ("T", "M"):
        n_all = s1 = s2 = 0.0
        w_ss = w_n = 0.0
        nb = len(RH_EDGES) - 1
        acc = np.zeros((nb, 5))                                # n, |lat|, tropical, h*500, h_ML
        for y in years:
            r = one(y, ev)
            key = r["cx"].astype(np.int64) * 100000 + r["cy"]
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            sm = np.bincount(inv, weights=r["hml"])
            keep = cnt[inv] >= 2
            dev = r["hml"] - (sm / cnt)[inv]
            w_ss += (dev[keep] ** 2).sum()
            w_n += keep.sum() - (cnt >= 2).sum()               # within-cell degrees of freedom
            x = r["hml"][keep]
            n_all += x.size
            s1 += x.sum()
            s2 += (x ** 2).sum()
            k = np.digitize(r["rh"], RH_EDGES) - 1
            u = (k >= 0) & (k < nb)
            for c, v in enumerate((np.ones_like(r["rh"]), r["alat"], r["alat"] < 23.5, r["hs"], r["hml"])):
                acc[:, c] += np.bincount(k[u], weights=v[u].astype(float), minlength=nb)
            print(f"  {y} {ev}: {len(key)} events", flush=True)
        sd_all = np.sqrt(s2 / n_all - (s1 / n_all) ** 2)
        sd_in = np.sqrt(w_ss / max(w_n, 1))
        sec1.append(f"| {ev} | {n_all:,.0f} | {sd_all:.2f} | {sd_in:.2f} | {sd_in / sd_all:.2f} | "
                    f"{100 * (1 - sd_in ** 2 / sd_all ** 2):.0f}% |")
        for b in range(nb):
            n = acc[b, 0]
            if n >= MIN_N:
                sec2.append(f"| {ev} | {RH_EDGES[b]:.2f}–{RH_EDGES[b + 1]:.2f} | {n:,.0f} | {acc[b, 1] / n:.1f} | "
                            f"{100 * acc[b, 2] / n:.0f}% | {acc[b, 3] / n:.1f} | {acc[b, 4] / n:.1f} |")
    text = "\n".join(L + sec1 + [""] + sec2) + "\n"
    out = Path(load("paths")["work"]) / "side_tests" / "cell_spread.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
