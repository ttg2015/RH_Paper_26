"""Task 2. Paper numbers straight from the stage g figure data (no rerun, no event files).

Reads   <work>/FigData/figdata_{T,M}_b06_{YEAR}.nc for every year found (via figures/figdata.py)
Writes  <out>/t2_report.md   (default out: <work>/side_tests)

    python side_tests/t2_figdata_numbers.py [--years 1980-2019] [--out DIR]

Weighting is stated for every number: the 1D histograms are given cos(lat) weighted and
unweighted side by side; the conditional means (lb_*, h2_*) are unweighted.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "figures"), str(REPO / "src")]
import figdata as FD                                          # noqa: E402

Q = {"delta_total": "δ (total)", "delta_S": "δ_S", "delta_FT": "δ_a (free trop.)"}


def pct(h, e, q):
    """Percentile (0–100) of a histogram, linear within the bin."""
    c = np.cumsum(h)
    if c[-1] == 0:
        return np.nan
    t = q / 100 * c[-1]
    i = min(int(np.searchsorted(c, t)), len(h) - 1)
    lo = c[i - 1] if i else 0.0
    return e[i] + (t - lo) / max(h[i], 1e-12) * (e[i + 1] - e[i])


def stats(h, e):
    c = 0.5 * (e[1:] + e[:-1])
    n = h.sum()
    return {"n": n, "mean": (h * c).sum() / n, ">0": h[e[:-1] >= 0].sum() / n, "<0": h[e[1:] <= 0].sum() / n,
            ">30": h[e[:-1] >= 30].sum() / n, "p99": pct(h, e, 99), "p99.9": pct(h, e, 99.9)}


def event_counts(event, years):
    """Events per year in the figure mask (one per cell per year) from each file's in_mask."""
    n = 0
    for y, _ in FD.available(event):
        if years[0] <= y <= years[1]:
            n += int(xr.load_dataset(FD.folder() / f"figdata_{event}_b06_{y}.nc")["in_mask"].sum())
    return n


def coarse_rows(e, width):
    """Bin groups of about `width` along edges e: list of (lo, hi) index ranges."""
    step = max(1, int(round(width / (e[1] - e[0]))))
    return [(i, min(i + step, len(e) - 1)) for i in range(0, len(e) - 1, step)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1980-2019")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    years = tuple(int(x) for x in a.years.split("-"))
    paths = FD.load("paths")
    out = Path(a.out or Path(paths["work"]) / "side_tests")
    out.mkdir(parents=True, exist_ok=True)
    L = ["# Task 2 · numbers from stage g figure data", "",
         f"Years {a.years}. Files: {FD.folder()}. 1D histograms: 0.25 kJ/kg bins from −40 to 60, zero is a bin edge.", ""]
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    shifts = {}
    for ev, d in data.items():
        e = d.edges["delta"]
        n_ev = event_counts(ev, years)
        L += [f"## {ev} events ({d.label()}, {len(d.years)} years)", "",
              f"Stored events in the figure mask: {n_ev:,}.", "",
              "| quantity | weighting | in histogram | outside −40…60 or NaN | mean | >0 | <0 | >30 | p99 | p99.9 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for q, lab in Q.items():
            for w in (False, True):
                h = d.h(f"h1_{q}", weighted=w)
                s = stats(h, e)
                miss = "" if w else f"{n_ev - int(s['n']):,}"
                L.append(f"| {lab} | {'cos(lat)' if w else 'none'} | {s['n']:,.0f}{' (weights)' if w else ''} | {miss} | "
                         f"{s['mean']:.2f} | {100 * s['>0']:.1f}% | {100 * s['<0']:.1f}% | {100 * s['>30']:.2f}% | "
                         f"{s['p99']:.1f} | {s['p99.9']:.1f} |")
            shifts[(ev, q)] = stats(d.h(f"h1_{q}", weighted=True), e)["mean"] - stats(d.h(f"h1_{q}"), e)["mean"]
        L += ["", "Mean from bin centres (error ≤ 0.125 kJ/kg). Means in kJ/kg.", ""]

        # conditional means, unweighted
        L += [f"### {ev}: δ_below / δ_above by z_LCL (LPL air, above ground) · unstable (δ_a > 0) · "
              f"ε = {d.var_attrs['lb_zLCL_LPL'].get('eps_km', '?')} km⁻¹ · unweighted", ""]
        s = d.hist["lb_zLCL_LPL"]
        ez = d.edges["zlcl"]
        L += ["| z_LCL [m] | n | mean δ_a | mean δ_below | mean δ_above |", "|---|---|---|---|---|"]
        for lo, hi in coarse_rows(ez, 250):
            n = s[lo:hi, 0, 0].sum()
            if n < 50:
                continue
            m = s[lo:hi, :, 1].sum(0) / n
            L.append(f"| {ez[lo]:g}–{ez[hi]:g} | {int(n):,} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} |")
        n_all = s[:, 0, 0]
        mb = np.where(n_all >= 100, s[:, 1, 1] / np.maximum(n_all, 1), np.nan)
        lt1 = ez[1:] <= 1000
        if np.isfinite(mb).any():
            k = int(np.nanargmax(mb))
            L.append(f"\n- largest bin-mean δ_below (25 m bins with n ≥ 100): **{mb[k]:.2f} kJ/kg** at z_LCL {ez[k]:g}–{ez[k + 1]:g} m")
        else:
            L.append("\n- largest bin-mean δ_below: no 25 m bin has n ≥ 100")
        L += [f"- mean δ_a for z_LCL < 1 km (unstable events): **{s[lt1, 0, 1].sum() / max(s[lt1, 0, 0].sum(), 1):.2f} kJ/kg**", ""]
        sr = d.hist["lb_RH_2m"]
        er = d.edges["rh"]
        L += [f"### {ev}: δ_below / δ_above by RH_2m · unstable · unweighted", "",
              "| RH | n | mean δ_a | mean δ_below | mean δ_above |", "|---|---|---|---|---|"]
        for lo, hi in coarse_rows(er, 0.1):
            n = sr[lo:hi, 0, 0].sum()
            if n < 50:
                continue
            m = sr[lo:hi, :, 1].sum(0) / n
            L.append(f"| {er[lo]:.2f}–{er[hi]:.2f} | {int(n):,} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} |")
        # all events (stable too): mean δ_a by z_LCL (2 m air) from h2_zlcl_dFT
        h2 = d.h("h2_zlcl_dFT")
        ex, ey = d.edges_of("h2_zlcl_dFT", "x"), d.edges_of("h2_zlcl_dFT", "y")
        cy = 0.5 * (ey[1:] + ey[:-1])
        L += ["", f"### {ev}: mean δ_a by z_LCL (2 m air), ALL events (stable included) · unweighted", "",
              "| z_LCL,2m [m] | n | mean δ_a |", "|---|---|---|"]
        for lo, hi in coarse_rows(ex, 500):
            hh = h2[lo:hi].sum(0)
            if hh.sum() >= 50:
                L.append(f"| {ex[lo]:g}–{ex[hi]:g} | {int(hh.sum()):,} | {(hh * cy).sum() / hh.sum():.2f} |")
        L.append("")

    L += ["## Weighted − unweighted mean shift (kJ/kg)", "", "| event | δ | δ_S | δ_a |", "|---|---|---|---|"]
    for ev in data:
        L.append(f"| {ev} | " + " | ".join(f"{shifts[(ev, q)]:+.2f}" for q in Q) + " |")
    L += ["", "## Draft claims to compare", "",
          "- δ_below ≤ about 1.3 kJ/kg: see the largest bin-mean δ_below per event above.",
          "- δ_a about 8–9 kJ/kg for z_LCL < 1 km, about 1 by 4 km, negative above about 3.5 km: the lb tables are",
          "  unstable events only (δ_a > 0 by construction), so 'negative' can only apply to δ_above there, or to",
          "  δ_a in the all-events table.", ""]
    text = "\n".join(L) + "\n"
    (out / "t2_report.md").write_text(text)
    print(text)
    print(f"Wrote {out / 't2_report.md'}")


if __name__ == "__main__":
    main()
