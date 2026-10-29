"""Every bold placeholder and blue number in the Draft 3 results/methods, from what is already on disk.

Reads   <work>/FigData (stage g), <work>/side_tests/lcl/lcl_*.npz (Task 1),
        <work>/side_tests/t345/t345_*.npz (Tasks 3, 4; or --t345 DIR, e.g. side_tests/nb)
Writes  <work>/side_tests/draft_numbers.md

    python side_tests/draft_numbers.py [--years 1980-2019] [--t345 DIR]

Weighting follows the paper: cos(lat) for the 1D histograms over the domain (Section 4 means,
shares, percentiles), none for plume shares, conditional means, RH0 and the side tests.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "figures"), str(REPO / "src"), str(REPO / "side_tests")]
import figdata as FD                                          # noqa: E402
import lcl_compare as LC                                      # noqa: E402
import t345_events as T                                       # noqa: E402
from t2_figdata_numbers import pct, stats                     # noqa: E402

EV = {"T": "temperature", "M": "wet-bulb"}


def sec4(data, L):
    L += ["## Section 4 · violation of the simple bound (cos(lat) weighted 1D histograms)", ""]
    for ev, d in data.items():
        e = d.edges["delta"]
        s = {q: stats(d.h(f"h1_{q}", weighted=True), e) for q in ("delta_total", "delta_S", "delta_FT")}
        su = stats(d.h("h1_delta_total"), e)
        L += [f"**{EV[ev]} ({ev})**",
              f"- mean δ = {s['delta_total']['mean']:.1f} (draft 3.9 / 16.7); p99 δ = **{s['delta_total']['p99']:.1f}**; "
              f"share δ > 0 = **{100 * s['delta_total']['>0']:.0f}%**",
              f"- mean δ_S = {s['delta_S']['mean']:.1f} (draft 5.9 / 10.2); share δ_S > 0 = **{100 * s['delta_S']['>0']:.0f}%**",
              f"- mean δ_a = {s['delta_FT']['mean']:.1f} (draft −2.0 / +6.5); share δ_a > 0 = **{100 * s['delta_FT']['>0']:.0f}%**, "
              f"δ_a < 0 = **{100 * s['delta_FT']['<0']:.0f}%**",
              f"- weighting shift of mean δ (weighted − unweighted) = **{s['delta_total']['mean'] - su['mean']:+.2f}** kJ/kg", ""]
    L += ["Tails sentence: use 'δ_a > 0' for T and 'δ_a < 0' for M.", ""]


def sec5(data, L):
    d0 = data["T"]
    eps, pc = d0.epsilon, FD.centers(d0.edges["ptop"])
    lo, hi = FD.load("analysis")["ztop_window_hPa"]
    kd = int(np.argmin(abs(eps - FD.load("figdata")["strike"]["eps_detail"])))
    L += [f"## Section 5 · plume tops (unweighted, shares of unstable = buoyant at ε = 0; detail ε = {eps[kd]:g})", ""]
    for ev, d in data.items():
        cat = d.h("strike_category")                              # (ε, code)
        total, stable, sh = FD.unstable_counts(cat)
        hp = d.h("strike_ptop")
        med = np.array([np.interp(0.5 * r.sum(), np.cumsum(r), pc) if r.sum() else np.nan for r in hp])
        inband = eps[(med <= hi) & (med >= lo)]
        b = int(np.nanargmax(sh[1]))
        L += [f"**{EV[ev]} ({ev})**",
              f"- stable share of all events = **{100 * stable[0] / (stable[0] + total[0]):.0f}%**",
              f"- best pass share = **{100 * sh[1][b]:.0f}%** of unstable at ε = **{eps[b]:.2f}** (draft 0.20 / 0.25)",
              f"- at ε = {eps[kd]:g}: pass {100 * sh[1][kd]:.0f}%, too shallow {100 * sh[2][kd]:.0f}%, too deep "
              f"{100 * sh[3][kd]:.0f}%, diluted out **{100 * sh[5][kd]:.0f}%**",
              f"- at ε = {eps[0]:g}: too deep {100 * sh[3][0]:.0f}% of unstable, "
              f"{100 * sh[3][0] / max(1 - sh[5][0], 1e-9):.0f}% of the plumes still buoyant",
              f"- median top: {med[0]:.0f} hPa at ε = {eps[0]:g} → {med[-1]:.0f} hPa at ε = {eps[-1]:g} (draft 300 → 640); "
              f"at ε = {eps[kd]:g}: {med[kd]:.0f} hPa",
              f"- median inside {lo}–{hi} hPa for ε = " + (f"**{inband.min():.2f}–{inband.max():.2f}**" if len(inband) else "none")
              + " (draft 0.15–0.4)", ""]
    r0 = FD.rh0(data)
    if not np.isfinite(r0):
        L += ["- RH0: no zero crossing, dry/moist split skipped", ""]
        return
    hc = sum(d.hist["strike_category_rh"] for d in data.values())   # (ε, rh, code), T + M
    dry, moist = FD.split_rh(hc, d0.edges["rh_strike"], r0, axis=1)
    for lab, h in (("dry (RH < RH0)", dry), ("moist", moist)):
        tot, stab, s = FD.unstable_counts(h)
        L.append(f"- {lab}, T + M: still buoyant at ε = {eps[kd]:g} = **{100 * (1 - s[5][kd]):.0f}%** of unstable "
                 f"(draft 9%), stable {100 * stab[0] / max(stab[0] + tot[0], 1):.0f}% of all")
    L += [f"- RH0 (pooled, unweighted) = {r0:.3f}", ""]


def lcl_split(data, L):
    L += ["## Section 5.3 · δ_below / δ_above (unweighted, unstable events, LPL LCL)", ""]
    for ev, d in data.items():
        s, ez = d.hist["lb_zLCL_LPL"], d.edges["zlcl"]
        n = s[:, 0, 0]
        mb = np.where(n >= 100, s[:, 1, 1] / np.maximum(n, 1), np.nan)
        ma = np.where(n >= 100, s[:, 2, 1] / np.maximum(n, 1), np.nan)
        lt1 = ez[1:] <= 1000
        da1 = s[lt1, 0, 1].sum() / max(s[lt1, 0, 0].sum(), 1)
        z = FD.centers(ez)
        neg = z[(ma < 0) & np.isfinite(ma)]
        at4 = np.nanmean(ma[(z > 3750) & (z < 4250)])
        L.append(f"- {ev}: max bin-mean δ_below = **{np.nanmax(mb) if np.isfinite(mb).any() else np.nan:.2f}** (draft 1.3); mean δ_a for z_LCL < 1 km = "
                 f"**{da1:.1f}** (draft 8–9); δ_above near 4 km = {at4:.1f} (draft 1); δ_above first < 0 at z_LCL ≈ "
                 + (f"**{neg.min():.0f} m**" if len(neg) else "never") + " (draft 3.5 km)")
    L.append("")


def methods_lcl(lcl_dir, L):
    files = sorted(Path(lcl_dir).glob("lcl_*_*.npz"))
    L += [f"## Methods · LPL height and LCL of LPL vs 2 m air (Task 1, unweighted, {len(files)} files)", ""]
    if not files:
        L += [f"no files in {lcl_dir}: run Task 1", ""]
        return
    t = {}
    for p in files:
        for k, v in np.load(p).items():
            t[k] = t.get(k, 0) + v
    nl = t["h_dz_lpl"].sum(1)                                     # events per LPL-height bin
    c = np.cumsum(nl) / nl.sum()
    e = LC.LPL_EDGES
    L += ["- z0 (LPL height above ground), share below: "
          + ", ".join(f"{e[i + 1]:.0f} m {100 * c[i]:.0f}%" for i in range(len(nl) - 1)),
          f"- dz = z_LCL,LPL − z_LCL,2m: median **{LC.pct(t['h_dz'], LC.DZ_EDGES, 50):.0f} m**, "
          f"p5–p95 **{LC.pct(t['h_dz'], LC.DZ_EDGES, 5):.0f} to {LC.pct(t['h_dz'], LC.DZ_EDGES, 95):.0f} m**",
          f"- split sensitivity: mean |Δδ_below| = **{t['s_sumabs_diff'][0] / max(t['s_both'][0], 1):.2f}** kJ/kg, "
          f"p5–p95 {LC.pct(t['h_dB_diff'], LC.DB_EDGES, 5):.2f} to {LC.pct(t['h_dB_diff'], LC.DB_EDGES, 95):.2f}", ""]


def side345(t_dir, L):
    files = sorted(Path(t_dir).glob("t345_*_*.npz"))
    L += [f"## Methods / Appendix · overlap, η, h* uniformity (Tasks 3–4, unweighted, {len(files)} files)", ""]
    by = {}
    for p in files:
        by.setdefault(p.stem.split("_")[1], []).append(dict(np.load(p)))
    for ev, ds in by.items():
        t3 = [d for d in ds if "t3_2x2" in d]
        if t3:
            m = sum(d["t3_2x2"] for d in t3)                      # [[U&+, U&−], [S&+, S&−]]
            n = m.sum()
            L.append(f"- {ev} overlap: unstable ∩ δ_a > 0 = {100 * m[0, 0] / n:.1f}%, unstable with δ_a ≤ 0 = "
                     f"**{100 * m[0, 1] / n:.1f}%**, stable with δ_a > 0 = **{100 * m[1, 0] / n:.1f}%** of events "
                     f"({len(t3)} years)")
        else:
            L.append(f"- {ev} overlap: no Task 3 data in {t_dir} (summaries predate the f rerun)")
        hu = sum(d["t4_band_mean_minus_500"] for d in ds)
        hr = sum(d["t4_band_range"] for d in ds)
        eps = ds[0].get("eps")
        if eps is not None and "t4_sums" in ds[0]:
            k = int(np.argmin(abs(eps - 0.2)))
            s = sum(d["t4_sums"] for d in ds)[k, 1]
            L.append(f"- {ev} η for pass plumes at ε = {eps[k]:g}: mean {s[1] / max(s[0], 1):.2f}, mean |η| "
                     f"**{s[2] / max(s[0], 1):.2f}** kJ/kg")
        L.append(f"- {ev} h* over 400–600 hPa: band mean − h*500 median **{T.pct(hu, T.UNI_EDGES, 50):.2f}** "
                 f"(p5 {T.pct(hu, T.UNI_EDGES, 5):.2f}, p95 {T.pct(hu, T.UNI_EDGES, 95):.2f}); range median "
                 f"{T.pct(hr, T.RNG_EDGES, 50):.2f}, p95 {T.pct(hr, T.RNG_EDGES, 95):.2f} kJ/kg")
    L.append("")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1980-2019")
    ap.add_argument("--t345", help="folder of t345_*.npz (default <work>/side_tests/t345)")
    a = ap.parse_args(argv)
    years = tuple(int(x) for x in a.years.split("-"))
    side = Path(FD.load("paths")["work"]) / "side_tests"
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    L = [f"# Draft 3 numbers · {a.years}", ""]
    sec4(data, L)
    sec5(data, L)
    lcl_split(data, L)
    methods_lcl(side / "lcl", L)
    side345(Path(a.t345) if a.t345 else side / "t345", L)
    text = "\n".join(L) + "\n"
    (side / "draft_numbers.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
