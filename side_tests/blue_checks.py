"""The blue numbers left in the draft that facts.json does not give, read from the stage g files.

    python side_tests/blue_checks.py [--years 1980-2019]          (login node, a minute or two)

Writes <work>/side_tests/blue_checks.md. All unweighted (conditional statistics), figure mask.
  1  cluster centres of T and M events in the (RH, δ_a) plane       (Section 6a, "RH ≈ 0.1, −3" etc.)
  2  T vs M at fixed RH: δ_a and δ_S, signed, per 0.05 RH bin       ("below about 0.6", "1.8 more δ_S")
  3  near-surface MSE across the RH bins of each cascade box         ("about 20 kJ/kg")
  4  ε for which a composite plume reaches h*500 at 500 hPa          ("about 0.1 km⁻¹")
  5  δ_below against the LCL height, per event type                 ("smallest for shallow layers")
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "figures"), str(REPO / "src"), str(REPO / "side_tests")]
import figdata as FD                                          # noqa: E402
from cascade_tests import eps_required                        # noqa: E402

MIN_N = 200
MIN_BOX = 500                                                 # events per cascade RH bin


def mode(d, product):
    h = d.h(product)
    h, ex = FD.coarsen(h, d.edges_of(product, "x"), 4, axis=0)     # 0.02 RH
    h, ey = FD.coarsen(h, d.edges_of(product, "y"), 4, axis=1)     # 1 kJ/kg
    a, b = np.unravel_index(np.argmax(h), h.shape)
    return FD.centers(ex)[a], FD.centers(ey)[b]


def rh_means(d, product, factor=10):
    h, ex = FD.coarsen(d.h(product), d.edges_of(product, "x"), factor, axis=0)   # 0.05 RH
    m, n = FD.binned_mean(h, ex, d.edges_of(product, "y"))
    return FD.centers(ex), m, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1980-2019")
    a = ap.parse_args(argv)
    years = tuple(int(x) for x in a.years.split("-"))
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    L = [f"# Blue numbers · {a.years} · unweighted", ""]

    L += ["## 1 · Cluster centres (densest 0.02 RH × 1 kJ/kg cell)", ""]
    for e, d in data.items():
        r, v = mode(d, "h2_rh_dFT")
        L.append(f"- {e}: RH ≈ **{r:.2f}**, δ_a ≈ **{v:+.1f}** kJ/kg")
    L.append("")

    L += ["## 2 · T vs M at fixed RH (bins of 0.05 with ≥ 200 events in both)", "",
          "| RH | n T | n M | δ_a T | δ_a M | T − M | δ_S T | δ_S M | M − T (δ_S) |", "|---|---|---|---|---|---|---|---|---|"]
    x, aT, nT = rh_means(data["T"], "h2_rh_dFT")
    _, aM, nM = rh_means(data["M"], "h2_rh_dFT")
    _, sT, _ = rh_means(data["T"], "h2_rh_dS")
    _, sM, _ = rh_means(data["M"], "h2_rh_dS")
    both = (nT >= MIN_N) & (nM >= MIN_N)
    for k in np.flatnonzero(both):
        L.append(f"| {x[k]:.3f} | {nT[k]:,.0f} | {nM[k]:,.0f} | {aT[k]:.2f} | {aM[k]:.2f} | {aT[k] - aM[k]:+.2f} | "
                 f"{sT[k]:.2f} | {sM[k]:.2f} | {sM[k] - sT[k]:+.2f} |")
    w = np.where(both, np.minimum(nT, nM), 0)
    if not both.any():
        L += ["", "No RH bin has enough events of both types.", ""]
    else:
        L += ["", f"Over these bins: mean |T − M| in δ_a = **{np.average(abs(aT - aM)[both], weights=w[both]):.2f}**, "
              f"mean (M − T) in δ_S = **{np.average((sM - sT)[both], weights=w[both]):+.2f}** kJ/kg "
              "(weights: the smaller of the two counts).", ""]

    L += ["## 3 · Near-surface MSE (h_LPL) across the 5 % RH bins of each cascade box, bins with ≥ 500 events", ""]
    edges = np.round(np.arange(0.0, 1.0001, 0.05), 3)
    eps_rows = []
    for case in data["T"].cases:
        d = data[case[0]]
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            c = FD.composite(d, case, lo, hi)
            if c["n"] < MIN_BOX:
                continue
            lev, z = d.levels.astype(float), c["Z"] / 9.8                    # kJ/kg → km
            k500 = int(np.argmin(abs(lev - 500)))
            er = eps_required(lev, z, c["MSE"], c["h_LPL"], c["p_LPL"], c["MSE_sat"][k500])
            rows.append((lo, hi, c["n"], c["h_LPL"], er))
            eps_rows.append((case, lo, er, c["n"]))
        if rows:
            hv = [r[3] for r in rows]
            L.append(f"- {case}: h_LPL from {hv[0]:.1f} (RH {rows[0][0]:.2f}–{rows[0][1]:.2f}) to {hv[-1]:.1f} "
                     f"(RH {rows[-1][0]:.2f}–{rows[-1][1]:.2f}): rise **{hv[-1] - hv[0]:+.1f}**, range "
                     f"{max(hv) - min(hv):.1f} kJ/kg over {len(rows)} bins")
    L.append("")

    L += ["## 4 · ε for a composite plume to reach h*500 at 500 hPa (bins with h_LPL > h*500)", "",
          "| box | RH bins | ε median | ε min–max |", "|---|---|---|---|"]
    allv = []
    for case in data["T"].cases:
        v = np.array([r[2] for r in eps_rows if r[0] == case and np.isfinite(r[2])])
        allv += list(v)
        if len(v):
            L.append(f"| {case} | {len(v)} | {np.median(v):.2f} | {v.min():.2f}–{v.max():.2f} |")
    if allv:
        L += ["", f"All boxes: median **{np.median(allv):.2f}** km⁻¹, range {min(allv):.2f}–{max(allv):.2f}."]
    L.append("")

    L += ["## 5 · Mean δ_below by LCL height of LPL air (unstable events, ε from figdata.yaml), 250 m bins", "",
          "| z_LCL [m] | n T | δ_below T | n M | δ_below M |", "|---|---|---|---|---|"]
    st = {}
    for e, d in data.items():
        s, ez = FD.coarsen(d.hist["lb_zLCL_LPL"], d.edges["zlcl"], 10, axis=0)
        n = s[:, 1, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            st[e] = (n, np.where(n >= MIN_N, s[:, 1, 1] / n, np.nan))
    zc = FD.centers(ez)
    for k in range(len(zc)):
        if st["T"][0][k] >= MIN_N or st["M"][0][k] >= MIN_N:
            L.append(f"| {zc[k]:.0f} | {st['T'][0][k]:,.0f} | {st['T'][1][k]:.2f} | {st['M'][0][k]:,.0f} | "
                     f"{st['M'][1][k]:.2f} |")
    L.append("")

    text = "\n".join(L) + "\n"
    out = Path(FD.load("paths")["work"]) / "side_tests" / "blue_checks.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
