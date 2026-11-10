"""RH composites (paper figure, merges old Figs 10 and 11): how the composites are built, then all three regimes.

Top row, the procedure on the regime of the plume schematic (fig_plume_schematic.REGIME, its two RH bins
from pair()):
(a) the pooled T + WBT events in the (h*500, h500) plane with all three regime boxes, lettered as the
    cascade panels below;
(b) the surface-RH distribution inside the schematic's regime in 5 % bins, the two Fig. 8 bins colored;
(c) the two mean profiles of those bins.
Bottom row, (d–f): the three regime cascades of fig_cascade_regimes.py, RH bins up to its RH_MAX.

Panels are drawn by the helpers of fig_rh_demo.py and fig_cascade_regimes.py, so the separate figures and
this one cannot drift apart. Reads <work>/side_tests/casc/casc_{E}_{Y}.npz once.
    python figures/fig_rh_composites.py [--years 1980-2019] [--casc DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

import figdata as FD
import fig_cascade_regimes as R
import fig_plume_schematic as S
import fig_rh_demo as D
import style

REPO = Path(__file__).resolve().parents[1]
WEIGHTED = False


def comps_of(c):
    """One regime composite in the per-bin dict layout of fig_plume_schematic.from_regime."""
    comps = {}
    for i, x in enumerate(c["rh_centers"]):
        b = (float(round(x - 0.025, 2)), float(round(x + 0.025, 2)))
        comps[b] = {"MSE": c["MSE"][i], "MSE_sat": c["MSE_sat"][i], "Z": c["Z"][i], "h_LPL": float(c["h_LPL"][i]),
                    "p_LPL": float(c["p_LPL"][i]), "n": float(c["n"][i]), "p_LCL": float(c["p_LCL_LPL"][i])}
    return comps


def draw(arrays, out_dir, label="", plane=None, rh_max=R.RH_MAX):
    style.apply()
    regs = R.regime_composites(arrays)
    names = list(regs)
    edges = arrays[0]["rh_edges"]
    norm = BoundaryNorm(edges, R.CMAP.N)

    fig = plt.figure(figsize=(15.0, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.32, wspace=0.28)

    # top row: the procedure
    ax_a = style.paper_axes(fig.add_subplot(gs[0, 0]))
    R.draw_plane(ax_a, arrays, plane, letters="def", title="select a regime")
    c_s, _ = regs[S.REGIME]
    comps = comps_of(c_s)
    levels = np.asarray(c_s["levels"], float)
    rh_low, rh_high, _ = S.pair(levels, comps)
    ax_b = style.paper_axes(fig.add_subplot(gs[0, 1]))
    ax_c = style.paper_axes(fig.add_subplot(gs[0, 2]))
    D.draw_hist_profiles(ax_b, ax_c, levels, comps, rh_low, rh_high)
    ax_b.set_title(f"bin by surface RH ({S.REGIME})", fontsize=10)
    for k, ax in enumerate((ax_a, ax_b, ax_c)):
        style.panel_letter(ax, k)

    # bottom row: the three regimes
    facts, axes = {}, []
    for k, name in enumerate(names):
        c, cov = regs[name]
        ax = style.paper_axes(fig.add_subplot(gs[1, k], sharey=axes[0] if axes else None))
        axes.append(ax)
        _, line = R.draw_cascade(ax, c, cov, norm, rh_max=rh_max)
        ax.set_title(name, fontsize=10, color=R.REGIME_COLOR[k])
        style.panel_letter(ax, k + 3)
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
        hs, h = R.REGIMES[name]
        facts[name] = f"h*500 {hs[0]:g}–{hs[1]:g}, h500 {h[0]:g}–{h[1]:g}; " + line
    axes[0].set_ylabel("Pressure [hPa]")
    R.rh_colorbar(fig, axes, edges, norm, rh_max)
    R.cascade_legend(axes[-1])

    pct = lambda r: f"{100 * r[0]:.0f}–{100 * r[1]:.0f}%"
    facts["top row"] = (f"{S.REGIME}: bins {pct(rh_low)} ({int(comps[rh_low]['n']):,} events) and "
                        f"{pct(rh_high)} ({int(comps[rh_high]['n']):,} events), as Fig. 8")
    facts["RH bins drawn (bottom)"] = "all" if rh_max is None else f"centers ≤ {100 * rh_max:.0f}%"
    facts["panel a"] = "figdata event plane" if plane is not None else "3 kJ/kg composite grid (no figdata plane)"
    facts["years"] = label
    path = FD.save(fig, out_dir, "rh_composites")
    plt.close(fig)
    return {"path": path, "title": "RH composites (old Figs 10 + 11)", "facts": facts,
            "script": "figures/fig_rh_composites.py"}


def make(out_dir, years=None, weighted=WEIGHTED, casc=None):
    folder = Path(casc) if casc else Path(FD.load("paths")["work"]) / "side_tests" / "casc"
    A = R.T.load_all(folder, years=years)
    label = f"{years[0]}–{years[1]}" if years else "all years in casc/"
    return [draw([A["T"], A["M"]], out_dir, label, plane=R.fine_plane(years))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", help="e.g. 1980-2019 (default: all casc files)")
    ap.add_argument("--casc", help="folder of casc_*.npz (default <work>/side_tests/casc)")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs"))
    a = ap.parse_args()
    for item in make(Path(a.out), FD.years_arg(a.years), casc=a.casc):
        print(f"Wrote {item['path']}")
        for k, v in item["facts"].items():
            print(f"  {k}: {v}")
