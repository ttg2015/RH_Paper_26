"""Excess parts vs RH (paper figure, between the maps and Fig. 5): how many events, and what excess, at each RH.

Top: event counts per RH bin for temperature and wet-bulb extremes and both together (the strip that used to
sit on Fig. 5): pooled, the events cover the RH range nearly uniformly. Bottom: the mean total excess δ and its
parts δ_S (surface layer) and δ_a (atmosphere) in each RH bin, temperature and wet-bulb extremes pooled.
Restyled from fig_delta_s_ft_vs_rh.py (orange/blue clashed with the event colors): δ black, δ_S green (the 2-m
color of Fig. 1), δ_a purple (the density color of Fig. 5). Means only for now, in BIN_PCT % RH bins, plain
(unweighted) conditional means like the other RH figures; bins with fewer than MIN_EVENTS events are dropped.
    python figures/fig_delta_parts_vs_rh.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style
from fig_excess_vs_rh import EVENT_LINE, count_strip

WEIGHTED = False
BIN_PCT = 2.5                        # % RH per bin (native 0.5 %)
MIN_EVENTS = 500
COLORS = {"delta": "k", "delta_S": "#2E8B57", "delta_a": "#6A51A3"}
LABELS = {"delta": r"total $\delta = h_{2\mathrm{m}} - h^*_{500}$",
          "delta_S": r"surface layer $\delta_S = h_{2\mathrm{m}} - h_{ML}$",
          "delta_a": r"atmosphere $\delta_a = h_{ML} - h^*_{500}$"}


def binned(H, ex, ey, step):
    """(bin centers, mean, count) of a pooled (RH, value) histogram in bins of `step` native bins."""
    hx, exx = FD.coarsen(H, ex, step, axis=0)
    m, n = FD.binned_mean(hx, exx, ey)
    m = np.where(n >= MIN_EVENTS, m, np.nan)
    return FD.centers(exx), m, n


def make(out_dir, years=None, weighted=WEIGHTED):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    d0 = data["T"]
    out = {}
    for key, product in (("delta_S", "h2_rh_dS"), ("delta_a", "h2_rh_dFT")):
        H = {e: d.h(product, weighted=weighted) for e, d in data.items()}
        ex, ey = d0.edges_of(product, "x"), d0.edges_of(product, "y")
        step = max(1, int(round(BIN_PCT / 100 / np.diff(ex).mean())))
        out[key] = binned(H["T"] + H["M"], ex, ey, step)
        if key == "delta_a":
            H_counts, ex_counts = H, ex
    x, mS, _ = out["delta_S"]
    xa, mA, n = out["delta_a"]
    if not np.allclose(x, xa):
        raise ValueError("δ_S and δ_a products have different RH bins")
    curves = {"delta": mS + mA, "delta_S": mS, "delta_a": mA}

    fig = plt.figure(figsize=(3.7, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 3.2], hspace=0.06)
    axc = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=axc)
    count_strip(axc, H_counts, ex_counts)
    axc.tick_params(labelbottom=False, labelsize=7)
    axc.legend(handles=[mlines.Line2D([], [], color=EVENT_LINE["T"], lw=1.2, label=style.EVENT_SHORT["T"]),
                        mlines.Line2D([], [], color=EVENT_LINE["M"], lw=1.2, label=style.EVENT_SHORT["M"]),
                        mlines.Line2D([], [], color="0.35", lw=1.2, label="both")],
               fontsize=6, ncol=3, loc="upper center", frameon=False, handlelength=1.5)
    ax.axhline(0, color="k", lw=0.9, ls=":")
    for key in ("delta", "delta_S", "delta_a"):
        ax.plot(x, curves[key], color=COLORS[key], lw=2.0 if key == "delta" else 1.8, marker="o", ms=2.8,
                label=LABELS[key], zorder=4)
    rh0 = FD.rh0(data)
    if np.isfinite(rh0):
        ax.plot(rh0, 0, "o", ms=6, mfc="white", mec=COLORS["delta_a"], mew=1.2, zorder=5)
    ax.set(xlim=(0, 1), ylabel=r"mean excess " + style.KJ)
    ax.legend(fontsize=6.3, loc="upper left", frameon=False)
    style.rh_percent(ax)
    fig.tight_layout()
    at = lambda m, r: float(np.interp(r, x[np.isfinite(m)], m[np.isfinite(m)])) if np.isfinite(m).any() else np.nan
    facts = {f"at {int(100 * r)}% RH": ", ".join(f"{k} {at(curves[k], r):+.1f}" for k in curves) + " kJ/kg"
             for r in (0.1, 0.3, 0.5, 0.7, 0.9)}
    facts.update({"RH0 (white circle)": f"{rh0:.3f}", "bins": f"{BIN_PCT:g} % RH, ≥ {MIN_EVENTS} events",
                  "weighting": "cos(lat)" if weighted else "none", "years": d0.label()})
    path = FD.save(fig, out_dir, "delta_parts_vs_rh")
    plt.close(fig)
    return [{"path": path, "title": "Excess parts vs RH", "facts": facts,
             "script": "figures/fig_delta_parts_vs_rh.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
