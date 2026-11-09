"""Excess Histograms (paper figure): δ_total, δ_S and δ_FT for T and WBT (h2m) extremes.

Port of Paper_Plots-2 cell 44 (Di_Histograms_publication.pdf): three stacked panels, light filled
curves, dashed means with labels, same colours and layout, now on 0.25 kJ/kg bins (old: 0.625).
Writes both versions: cos(lat) weighted (E02, default for global distributions) and plain counts
(as the old figure). fig_weighting_check.py compares them.
    python figures/fig_excess_hist.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figdata as FD
import style

WEIGHTED = True                  # a 1D distribution over the globe: cos(lat) weights (E02)

COARSEN = 1                      # native 0.25 kJ/kg bins
HIST_RANGE = (-25, 45)         # wide enough for the WBT tail of δ_total
QUANTITIES = [("delta_total", r"total excess $\delta$"),
              ("delta_S", r"surface-layer excess $\delta_{S}$"),
              ("delta_FT", r"atmospheric excess $\delta_{a}$")]
TEMP_COLOR, MSE_COLOR = style.EVENT_COLOR["T"], style.EVENT_COLOR["M"]
LW, MEAN_LW, ZERO_LW = 2.5, 1.8, 1.5
TXT, TITLE, LABEL, TICK = 13, 15, 13, 11
# the paper figure is drawn at one journal column (19pc); the shared constants above stay for fig_excess_hist_rh
COL = dict(LW=1.3, MEAN_LW=1.0, ZERO_LW=0.8, TXT=7, TITLE=8, LABEL=7.5, TICK=6.5, SPINE=0.8)


def hist(d, q, weighted):
    name = f"h1_{q}"
    c, e = FD.coarsen(d.h(name, weighted=weighted), d.edges_of(name), COARSEN)
    full = d.h(name, weighted=weighted)
    mean = (full * FD.centers(d.edges_of(name))).sum() / full.sum()      # all values, not just the range
    return FD.centers(e), c, mean


def panel(ax, q, title, data, weighted, show_x):
    ax.set_facecolor(style.PANEL_BG)
    for s in ("bottom", "left"):
        ax.spines[s].set_linewidth(COL["SPINE"])
        ax.spines[s].set_color("#444444")
    ax.axvline(0, c="#333333", lw=COL["ZERO_LW"], zorder=2)
    means = {}
    for e, color in (("T", TEMP_COLOR), ("M", MSE_COLOR)):
        x, c, mean = hist(data[e], q, weighted)
        c = c / 1000.0                                            # thousands of events
        ax.fill_between(x, c, alpha=0.12, color=color, zorder=1)
        ax.plot(x, c, c=color, lw=COL["LW"], zorder=3)
        ax.axvline(mean, c=color, alpha=0.85, ls="--", lw=COL["MEAN_LW"], zorder=4)
        means[e] = mean
    ax.text(0.98, 0.95, f"mean {means['T']:.1f}", color=TEMP_COLOR, transform=ax.transAxes, ha="right",
            va="top", fontsize=COL["TXT"])
    ax.text(0.98, 0.80, f"mean {means['M']:.1f}", color=MSE_COLOR, transform=ax.transAxes, ha="right",
            va="top", fontsize=COL["TXT"])
    ax.set_xlim(HIST_RANGE)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=COL["TICK"], length=2.5, width=0.7, pad=2)
    ax.set_title(title, fontsize=COL["TITLE"], pad=3)
    ax.set_ylabel(r"Events [$10^3$]", fontsize=COL["LABEL"])           # weighting stated in the caption
    if show_x:
        ax.set_xlabel("excess " + style.KJ, fontsize=COL["LABEL"])
    return means


def figure(data, weighted):
    band = style.band_text(FD.load("analysis"))
    fig, axes = plt.subplots(3, 1, figsize=(3.7, 5.6), facecolor="white")
    means = {}
    for k, (ax, (q, title)) in enumerate(zip(axes, QUANTITIES)):
        means[q] = panel(ax, q, title, data, weighted, k == 2)   # band and years go in the caption
        style.panel_letter(ax, k, x=0.015, y=0.97)
    fig.legend(handles=[Line2D([0], [0], color=TEMP_COLOR, lw=COL["LW"], label=style.EVENT_LABEL["T"]),
                        Line2D([0], [0], color=MSE_COLOR, lw=COL["LW"], label=style.EVENT_LABEL["M"]),
                        Line2D([0], [0], color="0.3", lw=COL["MEAN_LW"], ls="--", label="mean [kJ/kg]")],
               loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(rect=[0, 0.07, 1, 1], h_pad=0.6)
    return fig, means


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    items = []
    for w, name in ((True, "excess_histograms"), (False, "excess_histograms_unweighted")):
        fig, means = figure(data, w)
        facts = {f"mean {q} {e}": f"{means[q][e]:.2f} kJ/kg" for q, _ in QUANTITIES for e in ("T", "M")}
        facts.update(years=data["T"].label(), weighting="cos(lat)" if w else "none (plain counts, as the old figure)",
                     bins=f"{0.25 * COARSEN:g} kJ/kg (old figure: 0.625)")
        items.append({"path": FD.save(fig, out_dir, name), "facts": facts, "script": "figures/fig_excess_hist.py",
                      "title": "Excess Histograms" + ("" if w else " · unweighted")})
        plt.close(fig)
    return items


if __name__ == "__main__":
    FD.cli(make, __doc__)
