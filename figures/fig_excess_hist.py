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

COARSEN = 1                      # native 0.25 kJ/kg bins
HIST_RANGE = (-20, 30)
QUANTITIES = [("delta_total", r"Total Excess: $\mathbf{\delta_{\mathrm{total}}}$"),
              ("delta_S", r"Surface Jump: $\mathbf{\delta_{S}}$"),
              ("delta_FT", r"Free Tropo. Excess: $\mathbf{\delta_{FT}}$")]
TEMP_COLOR, MSE_COLOR = "#C0392B", "#2471A3"
LW, MEAN_LW, ZERO_LW = 2.5, 1.8, 1.5
TXT, TITLE, LABEL, TICK = 13, 15, 13, 11


def hist(d, q, weighted):
    name = f"h1_{q}"
    c, e = FD.coarsen(d.h(name, weighted=weighted), d.edges_of(name), COARSEN)
    full = d.h(name, weighted=weighted)
    mean = (full * FD.centers(d.edges_of(name))).sum() / full.sum()      # all values, not just the range
    return FD.centers(e), c, mean


def panel(ax, q, title, data, weighted, show_x):
    ax.set_facecolor("#F8F9FA")
    for s in ("bottom", "left"):
        ax.spines[s].set_linewidth(1.2)
        ax.spines[s].set_color("#444444")
    ax.axvline(0, c="#333333", lw=ZERO_LW, zorder=2)
    means = {}
    for e, color in (("T", TEMP_COLOR), ("M", MSE_COLOR)):
        x, c, mean = hist(data[e], q, weighted)
        ax.fill_between(x, c, alpha=0.12, color=color, zorder=1)
        ax.plot(x, c, c=color, lw=LW, zorder=3)
        ax.axvline(mean, c=color, alpha=0.85, ls="--", lw=MEAN_LW, zorder=4)
        means[e] = mean
    ax.text(0.04, 0.95, r"$\overline{\delta_{T_{\max}}}$" + f" = {means['T']:.2f} kJ kg$^{{-1}}$",
            color=TEMP_COLOR, transform=ax.transAxes, ha="left", va="top", fontsize=TXT, fontweight="bold")
    ax.text(0.04, 0.80, r"$\overline{\delta_{WBT_{\max}}}$" + f" = {means['M']:.2f} kJ kg$^{{-1}}$",
            color=MSE_COLOR, transform=ax.transAxes, ha="left", va="top", fontsize=TXT, fontweight="bold")
    ax.set_xlim(HIST_RANGE)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=TICK, length=4, width=1.1)
    ax.set_title(title, fontsize=TITLE, fontweight="bold", pad=8)
    ax.set_ylabel("Count" + (" (cos-lat weighted)" if weighted else ""), fontsize=LABEL)
    if show_x:
        ax.set_xlabel(r"$\delta$ [kJ kg$^{-1}$]", fontsize=LABEL)
    return means


def figure(data, weighted):
    band = style.band_text(FD.load("analysis"))
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), facecolor="white")
    means = {}
    for k, (ax, (q, title)) in enumerate(zip(axes, QUANTITIES)):
        means[q] = panel(ax, q, f"{title} — {band}", data, weighted, k == 2)
    fig.legend(handles=[Line2D([0], [0], color=TEMP_COLOR, lw=LW, label=r"Temperature $\delta$"),
                        Line2D([0], [0], color=MSE_COLOR, lw=LW, label=r"WBT $\delta$"),
                        Line2D([0], [0], color=TEMP_COLOR, lw=MEAN_LW, ls="--", alpha=0.85, label="Temperature mean"),
                        Line2D([0], [0], color=MSE_COLOR, lw=MEAN_LW, ls="--", alpha=0.85, label="WBT mean")],
               loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
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
