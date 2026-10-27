"""Excess Histograms, dry vs moist: T and h2m extremes pooled, split at RH0.

Same style as fig_excess_hist.py, but the two curves are events drier and moister than RH0 (the
RH where the pooled mean δ_FT crosses zero). Caution: RH0 is defined from δ_FT itself, so the
δ_FT panel (and partly δ_total) splits on its own sign by construction; δ_S is the clean test.
    python figures/fig_excess_hist_rh.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figdata as FD
import style

WEIGHTED = True                  # a 1D distribution over the globe: cos(lat) weights (E02)
from fig_excess_hist import HIST_RANGE, LW, MEAN_LW, ZERO_LW, TXT, TITLE, LABEL, TICK

PANELS = [("h2_rh_dtotal", r"Total Excess: $\mathbf{\delta_{\mathrm{total}}}$"),
          ("h2_rh_dS", r"Surface Jump: $\mathbf{\delta_{S}}$"),
          ("h2_rh_dFT", r"Free Tropo. Excess: $\mathbf{\delta_{FT}}$")]


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    r0, rh0_source = FD.rh_split_value(data, weighted)
    lab = FD.split_label(r0)
    d0 = data["T"]
    band = style.band_text(FD.load("analysis"))
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), facecolor="white")
    facts = {"RH0 (split)": f"{r0:.3f} ({rh0_source})"}
    for k, (ax, (name, title)) in enumerate(zip(axes, PANELS)):
        h = sum(d.h(name, weighted=weighted) for d in data.values())       # (RH, δ)
        dry, moist = FD.split_rh(h, d0.edges_of(name, "x"), r0, axis=0)
        x = FD.centers(d0.edges_of(name, "y"))
        ax.set_facecolor("#F8F9FA")
        ax.axvline(0, c="#333333", lw=ZERO_LW, zorder=2)
        for key, c, color, yt in (("dry", dry, FD.DRY_COLOR, 0.95), ("moist", moist, FD.MOIST_COLOR, 0.80)):
            ax.fill_between(x, c, alpha=0.12, color=color, zorder=1)
            ax.plot(x, c, c=color, lw=LW, zorder=3)
            mean = (c * x).sum() / c.sum()
            ax.axvline(mean, c=color, alpha=0.85, ls="--", lw=MEAN_LW, zorder=4)
            ax.text(0.04, yt, f"{key}: mean = {mean:.2f} kJ kg$^{{-1}}$  ({100 * c.sum() / (dry.sum() + moist.sum()):.0f}% of events)",
                    color=color, transform=ax.transAxes, ha="left", va="top", fontsize=TXT - 2, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#F8F9FA", ec="none", alpha=0.85), zorder=5)
            facts[f"{name[6:]} {key} mean"] = f"{mean:.2f} kJ/kg"
        ax.set_xlim(HIST_RANGE)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=TICK)
        ax.set_title(f"{title} — {band}", fontsize=TITLE, fontweight="bold", pad=8)
        ax.set_ylabel("Count" + (" (cos-lat weighted)" if weighted else ""), fontsize=LABEL)
    axes[2].set_xlabel(r"$\delta$ [kJ kg$^{-1}$]", fontsize=LABEL)
    fig.legend(handles=[Line2D([0], [0], color=FD.DRY_COLOR, lw=LW, label=lab["dry"]),
                        Line2D([0], [0], color=FD.MOIST_COLOR, lw=LW, label=lab["moist"])],
               loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    facts.update(years=d0.label(), events="T and h2m extremes pooled",
                 caution="δ_FT (and partly δ_total) is split on its own zero crossing; δ_S is the independent test")
    return [{"path": FD.save(fig, out_dir, "excess_histograms_rh"), "title": "Excess Histograms · dry vs moist",
             "facts": facts, "script": "figures/fig_excess_hist_rh.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
