"""Excess Histograms: distributions of δ_total, δ_S and δ_FT for T and M events.

Reads FigData (stage g). Default cos(lat) weighting (global distribution, E02).
    python figures/fig_excess_hist.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

QUANTITIES = ["delta_total", "delta_S", "delta_FT"]
COARSEN = 4            # 0.25 → 1 kJ/kg bins
XLIM = (-20, 30)


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    facts = {}
    for ax, q in zip(axes, QUANTITIES):
        for e, d in data.items():
            name = f"h1_{q}"
            c, ed = FD.coarsen(d.h(name, weighted=weighted), d.edges_of(name), COARSEN)
            frac = 100 * c / c.sum()
            x = FD.centers(ed)
            ax.plot(x, frac, color=style.EVENT_COLOR[e], lw=2, label=style.EVENT_LABEL[e])
            full = d.h(name, weighted=weighted)
            mean = (full * FD.centers(d.edges_of(name))).sum() / full.sum()
            ax.axvline(mean, color=style.EVENT_COLOR[e], ls="--", lw=1.2)
            facts[f"mean {q} {e}"] = f"{mean:.2f} kJ/kg"
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlim=XLIM, xlabel=f"{style.DELTA_LABEL[q]}  [kJ/kg]")
    axes[0].set_ylabel(f"% of events per {0.25 * COARSEN:g} kJ/kg")
    axes[0].legend(frameon=False)
    for ax, q in zip(axes, QUANTITIES):
        txt = "\n".join(f"{e}: mean {facts[f'mean {q} {e}']}" for e in data)
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    cfg = FD.load("analysis")
    d0 = data["T"]
    fig.suptitle(f"Energy excesses at annual extremes · {d0.label()} · {style.band_text(cfg, weighted)}")
    fig.tight_layout()
    facts.update(years=d0.label(), weighting="cos(lat)" if weighted else "none", bin=f"{0.25 * COARSEN:g} kJ/kg")
    return [{"path": FD.save(fig, out_dir, "excess_histograms"), "title": "Excess Histograms", "facts": facts,
             "script": "figures/fig_excess_hist.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
