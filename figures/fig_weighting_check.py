"""How much does cos(latitude) weighting change the excess distributions and means? (E02)

Overlays weighted (solid) and unweighted (dashed) distributions of each δ for T and M events,
normalised to unit area, and reports the shift in the mean for the whole band and each sub-band.
Unweighted counts over-represent high latitudes (a 0.25° cell at 60° is half the area of one at
the equator), so the shift shows how much those cells pull the answer.
    python figures/fig_weighting_check.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

QUANTITIES = ["delta_total", "delta_S", "delta_FT"]
COLORS = {"T": "#C0392B", "M": "#2471A3"}


def mean_of(d, q, weighted, subband="all"):
    name = f"h1_{q}"
    h = d.h(name, subband, weighted)
    return (h * FD.centers(d.edges_of(name))).sum() / h.sum()


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    facts = {}
    rows = []
    for ax, q in zip(axes, QUANTITIES):
        for e, d in data.items():
            name = f"h1_{q}"
            x = FD.centers(d.edges_of(name))
            for w, ls in ((True, "-"), (False, "--")):
                h = d.h(name, weighted=w)
                ax.plot(x, h / h.sum() / 0.25, color=COLORS[e], ls=ls, lw=1.6 if w else 1.3,
                        label=f"{e} {'weighted' if w else 'unweighted'}")
            mw, mu = mean_of(d, q, True), mean_of(d, q, False)
            facts[f"{q} {e}: weighted − unweighted mean"] = f"{mw:.2f} − {mu:.2f} = {mw - mu:+.2f} kJ/kg"
            for sb in d.subbands:
                rows.append((q, e, sb, mean_of(d, q, True, sb), mean_of(d, q, False, sb)))
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlim=(-20, 30), xlabel=f"{style.DELTA_LABEL[q]} [kJ/kg]", ylabel="density [per kJ/kg]")
        txt = "\n".join(f"{e}: {facts[f'{q} {e}: weighted − unweighted mean'].split('= ')[1]}" for e in data)
        ax.text(0.97, 0.95, "mean shift\n" + txt, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"cos(lat) weighting vs plain counts · {data['T'].label()} · {style.band_text(FD.load('analysis'))}")
    fig.tight_layout()
    for q, e, sb, mw, mu in rows:
        if sb != "all":
            facts[f"{q} {e} [{sb}] shift"] = f"{mw - mu:+.2f} kJ/kg"
    facts["note"] = "within a sub-band the weights vary less, so shifts there should be small"
    return [{"path": FD.save(fig, out_dir, "weighting_check"), "title": "Weighting check (E02)",
             "facts": facts, "script": "figures/fig_weighting_check.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
