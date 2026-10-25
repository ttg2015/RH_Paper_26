"""Excess Maps: multi-year mean δ_FT and δ_S at each land cell, for T and M events.

Mean = sum over years / number of years with a finite value. Cells outside the analysis mask
are blank. Plain lat-lon axes (no cartopy), so it runs anywhere.
    python figures/fig_excess_maps.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

QUANTITIES = ["delta_FT", "delta_S"]
LIM = 20     # kJ/kg colour limit


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.2), sharex=True, sharey=True)
    facts = {}
    for i, q in enumerate(QUANTITIES):
        for j, (e, d) in enumerate(data.items()):
            ax = axes[i][j]
            m = d.map_mean(q)
            pc = ax.pcolormesh(d.lon, d.lat, m, cmap="RdBu_r", vmin=-LIM, vmax=LIM, shading="nearest")
            ax.set_title(f"{style.DELTA_SHORT[q]} · {style.EVENT_LABEL[e]}", fontsize=10)
            facts[f"land mean {q} {e}"] = f"{np.nanmean(m):.2f} kJ/kg (unweighted over cells)"
            ax.set_facecolor("#eceff1")
    for ax in axes[:, 0]:
        ax.set_ylabel("latitude")
    for ax in axes[1]:
        ax.set_xlabel("longitude")
    lo, hi = FD.load("analysis")["lat_band"]
    axes[0][0].set_ylim(lo - 2, hi + 2)
    fig.colorbar(pc, ax=axes, label="kJ/kg", shrink=0.8)
    d0 = data["T"]
    fig.suptitle(f"Mean energy excess at annual extremes · {d0.label()} · {style.band_text(FD.load('analysis'))}")
    facts.update(years=d0.label(), colour_limit=f"±{LIM} kJ/kg")
    return [{"path": FD.save(fig, out_dir, "excess_maps"), "title": "Excess Maps", "facts": facts,
             "script": "figures/fig_excess_maps.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
