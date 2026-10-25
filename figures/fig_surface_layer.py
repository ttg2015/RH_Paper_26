"""Surface Layer Correction: h_2m vs h*_500 density with h_LPL contours, for M and T extremes.

Port of Paper_Plots-2 cell 46 (TempThing_overlay_right.pdf): filled density of (h*500, h_2m),
1:1 line, and green contours of the (h*500, h_LPL) density (from its 80th percentile up), showing
how much the surface layer lifts h_2m above the mixed layer.
    python figures/fig_surface_layer.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

RANGE = (280, 390)
COARSEN = 2                      # 0.5 → 1.1 kJ/kg-ish bins like the old 100 bins over 110 kJ/kg
SIGMA = 2
CONTOUR_LEVELS, CONTOUR_MIN_PCT = 4, 80


def density(d, name, weighted):
    h = d.h(name, weighted=weighted)
    ex, ey = d.edges_of(name, "x"), d.edges_of(name, "y")
    h, ex, ey = FD.crop(h, ex, ey, RANGE, RANGE)
    h, ex = FD.coarsen(h, ex, COARSEN, axis=0)
    h, ey = FD.coarsen(h, ey, COARSEN, axis=1)
    return FD.smooth(h, SIGMA).T, FD.centers(ex), FD.centers(ey)


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    fig, axes = plt.subplots(2, 1, figsize=(5, 8))
    for ax, e, cmap in ((axes[0], "M", "Blues"), (axes[1], "T", "Reds")):
        H, xc, yc = density(data[e], "h2_hs500_h2m", weighted)
        cf = ax.contourf(xc, yc, H, levels=20, cmap=cmap)
        ax.plot(RANGE, RANGE, lw=1.5, alpha=0.5, c="k")
        Hl, _, _ = density(data[e], "h2_hs500_hlpl", weighted)
        nz = Hl[Hl > 0]
        if nz.size:
            lv = np.linspace(np.percentile(nz, CONTOUR_MIN_PCT), Hl.max(), CONTOUR_LEVELS + 1)
            ax.contour(xc, yc, Hl, levels=lv, linewidths=1.5, alpha=0.8, colors="green")
        ax.set(xlim=RANGE, ylim=RANGE, xlabel=r"$h_{500}^*$ [kJ/kg]", ylabel=r"$h_{s}$ [kJ/kg]")
        fig.colorbar(cf, ax=ax, label="events" + (" (cos-wt)" if weighted else ""), pad=0.02)
        ax.legend(handles=[mpatches.Patch(color=plt.get_cmap(cmap)(0.6), label=r"$h_{2m}$ distribution"),
                           mlines.Line2D([], [], color="green", lw=1.5, label=r"$h_{LPL}$ contours")],
                  loc="upper left", fontsize=8)
        ax.set_title(style.EVENT_LABEL[e], fontsize=10)
    fig.suptitle("Surface Layer Correction")
    fig.tight_layout()
    return [{"path": FD.save(fig, out_dir, "surface_layer"), "title": "Surface Layer Correction",
             "facts": {"years": data["T"].label(), "weighting": "cos(lat)" if weighted else "none"},
             "script": "figures/fig_surface_layer.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
