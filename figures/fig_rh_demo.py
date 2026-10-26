"""RH Conditioning Procedure (paper figure): how events are selected and what RH does to the profile.

Port of Paper_Plots-2 cell 62 (RH_profiles_diagnostic.pdf), T events:
(a) (h*500, h500) density with the selected free-tropospheric box; (b) RH distribution of events
inside the box with the two RH bins shaded; (c) composite h (solid) and h* (dashed, ≤ 600 hPa) for
the two bins. Box and bins as in the old figure; one mask for every panel (fixes E10).
    python figures/fig_rh_demo.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import figdata as FD
import style

CASE = "T_TPL"
RH_LOW, RH_HIGH = (0.45, 0.50), (0.50, 0.55)
COLOR_LOW, COLOR_HIGH = "#c0392b", "#6c3483"
RANGE = (310, 360)


def make(out_dir, years=None, weighted=True):
    style.apply()
    d = FD.FigData("T", years)
    box = FD.load("figdata")["cascade"]["boxes"][CASE]
    fig, ax = plt.subplots(1, 3, figsize=(19 / 2.54 * 1.6, 6.5 / 2.54 * 1.6),
                           gridspec_kw=dict(wspace=0.55, left=0.08, right=0.97, top=0.85, bottom=0.18))
    # (a)
    name = "h2_hs500_h500"
    h = d.h(name, weighted=weighted)
    h, ex, ey = FD.crop(h, d.edges_of(name, "x"), d.edges_of(name, "y"), RANGE, RANGE)
    im = ax[0].pcolormesh(ex, ey, np.ma.masked_equal(h.T, 0), cmap="Reds")
    fig.colorbar(im, ax=ax[0], label="events", pad=0.02)
    (x0, x1), (y0, y1) = box["h_star"], box["h"]
    ax[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, lw=1.0, ec="0.25", fc="0.55", alpha=0.35, ls="--"))
    ax[0].set(xlabel=r"$h_{500}^{*}$ (kJ kg$^{-1}$)", ylabel=r"$h_{500}$ (kJ kg$^{-1}$)", title="2D Histogram")
    # (b)
    e = d.edges["rh_cascade"]
    cnt = d.hist["casc_count"][d.cases.index(CASE)]
    ax[1].bar(FD.centers(e), cnt, width=np.diff(e), color="steelblue", alpha=0.8, lw=0)
    for (lo, hi), col in ((RH_HIGH, COLOR_HIGH), (RH_LOW, COLOR_LOW)):
        ax[1].axvspan(lo, hi, color=col, alpha=0.18)
        for v in (lo, hi):
            ax[1].axvline(v, color=col, lw=1.2, ls="--")
    ax[1].set(xlabel=r"RH$_{2\,\mathrm{m}}$", ylabel="Counts", title="RH Distribution\nwithin Selected Region", xlim=(0, 1))
    # (c)
    facts = {}
    for (lo, hi), col, lab in ((RH_LOW, COLOR_LOW, "Low"), (RH_HIGH, COLOR_HIGH, "High")):
        c = FD.composite(d, CASE, lo, hi)
        ax[2].plot(c["MSE"], d.levels, color=col, lw=1.6)
        ax[2].plot(np.where(d.levels <= 600, c["MSE_sat"], np.nan), d.levels, color=col, lw=1.6, ls="--")
        facts[f"{lab} RH {lo:.2f}–{hi:.2f}"] = f"{int(c['n']):,} events"
    ax[2].set(ylim=(950, 100), xlim=(320, 350), xlabel=r"$h$ (kJ kg$^{-1}$)", ylabel="Pressure (hPa)",
              title="MSE Profiles by RH Bin")
    ax[2].legend(handles=[plt.Line2D([0], [0], color="0.2", lw=1.5, label=r"$h$ (environmental)"),
                          plt.Line2D([0], [0], color="0.2", lw=1.5, ls="--", label=r"$h^{*}$ (saturated)"),
                          plt.Line2D([0], [0], color=COLOR_HIGH, lw=1.5, label=f"High RH ({RH_HIGH[0]:.2f}–{RH_HIGH[1]:.2f})"),
                          plt.Line2D([0], [0], color=COLOR_LOW, lw=1.5, label=f"Low RH ({RH_LOW[0]:.2f}–{RH_LOW[1]:.2f})")],
                 fontsize=7, loc="upper left")
    for a, l in zip(ax, "abc"):
        a.text(-0.14, 1.04, f"({l})", transform=a.transAxes, fontweight="bold", va="bottom")
    facts.update(box=f"{CASE}: h* {box['h_star']}, h {box['h']}", years=d.label(),
                 mask="one mask for all panels (land, band, terrain kept; below-ground levels dropped)")
    return [{"path": FD.save(fig, out_dir, "rh_demo"), "title": "RH Conditioning Procedure", "facts": facts,
             "script": "figures/fig_rh_demo.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
