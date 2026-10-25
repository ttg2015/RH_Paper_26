"""Excess vs RH (paper figure): where T and h2m extremes live in the (RH, δ_FT) plane.

Port of Paper_Plots-2 cell 59 onto figure data: smoothed pooled density (purple) with white
density contours, the 50 % / 75 % regions of T and M events, the mean δ_FT, stable/unstable
labels and a count strip. Fixes: the combined count is T + M counts (E09), and RH0 is found
by interpolating the mean's zero crossing (E09/E11).
    python figures/fig_excess_vs_rh.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

PRODUCT = "h2_rh_dFT"
XRANGE, YRANGE = (0.0, 1.0), (-20.0, 20.0)
SIGMA = (4, 3)                  # smoothing in native bins (0.5 % RH, 0.25 kJ/kg) ≈ old σ=2 on 1 %×0.4
CONTOURF_LEVELS = 20
WHITE_LEVELS, WHITE_MIN_PCT = 4, 10
PCT = (0.50, 0.75)
EVENT_LINE = {"T": "darkred", "M": "darkblue"}
MEAN_COARSEN = 2                # mean line on 1 % RH bins
MIN_EVENTS = 20


def plane(years, weighted):
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    d0 = data["T"]
    ex, ey = d0.edges_of(PRODUCT, "x"), d0.edges_of(PRODUCT, "y")
    H = {e: d.h(PRODUCT, weighted=weighted) for e, d in data.items()}
    return data, ex, ey, H


def mean_curve(h, ex, ey):
    hx, exx = FD.coarsen(h, ex, MEAN_COARSEN, axis=0)
    m, n = FD.binned_mean(hx, exx, ey)
    m[n < MIN_EVENTS] = np.nan
    return FD.centers(exx), m, n


def draw_plane(ax, ex, ey, H, labels=True):
    """Density, white contours, T/M mass contours. Returns the contourf for a colorbar."""
    pooled = H["T"] + H["M"]
    hp, cx, cy = FD.crop(pooled, ex, ey, XRANGE, YRANGE)
    xc, yc = FD.centers(cx), FD.centers(cy)
    hs = FD.smooth(hp, SIGMA).T
    cf = ax.contourf(xc, yc, hs / hs.sum(), levels=CONTOURF_LEVELS, cmap="Purples")
    nz = hs[hs > 0]
    lv = np.linspace(np.percentile(nz, WHITE_MIN_PCT), hs.max(), WHITE_LEVELS + 1)
    ax.contour(xc, yc, hs, levels=lv, colors="white", linewidths=1.5, alpha=0.8)
    for e in ("T", "M"):
        he = FD.smooth(FD.crop(H[e], ex, ey, XRANGE, YRANGE)[0], SIGMA).T
        lev = FD.mass_levels(he, PCT)
        order = sorted(lev.items(), key=lambda kv: kv[1])          # ascending density
        cs = ax.contour(xc, yc, he, levels=[v for _, v in order], colors=EVENT_LINE[e],
                        linewidths=2.0, alpha=0.9)
        if labels:
            ax.clabel(cs, cs.levels, fmt={v: f"{int(f * 100)}%" for f, v in order}, fontsize=8, inline=True)
    return cf


def make(out_dir, years=None, weighted=True):
    style.apply()
    data, ex, ey, H = plane(years, weighted)
    d0 = data["T"]
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.05)
    axc = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=axc)
    cf = draw_plane(ax, ex, ey, H)

    x, m, _ = mean_curve(H["T"] + H["M"], ex, ey)
    ax.plot(x, m, color="gray", lw=1.5, marker=".", ms=6, alpha=0.9, zorder=5)
    rh0 = FD.zero_crossing(x[::-1], m[::-1])
    if np.isfinite(rh0):
        ax.plot(rh0, 0, "o", ms=7, mfc="white", mec="k", zorder=6)
        ax.annotate(f"RH$_0$ = {rh0:.3f}", (rh0, 0), xytext=(8, -16), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="k", lw=1.5, ls=":")
    ax.text(0.25, 13.0, "Unstable", fontsize=20, va="center")
    ax.text(0.42, -10.0, "Stable", fontsize=20, va="center")

    # count strip: T, M and their combined counts (E09: counts are added, not the RH values)
    for e, color, lab in (("T", EVENT_LINE["T"], None), ("M", EVENT_LINE["M"], None),
                          ("TM", "purple", "T + M")):
        h = H["T"] + H["M"] if e == "TM" else H[e]
        hx, exx = FD.coarsen(h, ex, MEAN_COARSEN, axis=0)
        n = hx.sum(axis=1)
        axc.step(FD.centers(exx), n, where="mid", color=color, lw=1.5)
    axc.set_ylabel("events" + (" (cos-wt)" if weighted else ""), fontsize=9)
    axc.tick_params(labelbottom=False, labelsize=8)
    axc.set_ylim(bottom=0)
    axc.set_title(fr"$\delta_{{FT}}$ vs RH · {d0.label()} · {style.band_text(FD.load('analysis'), weighted)}",
                  fontsize=10)

    fig.canvas.draw()
    pos = ax.get_position()
    cb = fig.colorbar(cf, cax=fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height]))
    cb.set_label("share of events")
    ax.legend(handles=[mlines.Line2D([], [], color=EVENT_LINE["T"], lw=2, label=r"$T_{2m}$ max"),
                       mlines.Line2D([], [], color=EVENT_LINE["M"], lw=2, label=r"$h_{2m}$ max"),
                       mlines.Line2D([], [], color="gray", lw=1.5, marker=".", ms=6,
                                     label=r"Mean $\delta_{FT}$")], loc="upper left", fontsize=8)
    ax.set(xlim=XRANGE, ylim=YRANGE, xlabel="RH", ylabel=r"$\delta_{FT}$ [kJ/kg]")
    facts = {"RH0 (pooled mean crosses 0)": f"{rh0:.3f}", "years": d0.label(),
             "weighting": "cos(lat)" if weighted else "none",
             "bins": "0.5 % RH × 0.25 kJ/kg, smoothed σ = 2 % RH × 0.75 kJ/kg",
             "contours": "white: density; red/blue: 50 % and 75 % of T / M events"}
    return [{"path": FD.save(fig, out_dir, "excess_vs_rh"), "title": "Excess vs RH", "facts": facts,
             "script": "figures/fig_excess_vs_rh.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
