"""Excess vs RH (paper figure): where T and h2m extremes live in the (RH, δ_FT) plane.

Port of Paper_Plots-2 cell 59 onto figure data: smoothed pooled density (purple) with white
density contours, the 50 % / 75 % regions of T and M events, the mean δ_FT, stable/unstable
labels and a count strip. Fixes: the combined count is T + M counts (E09), and RH0 is found
by interpolating the mean's zero crossing (E09/E11).
Also writes excess_vs_rh_model (paper Fig. 5, old Figs 5 + 11): the same plane with the log-RH model and its
linear form fitted to the mean (fig_excess_vs_rh_entrainment.draw_model) and the fit window; since 2026-11-11
without the count strip (now in fig_delta_parts_vs_rh), the residual strip or the white density contours.
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
EVENT_LINE = dict(style.EVENT_COLOR)                 # brick / navy
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


def draw_plane(ax, ex, ey, H, labels=True, white=True):
    """Density, white contours (white=False drops them), T/M mass contours. Returns the contourf for a colorbar."""
    pooled = H["T"] + H["M"]
    hp, cx, cy = FD.crop(pooled, ex, ey, XRANGE, YRANGE)
    xc, yc = FD.centers(cx), FD.centers(cy)
    hs = FD.smooth(hp, SIGMA).T
    cf = ax.contourf(xc, yc, 100 * hs / hs.sum(), levels=CONTOURF_LEVELS, cmap=style.DENSITY_CMAP)
    if white:
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
            ax.clabel(cs, cs.levels, fmt={v: f"{int(f * 100)}%" for f, v in order}, fontsize=6, inline=True)
    return cf


def count_strip(axc, H, ex):
    """T, M and combined event counts per RH bin (E09: counts are added, not the RH values)."""
    for e, color in (("T", EVENT_LINE["T"]), ("M", EVENT_LINE["M"]), ("TM", "0.35")):
        h = H["T"] + H["M"] if e == "TM" else H[e]
        hx, exx = FD.coarsen(h, ex, MEAN_COARSEN, axis=0)
        axc.step(FD.centers(exx), hx.sum(axis=1) / 1000.0, where="mid", color=color, lw=1.2)
    axc.set_ylabel(r"events [$10^3$]", fontsize=7)
    axc.set_ylim(bottom=0)


def make(out_dir, years=None, weighted=True):
    style.apply()
    data, ex, ey, H = plane(years, weighted)
    return [draw(data, ex, ey, H, out_dir, weighted, model=m) for m in (False, True)]


def draw(data, ex, ey, H, out_dir, weighted=False, model=False):
    d0 = data["T"]
    if model:                                          # one panel: clouds, mean and fits only
        fig, ax = plt.subplots(figsize=(6.8, 3.6))      # full page width (39pc), 2026-11-11
        axc = None
    else:
        fig = plt.figure(figsize=(3.7, 4.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.05)
        axc = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], sharex=axc)
    cf = draw_plane(ax, ex, ey, H, labels=not model, white=not model)

    fit_facts = {}
    if model:                                          # mean as dots, with the fitted models over it
        import fig_excess_vs_rh_entrainment as ENT
        _, fit_facts = ENT.draw_model(ax, None, H, ex, ey)
    else:
        x, m, _ = mean_curve(H["T"] + H["M"], ex, ey)
        ax.plot(x, m, color="gray", lw=1.5, marker=".", ms=6, alpha=0.9, zorder=5)
    rh0 = FD.rh0(data)                                 # unweighted, same number every figure uses
    if np.isfinite(rh0):
        ax.plot(rh0, 0, "o", ms=7, mfc="white", mec="k", zorder=6)
        ax.annotate(f"RH$_0$ = {100 * rh0:.1f}%", (rh0, 0), xytext=(8, -16), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="k", lw=1.5, ls=":")
    ax.text(0.25, 13.0, "Unstable", fontsize=11, va="center")
    ax.text(*((0.26, -17.0) if model else (0.42, -10.0)), "Stable", fontsize=11, va="center")   # model: clear of the legend

    if axc is not None:
        count_strip(axc, H, ex)
        axc.tick_params(labelbottom=False, labelsize=7)

    if model:
        cb = fig.colorbar(cf, ax=ax, pad=0.015, fraction=0.03)
    else:
        fig.canvas.draw()
        pos = ax.get_position()
        cb = fig.colorbar(cf, cax=fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height]))
    cb.set_label("events per bin [%]")
    handles = [mlines.Line2D([], [], color=EVENT_LINE["T"], lw=2, label=style.EVENT_SHORT["T"]),
               mlines.Line2D([], [], color=EVENT_LINE["M"], lw=2, label=style.EVENT_SHORT["M"]),
               mlines.Line2D([], [], color="gray", lw=0 if model else 1.5, marker=".", ms=6,
                             label=r"Mean $\delta_{a}$")]
    if model:
        handles += [h for h in ax.get_legend_handles_labels()[0] if h.get_label().startswith("$\\Gamma")]
    ax.legend(handles=handles, loc="lower right" if model else "upper left", fontsize=6,   # lower right is empty (dry, unstable) framealpha=0.85 if model else None,
              edgecolor="none" if model else None)
    ax.set(xlim=XRANGE, ylim=YRANGE, ylabel=r"$\delta_{a}$ " + style.KJ)
    style.rh_percent(ax)
    if model:
        fig.tight_layout()
    facts = {"RH0 (pooled mean crosses 0)": f"{rh0:.3f}", "years": d0.label(),
             "weighting": "cos(lat)" if weighted else "none",
             "bins": "0.5 % RH × 0.25 kJ/kg, smoothed σ = 2 % RH × 0.75 kJ/kg",
             "contours": "white: density; red/blue: 50 % and 75 % of T / M events"}
    facts.update(fit_facts)
    name = "excess_vs_rh_model" if model else "excess_vs_rh"
    path = FD.save(fig, out_dir, name)
    plt.close(fig)
    return {"path": path, "title": "Excess vs RH + model (old Figs 5 + 11)" if model else "Excess vs RH",
            "facts": facts, "script": "figures/fig_excess_vs_rh.py"}


if __name__ == "__main__":
    FD.cli(make, __doc__)
