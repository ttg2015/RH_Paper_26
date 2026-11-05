"""Strike points: where do entraining plumes from the LPL top out?

From stage f's plume tops, summed in stage g (strike_* products), pooled over years.
Figure 1 (strike_points):
  (a, b) for T and M events: ε × plume-top pressure, colour = share of the unstable events (buoyant
         at ε = 0) whose top lands in each 10 hPa bin, so columns fade as plumes dilute out.
         400–600 hPa window, median top (highest crossing, solid) and median first crossing
         (dashed); % of unstable events still buoyant on top.
  (c)    top distributions at ε_detail for T and M: highest crossing (solid) vs first (dashed).
  (d)    where plumes top out vs surface RH at ε_detail (T + M), with the median top per RH bin.
Figure 2 (strike_map): mean plume-top pressure at ε_detail per land cell, T and M.
    python figures/fig_strike_points.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe_
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import maps
import style

P_COARSEN = 2                    # 5 → 10 hPa for the distributions
WINDOW = None                    # read from config/analysis.yaml
HALO = [pe_.withStroke(linewidth=3.6, foreground="#1b1b1b")]   # white lines readable on light and dark bins


def quantile_p(h, pc, q):
    """Pressure at cumulative fraction q of a top-distribution (counts over pressure bins)."""
    c = np.cumsum(h)
    if c[-1] == 0:
        return np.nan
    return float(np.interp(q * c[-1], c, pc))


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    d0 = data["T"]
    if "strike_ptop" not in d0.hist:
        raise FileNotFoundError("no strike_* products: run stage f, then stage g")
    lo, hi = FD.load("analysis")["ztop_window_hPa"]
    eps = d0.epsilon
    ed = FD.load("figdata")["strike"]["eps_detail"]
    kd = int(np.argmin(abs(eps - ed)))
    suf = "_w" if weighted else ""
    pe = d0.edges["ptop"]
    facts = {}

    fig = plt.figure(figsize=(12, 9.6))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)
    for j, e in enumerate(("T", "M")):
        d = data[e]
        ax = fig.add_subplot(gs[0, j])
        h = d.hist["strike_ptop" + suf][d.subbands.index("all")]           # (ε, p)
        hf = d.hist["strike_ptop_first" + suf][d.subbands.index("all")]
        cat = d.hist["strike_category"][d.subbands.index("all")]          # (ε, code)
        total, stable, _ = FD.unstable_counts(cat)                       # unstable events per ε
        hc, pc_e = FD.coarsen(h, pe, P_COARSEN, axis=1)
        share = hc / np.where(total[:, None] > 0, total[:, None], np.nan)
        x_e = np.concatenate([[eps[0] - (eps[1] - eps[0]) / 2], (eps[1:] + eps[:-1]) / 2,
                              [eps[-1] + (eps[-1] - eps[-2]) / 2]])
        style.paper_axes(ax, minor=False)
        im = ax.pcolormesh(x_e, pc_e, share.T * 100, cmap=style.DENSITY_CMAP, shading="flat", vmin=0,
                           rasterized=True)                      # empty bins show the palest colour, not white gaps
        cb = fig.colorbar(im, ax=ax, label="% of unstable events per 10 hPa", pad=0.02)
        cb.outline.set_edgecolor(style.SPINE)
        pc = FD.centers(pe)
        med = [quantile_p(h[k], pc, 0.5) for k in range(len(eps))]
        medf = [quantile_p(hf[k], pc, 0.5) for k in range(len(eps))]
        ax.plot(eps, med, "w-o", ms=4.5, lw=2.2, mec="#1b1b1b", mew=0.8, label="median top", path_effects=HALO)
        ax.plot(eps, medf, "w--", lw=1.6, label="median first crossing", path_effects=HALO)
        for p in (lo, hi):
            ax.axhline(p, color=style.WINDOW_COLOR, lw=1.6, ls=(0, (5, 3)))
        buoy = 1 - FD.unstable_counts(cat)[2][5]                        # not diluted out
        from matplotlib.transforms import blended_transform_factory
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        for k, b in enumerate(buoy):
            ax.text(eps[k], 1.01, f"{100 * b:.0f}%", transform=tr, ha="center", va="bottom", fontsize=7, color=style.SPINE,
                    rotation=90 if len(eps) > 12 else 0)
        ax.set(ylim=(1000, 100), xlabel="ε [km⁻¹]", ylabel="plume top [hPa]")
        ax.set_title(f"({'ab'[j]}) {style.EVENT_LABEL[e]} · {100 * stable[0] / (stable[0] + total[0]):.0f}% stable",
                     pad=16, fontweight="bold", loc="left")
        ax.legend(fontsize=7.5, loc="lower right", facecolor="white", framealpha=0.85, edgecolor="none")
        frac_window = h[kd][(pc >= lo) & (pc <= hi)].sum() / total[kd] if total[kd] else np.nan
        facts[f"{e}: median top at ε={eps[kd]:g}"] = f"{med[kd]:.0f} hPa (first crossing {medf[kd]:.0f} hPa)"
        facts[f"{e}: share of unstable events topping in {lo}–{hi} hPa at ε={eps[kd]:g}"] = f"{100 * frac_window:.0f}%"

    # (c) distributions at ε_detail
    ax = style.paper_axes(fig.add_subplot(gs[1, 0]))
    for e, d in data.items():
        for var, ls in (("strike_ptop", "-"), ("strike_ptop_first", "--")):
            h = d.hist[var + suf][d.subbands.index("all"), kd]
            hc, pc_e = FD.coarsen(h, pe, P_COARSEN)
            xs = 100 * hc / max(hc.sum(), 1)
            if ls == "-":
                ax.fill_betweenx(FD.centers(pc_e), 0, xs, color=style.EVENT_COLOR[e], alpha=0.12, lw=0)
            ax.plot(xs, FD.centers(pc_e), color=style.EVENT_COLOR[e], ls=ls, lw=2.4 if ls == "-" else 1.5,
                    label=f"{style.EVENT_LABEL[e].split(' extremes')[0]} {'top' if var == 'strike_ptop' else 'first crossing'}")
    ax.axhspan(hi, lo, color=style.WINDOW_COLOR, alpha=0.08, lw=0)
    for p in (lo, hi):
        ax.axhline(p, color=style.WINDOW_COLOR, lw=1.2, ls=(0, (5, 3)))
    ax.set(ylim=(1000, 100), xlabel="% of buoyant plumes per 10 hPa", ylabel="plume top [hPa]")
    ax.set_xlim(left=0)
    ax.set_title(f"(c) Top distributions at ε = {eps[kd]:g} km⁻¹", fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")

    # (d) RH × top
    ax = style.paper_axes(fig.add_subplot(gs[1, 1]), minor=False)
    h2 = sum(d.hist["strike_rh_ptop"] for d in data.values())
    ex = d0.edges["rh"]
    h2, ey = FD.coarsen(h2, pe, P_COARSEN, axis=1)
    im = ax.pcolormesh(ex, ey, np.ma.masked_equal(h2.T, 0), cmap=style.DENSITY_CMAP, shading="flat", rasterized=True)
    cb = fig.colorbar(im, ax=ax, label="events", pad=0.02)
    cb.outline.set_edgecolor(style.SPINE)
    pcs = FD.centers(ey)
    med = np.array([quantile_p(row, pcs, 0.5) if row.sum() >= 20 else np.nan for row in h2])   # 0.5 % RH bins
    ax.plot(FD.centers(ex), med, color="white", lw=2.2, label="median top", path_effects=HALO)
    for p in (lo, hi):
        ax.axhline(p, color=style.WINDOW_COLOR, lw=1.6, ls=(0, (5, 3)))
    ax.set(xlim=(0, 1), ylim=(1000, 100), xlabel="RH$_{2m}$", ylabel="plume top [hPa]")
    ax.set_title(f"(d) Plume tops vs surface RH (T + M, ε = {eps[kd]:g} km⁻¹)", fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, loc="lower right", facecolor="white", framealpha=0.85, edgecolor="none")
    # years, band, weighting and mask are in the caption
    facts.update(years=d0.label(), weighting="cos(lat)" if weighted else "none",
                 note="share in (a, b) is of the unstable events (buoyant at ε = 0), so it fades where plumes dilute out")
    items = [{"path": FD.save(fig, out_dir, "strike_points"), "title": "Plume strike points", "facts": facts,
              "script": "figures/fig_strike_points.py"}]
    plt.close(fig)

    # map of mean top
    band = FD.load("analysis")["lat_band"]
    fig, axes = maps.grid(1, 2, (18, 5))
    for j, e in enumerate(("T", "M")):
        m = data[e].map_mean("strike_ptop", masked=False)
        im = maps.draw(axes[0][j], data[e].lon, data[e].lat, m, band=band, labels=(j == 0),
                       cmap="viridis_r", vmin=200, vmax=900)
        axes[0][j].set_title(f"Mean plume top · {style.EVENT_LABEL[e]} · ε = {eps[kd]:g}", fontsize=12)
        fig.colorbar(im, ax=axes[0][j], shrink=0.8, pad=0.02, label="hPa")
    items.append({"path": FD.save(fig, out_dir, "strike_map"), "title": "Plume strike points · map",
                  "facts": {"eps": f"{eps[kd]:g} km⁻¹", "years": d0.label(),
                            "cells": "land in band, low terrain, years with a buoyant plume"},
                  "script": "figures/fig_strike_points.py"})
    plt.close(fig)
    return items


if __name__ == "__main__":
    FD.cli(make, __doc__)
