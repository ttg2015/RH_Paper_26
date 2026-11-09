"""LCL-Binned Decomposition (paper figure): δ_FT = δ_below + δ_above against LCL height and RH.

Port of Paper_Plots-2 cells 135–137 (binned_combined.pdf), T and M unstable events pooled:
(a) RH_2m × z_LCL,LPL density (z_LCL,2m for stage g files before 2026-11-02) with an RH count strip; (b) mean δ_FT, δ_below, δ_above vs z_LCL,LPL
(100 m bins) with the 25–75th percentile range shaded (from the lbh_* distributions; ±1 standard
error for older stage g files) and a count strip; (c) the same
vs RH_2m (1 % bins). Heights are above ground (E07 corrected). ε from config/figdata.yaml.
    python figures/fig_lcl_binned.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

Z_COARSEN, RH_COARSEN = 4, 2         # 25 m → 100 m, 0.5 % → 1 %
MIN_PER_BIN = 5
COLORS = {"delta_FT": "#1F2937", "delta_below": "#8C7346", "delta_above": "#0A9396"}   # δ_above the bright one
LABELS = {"delta_FT": r"$\delta_\mathrm{a}$", "delta_below": r"$\delta_\mathrm{below}$",
          "delta_above": r"$\delta_\mathrm{above}$"}
LW, ALPHA_IQR, ALPHA_STERR = 2.4, 0.16, 0.35
QUANTILES = (0.25, 0.75)
SHOW_ZLCL_PANEL = False              # the δ vs z_LCL panel; dropped for the advisor's storyline (new Fig 9)
STRIP = "#9CA3AF"


def pooled(data, name):
    return sum(d.hist[name] for d in data.values())


def stats(lb, edges, factor):
    """Coarsen (bin, quantity, stat) sums and return centres, mean, std, sterr, count per quantity."""
    s, e = FD.coarsen(lb, edges, factor, axis=0)
    n, tot, sq = s[..., 0], s[..., 1], s[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = tot / n
        std = np.sqrt(np.maximum(sq / n - mean ** 2, 0))
        se = std / np.sqrt(n)
    bad = n < MIN_PER_BIN
    for a in (mean, std, se):
        a[bad] = np.nan
    return FD.centers(e), mean, std, se, n[:, 0], e


def quantile_band(lbh, edges, factor, d_edges, qs=QUANTILES):
    """(bin, quantity, q) quantiles from per-bin counts over δ bins, coarsened like stats()."""
    c, _ = FD.coarsen(lbh, edges, factor, axis=0)
    cum = np.cumsum(c, axis=2)
    out = np.full(c.shape[:2] + (len(qs),), np.nan)
    for i in range(c.shape[0]):
        for k in range(c.shape[1]):
            tot = cum[i, k, -1]
            if tot >= MIN_PER_BIN:
                out[i, k] = np.interp(np.array(qs) * tot, cum[i, k], d_edges[1:])
    return out


def style_hist(ax):
    ax.tick_params(labelbottom=False, left=False, labelleft=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylabel("N", fontsize=7, labelpad=2, color=STRIP)


def style_ax(ax):
    style.paper_axes(ax)
    ax.axhline(0, color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=1)


def lines(ax, ax_h, x, mean, band, se, cnt, width, quantities, xlabel):
    """Means with the 25–75th percentile band (band: (bin, quantity, 2)), or ±1 standard error if band is None."""
    style_ax(ax)
    for k, q in enumerate(quantities):
        if band is not None:
            ax.fill_between(x, band[:, k, 0], band[:, k, 1], color=COLORS[q], alpha=ALPHA_IQR, lw=0)
        else:
            ax.fill_between(x, mean[:, k] - se[:, k], mean[:, k] + se[:, k], color=COLORS[q], alpha=ALPHA_STERR, lw=0)
        ax.plot(x, mean[:, k], color=COLORS[q], lw=LW, label=LABELS[q], solid_capstyle="round")
    ax.set(xlabel=xlabel, ylabel="excess " + style.KJ)
    ax.legend(loc="upper right", frameon=False)
    ax_h.bar(x, cnt, width=width * 0.9, color=STRIP, lw=0)
    style_hist(ax_h)


def trim(x, cnt):
    """x-range covering bins that have events."""
    nz = np.nonzero(cnt >= MIN_PER_BIN)[0]
    return (x[nz[0]], x[nz[-1]]) if nz.size else (x[0], x[-1])


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    d0 = data["T"]
    quantities = ["delta_FT", "delta_below", "delta_above"]
    ncol = 3 if SHOW_ZLCL_PANEL else 2
    fig = plt.figure(figsize=(14 if SHOW_ZLCL_PANEL else 9.6, 6))
    gs = gridspec.GridSpec(2, ncol, figure=fig, height_ratios=[0.18, 1], hspace=0.04, wspace=0.32)
    axh = [fig.add_subplot(gs[0, k]) for k in range(ncol)]
    ax = [fig.add_subplot(gs[1, k]) for k in range(ncol)]
    if not SHOW_ZLCL_PANEL:                                       # (b) is computed but not drawn
        axh.insert(1, None)
        ax.insert(1, None)

    # (a) RH × z_LCL density of unstable events: mixed-layer (LPL) air, as in (b); older stage g
    # files only have the 2 m parcel
    ml = all("h2_rh_zlclLPL_unstable" in d.hist for d in data.values())
    name = "h2_rh_zlclLPL_unstable" if ml else "h2_rh_zlcl2m_unstable"
    h = pooled(data, name)
    ex, ey = d0.edges_of(name, "x"), d0.edges_of(name, "y")
    h, ex = FD.coarsen(h, ex, 2, axis=0)
    h, ey = FD.coarsen(h, ey, 4, axis=1)
    style.paper_axes(ax[0], minor=False)
    im = ax[0].pcolormesh(ex * 100, ey, h.T, cmap=style.DENSITY_CMAP, rasterized=True)
    cb = fig.colorbar(im, ax=ax[0], label="events", pad=0.02, shrink=0.85)
    cb.outline.set_edgecolor(style.SPINE)
    ax[0].set(xlabel=style.RH_LABEL, ylabel="LCL of mixed-layer air [m]" if ml else "LCL of 2-m air [m]")
    nrh = h.sum(axis=1)
    axh[0].bar(FD.centers(ex) * 100, nrh, width=np.diff(ex)[0] * 100 * 0.9, color=STRIP, lw=0)
    style_hist(axh[0])
    xr_a = trim(FD.centers(ex) * 100, nrh)
    zr = trim(FD.centers(ey), h.sum(axis=0))
    ax[0].set(xlim=xr_a, ylim=zr)
    axh[0].set_xlim(xr_a)

    # (b) vs z_LCL,LPL
    has_q = all("lbh_zLCL_LPL" in d.hist and "lbh_RH_2m" in d.hist for d in data.values())
    de = d0.edges["delta"]
    x, mean, std, se, cnt, e = stats(pooled(data, "lb_zLCL_LPL"), d0.edges["zlcl"], Z_COARSEN)
    band = quantile_band(pooled(data, "lbh_zLCL_LPL"), d0.edges["zlcl"], Z_COARSEN, de) if has_q else None
    if SHOW_ZLCL_PANEL:
        lines(ax[1], axh[1], x, mean, band, se, cnt, np.diff(e)[0], quantities, r"$z_{\mathrm{LCL}}$ of mixed-layer air (m)")
        xr_b = trim(x, cnt)
        ax[1].set_xlim(xr_b); axh[1].set_xlim(xr_b)

    # (c) vs RH
    x, mean2, std2, se2, cnt2, e2 = stats(pooled(data, "lb_RH_2m"), d0.edges["rh"], RH_COARSEN)
    band2 = quantile_band(pooled(data, "lbh_RH_2m"), d0.edges["rh"], RH_COARSEN, de) if has_q else None
    lines(ax[2], axh[2], x * 100, mean2, band2, se2, cnt2, np.diff(e2)[0] * 100, quantities, style.RH_LABEL)
    xr_c = trim(x * 100, cnt2)
    ax[2].set_xlim(xr_c); axh[2].set_xlim(xr_c)

    for k, a in enumerate(a for a in ax if a is not None):
        style.panel_letter(a, k)
    eps = FD.load("figdata")["convective"]["eps"]                 # years, band and ε are in the caption
    n = int(cnt.sum())
    facts = {"unstable events": f"{n:,}", "eps": f"{eps:g} km⁻¹", "years": d0.label(),
             "z_LCL": "above ground (E07 corrected); LPL parcel for (b) and " + ("(a)" if ml else "2 m parcel for (a): rerun stage g"),
             "bands": "25–75th percentiles" if has_q else "±1 standard error (rerun stage g for percentile bands)"}
    return [{"path": FD.save(fig, out_dir, "lcl_binned"), "title": "LCL-Binned Decomposition",
             "facts": facts, "script": "figures/fig_lcl_binned.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
