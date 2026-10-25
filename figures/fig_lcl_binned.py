"""LCL-Binned Decomposition (paper figure): δ_FT = δ_below + δ_above against LCL height and RH.

Port of Paper_Plots-2 cells 135–137 (binned_combined.pdf), T and M unstable events pooled:
(a) RH_2m × z_LCL,2m density with an RH count strip; (b) mean δ_FT, δ_below, δ_above vs z_LCL,LPL
(100 m bins) with ±1 std (light) and ±1 standard error (dark) bands and a count strip; (c) the same
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
COLORS = {"delta_FT": "black", "delta_below": "blue", "delta_above": "red"}
LABELS = {"delta_FT": r"$\delta_\mathrm{FT}$", "delta_below": r"$\delta_\mathrm{below}$",
          "delta_above": r"$\delta_\mathrm{above}$"}
LW, ALPHA_STD, ALPHA_STERR = 3.2, 0.12, 0.35
SCATTER_CMAP = "Blues"


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


def style_hist(ax):
    ax.tick_params(labelbottom=False, left=False, labelleft=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylabel("N", fontsize=7, labelpad=2)


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="0.5", lw=0.7, ls="--")


def lines(ax, ax_h, x, mean, std, se, cnt, width, quantities, xlabel):
    for k, q in enumerate(quantities):
        ax.plot(x, mean[:, k], color=COLORS[q], lw=LW, label=LABELS[q])
        ax.fill_between(x, mean[:, k] - std[:, k], mean[:, k] + std[:, k], color=COLORS[q], alpha=ALPHA_STD)
        ax.fill_between(x, mean[:, k] - se[:, k], mean[:, k] + se[:, k], color=COLORS[q], alpha=ALPHA_STERR)
    style_ax(ax)
    ax.set(xlabel=xlabel, ylabel="MSE anomaly (kJ kg$^{-1}$)")
    ax.legend(loc="upper right")
    ax_h.bar(x, cnt, width=width * 0.9, color="0.5", lw=0)
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
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[0.18, 1], hspace=0.04, wspace=0.32)
    axh = [fig.add_subplot(gs[0, k]) for k in range(3)]
    ax = [fig.add_subplot(gs[1, k]) for k in range(3)]

    # (a) RH × z_LCL,2m density of unstable events
    name = "h2_rh_zlcl2m_unstable"
    h = pooled(data, name)
    ex, ey = d0.edges_of(name, "x"), d0.edges_of(name, "y")
    h, ex = FD.coarsen(h, ex, 2, axis=0)
    h, ey = FD.coarsen(h, ey, 4, axis=1)
    im = ax[0].pcolormesh(ex * 100, ey, h.T, cmap=SCATTER_CMAP, rasterized=True)
    fig.colorbar(im, ax=ax[0], label="Count", pad=0.02, shrink=0.85)
    ax[0].set(xlabel=r"RH$_{2\mathrm{m}}$ (%)", ylabel=r"$z_{\mathrm{LCL,2m}}$ (m)")
    ax[0].spines[["top", "right"]].set_visible(False)
    nrh = h.sum(axis=1)
    axh[0].bar(FD.centers(ex) * 100, nrh, width=np.diff(ex)[0] * 100 * 0.9, color="0.5", lw=0)
    style_hist(axh[0])
    xr_a = trim(FD.centers(ex) * 100, nrh)
    zr = trim(FD.centers(ey), h.sum(axis=0))
    ax[0].set(xlim=xr_a, ylim=zr)
    axh[0].set_xlim(xr_a)

    # (b) vs z_LCL,LPL
    x, mean, std, se, cnt, e = stats(pooled(data, "lb_zLCL_LPL"), d0.edges["zlcl"], Z_COARSEN)
    lines(ax[1], axh[1], x, mean, std, se, cnt, np.diff(e)[0], quantities, r"$z_{\mathrm{LCL,LPL}}$ (m)")
    xr_b = trim(x, cnt)
    ax[1].set_xlim(xr_b); axh[1].set_xlim(xr_b)

    # (c) vs RH
    x, mean2, std2, se2, cnt2, e2 = stats(pooled(data, "lb_RH_2m"), d0.edges["rh"], RH_COARSEN)
    lines(ax[2], axh[2], x * 100, mean2, std2, se2, cnt2, np.diff(e2)[0] * 100, quantities, r"RH$_{2\mathrm{m}}$ (%)")
    xr_c = trim(x * 100, cnt2)
    ax[2].set_xlim(xr_c); axh[2].set_xlim(xr_c)

    for a, lbl in zip(ax, ["(a)", "(b)", "(c)"]):
        a.text(0.02, 0.97, lbl, transform=a.transAxes, fontsize=10, fontweight="bold", va="top")
    eps = FD.load("figdata")["convective"]["eps"]
    fig.suptitle(f"T + M combined · unstable events · ε = {eps:g} km⁻¹ · {d0.label()} · "
                 f"{style.band_text(FD.load('analysis'))}", fontsize=12, y=1.01)
    n = int(cnt.sum())
    facts = {"unstable events": f"{n:,}", "eps": f"{eps:g} km⁻¹", "years": d0.label(),
             "z_LCL": "above ground (E07 corrected); LPL parcel for (b), 2 m parcel for (a)",
             "bands": "light = ±1 std, dark = ±1 standard error"}
    return [{"path": FD.save(fig, out_dir, "lcl_binned"), "title": "LCL-Binned Decomposition",
             "facts": facts, "script": "figures/fig_lcl_binned.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
