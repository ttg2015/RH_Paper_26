"""RH cascades by regime (paper figure): composite MSE profiles binned by surface RH, T and wet-bulb pooled.

Three free-tropospheric regimes in the (h*500, h500) plane, with temperature and wet-bulb extremes pooled
in each: middle latitudes, tropics with a drier free troposphere, tropics with a moister one. The regimes
are the 3 kJ/kg grid cells of side_tests/cascade_composites.py set in side_tests/plume_top_vs_rh.REGIMES,
so this figure and the plume-top check use the same events.

(a) where the pooled events sit in the (h*500, h500) plane, with the regimes outlined: the 0.5 kJ/kg
    figdata histogram h2_hs500_h500 (T + M, unweighted, all terrain), or the 3 kJ/kg composite grid if no
    figdata files are found.
(b–d) one cascade per regime: h solid, h* dashed (at and above 600 hPa), one color per 5 % RH bin, and a
dot at each bin's mean LCL so the mixed layer can be seen to shallow. Levels are solid where at least
COVERAGE_MIN of the bin's events have them above ground and faded down to COVERAGE_FAINT, as in
fig_cascade.py. Plain (unweighted) means, terrain below 0.5 km.

Reads   <work>/side_tests/casc/casc_{E}_{Y}.npz
    python figures/fig_cascade_regimes.py [--years 1980-2019] [--casc DIR]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import figdata as FD
import style

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "side_tests"))
import cascade_composites as CC                               # noqa: E402
import cascade_tests as T                                     # noqa: E402
from plume_top_vs_rh import REGIMES, cells, pooled            # noqa: E402

WEIGHTED = False
MIN_COUNT = 500
SAT_BELOW = 600.0                                             # h* drawn at and above this level
COVERAGE_MIN, COVERAGE_FAINT = 0.9, 0.5
# RH colors: YlGnBu without its palest end, so the driest bins stay visible on the pale background
CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list("rh", plt.cm.YlGnBu(np.linspace(0.3, 1.0, 256)))
PLANE_CMAP = "Greys"                                          # neutral, so the colored regime boxes stand out
PLANE_KEEP = 0.002                                            # crop (a) to cells above this share of the peak
REGIME_COLOR = ["#7C3AED", "#D97706", "#0F766E"]              # box outlines in (a), and panel titles
PLANE = (312.0, 348.0), (305.0, 341.0)
X_FROM_HPA = 200.0                                            # x-limits from levels at or below this (the casc
                                                              # levels reach 1 hPa, where MSE is far larger)


def plane_counts(arrays):
    """Pooled event counts on the cascade_composites grid: (h* cells, h cells)."""
    ni, nj = len(CC.GRID_HS) - 1, len(CC.GRID_H) - 1
    out = np.zeros((ni, nj))
    for a in arrays:
        names = list(a["names"])
        for i in range(ni):
            for j in range(nj):
                out[i, j] += a["n"][names.index(f"g{i:02d}_{j:02d}")].sum()
    return out


def plane_limits(cnt, ex, ey, keep=PLANE_KEEP, pad=1.5):
    """Axis limits of (a): the cells holding events (above `keep` of the peak) and the regime boxes."""
    i, j = np.nonzero(cnt > keep * np.nanmax(cnt))
    xs = [ex[i.min()], ex[i.max() + 1]] + [v for hs, _ in REGIMES.values() for v in hs]
    ys = [ey[j.min()], ey[j.max() + 1]] + [v for _, h in REGIMES.values() for v in h]
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def coverage(arrays, names):
    """Share of each (RH bin, level)'s events with the level above ground, pooled."""
    n = sum(a["n"][[list(a["names"]).index(x) for x in names]].sum(0) for a in arrays)
    w = sum(a["pw_MSE"][[list(a["names"]).index(x) for x in names]].sum(0) for a in arrays)
    with np.errstate(invalid="ignore", divide="ignore"):
        return w / n[:, None]


def fine_plane(years):
    """Pooled T + M (h*500, h500) counts at 0.5 kJ/kg from the figdata files, or None if there are none."""
    try:
        data = [FD.FigData(e, years) for e in ("T", "M")]
    except FileNotFoundError:
        return None
    name = "h2_hs500_h500"
    if not all(name in d.hist for d in data):
        return None
    h = sum(d.h(name, weighted=False) for d in data)
    return FD.crop(h, data[0].edges_of(name, "x"), data[0].edges_of(name, "y"), *PLANE)


def draw(arrays, out_dir, label="", plane=None):
    style.apply()
    edges = arrays[0]["rh_edges"]
    norm = BoundaryNorm(edges, CMAP.N)
    fig = plt.figure(figsize=(15.5, 5.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.82, 0.16, 1, 1, 1], wspace=0.08)   # column 1 is a spacer

    ax = style.paper_axes(fig.add_subplot(gs[0, 0]))            # (a) selection plane
    if plane is not None:
        cnt, ex, ey = plane
    else:
        cnt, ex, ey = plane_counts(arrays), CC.GRID_HS, CC.GRID_H
    ax.pcolormesh(ex, ey, np.ma.masked_equal(cnt.T, 0), cmap=PLANE_CMAP, shading="flat", rasterized=True,
                  vmin=0, vmax=np.nanmax(cnt) * 1.05)
    xl, yl = plane_limits(cnt, ex, ey)
    lo, hi = min(xl[0], yl[0]), max(xl[1], yl[1])
    ax.plot((lo, hi), (lo, hi), color="k", lw=0.8, alpha=0.5)
    for k, (name, (hs, h)) in enumerate(REGIMES.items()):
        ax.add_patch(Rectangle((hs[0], h[0]), hs[1] - hs[0], h[1] - h[0], fill=False, lw=2.2, ec=REGIME_COLOR[k]))
        ax.text(hs[1] + 0.4, h[1] + 0.3, "bcd"[k], color=REGIME_COLOR[k], fontweight="bold", fontsize=10)
    ax.set(xlim=xl, ylim=yl, xlabel=r"$h^*_{500}$ (kJ kg$^{-1}$)", ylabel=r"$h_{500}$ (kJ kg$^{-1}$)")
    ax.set_title("Events, T + wet-bulb", fontsize=11)
    style.panel_letter(ax, 0)

    facts, axes = {}, []
    for k, (name, (hs, h)) in enumerate(REGIMES.items()):
        ax = style.paper_axes(fig.add_subplot(gs[0, k + 2], sharey=axes[0] if axes else None))
        axes.append(ax)
        names = cells(hs, h)
        c = pooled(arrays, names)
        cov = coverage(arrays, names)
        lev = np.asarray(c["levels"], float)
        ax.axhspan(600, 400, color=style.WINDOW_COLOR, alpha=0.07, lw=0, zorder=0)
        xs, kept = [], 0
        for b in range(len(c["n"])):
            if c["n"][b] < MIN_COUNT:
                continue
            col = CMAP(norm(c["rh_centers"][b]))
            solid = cov[b] >= COVERAGE_MIN
            faint = (cov[b] >= COVERAGE_FAINT) & ~solid
            if solid.any():
                faint[np.nonzero(solid)[0].max()] = True              # join at the deepest solid level
            ax.plot(np.where(solid, c["MSE"][b], np.nan), lev, color=col, lw=1.8, zorder=3)
            ax.plot(np.where(faint, c["MSE"][b], np.nan), lev, color=col, lw=0.9, alpha=0.5, zorder=3)
            ax.plot(np.where(lev <= SAT_BELOW, c["MSE_sat"][b], np.nan), lev, color=col, lw=1.6, ls="--", zorder=3)
            p_lcl = c["p_LCL_LPL"][b]
            ok = np.isfinite(c["MSE"][b])
            if np.isfinite(p_lcl) and ok.sum() > 1:
                o = np.argsort(lev[ok])
                h_lcl = np.interp(np.log(p_lcl), np.log(lev[ok][o]), c["MSE"][b][ok][o])
                ax.plot(h_lcl, p_lcl, "o", color=col, mec="k", mew=0.6, ms=5, zorder=5)
            shown = (lev >= X_FROM_HPA) & (cov[b] >= COVERAGE_FAINT)       # x-range from the drawn troposphere
            xs.append(np.nanmin(np.where(shown, c["MSE"][b], np.nan)))
            xs.append(np.nanmax(np.where(shown, c["MSE"][b], np.nan)))
            kept += c["n"][b]
        if xs:
            ax.set_xlim(np.nanmin(xs) - 2, np.nanmax(xs) + 2)
        ax.set(ylim=(1000, 150), xlabel=r"MSE (kJ kg$^{-1}$)")
        ax.set_yticks(np.arange(200, 1001, 100))
        ax.set_title(f"{name}", fontsize=11, color=REGIME_COLOR[k])
        ax.text(0.03, 0.91, f"n = {int(kept):,}", transform=ax.transAxes, ha="left", va="top",
                color="#444", fontstyle="italic", fontsize=8)
        style.panel_letter(ax, k + 1)
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
        facts[name] = (f"h*500 {hs[0]:g}–{hs[1]:g}, h500 {h[0]:g}–{h[1]:g}; {int(kept):,} events in bins with "
                       f"≥ {MIN_COUNT}; LCL {np.nanmax(np.where(c['n'] >= MIN_COUNT, c['p_LCL_LPL'], np.nan)):.0f}"
                       f" → {np.nanmin(np.where(c['n'] >= MIN_COUNT, c['p_LCL_LPL'], np.nan)):.0f} hPa across bins")
    axes[0].set_ylabel("Pressure (hPa)")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=axes, pad=0.012, fraction=0.025, ticks=edges[::2])
    cb.set_label("surface RH (5 % bins)")
    cb.outline.set_edgecolor(style.SPINE)
    axes[-1].legend(handles=[Line2D([0], [0], color="k", lw=1.8, label="$h$"),
                             Line2D([0], [0], color="k", lw=1.6, ls="--", label=f"$h^*$ (≤ {SAT_BELOW:g} hPa)"),
                             Line2D([0], [0], ls="", marker="o", mfc="w", mec="k", label="LCL")],
                    loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="none")
    facts["years"] = label
    path = FD.save(fig, out_dir, "cascade_regimes")
    plt.close(fig)
    return {"path": path, "title": "RH cascades by regime", "facts": facts,
            "script": "figures/fig_cascade_regimes.py"}


def make(out_dir, years=None, weighted=WEIGHTED, casc=None):
    folder = Path(casc) if casc else Path(FD.load("paths")["work"]) / "side_tests" / "casc"
    A = T.load_all(folder, years=years)
    label = f"{years[0]}–{years[1]}" if years else "all years in casc/"
    return [draw([A["T"], A["M"]], out_dir, label, plane=fine_plane(years))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", help="e.g. 1980-2019 (default: all casc files)")
    ap.add_argument("--casc", help="folder of casc_*.npz (default <work>/side_tests/casc)")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs"))
    a = ap.parse_args()
    for item in make(Path(a.out), FD.years_arg(a.years), casc=a.casc):
        print(f"Wrote {item['path']}")
        for k, v in item["facts"].items():
            print(f"  {k}: {v}")
