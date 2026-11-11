"""RH cascades by regime (paper figure): composite MSE profiles binned by surface RH, T and wet-bulb pooled.

Three free-tropospheric regimes in the (h*500, h500) plane, with temperature and wet-bulb extremes pooled
in each: middle latitudes, tropics with a drier free troposphere, tropics with a moister one. The regimes
are the 3 kJ/kg grid cells of side_tests/cascade_composites.py set in side_tests/plume_top_vs_rh.REGIMES,
so this figure and the plume-top check use the same events.

(a) where the pooled events sit in the (h*500, h500) plane, with the regimes outlined: the 0.2 kJ/kg
    figdata histogram h2_hs500_h500_fine if it exists, else the 0.5 kJ/kg h2_hs500_h500 (T + M, unweighted,
    all terrain), else the 3 kJ/kg composite grid. Equal aspect, so the boxes stay square.
    To make the fine one: in config/figdata.yaml add the bins `energy_fine: [295.0, 355.0, 0.2]` and the hist2d
    `h2_hs500_h500_fine: [h_star500, energy_fine, h500, energy_fine]`, add "h2_hs500_h500_fine" to FEEDS in
    pipeline/g.py, and rerun stage g (both files are in its fingerprint, so any edit marks it stale).
(b–d) one cascade per regime, RH bins up to RH_MAX (cascade_regimes_full, for the supplement, keeps all): h solid, h* dashed (at and above 600 hPa), one color per 5 % RH bin, and a
dot at each bin's mean LCL so the mixed layer can be seen to shallow. Levels are solid where at least
COVERAGE_MIN of the bin's events have them above ground and faded down to COVERAGE_FAINT, as in
fig_cascade.py. Plain (unweighted) means, terrain below 0.5 km.

Trial version (separate file, toss if it looks bad): cascade_regimes_slope adds to each regime panel a thin
gray guide of slope GAMMA_GUIDE (the minimal model's fitted Γ_h), so the reader can check it against the
falling part of the profiles. The same run prints, per regime, the numbers the draft needs from the three
pooled regimes (RH window 0.30–0.60, as cascade_tests.py):
    Γ_h across bins   −d h_LPL / d z_LCL fitted across the RH bins (the paper's definition of Γ_h)
    Γ_h of profile    −(sickle slope) of the three-segment fit, the slope between the kink and the break
    C                 −d z_LCL / d ln RH;  kink/LCL coupling;  d h*500 / d ln RH;  ε for a plume to reach h*500
    T drop            lowest-level temperature, driest minus moistest bin

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
CMAP = style.RH_CMAP
PLANE_CMAP = "Greys"            # neutral, so the colored regime boxes stand out (purples would hide the violet box)
PLANE_KEEP = 0.002                                            # crop (a) to cells above this share of the peak
REGIME_COLOR = style.REGIME_COLOR                             # box outlines in (a), and panel titles
PLANE = (312.0, 348.0), (305.0, 341.0)
RH_MAX = 0.80                                                 # paper version: bins centered at or below this
                                                              # (above it the kink fit is erratic and h_ML has
                                                              # plateaued); cascade_regimes_full keeps every bin
GAMMA_GUIDE = 5.8                                             # kJ/kg/km, fitted in fig_excess_vs_rh_entrainment
GUIDE_RH, GUIDE_DZ, GUIDE_OFFSET = 0.45, 1.5, 4.0             # anchor bin, length [km], shift right of the LCL
                                                              # [kJ/kg], into the empty space under h* (left was
                                                              # off the axis on the 2026-11-10 run)
GUIDE_COLOR = "#6B7280"
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
    name = next((n for n in ("h2_hs500_h500_fine", "h2_hs500_h500") if all(n in d.hist for d in data)), None)
    if name is None:
        return None
    h = sum(d.h(name, weighted=False) for d in data)
    return FD.crop(h, data[0].edges_of(name, "x"), data[0].edges_of(name, "y"), *PLANE)


def with_heights(c):
    """Add the cascade_tests height fields: z [km above the bin's mean surface] and zLCL [km]."""
    c["z"] = c["Z"] / T.G_KM - c["zs_km"][:, None]
    c["zLCL"] = c["zLCL_LPL"] / 1000.0
    return c


def regime_numbers(c, window=T.WINDOW):
    """The per-regime numbers for the draft (see the module docstring)."""
    r = T.candidates(c)
    x = c["RH_2m"]
    ok = (c["n"] >= T.MIN_N) & np.isfinite(c["zLCL"]) & np.isfinite(c["h_LPL"]) & (x >= window[0]) & (x <= window[1])
    g_bins = -np.polyfit(c["zLCL"][ok], c["h_LPL"][ok], 1, w=np.sqrt(c["n"][ok]))[0] if ok.sum() >= T.MIN_BINS else np.nan
    lev = np.asarray(c["levels"], float)
    use = np.nonzero(c["n"] >= MIN_COUNT)[0]
    t_low = np.array([T.interp_p(lev, c["T"][i], c["p_LPL"][i]) for i in use])
    t_drop = t_low[0] - t_low[-1] if t_low.size > 1 else np.nan
    return {"Gamma_bins": g_bins, "Gamma_profile": -r["G_h"], "C": r["C"], "coupling": r["coupling"],
            "dhstar_dlnRH": -r["K_hstar"], "eps_req": r["eps_req"], "T_drop": t_drop}


def slope_guide(ax, c, gamma=GAMMA_GUIDE):
    """A straight segment of slope −gamma in height, drawn in pressure through the anchor bin's heights."""
    lev = np.asarray(c["levels"], float)
    use = np.nonzero(c["n"] >= MIN_COUNT)[0]
    if not use.size:
        return
    i = use[np.argmin(abs(c["rh_centers"][use] - GUIDE_RH))]
    z, h = c["z"][i], c["MSE"][i]
    ok = np.isfinite(z) & np.isfinite(h)
    if ok.sum() < 3 or not np.isfinite(c["p_LCL_LPL"][i]):
        return
    z0 = T.interp_p(lev, z, c["p_LCL_LPL"][i])
    h0 = T.interp_p(lev, h, c["p_LCL_LPL"][i])
    zs = np.linspace(z0, z0 + GUIDE_DZ, 20)
    o = np.argsort(z[ok])
    ps = np.exp(np.interp(zs, z[ok][o], np.log(lev[ok][o]), left=np.nan, right=np.nan))
    hs = h0 + GUIDE_OFFSET - gamma * (zs - z0)
    ax.plot(hs, ps, color=GUIDE_COLOR, lw=1.6, solid_capstyle="round", zorder=6)
    ax.text(hs[-1], ps[-1] - 12, r"$\Gamma_h$", color=GUIDE_COLOR, ha="center", va="bottom", fontsize=9, zorder=6)


def draw_plane(ax, arrays, plane=None, letters="bcd", title="events"):
    """The (h*500, h500) plane with the three regime boxes, lettered as the cascade panels."""
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
        ax.text(hs[1] + 0.4, h[1] + 0.3, letters[k], color=REGIME_COLOR[k], fontweight="bold", fontsize=10)
    ax.set(xlim=xl, ylim=yl, xlabel=r"$h^*_{500}$ " + style.KJ, ylabel=r"$h_{500}$ " + style.KJ,
           aspect="equal")
    if title:
        ax.set_title(title, fontsize=10)


def draw_cascade(ax, c, cov, norm, rh_max=None, guide=False):
    """One regime's cascade into ax; returns the number of events drawn and the facts line."""
    lev = np.asarray(c["levels"], float)
    style.band(ax)
    xs, kept, used = [], 0, []
    for b in range(len(c["n"])):
        if c["n"][b] < MIN_COUNT or (rh_max is not None and c["rh_centers"][b] > rh_max):
            continue
        used.append(b)
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
    if guide:
        slope_guide(ax, c)
    if xs:
        ax.set_xlim(np.nanmin(xs) - 2, np.nanmax(xs) + 2)
    ax.set(ylim=(1000, 150), xlabel="MSE " + style.KJ)
    ax.set_yticks(np.arange(200, 1001, 100))
    ax.text(0.03, 0.91, f"n = {int(kept):,}", transform=ax.transAxes, ha="left", va="top",
            color="#444", fontstyle="italic", fontsize=8)
    pl = c["p_LCL_LPL"][used] if used else np.array([np.nan])
    return kept, f"{int(kept):,} events in bins with ≥ {MIN_COUNT}; LCL {np.nanmax(pl):.0f} → {np.nanmin(pl):.0f} hPa across bins"


def regime_composites(arrays):
    """{regime name: (composite with heights, coverage)} for the three regimes."""
    out = {}
    for name, (hs, h) in REGIMES.items():
        names = cells(hs, h)
        out[name] = (with_heights(pooled(arrays, names)), coverage(arrays, names))
    return out


def cascade_legend(ax, guide=False):
    ax.legend(handles=[Line2D([0], [0], color="k", lw=1.8, label="$h$"),
                       Line2D([0], [0], color="k", lw=1.6, ls="--", label=f"$h^*$ (≤ {SAT_BELOW:g} hPa)"),
                       Line2D([0], [0], ls="", marker="o", mfc="w", mec="k", label="LCL")]
              + ([Line2D([0], [0], color=GUIDE_COLOR, lw=1.6, label=rf"$\Gamma_h$ = {GAMMA_GUIDE:g} kJ/kg/km")]
                 if guide else []),
              loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=7, handlelength=1.6, frameon=True,
              framealpha=0.9, edgecolor="none")   # under the n label, left of the profiles (was upper right, on them)


def rh_colorbar(fig, axes, edges, norm, rh_max=None):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=axes, pad=0.012, fraction=0.025, ticks=edges[::2])
    cb.set_label(style.RH_LABEL + ", 5 % bins")
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    cb.outline.set_edgecolor(style.SPINE)
    if rh_max is not None:
        top = edges[min(np.searchsorted(edges, rh_max - 1e-9), len(edges) - 1)]   # edge of the last drawn bin
        cb.ax.set_ylim(edges[0], top)
    return cb


def draw(arrays, out_dir, label="", plane=None, guide=False, fname="cascade_regimes", rh_max=RH_MAX,
         numbers=True):
    style.apply()
    edges = arrays[0]["rh_edges"]
    norm = BoundaryNorm(edges, CMAP.N)
    fig = plt.figure(figsize=(17.0, 5.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 0.16, 1, 1, 1], wspace=0.08)    # column 1 is a spacer
    ax = style.paper_axes(fig.add_subplot(gs[0, 0]))            # (a) selection plane
    draw_plane(ax, arrays, plane)
    style.panel_letter(ax, 0)

    facts, axes = {}, []
    for k, (name, (c, cov)) in enumerate(regime_composites(arrays).items()):
        ax = style.paper_axes(fig.add_subplot(gs[0, k + 2], sharey=axes[0] if axes else None))
        axes.append(ax)
        _, line = draw_cascade(ax, c, cov, norm, rh_max=rh_max, guide=guide)
        ax.set_title(f"{name}", fontsize=10, color=REGIME_COLOR[k])
        style.panel_letter(ax, k + 1)
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
        hs, h = REGIMES[name]
        facts[name] = f"h*500 {hs[0]:g}–{hs[1]:g}, h500 {h[0]:g}–{h[1]:g}; " + line
        if numbers:
            q = regime_numbers(c)
            facts[f"{name} numbers"] = (f"Γ_h across bins {q['Gamma_bins']:.1f}, of profile {q['Gamma_profile']:.1f} kJ/kg/km; "
                                        f"C {q['C']:.2f} km; kink/LCL {q['coupling']:.2f}; dh*500/dlnRH {q['dhstar_dlnRH']:.2f} kJ/kg; "
                                        f"ε to reach h*500 {q['eps_req']:.2f} /km; T drop {q['T_drop']:.1f} K (RH window "
                                        f"{T.WINDOW[0]:g}–{T.WINDOW[1]:g}; T drop over all drawn bins)")
    axes[0].set_ylabel("Pressure [hPa]")
    rh_colorbar(fig, axes, edges, norm, rh_max)
    cascade_legend(axes[-1], guide)
    facts["years"] = label
    facts["RH bins drawn"] = "all" if rh_max is None else f"centers ≤ {100 * rh_max:.0f}%"
    path = FD.save(fig, out_dir, fname)
    plt.close(fig)
    title = "RH cascades by regime" + (" (Γ_h guide, trial)" if guide else "") + (" (full RH, supplement)" if rh_max is None else "")
    return {"path": path, "title": title, "facts": facts, "script": "figures/fig_cascade_regimes.py"}


def make(out_dir, years=None, weighted=WEIGHTED, casc=None):
    folder = Path(casc) if casc else Path(FD.load("paths")["work"]) / "side_tests" / "casc"
    A = T.load_all(folder, years=years)
    label = f"{years[0]}–{years[1]}" if years else "all years in casc/"
    plane = fine_plane(years)
    arrays = [A["T"], A["M"]]
    return [draw(arrays, out_dir, label, plane=plane),                                   # paper: RH ≤ RH_MAX
            draw(arrays, out_dir, label, plane=plane, rh_max=None, fname="cascade_regimes_full", numbers=False),
            draw(arrays, out_dir, label, plane=plane, guide=True, fname="cascade_regimes_slope", numbers=False)]


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
