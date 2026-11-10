"""RH Conditioning Procedure (paper figure): how the composites are built, on the plume schematic's profiles.

Same events and the same two RH bins as the plume schematic (fig_plume_schematic.py: its SOURCE, REGIME and
pair()), so Fig. 10 shows how the two profiles of Fig. 8 are made:
(a) the pooled T + WBT events in the (h*500, h500) plane with the regime box;
(b) the surface-RH distribution inside the regime in 5 % bins, the two chosen bins colored;
(c) the two mean profiles, h solid from each bin's mean LPL and h* dashed at and above 600 hPa.

    python figures/fig_rh_demo.py [--years 1980-2019] [--cache schematic_cache.npz] [--out DIR]
With --cache it runs anywhere from the schematic's cache (panel a needs a cache made with --dump after
2026-11-09, which stores the event plane).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import figdata as FD
import fig_plume_schematic as S
import style

REPO = Path(__file__).resolve().parents[1]
PAD = 6.0                       # kJ/kg around the regime box in panel (a)


def regime_box():
    from plume_top_vs_rh import REGIMES
    names = list(REGIMES)
    return REGIMES[S.REGIME], style.REGIME_COLOR[names.index(S.REGIME) % len(style.REGIME_COLOR)]


def draw_hist_profiles(ax_hist, ax_prof, levels, comps, rh_low, rh_high, titles=True):
    """Panels (b) and (c): the RH distribution in the regime with the two bins colored, and their profiles."""
    ax = {1: ax_hist, 2: ax_prof}
    # (b) RH distribution inside the regime, the two bins colored
    for b in sorted(comps):
        col = S.COLOR_LOW if b == rh_low else S.COLOR_HIGH if b == rh_high else "0.7"
        ax[1].bar(0.5 * (b[0] + b[1]), comps[b]["n"] / 1000.0, width=0.048, color=col, lw=0)
    ax[1].set(ylabel=r"events [$10^3$]", title="bin by surface RH" if titles else "", xlim=(0, 1))
    style.rh_percent(ax[1])

    # (c) the two mean profiles
    pct = lambda r: f"{100 * r[0]:.0f}–{100 * r[1]:.0f}%"
    for b, col in ((rh_low, S.COLOR_LOW), (rh_high, S.COLOR_HIGH)):
        c = comps[b]
        up = (levels <= c["p_LPL"]) & np.isfinite(c["MSE"])
        o = np.argsort(levels[up])
        ax[2].plot(np.r_[c["MSE"][up][o], c["h_LPL"]], np.r_[levels[up][o], c["p_LPL"]], color=col, lw=1.8,
                   label=f"RH {pct(b)}")
        ax[2].plot(np.where(levels <= 600, c["MSE_sat"], np.nan), levels, color=col, lw=1.6, ls="--")
    p_sfc = max(comps[rh_low]["p_LPL"], comps[rh_high]["p_LPL"])
    ax[2].plot([], [], color="0.3", lw=1.6, ls="--", label=r"$h^*$")
    ax[2].set(ylim=(p_sfc, 100), xlim=(320, 355), xlabel="MSE " + style.KJ, ylabel="Pressure [hPa]",
              title="average each bin" if titles else "")
    ax[2].legend(fontsize=7, loc="lower left", frameon=False)


def draw(levels, comps, label, out_dir, plane=None, src=""):
    style.apply()
    rh_low, rh_high, _ = S.pair(levels, comps)
    (hs, h), box_color = regime_box()
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1, 0.9]))

    # (a) where the events sit, and the regime
    if plane is not None:
        cnt, ex, ey = plane
        im = ax[0].pcolormesh(ex, ey, np.ma.masked_equal(cnt.T, 0), cmap="Greys", rasterized=True)
        fig.colorbar(im, ax=ax[0], label="events", pad=0.02)
        lo_, hi_ = min(hs[0], h[0]) - PAD, max(hs[1], h[1]) + PAD
        ax[0].plot((lo_, hi_), (lo_, hi_), color="k", lw=0.8, alpha=0.5)
        ax[0].set(xlim=(hs[0] - PAD, hs[1] + PAD), ylim=(h[0] - PAD, h[1] + PAD))
    else:
        ax[0].text(0.5, 0.5, "event plane not in this cache\n(re-dump on Casper)", transform=ax[0].transAxes,
                   ha="center", va="center", fontsize=8, color="0.4")
        ax[0].set(xlim=(hs[0] - PAD, hs[1] + PAD), ylim=(h[0] - PAD, h[1] + PAD))
    ax[0].add_patch(Rectangle((hs[0], h[0]), hs[1] - hs[0], h[1] - h[0], lw=1.8, ec=box_color, fill=False))
    ax[0].set(xlabel=r"$h^*_{500}$ " + style.KJ, ylabel=r"$h_{500}$ " + style.KJ, title="select a regime")

    draw_hist_profiles(ax[1], ax[2], levels, comps, rh_low, rh_high)
    pct = lambda r: f"{100 * r[0]:.0f}–{100 * r[1]:.0f}%"
    for k, a in enumerate(ax):
        style.panel_letter(a, k)
    facts = {"source": src or S.source_name(), "regime box": f"h*500 {hs[0]:g}–{hs[1]:g}, h500 {h[0]:g}–{h[1]:g}",
             "bins (as Fig. 8)": f"{pct(rh_low)} ({int(comps[rh_low]['n']):,} events), "
                                 f"{pct(rh_high)} ({int(comps[rh_high]['n']):,} events)",
             "years": label, "panel a": "event plane stored" if plane is not None else "no event plane (re-dump)"}
    path = FD.save(fig, out_dir, "rh_demo")
    plt.close(fig)
    return {"path": path, "title": "RH Conditioning Procedure", "facts": facts, "script": "figures/fig_rh_demo.py"}


def make(out_dir, years=None, weighted=False, cache=None):
    if cache:
        levels, comps, label, src = S.from_cache(cache)
        plane = S.plane_from_cache(cache)
    else:
        levels, comps, label = S.from_regime(years) if S.SOURCE == "regime" else S.from_figdata(years)
        src = S.source_name()
        try:
            from fig_cascade_regimes import fine_plane
            plane = fine_plane(years)
        except Exception:                                     # noqa: BLE001
            plane = None
    return [draw(levels, comps, label, out_dir, plane, src)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", help="e.g. 1980-2019")
    ap.add_argument("--cache", help="draw from the plume schematic's cache instead of the Casper files")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs"))
    a = ap.parse_args()
    for item in make(Path(a.out), FD.years_arg(a.years), cache=a.cache):
        print(f"Wrote {item['path']}")
        for k, v in item["facts"].items():
            print(f"  {k}: {v}")
