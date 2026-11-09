"""Plume Schematic (paper figure): composite low- and high-RH profiles with their entraining plumes.

Port of Paper_Plots-2 cells 73–76 (meeting.pdf). Composites of the T_TPL box at two 5 % RH bins: by default
the driest bin from 30 % whose plume (ε = EPS) tops out, and one about GAP moister that also tops out, so
both convect and the mixed-layer change is visible. Set RH_LOW / RH_HIGH to override.
Each plume starts at the composite's LPL with its mean h_LPL (old version: fixed 1000 hPa) and is
drawn up to where it meets h* (old version: hand-tuned cutoffs 555 / 500 hPa). Shading: green =
δ_FT/ε region of the low-RH plume (gray); growth of that region at high RH (teal).
    python figures/fig_plume_schematic.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figdata as FD
import style
from fig_rh_demo import CASE
from tpl_mll import ztop

EPS = 0.1                      # as in the old schematic
RH_LOW = RH_HIGH = None        # e.g. (0.40, 0.45); None = choose automatically (choose_bins)
SCAN = np.round(np.arange(0.30, 0.70, 0.05), 2)              # lower edges of the 5 % bins scanned
GAP = 0.15                     # aim for the moister bin this far above the drier one
COLOR_LOW, COLOR_HIGH = "#A07C45", "#005F73"                 # dry sand / moist teal: the ends of style.RH_CMAP, darkened
SHADE_L, SHADE_H = "0.55", "#0A9396"                         # δ_a/ε at low RH (gray); its growth at high RH (teal)
MIN_N = 300                                                  # events per composite
ANNOTATE = True                # the advisor's pathways: (i) lower ML top, (ii) top, (iii) ML–FT contrast
ML_DROP = 1.0                  # kJ/kg below h_ML that marks the mixed-layer top (the kink)
CONTRAST_P = 650.0             # hPa, level of the (iii) arrow


def plume_on(levels, c):
    ok = np.isfinite(c["MSE"]) & np.isfinite(c["MSE_sat"]) & np.isfinite(c["Z"])
    lv = levels[ok]
    p0 = min(c["p_LPL"], lv.max())
    cols = ztop.Columns(lv, c["MSE"][ok][None], c["MSE_sat"][ok][None], c["Z"][ok][None],
                        np.array([c["h_LPL"]]), np.array([p0]))
    hp, _, p_top, *_ = cols.top(EPS)
    return cols.p[0], hp[0], cols.h[0], float(p_top[0])


def tops_out(levels, c):
    """True if the composite's plume at EPS reaches a top below 100 hPa (neither diluted out nor still rising)."""
    _, _, _, top = plume_on(levels, c)
    return np.isfinite(top) and top > ztop.P_TOP, top


def choose_bins(d):
    """Driest 5 % bin (from 30 %) whose plume tops out with ≥ MIN_N events, and the bin about GAP moister
    that does too. Returns (low, high, table of every candidate) for the facts."""
    rows, ok = [], {}
    for lo in SCAN:
        b = (float(lo), float(round(lo + 0.05, 2)))
        c = FD.composite(d, CASE, *b)
        good, top = (False, np.nan) if c["n"] < MIN_N else tops_out(d.levels, c)
        ok[b] = good
        what = ("too few events" if c["n"] < MIN_N else "no top (diluted out or above 100 hPa)"
                if not np.isfinite(top) else f"top {top:.0f} hPa")
        rows.append(f"{100 * b[0]:.0f}–{100 * b[1]:.0f}%: n {int(c['n'])}, {what}{' ✓' if good else ''}")
    good = [b for b, g in ok.items() if g]
    if len(good) < 2:
        raise ValueError("fewer than two RH bins whose plume tops out:\n  " + "\n  ".join(rows))
    low = good[0]
    higher = [b for b in good if b[0] >= low[0] + 0.10 - 1e-9]
    if not higher:
        raise ValueError("no plume-topping bin 10+ points moister than the driest:\n  " + "\n  ".join(rows))
    high = min(higher, key=lambda b: abs(b[0] - (low[0] + GAP)))
    return low, high, rows


def ml_top(levels, c, drop=ML_DROP):
    """Pressure of the mixed-layer top of a composite: going up from the LPL, the first level where h
    falls `drop` below h_LPL (linear in ln p between levels). NaN if it never does."""
    lev = np.asarray(levels, float)
    o = np.argsort(lev)[::-1]                                       # surface → top
    p, h = lev[o], c["MSE"][o]
    ok = np.isfinite(h) & (p <= c["p_LPL"])
    p, h = p[ok], h[ok]
    below = h < c["h_LPL"] - drop
    if not below.any():
        return np.nan
    k = int(np.argmax(below))
    if k == 0:
        return float(p[0])
    f = (c["h_LPL"] - drop - h[k - 1]) / (h[k] - h[k - 1])
    return float(np.exp(np.log(p[k - 1]) + f * (np.log(p[k]) - np.log(p[k - 1]))))


def annotate(ax, levels, lo, hi, topL, topH, cL, cH, eH, pg):
    """Arrows for the RH pathways on the low- and high-RH composites."""
    ann = dict(fontsize=7, fontweight="bold", color="#111827")
    arrow = dict(arrowstyle="<->", color="#111827", lw=1.0, shrinkA=0, shrinkB=0)
    mlL, mlH = ml_top(levels, lo), ml_top(levels, hi)
    x_r = max(lo["h_LPL"], hi["h_LPL"]) + 0.8
    if np.isfinite(mlL) and np.isfinite(mlH):                       # (i) the mixed-layer top drops
        for p_, col in ((mlL, COLOR_LOW), (mlH, COLOR_HIGH)):
            ax.plot([x_r - 0.6, x_r + 0.6], [p_, p_], color=col, lw=1.6)
        ax.annotate("", xy=(x_r, mlH), xytext=(x_r, mlL), arrowprops=arrow, zorder=7)
        ax.text(x_r + 0.5, np.sqrt(mlL * mlH), "(i) ML top\nlowers", va="center", ha="left", **ann)
    # (iii) the contrast: the gap between the high-RH environment and its plume, h_c − h_e
    at_p = lambda v: np.interp(np.log(CONTRAST_P), np.log(pg[::-1]), v[::-1])
    e_hi, c_hi = at_p(eH), at_p(cH)
    if np.isfinite(e_hi) and np.isfinite(c_hi):
        ax.annotate("", xy=(c_hi, CONTRAST_P), xytext=(e_hi, CONTRAST_P), arrowprops=arrow, zorder=7)
        ax.text(0.5 * (e_hi + c_hi), CONTRAST_P - 12, "(iii) contrast", ha="center", va="bottom", **ann)
    xs = []
    for top, c_, col in ((topL, cL, COLOR_LOW), (topH, cH, COLOR_HIGH)):   # (ii) the tops
        if np.isfinite(top):
            x_top = np.interp(np.log(top), np.log(pg[::-1]), c_[::-1])
            ax.plot(x_top, top, "o", color=col, ms=4.5, mec="white", mew=0.7, zorder=6)
            xs.append(x_top)
    if len(xs) == 2:
        ax.text(max(xs) + 0.6, min(topL, topH), "(ii) tops\nnearly fixed", ha="left", va="center", **ann)
    return mlL, mlH


def make(out_dir, years=None, weighted=True):
    style.apply()
    d = FD.FigData("T", years)
    if RH_LOW and RH_HIGH:
        rh_low, rh_high, scan = RH_LOW, RH_HIGH, ["set by hand"]
    else:
        rh_low, rh_high, scan = choose_bins(d)
    lo, hi = FD.composite(d, CASE, *rh_low), FD.composite(d, CASE, *rh_high)
    for c, rng in ((lo, rh_low), (hi, rh_high)):
        if c["n"] < MIN_N:
            raise ValueError(f"only {int(c['n'])} {CASE} events with RH {rng[0]:.2f}–{rng[1]:.2f}; move RH_LOW/RH_HIGH")
    pL, hpL, envL, topL = plume_on(d.levels, lo)
    pH, hpH, envH, topH = plume_on(d.levels, hi)
    fig, ax = plt.subplots(figsize=(3.6, 3.9))                        # compact, as the Paper_Plots-2 original
    style.paper_axes(ax)
    # common pressure axis for shading (log-p interpolation)
    pg = np.exp(np.linspace(np.log(min(pL[0], pH[0])), np.log(100), 300))
    at = lambda p, v: np.interp(np.log(pg), np.log(p[::-1]), v[::-1], left=np.nan, right=np.nan)
    cL, eL, cH, eH = at(pL, hpL), at(pL, envL), at(pH, hpH), at(pH, envH)
    below_L = pg >= (topL if np.isfinite(topL) else 1e9)
    below_H = pg >= (topH if np.isfinite(topH) else 1e9)
    ax.fill_betweenx(pg, cL, eL, where=below_L & (cL > eL), color=SHADE_L, alpha=0.18, lw=0)
    left = np.where(below_L, np.maximum(eH, cL), eH)
    ax.fill_betweenx(pg, left, cH, where=below_H & (cH > left), color=SHADE_H, alpha=0.22, lw=0)
    for c, col in ((lo, COLOR_LOW), (hi, COLOR_HIGH)):
        ax.plot(c["MSE"], d.levels, color=col, lw=2.5)
        ax.plot(np.where(d.levels <= 600, c["MSE_sat"], np.nan), d.levels, color=col, lw=2.5, ls="--")
    ax.plot(np.where(below_L, cL, np.nan), pg, color=COLOR_LOW, lw=2, ls=":")
    ax.plot(np.where(below_H, cH, np.nan), pg, color=COLOR_HIGH, lw=2, ls=":")
    mlL = mlH = np.nan
    if ANNOTATE:
        mlL, mlH = annotate(ax, d.levels, lo, hi, topL, topH, cL, cH, eH, pg)
    x_hi = max(345.0, max(lo["h_LPL"], hi["h_LPL"]) + 6.5) if ANNOTATE else 345.0   # room for the (i) label
    ax.set(ylim=(1000, 100), xlim=(320, x_hi), xlabel="MSE " + style.KJ,
           ylabel="Pressure [hPa]")
    ax.set_yticks(np.arange(100, 1001, 100))
    pct = lambda r: f"{100 * r[0]:.0f}–{100 * r[1]:.0f}%"
    ax.legend(handles=[Line2D([0], [0], color="k", lw=2.5, label="environment $h$"),
                       Line2D([0], [0], color="k", lw=2.5, ls="--", label="saturated $h^*$"),
                       Line2D([0], [0], color="k", lw=2, ls=":", label=f"plume, $\\varepsilon$ = {EPS:g} km$^{{-1}}$"),
                       Line2D([0], [0], color=COLOR_LOW, lw=2.5, label=f"RH {pct(rh_low)}"),
                       Line2D([0], [0], color=COLOR_HIGH, lw=2.5, label=f"RH {pct(rh_high)}"),
                       Patch(fc=SHADE_L, alpha=0.18, label=r"$\delta_{a}/\varepsilon$, drier"),
                       Patch(fc=SHADE_H, alpha=0.22, label=r"growth at higher RH")],
              fontsize=5.5, loc="upper left", framealpha=0.9, edgecolor="#9CA3AF", handlelength=2.2,
              borderpad=0.4, labelspacing=0.3)
    fig.tight_layout()
    facts = {"eps": f"{EPS} km⁻¹", "plume top low RH": f"{topL:.0f} hPa", "plume top high RH": f"{topH:.0f} hPa",
             "start": f"LPL of composite (low {lo['p_LPL']:.0f}, high {hi['p_LPL']:.0f} hPa)", "years": d.label(),
             "ML top (h_LPL − 1 kJ/kg)": f"low {mlL:.0f}, high {mlH:.0f} hPa",
             "RH bins": f"{100 * rh_low[0]:.0f}–{100 * rh_low[1]:.0f}% and {100 * rh_high[0]:.0f}–{100 * rh_high[1]:.0f}%",
             "candidates (✓ = tops out, enough events)": "; ".join(scan)}
    return [{"path": FD.save(fig, out_dir, "plume_schematic"), "title": "Plume Schematic", "facts": facts,
             "script": "figures/fig_plume_schematic.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
