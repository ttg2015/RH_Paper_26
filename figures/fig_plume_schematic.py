"""Plume Schematic (paper figure): composite low- and high-RH profiles with their entraining plumes.

Port of Paper_Plots-2 cells 73–76 (meeting.pdf). Composites are the RH-demo bins of the T_TPL box.
Each plume starts at the composite's LPL with its mean h_LPL (old version: fixed 1000 hPa) and is
drawn up to where it meets h* (old version: hand-tuned cutoffs 555 / 500 hPa). Shading: green =
δ_FT/ε region of the low-RH plume; orange = growth of that region at high RH.
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
from fig_rh_demo import CASE, RH_LOW, RH_HIGH
from tpl_mll import ztop

EPS = 0.1                      # as in the old schematic
COLOR_LOW, COLOR_HIGH = "firebrick", "purple"
SHADE_L, SHADE_H = "green", "#e67e22"
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


def annotate(ax, levels, lo, hi, topL, topH, cL, cH, pg):
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
    e_hi = np.interp(np.log(CONTRAST_P), np.log(levels), hi["MSE"])  # (iii) ML–FT contrast widens
    if np.isfinite(e_hi):
        ax.annotate("", xy=(hi["h_LPL"], CONTRAST_P), xytext=(e_hi, CONTRAST_P), arrowprops=arrow, zorder=7)
        ax.text(0.5 * (e_hi + hi["h_LPL"]), CONTRAST_P - 12, "(iii) ML–FT contrast", ha="center", va="bottom", **ann)
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
    lo, hi = FD.composite(d, CASE, *RH_LOW), FD.composite(d, CASE, *RH_HIGH)
    for c, rng in ((lo, RH_LOW), (hi, RH_HIGH)):
        if not c["n"]:
            raise ValueError(f"no {CASE} events with RH {rng[0]:.2f}–{rng[1]:.2f} in these years")
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
    ax.fill_betweenx(pg, cL, eL, where=below_L & (cL > eL), color=SHADE_L, alpha=0.15, lw=0)
    left = np.where(below_L, np.maximum(eH, cL), eH)
    ax.fill_betweenx(pg, left, cH, where=below_H & (cH > left), color=SHADE_H, alpha=0.20, lw=0)
    for c, col in ((lo, COLOR_LOW), (hi, COLOR_HIGH)):
        ax.plot(c["MSE"], d.levels, color=col, lw=2.5)
        ax.plot(np.where(d.levels <= 600, c["MSE_sat"], np.nan), d.levels, color=col, lw=2.5, ls="--")
    ax.plot(np.where(below_L, cL, np.nan), pg, color=COLOR_LOW, lw=2, ls=":")
    ax.plot(np.where(below_H, cH, np.nan), pg, color=COLOR_HIGH, lw=2, ls=":")
    mlL = mlH = np.nan
    if ANNOTATE:
        mlL, mlH = annotate(ax, d.levels, lo, hi, topL, topH, cL, cH, pg)
    x_hi = max(345.0, max(lo["h_LPL"], hi["h_LPL"]) + 6.5) if ANNOTATE else 345.0   # room for the (i) label
    ax.set(ylim=(1000, 100), xlim=(320, x_hi), xlabel=r"MSE (kJ kg$^{-1}$)",
           ylabel="Pressure (hPa)")
    ax.set_title("Moist Static Energy Profiles", fontweight="bold", fontsize=12)
    ax.set_yticks(np.arange(100, 1001, 100))
    ax.legend(handles=[Line2D([0], [0], color="k", lw=2.5, label="Environmental MSE"),
                       Line2D([0], [0], color="k", lw=2.5, ls="--", label="Saturated Environmental MSE"),
                       Line2D([0], [0], color="k", lw=2, ls=":", label="Convective Parcel Path"),
                       Line2D([0], [0], color=COLOR_LOW, lw=2.5, label="Low RH"),
                       Line2D([0], [0], color=COLOR_HIGH, lw=2.5, label="High RH"),
                       Patch(fc=SHADE_L, alpha=0.15, label=r"$\delta_{a}/\epsilon$ (Low RH)"),
                       Patch(fc=SHADE_H, alpha=0.2, label=r"$\Delta\delta_{a}/\epsilon$ (High RH growth)")],
              fontsize=5.5, loc="upper left", framealpha=0.9, edgecolor="#9CA3AF", handlelength=2.2,
              borderpad=0.4, labelspacing=0.3)
    fig.tight_layout()
    facts = {"eps": f"{EPS} km⁻¹", "plume top low RH": f"{topL:.0f} hPa", "plume top high RH": f"{topH:.0f} hPa",
             "start": f"LPL of composite (low {lo['p_LPL']:.0f}, high {hi['p_LPL']:.0f} hPa)", "years": d.label(),
             "ML top (h_LPL − 1 kJ/kg)": f"low {mlL:.0f}, high {mlH:.0f} hPa"}
    return [{"path": FD.save(fig, out_dir, "plume_schematic"), "title": "Plume Schematic", "facts": facts,
             "script": "figures/fig_plume_schematic.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
