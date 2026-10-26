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


def plume_on(levels, c):
    ok = np.isfinite(c["MSE"]) & np.isfinite(c["MSE_sat"]) & np.isfinite(c["Z"])
    lv = levels[ok]
    p0 = min(c["p_LPL"], lv.max())
    cols = ztop.Columns(lv, c["MSE"][ok][None], c["MSE_sat"][ok][None], c["Z"][ok][None],
                        np.array([c["h_LPL"]]), np.array([p0]))
    hp, _, p_top, *_ = cols.top(EPS)
    return cols.p[0], hp[0], cols.h[0], float(p_top[0])


def make(out_dir, years=None, weighted=True):
    style.apply()
    d = FD.FigData("T", years)
    lo, hi = FD.composite(d, CASE, *RH_LOW), FD.composite(d, CASE, *RH_HIGH)
    for c, rng in ((lo, RH_LOW), (hi, RH_HIGH)):
        if not c["n"]:
            raise ValueError(f"no {CASE} events with RH {rng[0]:.2f}–{rng[1]:.2f} in these years")
    pL, hpL, envL, topL = plume_on(d.levels, lo)
    pH, hpH, envH, topH = plume_on(d.levels, hi)
    fig, ax = plt.subplots(figsize=(4.2, 4.8))
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
    ax.set(ylim=(1000, 100), xlim=(320, 345), xlabel=r"MSE (kJ kg$^{-1}$)", ylabel="Pressure (hPa)")
    ax.set_title("Moist Static Energy Profiles", fontweight="bold")
    ax.legend(handles=[Line2D([0], [0], color="k", lw=2.5, label="Environmental MSE"),
                       Line2D([0], [0], color="k", lw=2.5, ls="--", label="Saturated Environmental MSE"),
                       Line2D([0], [0], color="k", lw=2, ls=":", label="Convective Parcel Path"),
                       Line2D([0], [0], color=COLOR_LOW, lw=2.5, label="Low RH"),
                       Line2D([0], [0], color=COLOR_HIGH, lw=2.5, label="High RH"),
                       Patch(fc=SHADE_L, alpha=0.15, label=r"$\delta_{FT}/\epsilon$ (Low RH)"),
                       Patch(fc=SHADE_H, alpha=0.2, label=r"$\Delta\delta_{FT}/\epsilon$ (High RH growth)")],
              fontsize=6, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    facts = {"eps": f"{EPS} km⁻¹", "plume top low RH": f"{topL:.0f} hPa", "plume top high RH": f"{topH:.0f} hPa",
             "start": f"LPL of composite (low {lo['p_LPL']:.0f}, high {hi['p_LPL']:.0f} hPa)", "years": d.label()}
    return [{"path": FD.save(fig, out_dir, "plume_schematic"), "title": "Plume Schematic", "facts": facts,
             "script": "figures/fig_plume_schematic.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
