"""WBT Increase from Excess (paper figure): wet-bulb warming produced by an MSE excess.

Port of Paper_Plots-2 cell 138 (wbt_increase_map.png), self-contained (no data): for each starting
wet-bulb temperature and excess dH, solve cp·dTw + Lv·(q*(Tw+dTw) − q*(Tw)) = dH exactly at p.
Constants as in the original (Magnus e_s, cp = 1005.7, Lv = 2.501e6); p = 1000 hPa (state in caption).

Dots: the two extremes of Fig. 1 (fig_example_profiles.py), each at (reference WBT, δ). The reference WBT is
the wet-bulb temperature the surface would have with h_2m = h*500, i.e. at convective neutrality, solved with
this figure's constants at P_HPA after removing the surface geopotential; the dot then sits where the
contours give the warming its excess δ = h_2m − h*500 produced. Values come from Fig. 1's cache
(<out>/example_profiles_cache.npz, written when Fig. 1 is made) or, failing that, from the pipeline files; with neither, no dots.
    python figures/fig_wbt_increase.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from scipy.optimize import brentq

import figdata as FD
import style

CP, LV, EPS = 1005.7, 2.501e6, 0.622
P_HPA = 1000.0


def esat(T):
    Tc = T - 273.15
    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))


def qsat(T, p):
    es = esat(T)
    return EPS * es / (p - (1.0 - EPS) * es)


def dTw_from_excess(Tw, p, dMSE):
    qs0 = qsat(Tw, p)
    return brentq(lambda dT: CP * dT + LV * (qsat(Tw + dT, p) - qs0) - dMSE, -80.0, 80.0)


def tw_from_h(h_kj, phi_s_kj, p=P_HPA):
    """WBT [K] of saturated air with MSE h at pressure p, the surface geopotential removed (this figure's constants)."""
    target = (h_kj - phi_s_kj) * 1000.0
    return brentq(lambda T: CP * T + LV * qsat(T, p) - target, 200.0, 340.0)


def beijing_dots(out_dir):
    """[(event, reference WBT °C, δ kJ/kg, 2-m WBT °C)] for the Fig. 1 extremes, or [] if no data."""
    import fig_example_profiles as E
    from tpl_mll import thermo
    try:
        cache = Path(out_dir) / E.CACHE_NAME
        cols, _ = E.load(cache=cache if cache.exists() else None)
    except Exception:                                          # noqa: BLE001  no cache and no pipeline files
        return []
    out = []
    for e in ("T", "M"):
        c = cols[e]
        ex = E.excesses(c)
        phi_s = c["h_2m"] - thermo.mse(c["T2m"], c["q2m"], 0.0)
        if not np.isfinite(phi_s):                             # no 2-m T or q in the files
            continue
        out.append((e, tw_from_h(ex["h*500"], phi_s) - 273.15, ex["delta"], tw_from_h(c["h_2m"], phi_s) - 273.15))
    return out


def make(out_dir, years=None, weighted=True, dots=True):
    Tw_C = np.linspace(15.0, 40.0, 150)
    dH = np.linspace(0.0, 30.0, 150)
    Z = np.zeros((dH.size, Tw_C.size))
    for i, dh in enumerate(dH * 1000.0):
        if dh:
            for j, T in enumerate(Tw_C + 273.15):
                Z[i, j] = dTw_from_excess(T, P_HPA, dh)
    style.apply()
    fig, ax = plt.subplots(figsize=(3.7, 3.0))  # one journal column (19pc)
    ax.tick_params(labelsize=8)
    cf = ax.contourf(Tw_C, dH, Z, levels=np.linspace(0.0, Z.max(), 40), cmap="magma")
    cs = ax.contour(Tw_C, dH, Z, levels=np.arange(1.0, Z.max(), 1.0), colors="white", linewidths=0.7, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f K")
    cs2 = ax.contour(Tw_C, dH, Tw_C[None, :] + Z, levels=np.arange(20.0, 51.0, 5.0), colors="#34d8d8",
                     linewidths=1.0, linestyles="--")
    ax.clabel(cs2, inline=True, fontsize=6, fmt="%.0f°C")
    ax.legend(handles=[Line2D([0], [0], color="0.55", lw=0.9, label="increase ΔTw [K]"),
                       Line2D([0], [0], color="#34d8d8", lw=1.0, ls="--", label="resulting WBT [°C]")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5, frameon=False,
              facecolor="0.3")
    fig.colorbar(cf, ax=ax, ticks=np.arange(0.0, Z.max() + 0.5, 2.0)).set_label(r"WBT increase $\Delta T_w$ [K]")
    dot_facts = {}
    for e, tw0, d, tw2m in (beijing_dots(out_dir) if dots else []):
        guide = [pe.withStroke(linewidth=2.0, foreground="white", alpha=0.4)]       # readable on dark and light
        ax.plot([tw0, tw0, Tw_C[0]], [0.0, d, d], color=style.EVENT_COLOR[e], lw=1.1, ls=(0, (3, 2)), zorder=5,
                path_effects=guide)                                                 # guides to both axes
        ax.plot(tw0, d, "o", color=style.EVENT_COLOR[e], mec="white", mew=1.0, ms=7, zorder=6, clip_on=False)
        ax.annotate(style.EVENT_SHORT[e], (tw0, d), xytext=(5, 4), textcoords="offset points", fontsize=7,
                    color="white", zorder=6, path_effects=[pe.withStroke(linewidth=1.5, foreground="0.15")])
        dot_facts[f"Fig. 1 {e} extreme"] = (f"reference WBT {tw0:.1f} °C, δ {d:+.1f} kJ/kg → "
                                             f"ΔTw ≈ {dTw_from_excess(tw0 + 273.15, P_HPA, d * 1000.0):.1f} K "
                                             f"(2-m WBT from h_2m: {tw2m:.1f} °C)")
    ax.set(xlabel="reference WBT [°C]", ylabel=r"excess $\delta$ [kJ/kg]")
    fig.tight_layout()
    k = np.argmin(abs(Tw_C - 30)), np.argmin(abs(dH - 10))
    return [{"path": FD.save(fig, out_dir, "wbt_increase"), "title": "WBT Increase from Excess",
             "facts": {"pressure": f"{P_HPA:g} hPa (state in caption)",
                       "example": f"dH = 10 kJ/kg at Tw = 30 °C → ΔTw ≈ {Z[k[1], k[0]]:.2f} K",
                       "constants": "Magnus e_s, cp = 1005.7 J/kg/K, Lv = 2.501e6 J/kg (as original)",
                       **(dot_facts or {"Fig. 1 dots": "none (no example_profiles cache or pipeline files)"})},
             "script": "figures/fig_wbt_increase.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
