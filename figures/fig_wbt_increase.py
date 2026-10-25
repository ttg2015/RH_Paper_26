"""WBT Increase from Excess (paper figure): wet-bulb warming produced by an MSE excess.

Port of Paper_Plots-2 cell 138 (wbt_increase_map.png), self-contained (no data): for each starting
wet-bulb temperature and excess dH, solve cp·dTw + Lv·(q*(Tw+dTw) − q*(Tw)) = dH exactly at p.
Constants as in the original (Magnus e_s, cp = 1005.7, Lv = 2.501e6); p = 1000 hPa (state in caption).
    python figures/fig_wbt_increase.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq

import figdata as FD

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


def make(out_dir, years=None, weighted=True):
    Tw_C = np.linspace(15.0, 40.0, 150)
    dH = np.linspace(0.0, 30.0, 150)
    Z = np.zeros((dH.size, Tw_C.size))
    for i, dh in enumerate(dH * 1000.0):
        if dh:
            for j, T in enumerate(Tw_C + 273.15):
                Z[i, j] = dTw_from_excess(T, P_HPA, dh)
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(Tw_C, dH, Z, levels=np.linspace(0.0, Z.max(), 40), cmap="magma")
    cs = ax.contour(Tw_C, dH, Z, levels=np.arange(1.0, Z.max(), 1.0), colors="white", linewidths=0.7, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f K")
    cs2 = ax.contour(Tw_C, dH, Tw_C[None, :] + Z, levels=np.arange(20.0, 51.0, 5.0), colors="#34d8d8",
                     linewidths=1.0, linestyles="--")
    ax.clabel(cs2, inline=True, fontsize=7, fmt="%.0f°C")
    ax.legend(handles=[Line2D([0], [0], color="white", lw=0.9, label="Increase ΔTw [K]"),
                       Line2D([0], [0], color="#34d8d8", lw=1.0, ls="--", label="Resulting Tw [°C]")],
              loc="lower right", fontsize=8, framealpha=0.3)
    fig.colorbar(cf, ax=ax).set_label("Wet-bulb temperature increase [K]")
    ax.set(xlabel="Wet-bulb temperature [°C]", ylabel="Excess MSE, dH [kJ/kg]",
           title="Wet-bulb amplification of excess energy")
    fig.tight_layout()
    k = np.argmin(abs(Tw_C - 30)), np.argmin(abs(dH - 10))
    return [{"path": FD.save(fig, out_dir, "wbt_increase"), "title": "WBT Increase from Excess",
             "facts": {"pressure": f"{P_HPA:g} hPa (state in caption)",
                       "example": f"dH = 10 kJ/kg at Tw = 30 °C → ΔTw ≈ {Z[k[1], k[0]]:.2f} K",
                       "constants": "Magnus e_s, cp = 1005.7 J/kg/K, Lv = 2.501e6 J/kg (as original)"},
             "script": "figures/fig_wbt_increase.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
