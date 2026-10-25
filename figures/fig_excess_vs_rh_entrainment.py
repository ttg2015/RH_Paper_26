"""Excess vs RH, entrainment view: the log-RH plume model on the event plane.

The plume picture gives δ_FT = Γ_h · C · ln(RH0 / RH) (Cascade_Summary log-RH model, constant
C = 2 km). RH0 is fixed at the pooled mean's zero crossing (E11), so only K = Γ_h·C is fitted,
by count-weighted least squares on the mean curve inside the RH window. Also shown: the first-
order (linear) Taylor form K·(1 − RH/RH0). The window is drawn exactly as fitted (E11).
    python figures/fig_excess_vs_rh_entrainment.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style
from fig_excess_vs_rh import (XRANGE, YRANGE, EVENT_LINE, draw_plane, mean_curve, plane)

C_KM = 2.0                       # constant C in dz_LCL/dRH ≈ −C/RH
WINDOW = (0.30, 0.60)            # RH fit window (E11: still to be chosen; drawn as used)


def fit_k(x, m, n, rh0, window, form):
    ok = np.isfinite(m) & (x >= window[0]) & (x <= window[1])
    g = np.log(rh0 / x[ok]) if form == "log" else 1 - x[ok] / rh0
    w = n[ok]
    return (w * g * m[ok]).sum() / (w * g * g).sum()


def make(out_dir, years=None, weighted=True):
    style.apply()
    data, ex, ey, H = plane(years, weighted)
    d0 = data["T"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1.3]})
    draw_plane(ax, ex, ey, H, labels=False)
    x, m, n = mean_curve(H["T"] + H["M"], ex, ey)
    rh0 = FD.zero_crossing(x[::-1], m[::-1])
    ax.plot(x, m, color="gray", lw=0, marker=".", ms=6, label=r"mean $\delta_{FT}$", zorder=5)
    facts = {"RH0 (fixed, from crossing)": f"{rh0:.3f}", "fit window": f"{WINDOW[0]:.2f}–{WINDOW[1]:.2f}"}
    xs = np.linspace(0.05, 1.0, 400)
    in_window = np.isfinite(m) & (x >= WINDOW[0]) & (x <= WINDOW[1])
    forms = (("log", "k", "-", r"$\Gamma_h C\,\ln(RH_0/RH)$"),
             ("lin", "#00897b", "--", r"$\Gamma_h C\,(1-RH/RH_0)$"))
    if not np.isfinite(rh0) or not in_window.any():
        forms = ()
        facts["fit"] = "skipped: no zero crossing or no populated bins in the window"
        ax.text(0.5, 0.5, "not enough events for the fit", transform=ax.transAxes, ha="center")
    for form, color, ls, label in forms:
        k = fit_k(x, m, n, rh0, WINDOW, form)
        model = k * (np.log(rh0 / xs) if form == "log" else 1 - xs / rh0)
        ax.plot(xs, model, color=color, ls=ls, lw=2, label=label, zorder=6)
        pred = k * (np.log(rh0 / x) if form == "log" else 1 - x / rh0)
        axr.plot(x, m - pred, color=color, ls=ls, lw=1.5)
        ok = np.isfinite(m) & (x >= WINDOW[0]) & (x <= WINDOW[1])
        rmse = np.sqrt(np.average((m[ok] - pred[ok]) ** 2, weights=n[ok]))
        facts[f"{form} fit"] = f"Γ_h·C = {k:.2f} kJ/kg → Γ_h = {k / C_KM:.2f} kJ/kg/km (C = {C_KM:g} km), RMSE {rmse:.2f} kJ/kg in window"
    for a in (ax, axr):
        a.axvspan(*WINDOW, color="#fff59d", alpha=0.25, zorder=0)
        for v in WINDOW:
            a.axvline(v, color="#f9a825", lw=1, ls="--")
    ax.axhline(0, color="k", lw=1.2, ls=":")
    if np.isfinite(rh0):
        ax.axvline(rh0, color="k", lw=0.8, alpha=0.6)
        ax.text(rh0, YRANGE[1] * 0.93, f" RH$_0$ = {rh0:.3f}", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set(xlim=XRANGE, ylim=YRANGE, ylabel=r"$\delta_{FT}$ [kJ/kg]",
           title=f"Entrainment view: log-RH plume model · {d0.label()} · {style.band_text(FD.load('analysis'), weighted)}")
    axr.axhline(0, color="k", lw=0.8)
    axr.set(xlabel="RH", ylabel="mean − model\n[kJ/kg]", ylim=(-4, 4))
    fig.tight_layout()
    facts.update(years=d0.label(), weighting="cos(lat)" if weighted else "none")
    return [{"path": FD.save(fig, out_dir, "excess_vs_rh_entrainment"), "title": "Excess vs RH · entrainment view",
             "facts": facts, "script": "figures/fig_excess_vs_rh_entrainment.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
