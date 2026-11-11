"""Excess vs RH, entrainment view: the log-RH plume model on the event plane.

The plume picture gives δ_a = Γ_h · C · ln(RH / RH0) with Γ_h = −dh_ML/dz_LCL > 0 (Cascade_Summary
log-RH model, constant C = 2 km). RH0 is fixed at the pooled mean's zero crossing (E11), so only
K = Γ_h·C is fitted, by count-weighted least squares on the mean curve inside the RH window. Also
shown: the first-order (linear) Taylor form K·(RH/RH0 − 1). The fit runs on the old basis
ln(RH0/RH), so K = −k. The window is drawn exactly as fitted (E11).
The model drawing is draw_model(); fig_excess_vs_rh.py also puts it on Fig. 5 (excess_vs_rh_model).
    python figures/fig_excess_vs_rh_entrainment.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style
from fig_excess_vs_rh import (XRANGE, YRANGE, EVENT_LINE, draw_plane, mean_curve, plane)

C_KM = 1.8                       # constant C in dz_LCL/dRH ≈ −C/RH (mid-range of the Romps-LCL derivation, Appendix A)
WINDOW = (0.30, 0.60)            # RH fit window (E11: still to be chosen; drawn as used)


def fit_k(x, m, n, rh0, window, form):
    ok = np.isfinite(m) & (x >= window[0]) & (x <= window[1])
    g = np.log(rh0 / x[ok]) if form == "log" else 1 - x[ok] / rh0
    w = n[ok]
    return (w * g * m[ok]).sum() / (w * g * g).sum()


def draw_model(ax, axr, H, ex, ey):
    """Mean δ_a dots, the fitted log and linear models, the fit window, and residuals on axr.
    Returns (rh0 of the mean curve, facts)."""
    x, m, n = mean_curve(H["T"] + H["M"], ex, ey)
    rh0 = FD.zero_crossing(x[::-1], m[::-1])
    ax.plot(x, m, color="gray", lw=0, marker=".", ms=6, label=r"mean $\delta_{a}$", zorder=5)
    facts = {"RH0 (fixed, from crossing)": f"{rh0:.3f}", "fit window": f"{WINDOW[0]:.2f}–{WINDOW[1]:.2f}"}
    xs = np.linspace(0.05, 1.0, 400)
    in_window = np.isfinite(m) & (x >= WINDOW[0]) & (x <= WINDOW[1])
    forms = (("log", "k", "-", r"$\Gamma_h C\,\ln(RH/RH_0)$"),
             ("lin", "#0A9396", "--", r"$\Gamma_h C\,(RH/RH_0-1)$"))
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
        facts[f"{form} fit"] = f"Γ_h·C = {-k:.2f} kJ/kg → Γ_h = {-k / C_KM:.2f} kJ/kg/km (C = {C_KM:g} km), RMSE {rmse:.2f} kJ/kg in window"
        for e in ("T", "M"):                                      # same fit, same RH0, one event type
            xe, me, ne = mean_curve(H[e], ex, ey)
            facts[f"{form} fit {e} only"] = f"Γ_h·C = {-fit_k(xe, me, ne, rh0, WINDOW, form):.2f} kJ/kg (RH0 pooled)"
    for a in (ax, axr):                                           # the fit window, in the band style
        style.band(a, *WINDOW, edges=True, axis="x")
    axr.axhline(0, color="k", lw=0.8)
    axr.set(ylabel="mean − model\n" + style.KJ, ylim=(-4, 4))
    return rh0, facts


def make(out_dir, years=None, weighted=True):
    style.apply()
    data, ex, ey, H = plane(years, weighted)
    d0 = data["T"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(3.7, 4.6), sharex=True, gridspec_kw={"height_ratios": [4, 1.3]})
    draw_plane(ax, ex, ey, H, labels=False)
    rh0, facts = draw_model(ax, axr, H, ex, ey)
    ax.axhline(0, color="k", lw=1.2, ls=":")
    if np.isfinite(rh0):
        ax.axvline(rh0, color="k", lw=0.8, alpha=0.6)
        ax.text(rh0, YRANGE[0] * 0.93, f" RH$_0$ = {100 * rh0:.1f}%", fontsize=7, va="bottom")
    ax.legend(loc="upper left", fontsize=6, framealpha=0.85, edgecolor="none")
    ax.set(xlim=XRANGE, ylim=YRANGE, ylabel=r"$\delta_{a}$ " + style.KJ)
    style.rh_percent(axr)
    fig.tight_layout()
    facts.update(years=d0.label(), weighting="cos(lat)" if weighted else "none")
    return [{"path": FD.save(fig, out_dir, "excess_vs_rh_entrainment"), "title": "Excess vs RH · entrainment view",
             "facts": facts, "script": "figures/fig_excess_vs_rh_entrainment.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
