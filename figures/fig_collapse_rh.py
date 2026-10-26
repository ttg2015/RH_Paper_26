"""Collapse test: at the same surface RH, do T and h2m extremes look alike?

For each quantity, the mean against RH_2m for T events (red), M events (blue) and both pooled
(black dashed), with RH0 marked. If the red and blue curves lie on each other where both have
events, RH controls the quantity and pooling by RH is justified; if they stay apart, event type
carries extra information. The gap is reported as the event-weighted mean |T − M| difference over
the RH bins where both types have at least MIN_BOTH events.
    python figures/fig_collapse_rh.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

COARSEN = 2                       # 0.5 % → 1 % RH
MIN_N, MIN_BOTH = 20, 200
PANELS = [("h2_rh_dFT", r"mean $\delta_{FT}$ [kJ/kg]"), ("h2_rh_dS", r"mean $\delta_S$ [kJ/kg]"),
          ("h2_rh_dtotal", r"mean $\delta_{total}$ [kJ/kg]"), ("lb:delta_below", r"mean $\delta_{below}$ [kJ/kg] (unstable)"),
          ("lb:delta_above", r"mean $\delta_{above}$ [kJ/kg] (unstable)"), ("strike", "median plume top [hPa]")]


def curve(d, key, weighted):
    """(RH centres, value, n) for one event type."""
    if key.startswith("h2_"):
        if key + ("_w" if weighted else "") not in d.hist:
            return None
        h = d.h(key, weighted=weighted)
        hx, ex = FD.coarsen(h, d.edges_of(key, "x"), COARSEN, axis=0)
        m, n = FD.binned_mean(hx, ex, d.edges_of(key, "y"))
        return FD.centers(ex), m, n
    if key.startswith("lb:"):
        q = ["delta_FT", "delta_below", "delta_above"].index(key[3:])
        s, ex = FD.coarsen(d.hist["lb_RH_2m"], d.edges["rh"], COARSEN, axis=0)
        n = s[:, q, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            return FD.centers(ex), s[:, q, 1] / n, n
    if "strike_rh_ptop" not in d.hist:
        return None
    h, ex = FD.coarsen(d.hist["strike_rh_ptop"], d.edges["rh"], COARSEN, axis=0)
    pc = FD.centers(d.edges["ptop"])
    med = np.array([np.interp(0.5 * r.sum(), np.cumsum(r), pc) if r.sum() else np.nan for r in h])
    return FD.centers(ex), med, h.sum(1)


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    rh0, rh0_source = FD.rh_split_value(data, weighted)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    facts = {"RH0": f"{rh0:.3f} ({rh0_source})"}
    for ax, (key, ylabel) in zip(axes.ravel(), PANELS):
        curves = {e: curve(d, key, weighted) for e, d in data.items()}
        if any(c is None for c in curves.values()):
            ax.text(0.5, 0.5, "not in figure data yet", ha="center", transform=ax.transAxes)
            continue
        x = curves["T"][0]
        for e, (xx, m, n) in curves.items():
            ax.plot(xx, np.where(n >= MIN_N, m, np.nan), color=style.EVENT_COLOR[e], lw=2,
                    label=style.EVENT_LABEL[e])
        nT, nM = curves["T"][2], curves["M"][2]
        mT, mM = curves["T"][1], curves["M"][1]
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled = np.where(nT + nM >= MIN_N, (np.nan_to_num(mT) * nT + np.nan_to_num(mM) * nM) / (nT + nM), np.nan)
        ax.plot(x, pooled, "k--", lw=1.4, label="pooled")
        both = (nT >= MIN_BOTH) & (nM >= MIN_BOTH) & np.isfinite(mT) & np.isfinite(mM)
        if both.any():
            w = np.minimum(nT, nM)[both]
            gap = np.average(np.abs(mT - mM)[both], weights=w)
            spread = np.nanmax(pooled) - np.nanmin(pooled)
            ax.axvspan(x[both].min(), x[both].max(), color="0.9", zorder=0)
            facts[f"{key.replace('lb:', '')}: mean |T − M| at fixed RH"] = (
                f"{gap:.2f} (vs {spread:.2f} range of the pooled curve) over RH {x[both].min():.2f}–{x[both].max():.2f}")
            ax.text(0.03, 0.04, f"|T−M| = {gap:.2f}", transform=ax.transAxes, fontsize=9)
        if np.isfinite(rh0):
            ax.axvline(rh0, color="k", lw=0.8, ls=":")
        if key == "strike":
            ax.set_ylim(1000, 100)
        ax.set(xlim=(0, 1), xlabel="RH$_{2m}$", ylabel=ylabel)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Collapse test: T vs h2m extremes at fixed RH · grey = RH range where both have ≥ {MIN_BOTH} "
                 f"events · dotted = RH0 · {data['T'].label()}", fontsize=11)
    fig.tight_layout()
    facts.update(years=data["T"].label(), weighting="cos(lat)" if weighted else "none",
                 reading="small |T − M| compared with the pooled range → RH controls it; large → event type matters")
    return [{"path": FD.save(fig, out_dir, "collapse_rh"), "title": "Collapse test: T vs M at fixed RH",
             "facts": facts, "script": "figures/fig_collapse_rh.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
