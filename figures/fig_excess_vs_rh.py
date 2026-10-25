"""Excess vs RH: density of δ_FT against surface RH, mean curves, and the RH0 zero crossing.

RH0 is where the pooled (T + M) mean δ_FT crosses zero, found by interpolation (E09). It feeds
the log-RH model (E11). The count strip shows T, M and their combined counts (E09 fix).
    python figures/fig_excess_vs_rh.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

C_RH, C_D = 2, 4          # coarsen to 1 % RH × 1 kJ/kg
XLIM, YLIM = (0.0, 1.0), (-20, 20)


def make(out_dir, years=None, weighted=True, product="h2_rh_dFT"):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    d0 = data["T"]
    ex, ey = d0.edges_of(product, "x"), d0.edges_of(product, "y")
    H = {e: d.h(product, weighted=weighted) for e, d in data.items()}
    pooled = H["T"] + H["M"]
    # density panel (coarse) and mean curves (1 % RH bins)
    Hc, ex_c = FD.coarsen(pooled, ex, C_RH, axis=0)
    Hc, ey_c = FD.coarsen(Hc, ey, C_D, axis=1)
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1]})
    dens = Hc / Hc.sum()
    pc = ax.pcolormesh(ex_c, ey_c, np.ma.masked_equal(dens.T, 0), cmap="Purples", shading="flat",
                       vmax=np.percentile(dens[dens > 0], 99.5))
    fig.colorbar(pc, ax=ax, label="share of events", pad=0.01)
    facts = {}
    for e, h in list(H.items()) + [("pooled", pooled)]:
        hx, exx = FD.coarsen(h, ex, C_RH, axis=0)
        mean, n = FD.binned_mean(hx, exx, ey)
        mean[n < 20] = np.nan
        x = FD.centers(exx)
        color = style.EVENT_COLOR.get(e, "k")
        ax.plot(x, mean, color=color, lw=2 if e == "pooled" else 1.3,
                label="pooled mean" if e == "pooled" else f"mean, {style.EVENT_LABEL[e]}")
        rh0 = FD.zero_crossing(x[::-1], mean[::-1])          # crossing going from high to low RH
        facts[f"RH0 ({e})"] = f"{rh0:.3f}" if np.isfinite(rh0) else "none"
        axn.plot(x, n / n.sum() * 100, color=color, lw=1.2)
    rh0 = float(facts["RH0 (pooled)"]) if facts["RH0 (pooled)"] != "none" else np.nan
    if np.isfinite(rh0):
        ax.axvline(rh0, color="k", ls="--", lw=1)
        ax.text(rh0, YLIM[1] * 0.92, f" RH$_0$ = {rh0:.3f}", fontsize=9)
    ax.axhline(0, color="k", ls=":", lw=1.2)
    ax.set(ylim=YLIM, ylabel=f"{style.DELTA_LABEL['delta_FT']}  [kJ/kg]",
           title=f"{style.DELTA_SHORT['delta_FT']} vs surface RH · {d0.label()} · "
                 f"{style.band_text(FD.load('analysis'), weighted)}")
    ax.legend(frameon=False, loc="lower left")
    axn.set(xlim=XLIM, xlabel="RH$_{2m}$", ylabel="% events")
    fig.tight_layout()
    facts.update(years=d0.label(), weighting="cos(lat)" if weighted else "none",
                 density_bins="1 % RH × 1 kJ/kg", mean_bins="1 % RH (≥ 20 events)")
    return [{"path": FD.save(fig, out_dir, "excess_vs_rh"), "title": "Excess vs RH", "facts": facts,
             "script": "figures/fig_excess_vs_rh.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
