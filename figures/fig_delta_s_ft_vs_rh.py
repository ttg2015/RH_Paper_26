"""Surface RH raises both terms: mean δ_S and δ_FT (and their sum) against surface RH.

Port of Paper_Plots-2 cell 160 (deltaS_deltaFT_vs_RH.pdf), T and M pooled, plain (unweighted) means
in 12 RH bins. Window: RH_LO–RH_HI (old values lived in the missing delta_ft_budget module; the
defaults here are stated on the figure). Old version also restricted to SP 995–1015 hPa; this one
keeps all terrain, as for the other figure data.
    python figures/fig_delta_s_ft_vs_rh.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

RH_LO, RH_HI, NBINS = 0.15, 0.85, 12


def means(data, product, weighted):
    d0 = data["T"]
    h = sum(d.h(product, weighted=weighted) for d in data.values())
    ex, ey = d0.edges_of(product, "x"), d0.edges_of(product, "y")
    edges = np.linspace(RH_LO, RH_HI, NBINS + 1)
    idx = np.digitize(FD.centers(ex), edges) - 1
    out = np.full(NBINS, np.nan)
    for k in range(NBINS):
        hk = h[idx == k].sum(0)
        out[k] = (hk * FD.centers(ey)).sum() / hk.sum() if hk.sum() else np.nan
    return FD.centers(edges), out


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    x, mS = means(data, "h2_rh_dS", weighted)
    _, mF = means(data, "h2_rh_dFT", weighted)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x, mS, "o-", color="#e67e22", label=r"$\delta_S = h_{2m}-h_{ML}$")
    ax.plot(x, mF, "o-", color="#2980b9", label=r"$\delta_{FT} = h_{ML}-h^*_{500}$")
    ax.plot(x, mS + mF, "k--", lw=1.5, label=r"total $\delta$")
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.grid(alpha=0.3)
    ax.set(xlabel="RH", ylabel="kJ kg$^{-1}$", title="Surface RH raises both terms")
    ax.legend(fontsize=9)
    fig.tight_layout()
    slope = lambda y: np.polyfit(x[np.isfinite(y)], y[np.isfinite(y)], 1)[0]
    facts = {"δ_S slope": f"{slope(mS):.2f} kJ/kg per unit RH", "δ_FT slope": f"{slope(mF):.2f} kJ/kg per unit RH",
             "window": f"{RH_LO}–{RH_HI}, {NBINS} bins", "weighting": "cos(lat)" if weighted else "none",
             "years": data["T"].label()}
    return [{"path": FD.save(fig, out_dir, "delta_s_ft_vs_rh"), "title": "δ_S and δ_FT vs RH", "facts": facts,
             "script": "figures/fig_delta_s_ft_vs_rh.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
