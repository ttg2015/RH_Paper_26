"""δ_FT vs LCL height above ground, for T and M events (first step towards the LCL-binned figure).

z_LCL is the LCL of 2 m air above the ground; legacy d files are corrected on read in stage g (E07).
    python figures/fig_excess_vs_zlcl.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

C_Z, C_D = 4, 4           # 100 m × 1 kJ/kg
PRODUCT = "h2_zlcl_dFT"


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    facts = {}
    for ax, (e, d) in zip(axes, data.items()):
        ex, ey = d.edges_of(PRODUCT, "x"), d.edges_of(PRODUCT, "y")
        h = d.h(PRODUCT, weighted=weighted)
        hc, exc = FD.coarsen(h, ex, C_Z, axis=0)
        hc2, eyc = FD.coarsen(hc, ey, C_D, axis=1)
        dens = hc2 / hc2.sum()
        ax.pcolormesh(exc / 1000, eyc, np.ma.masked_equal(dens.T, 0), cmap="Greys", shading="flat",
                      vmax=np.percentile(dens[dens > 0], 99.5))
        mean, n = FD.binned_mean(hc, exc, ey)
        mean[n < 20] = np.nan
        ax.plot(FD.centers(exc) / 1000, mean, color=style.EVENT_COLOR[e], lw=2, label="mean")
        ax.axhline(0, color="k", ls=":", lw=1)
        zc = FD.centers(exc)
        facts[f"median z_LCL {e}"] = f"{np.interp(0.5, np.cumsum(n) / n.sum(), zc):.0f} m"
        ax.set(xlim=(0, 4), ylim=(-25, 25), xlabel="z$_{LCL}$ of 2 m air above ground [km]",
               title=style.EVENT_LABEL[e])
    axes[0].set_ylabel(f"{style.DELTA_LABEL['delta_FT']}  [kJ/kg]")
    d0 = data["T"]
    fig.suptitle(f"{style.DELTA_SHORT['delta_FT']} vs LCL height · {d0.label()} · "
                 f"{style.band_text(FD.load('analysis'), weighted)}")
    fig.tight_layout()
    facts.update(years=d0.label(), zlcl="above ground (E07 corrected)", bins="100 m × 1 kJ/kg")
    return [{"path": FD.save(fig, out_dir, "excess_vs_zlcl"), "title": "δ_FT vs LCL height", "facts": facts,
             "script": "figures/fig_excess_vs_zlcl.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
