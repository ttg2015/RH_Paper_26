"""RH Cascades: RH-binned composite MSE profiles in four free-tropospheric energy boxes.

Port of plot_cascade (Cascade_Summary cell 14) onto stage g figure data: Cleaned 1980–2019 only
(fixes E03), boxes from config/figdata.yaml, cos(lat)-weighted means, below-ground levels left
out of each composite. Solid h, dashed h* (≤ 800 hPa); one colour per RH bin.
    python figures/fig_cascade.py [--years 1980-2019] [--min-count 500]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

PANELS = [["T_MLL", "M_MLL"], ["T_TPL", "M_TPL"]]
TITLES = [[r"Middle Latitude Like ($T_{2m}$ Max)", r"Middle Latitude Like ($h_{2m}$ Max)"],
          [r"Tropics Like ($T_{2m}$ Max)", r"Tropics Like ($h_{2m}$ Max)"]]
RH_COARSEN = 5                  # 1 % → 5 % bins
MIN_COUNT = 500
SAT_BELOW = 800.0
XLIM, YLIM = (315, 355), (1000, 100)


def composites(d):
    """(case, rh_bin, level) means of h and h*, event counts, and the 5 % edges."""
    e = d.edges["rh_cascade"]
    out = {}
    w, _ = FD.coarsen(d.hist["casc_weight"], e, RH_COARSEN, axis=1)
    n, e5 = FD.coarsen(d.hist["casc_count"], e, RH_COARSEN, axis=1)
    for v in ("MSE", "MSE_sat"):
        s, _ = FD.coarsen(d.hist[f"casc_sum_{v}"], e, RH_COARSEN, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[v] = s / w
    return out, n, e5


def make(out_dir, years=None, weighted=True, min_count=MIN_COUNT):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    legend = {0: {}, 1: {}}
    facts = {}
    for i in range(2):
        for j in range(2):
            ax, case = axes[i][j], PANELS[i][j]
            d = data[case[0]]
            comp, n, e5 = composites(d)
            c = d.cases.index(case)
            cmap = plt.cm.Reds if j == 0 else plt.cm.Blues
            for p in (600, 400):
                ax.axhline(p, c="gray", alpha=0.4)
            kept = 0
            for b in range(len(e5) - 1):
                if n[c, b] < min_count:
                    continue
                rh = 0.5 * (e5[b] + e5[b + 1])
                col = cmap(0.15 + 0.85 * rh)
                lab = f"RH {e5[b] * 100:.0f}–{e5[b + 1] * 100:.0f}%"
                ax.plot(comp["MSE"][c, b], d.levels, color=col, lw=2)
                sat = np.where(d.levels <= SAT_BELOW, comp["MSE_sat"][c, b], np.nan)
                ax.plot(sat, d.levels, color=col, lw=2, ls="--")
                legend[j].setdefault(lab, col)
                kept += n[c, b]
            ax.text(0.03, 0.97, f"n = {int(kept):,}", transform=ax.transAxes, ha="left", va="top",
                    color="#444", fontstyle="italic")
            ax.set_title(TITLES[i][j], fontsize=13)
            facts[f"{case} events"] = f"{int(kept):,} in bins with ≥ {min_count}"
    axes[0][0].set(xlim=XLIM, ylim=YLIM)
    for a in axes[:, 0]:
        a.set_ylabel("Pressure [hPa]")
    for a in axes[1]:
        a.set_xlabel("MSE [kJ/kg]")
    styl = [plt.Line2D([0], [0], color="k", lw=2, label="$h$"),
            plt.Line2D([0], [0], color="k", lw=2, ls="--", label=f"$h^*$ (≤{int(SAT_BELOW)} hPa)")]
    for j in (0, 1):
        items = sorted(legend[j].items(), key=lambda kv: float(kv[0].split()[1].split("–")[0]))
        fig.legend(handles=styl + [plt.Line2D([0], [0], color=c, lw=2, label=l) for l, c in items],
                   loc="lower center", bbox_to_anchor=(0.27 if j == 0 else 0.75, 0.0), ncol=4,
                   title="$T_{2m}$ RH bins" if j == 0 else "$h_{2m}$ RH bins", fontsize=9, frameon=True)
    d0 = data["T"]
    fig.suptitle(f"RH cascades · {d0.label()} · {style.band_text(FD.load('analysis'), True)}", fontsize=12)
    fig.tight_layout(rect=(0, 0.14, 1, 0.97))
    facts.update(years=d0.label(), bins="5 % RH", boxes="config/figdata.yaml → cascade.boxes",
                 note="cos(lat)-weighted means; below-ground levels excluded")
    return [{"path": FD.save(fig, out_dir, "cascade"), "title": "RH Cascades", "facts": facts,
             "script": "figures/fig_cascade.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
