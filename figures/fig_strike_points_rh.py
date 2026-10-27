"""Plume strike points and plausibility, dry vs moist: T and h2m extremes pooled, split at RH0.

Figure 1 (strike_points_rh): (a, b) ε × plume-top pressure for dry and moist events, colour =
share of the unstable events (buoyant at ε = 0) in the class per 10 hPa, window, median top,
% of unstable events still buoyant above each column; (c) top distributions at ε_detail.
Figure 2 (ztop_fractions_rh): shares of the unstable events vs ε for dry and moist (too shallow ·
pass · too deep · diluted out, as fig_ztop_fractions.py); stable share printed in each panel.
    python figures/fig_strike_points_rh.py [--years 1980-2019] [--unweighted]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

import figdata as FD
import style

P_COARSEN = 2
CAT = FD.STRIKE_CAT


def pooled(data, name):
    return sum(d.hist[name] for d in data.values())


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    d0 = data["T"]
    suf = "_w" if weighted else ""
    if "strike_ptop_rh" + suf not in d0.hist:
        raise FileNotFoundError("no strike_*_rh products: rerun stage g after stage f")
    r0, rh0_source = FD.rh_split_value(data, weighted)
    lab = FD.split_label(r0)
    eps, pe, re_ = d0.epsilon, d0.edges["ptop"], d0.edges["rh_strike"]
    lo, hi = FD.load("analysis")["ztop_window_hPa"]
    kd = int(np.argmin(abs(eps - FD.load("figdata")["strike"]["eps_detail"])))
    hp = {}
    hc = {}
    hp["dry"], hp["moist"] = FD.split_rh(pooled(data, "strike_ptop_rh" + suf), re_, r0, axis=1)     # (ε, p)
    hc["dry"], hc["moist"] = FD.split_rh(pooled(data, "strike_category_rh" + suf), re_, r0, axis=1)  # (ε, code)
    pc = FD.centers(pe)
    facts = {"RH0 (split)": f"{r0:.3f} ({rh0_source})"}

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    x_e = np.concatenate([[eps[0] - (eps[1] - eps[0]) / 2], (eps[1:] + eps[:-1]) / 2,
                          [eps[-1] + (eps[-1] - eps[-2]) / 2]])
    for j, key in enumerate(("dry", "moist")):
        ax = fig.add_subplot(gs[0, j])
        total, stable, sh = FD.unstable_counts(hc[key])
        h, ep = FD.coarsen(hp[key], pe, P_COARSEN, axis=1)
        share = h / np.where(total[:, None] > 0, total[:, None], np.nan)
        im = ax.pcolormesh(x_e, ep, np.ma.masked_equal(share.T, 0) * 100, cmap="viridis", shading="flat")
        fig.colorbar(im, ax=ax, label="% of unstable events per 10 hPa", pad=0.02)
        med = [np.interp(0.5 * r.sum(), np.cumsum(r), pc) if r.sum() else np.nan for r in hp[key]]
        ax.plot(eps, med, "w-o", ms=4, lw=2, label="median top")
        for p in (lo, hi):
            ax.axhline(p, color="#ff7043", lw=1.5, ls="--")
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        buoy = 1 - sh[5]                                              # not diluted out
        for k, b in enumerate(buoy):
            ax.text(eps[k], 1.01, f"{100 * b:.0f}", transform=tr, ha="center", va="bottom", fontsize=7)
        ax.set(ylim=(1000, 100), xlabel="ε [km⁻¹]", ylabel="plume top [hPa]")
        ax.set_title(f"{lab[key]} · stable {100 * stable[0] / max(stable[0] + total[0], 1):.0f}% left out"
                     "\n(numbers above: % of unstable still buoyant)", pad=16)
        ax.legend(fontsize=7, loc="lower right", facecolor="0.3", labelcolor="w", framealpha=0.6)
        inwin = hp[key][kd][(pc >= lo) & (pc <= hi)].sum() / total[kd] if total[kd] else np.nan
        facts[f"{key}: median top at ε={eps[kd]:g}"] = f"{med[kd]:.0f} hPa"
        facts[f"{key}: share of unstable events topping in {lo}–{hi} hPa at ε={eps[kd]:g}"] = f"{100 * inwin:.0f}%"
        facts[f"{key}: % of unstable still buoyant at ε={eps[kd]:g}"] = f"{100 * buoy[kd]:.0f}%"
        facts[f"{key}: stable (never buoyant)"] = f"{100 * stable[0] / max(stable[0] + total[0], 1):.0f}%"
    ax = fig.add_subplot(gs[1, 0])
    for key, color in (("dry", FD.DRY_COLOR), ("moist", FD.MOIST_COLOR)):
        h, ep = FD.coarsen(hp[key][kd], pe, P_COARSEN)
        ax.plot(100 * h / max(h.sum(), 1), FD.centers(ep), color=color, lw=2, label=lab[key])
    ax.axhspan(hi, lo, color="#ff7043", alpha=0.12)
    ax.set(ylim=(1000, 100), xlabel="% of buoyant plumes per 10 hPa", ylabel="plume top [hPa]",
           title=f"Top distributions at ε = {eps[kd]:g} km⁻¹")
    ax.legend(fontsize=8)
    fig.suptitle(f"Plume strike points, dry vs moist (T + h2m pooled, split at RH0 = {r0:.2f}) · {d0.label()} · "
                 f"{style.band_text(FD.load('analysis'), weighted)}", fontsize=11)
    items = [{"path": FD.save(fig, out_dir, "strike_points_rh"), "title": "Plume strike points · dry vs moist",
              "facts": dict(facts, years=d0.label()), "script": "figures/fig_strike_points_rh.py"}]
    plt.close(fig)

    # category shares vs ε, dry and moist
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    ff = {}
    for ax, key in zip(axes, ("dry", "moist")):
        n, stable, sh = FD.unstable_counts(hc[key])
        order = FD.STRIKE_ORDER
        c6 = np.concatenate([hc[key], np.zeros(hc[key].shape[:-1] + (max(0, 6 - hc[key].shape[-1]),))], -1)
        allev = c6[:, 1:6].sum(1)
        frac = {k: c6[:, k] / np.where(allev > 0, allev, np.nan) for k in order}
        ax.stackplot(eps, *[100 * np.nan_to_num(frac[k]) for k in order], colors=[CAT[k][1] for k in order],
                     labels=[CAT[k][0] for k in order], alpha=0.9, edgecolor="white", lw=0.6)
        ratio = 100 * sh[1]
        st = 100 * stable[0] / max(stable[0] + n[0], 1)
        ff[f"{key}: stable (never buoyant)"] = f"{st:.0f}%"
        if np.nanmax(np.nan_to_num(ratio)) > 0:
            b = int(np.nanargmax(ratio))
            ax.text(0.98, 0.97 - st / 100, f"best: {ratio[b]:.0f}% of unstable events pass\nat ε = {eps[b]:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
            ff[f"{key}: best pass / unstable"] = f"{ratio[b]:.0f}% at ε = {eps[b]:.2f}"
        else:
            ff[f"{key}: best pass / unstable"] = "no passes"
        ax.set_xticks(np.arange(0.1, eps[-1] + 1e-9, 0.1))
        ax.set(xlim=(eps[0], eps[-1]), ylim=(0, 100), xlabel="ε [km⁻¹]",
               title=lab[key])
    axes[0].set_ylabel("share of events [%]")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h[::-1], l[::-1], fontsize=7, loc="upper right")
    fig.suptitle(f"Where do plumes top out? dry vs moist · pass = {lo}–{hi} hPa · unstable = buoyant at ε = 0",
                 fontsize=10)
    fig.tight_layout()
    items.append({"path": FD.save(fig, out_dir, "ztop_fractions_rh"), "title": "Plume tops vs ε · dry vs moist",
                  "facts": dict(ff, **{"RH0 (split)": f"{r0:.3f}"}), "script": "figures/fig_strike_points_rh.py"})
    plt.close(fig)
    return items


if __name__ == "__main__":
    FD.cli(make, __doc__)
