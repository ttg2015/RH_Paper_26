"""RH Cascades: RH-binned composite MSE profiles in four free-tropospheric energy boxes.

Port of plot_cascade (Cascade_Summary cell 14) onto stage g figure data: Cleaned 1980–2019 only
(fixes E03), boxes from config/figdata.yaml, cos(lat)-weighted means, below-ground levels left
out of each composite. Solid h, dashed h* (at and above 500 hPa only); one colour per RH bin.

Surface handling: a composite level is drawn only where at least COVERAGE_MIN of the bin's events
have that level above ground, so every drawn level averages (almost) the same events and the
surface-pressure mixing that made hooks near 1000 hPa is cut off.
Top row: where the events sit in (h*500, h500) space, with the MLL and TPL boxes outlined.

Two versions are written:
  cascade            y = pressure (hPa), levels cut where < COVERAGE_MIN of events reach them
  cascade_surface    y = pressure above the surface, Δp = SP − p, so every event contributes at
                     every height and the surface (h_2m at Δp = 0) lines up across events
    python figures/fig_cascade.py [--years 1980-2019]
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
SAT_BELOW = 500.0               # h* drawn at and above this level only
COVERAGE_MIN = 0.9              # share of a bin's events that must have the level above ground
XLIM, YLIM = (315, 355), (1000, 100)
PLANE = (310, 360)


DP_SAT_FROM = 500.0             # surface version: h* drawn from this Δp upward
DP_LIM = (0, 900)


def composites_dp(d):
    """(case, rh_bin, Δp) means of h and h* on the surface-relative grid."""
    e = d.edges["rh_cascade"]
    n, e5 = FD.coarsen(d.hist["casc_count"], e, RH_COARSEN, axis=1)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for v in ("MSE", "MSE_sat"):
            s, _ = FD.coarsen(d.hist[f"casc_dp_sum_{v}"], e, RH_COARSEN, axis=1)
            w, _ = FD.coarsen(d.hist[f"casc_dp_weight_{v}"], e, RH_COARSEN, axis=1)
            out[v] = s / w
    return out, n, e5


def composites(d):
    """(case, rh_bin, level) means of h and h*, event counts, and the 5 % edges."""
    e = d.edges["rh_cascade"]
    out = {}
    w, _ = FD.coarsen(d.hist["casc_weight"], e, RH_COARSEN, axis=1)
    n, e5 = FD.coarsen(d.hist["casc_count"], e, RH_COARSEN, axis=1)
    if "casc_weight_scalar" in d.hist:                     # total weight of each bin's events
        wt, _ = FD.coarsen(d.hist["casc_weight_scalar"], e, RH_COARSEN, axis=1)
    else:
        wt = w.max(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cover = w / wt[..., None]
        for v in ("MSE", "MSE_sat"):
            s, _ = FD.coarsen(d.hist[f"casc_sum_{v}"], e, RH_COARSEN, axis=1)
            out[v] = np.where(cover >= COVERAGE_MIN, s / w, np.nan)
    return out, n, e5


def make(out_dir, years=None, weighted=True, min_count=MIN_COUNT):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    items = [figure(data, weighted, min_count, out_dir, surface=False)]
    if "casc_dp_sum_MSE" in data["T"].hist:
        items.append(figure(data, weighted, min_count, out_dir, surface=True))
    return items


def figure(data, weighted, min_count, out_dir, surface):
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.75, 1, 1], hspace=0.28, wspace=0.12)
    boxes = FD.load("figdata")["cascade"]["boxes"]
    from matplotlib.patches import Rectangle
    for j, e in enumerate(("T", "M")):                     # where are we selecting from?
        ax = fig.add_subplot(gs[0, j])
        d = data[e]
        name = "h2_hs500_h500"
        if name in d.hist or name + "_w" in d.hist:
            h = d.h(name, weighted=weighted)
            h, ex, ey = FD.crop(h, d.edges_of(name, "x"), d.edges_of(name, "y"), PLANE, PLANE)
            ax.pcolormesh(ex, ey, np.ma.masked_equal(h.T, 0), cmap="Reds" if e == "T" else "Blues")
        ax.plot(PLANE, PLANE, color="k", lw=0.8, alpha=0.5)
        for case, box in boxes.items():
            if not case.startswith(e + "_"):
                continue
            (x0, x1), (y0, y1) = box["h_star"], box["h"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2, ec="k",
                                   ls="-" if "TPL" in case else "--"))
            ax.text(x1 + 0.5, y1 + 0.5, case.split("_")[1], fontsize=9, fontweight="bold")
        ax.set(xlim=PLANE, ylim=PLANE, xlabel=r"$h^*_{500}$ [kJ/kg]", ylabel=r"$h_{500}$ [kJ/kg]",
               title=f"Event selection · {style.EVENT_LABEL[e]}")
    axes = np.array([[fig.add_subplot(gs[1 + i, j]) for j in range(2)] for i in range(2)])
    for a in axes.ravel()[1:]:
        a.sharex(axes[0][0]); a.sharey(axes[0][0])
    legend = {0: {}, 1: {}}
    facts = {}
    for i in range(2):
        for j in range(2):
            ax, case = axes[i][j], PANELS[i][j]
            d = data[case[0]]
            comp, n, e5 = composites_dp(d) if surface else composites(d)
            y = d.dp_levels if surface else d.levels
            c = d.cases.index(case)
            cmap = plt.cm.Reds if j == 0 else plt.cm.Blues
            for p in ((400, 600) if surface else (600, 400)):
                ax.axhline(p, c="gray", alpha=0.4)
            kept = 0
            for b in range(len(e5) - 1):
                if n[c, b] < min_count:
                    continue
                rh = 0.5 * (e5[b] + e5[b + 1])
                col = cmap(0.15 + 0.85 * rh)
                lab = f"RH {e5[b] * 100:.0f}–{e5[b + 1] * 100:.0f}%"
                ax.plot(comp["MSE"][c, b], y, color=col, lw=2)
                cut = (y >= DP_SAT_FROM) if surface else (y <= SAT_BELOW)
                ax.plot(np.where(cut, comp["MSE_sat"][c, b], np.nan), y, color=col, lw=2, ls="--")
                legend[j].setdefault(lab, col)
                kept += n[c, b]
            ax.text(0.03, 0.97, f"n = {int(kept):,}", transform=ax.transAxes, ha="left", va="top",
                    color="#444", fontstyle="italic")
            ax.set_title(TITLES[i][j], fontsize=13)
            facts[f"{case} events"] = f"{int(kept):,} in bins with ≥ {min_count}"
    axes[0][0].set(xlim=XLIM, ylim=DP_LIM if surface else YLIM)
    for a in axes[:, 0]:
        a.set_ylabel("Pressure above surface, SP − p [hPa]" if surface else "Pressure [hPa]")
    for a in axes[1]:
        a.set_xlabel("MSE [kJ/kg]")
    styl = [plt.Line2D([0], [0], color="k", lw=2, label="$h$"),
            plt.Line2D([0], [0], color="k", lw=2, ls="--",
                       label=f"$h^*$ (Δp ≥ {int(DP_SAT_FROM)} hPa)" if surface else f"$h^*$ (≤{int(SAT_BELOW)} hPa)")]
    for j in (0, 1):
        items = sorted(legend[j].items(), key=lambda kv: float(kv[0].split()[1].split("–")[0]))
        fig.legend(handles=styl + [plt.Line2D([0], [0], color=c, lw=2, label=l) for l, c in items],
                   loc="lower center", bbox_to_anchor=(0.27 if j == 0 else 0.75, 0.0), ncol=4,
                   title="$T_{2m}$ RH bins" if j == 0 else "$h_{2m}$ RH bins", fontsize=9, frameon=True)
    d0 = data["T"]
    fig.suptitle(f"RH cascades{' · surface-relative' if surface else ''} · {d0.label()} · "
                 f"{style.band_text(FD.load('analysis'), True)}", fontsize=12)
    fig.subplots_adjust(bottom=0.13, top=0.95)
    facts.update(years=d0.label(), bins="5 % RH", boxes="config/figdata.yaml → cascade.boxes",
                 note=f"cos(lat)-weighted means; a level is drawn only where ≥ {COVERAGE_MIN:.0%} of the bin's "
                      "events have it above ground; h* drawn at and above 500 hPa")
    if surface:
        facts["note"] = ("cos(lat)-weighted means on Δp = SP − p; h uses h_2m at Δp = 0 and blends to h_LPL; "
                         f"h* drawn from Δp = {DP_SAT_FROM:g} hPa up")
    name = "cascade_surface" if surface else "cascade"
    item = {"path": FD.save(fig, out_dir, name), "facts": facts, "script": "figures/fig_cascade.py",
            "title": "RH Cascades" + (" · surface-relative" if surface else "")}
    plt.close(fig)
    return item


if __name__ == "__main__":
    FD.cli(make, __doc__)
