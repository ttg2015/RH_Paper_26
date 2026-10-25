"""Stage f: where do entraining plumes top out? Category shares vs ε, one panel per event type.

Stacked bottom → top: too shallow · no buoyancy · pass · too deep, so "too low" sits under
the pass band, "too high" sits over it, and the stable (no buoyancy) events lie next to pass.
The dashed line (right axis) is pass / buoyant: the share of plumes that can rise at all whose
top lands in the window.

    python figures/fig_ztop_fractions.py                          # every ztop file, both events
    python figures/fig_ztop_fractions.py --years 1980-1989 --weighted
    python figures/fig_ztop_fractions.py --order 2 4 1 3 --out runs/figs/ztop_fractions

Reads Ztop/ztop_{E}_b06_{YEAR}.nc only. Writes <out>.pdf and <out>.png.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage_f_checks as F                                   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ORDER = [2, 4, 1, 3]          # too shallow, no buoyancy, pass, too deep (bottom → top)
EVENT_NAME = {"T": "2 m temperature extremes", "M": "2 m MSE (wet-bulb) extremes"}


def plot(by_event, order=ORDER, weighted=False, window=(400, 600)):
    fig, axes = plt.subplots(1, len(by_event), figsize=(5.2 * len(by_event), 4.0),
                             squeeze=False, sharey=True)
    for ax, (event, dss) in zip(axes[0], by_event.items()):
        eps, fr = F.fractions(dss, weighted)
        ax.stackplot(eps, *[100 * fr[c] for c in order], colors=[F.CAT[c][1] for c in order],
                     labels=[F.CAT[c][0] for c in order], alpha=0.9, edgecolor="white", lw=0.6)
        # label each band where it is thickest
        base = np.zeros_like(eps)
        for c in order:
            k = 1 + int(np.argmax(fr[c][1:-1]))          # thickest point away from the edges
            if fr[c][k] > 0.06:
                ax.text(eps[k], 100 * (base[k] + fr[c][k] / 2), F.CAT[c][0], ha="center", va="center",
                        fontsize=8, color="white" if c in (1, 3) else "#263238", weight="bold")
            base = base + fr[c]
        buoy = 1 - fr[4]
        ratio = 100 * fr[1] / np.where(buoy > 0, buoy, np.nan)
        ax2 = ax.twinx()
        ax2.plot(eps, ratio, "k--o", ms=3.5, lw=1.3, label="pass / buoyant")
        best = int(np.nanargmax(ratio))
        ax2.annotate(f"{ratio[best]:.0f}% at ε = {eps[best]:.2f}", (eps[best], ratio[best]),
                     xytext=(6, -14), textcoords="offset points", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax2.set(ylim=(0, 100))
        ax2.set_ylabel("pass / buoyant [%]" if ax is axes[0][-1] else "")
        if ax is not axes[0][-1]:
            ax2.set_yticklabels([])
        n_years = len({d.attrs["year"] for d in dss})
        ax.set_xticks(np.arange(0.1, eps[-1] + 1e-9, 0.1))
        ax.set(xlim=(eps[0], eps[-1]), ylim=(0, 100), xlabel="entrainment rate ε [km⁻¹]",
               title=f"{EVENT_NAME.get(event, event)}\n{n_years} year{'s' * (n_years > 1)}"
                     f"{' · cos(lat) weighted' if weighted else ''}")
    axes[0][0].set_ylabel("share of events [%]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[::-1] + [plt.Line2D([], [], color="k", ls="--", marker="o", ms=3.5)],
               labels[::-1] + ["pass / buoyant (right axis)"], loc="lower center",
               ncol=len(order) + 1, fontsize=8, frameon=False)
    fig.suptitle(f"Where do entraining plumes from the LPL top out? Pass = top within "
                 f"{window[0]}–{window[1]} hPa", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def make(out_dir, years=None, weighted=True):
    """For make_all.py: both events, every ztop file in the year range."""
    y0, y1 = years or (0, 9999)
    by_event = {}
    for year, event in F.available():
        if y0 <= year <= y1:
            by_event.setdefault(event, []).append(F.open_ztop(year, event))
    if not by_event:
        raise FileNotFoundError("no ztop files")
    by_event = {e: by_event[e] for e in ("T", "M") if e in by_event}
    fig = plot(by_event, ORDER, weighted, tuple(F.load("analysis")["ztop_window_hPa"]))
    out = Path(out_dir) / "ztop_fractions"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    facts = {}
    for e, dss in by_event.items():
        eps, fr = F.fractions(dss, weighted)
        ratio = fr[1] / (1 - fr[4])
        k = int(np.nanargmax(ratio))
        facts[f"best pass/buoyant {e}"] = f"{100 * ratio[k]:.0f}% at ε = {eps[k]:.2f} ({len(dss)} years)"
    facts["weighting"] = "cos(lat)" if weighted else "none"
    return [{"path": out.with_suffix(".png"), "title": "Plume tops vs ε (stage f)", "facts": facts,
             "script": "figures/fig_ztop_fractions.py"}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", help="e.g. 1980-2019 (default: every ztop file found)")
    ap.add_argument("--events", nargs="+", default=["T", "M"], choices=["T", "M"])
    ap.add_argument("--weighted", action="store_true", help="cos(latitude) area weights (E02)")
    ap.add_argument("--order", nargs=4, type=int, default=ORDER,
                    help="category codes bottom → top: 1 pass, 2 too shallow, 3 too deep, 4 no buoyancy")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs" / "ztop_fractions"))
    a = ap.parse_args(argv)
    y0, y1 = (map(int, a.years.split("-")) if a.years and "-" in a.years
              else (int(a.years), int(a.years)) if a.years else (0, 9999))
    by_event = {}
    for year, event in F.available():
        if event in a.events and y0 <= year <= y1:
            by_event.setdefault(event, []).append(F.open_ztop(year, event))
    if not by_event:
        sys.exit("No ztop files found for that selection.")
    by_event = {e: by_event[e] for e in a.events if e in by_event}
    window = tuple(F.load("analysis")["ztop_window_hPa"])
    fig = plot(by_event, a.order, a.weighted, window)
    out = Path(a.out + ("_weighted" if a.weighted else ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    print(f"Wrote {out}.pdf and .png  ({', '.join(f'{e}: {len(d)} yr' for e, d in by_event.items())})")
    print(F.table(by_event))


if __name__ == "__main__":
    main()
