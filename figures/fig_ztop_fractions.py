"""Stage f: where do entraining plumes top out? Category shares vs ε, one panel per event type.

Stacked bottom → top: too deep · pass · too shallow · diluted out · stable, as shares of all
events. Stable (never buoyant, even with no entrainment) is the same grey cap at every ε.
The pass rate quoted on the plot is pass / unstable (unstable = buoyant at ε = 0), a fixed
denominator at every ε.

    python figures/fig_ztop_fractions.py                          # every ztop file, both events
    python figures/fig_ztop_fractions.py --years 1980-1989 --weighted
    python figures/fig_ztop_fractions.py --out runs/figs/ztop_fractions

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
EVENT_NAME = {"T": "2 m temperature extremes", "M": "2 m MSE (wet-bulb) extremes"}


def plot(by_event, weighted=False, window=(400, 600)):
    fig, axes = plt.subplots(1, len(by_event), figsize=(5.2 * len(by_event), 4.0),
                             squeeze=False, sharey=True)
    for ax, (event, dss) in zip(axes[0], by_event.items()):
        eps, fr = F.fractions(dss, weighted)
        sh, stable = F.of_unstable(fr)
        order = F.ORDER
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
        best = int(np.nanargmax(sh[1]))
        ax.text(0.98, 0.97 - stable[0], f"best: {100 * sh[1][best]:.0f}% of unstable events pass\n"
                f"at ε = {eps[best]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        n_years = len({d.attrs["year"] for d in dss})
        ax.set_xticks(np.arange(0.1, eps[-1] + 1e-9, 0.1))
        ax.set(xlim=(eps[0], eps[-1]), ylim=(0, 100), xlabel="entrainment rate ε [km⁻¹]",
               title=f"{EVENT_NAME.get(event, event)}\n{n_years} year{'s' * (n_years > 1)}"
                     f"{' · cos(lat) weighted' if weighted else ''}")
    axes[0][0].set_ylabel("share of events [%]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=len(F.ORDER), fontsize=8, frameon=False)
    fig.suptitle(f"Where do entraining plumes from the LPL top out? Pass = top within "
                 f"{window[0]}–{window[1]} hPa · unstable = buoyant at ε = 0 (not stable)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    return fig


def make(out_dir, years=None, weighted=False):
    """For make_all.py: both events, every ztop file in the year range."""
    y0, y1 = years or (0, 9999)
    by_event = {}
    for year, event in F.available():
        if y0 <= year <= y1:
            by_event.setdefault(event, []).append(F.open_ztop(year, event))
    if not by_event:
        raise FileNotFoundError("no ztop files")
    by_event = {e: by_event[e] for e in ("T", "M") if e in by_event}
    fig = plot(by_event, weighted, tuple(F.load("analysis")["ztop_window_hPa"]))
    out = Path(out_dir) / "ztop_fractions"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    facts = {}
    for e, dss in by_event.items():
        eps, fr = F.fractions(dss, weighted)
        sh, stable = F.of_unstable(fr)
        k = int(np.nanargmax(sh[1]))
        facts[f"best pass/unstable {e}"] = (f"{100 * sh[1][k]:.0f}% at ε = {eps[k]:.2f} "
                                            f"(diluted out {100 * sh[5][k]:.0f}%, {len(dss)} years)")
        facts[f"stable {e}"] = f"{100 * stable[0]:.0f}% of all events"
    facts["weighting"] = "cos(lat)" if weighted else "none"
    return [{"path": out.with_suffix(".png"), "title": "Plume tops vs ε (stage f)", "facts": facts,
             "script": "figures/fig_ztop_fractions.py"}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", help="e.g. 1980-2019 (default: every ztop file found)")
    ap.add_argument("--events", nargs="+", default=["T", "M"], choices=["T", "M"])
    ap.add_argument("--weighted", action="store_true", help="cos(latitude) area weights (E02)")
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
    fig = plot(by_event, a.weighted, window)
    out = Path(a.out + ("_weighted" if a.weighted else ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    print(f"Wrote {out}.pdf and .png  ({', '.join(f'{e}: {len(d)} yr' for e, d in by_event.items())})")
    print(F.table(by_event))


if __name__ == "__main__":
    main()
