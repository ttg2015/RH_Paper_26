"""Plume tops (paper figure, merges old Figs 6 and 7): one column's plumes beside where all plumes top out.

(a) the plume case of fig_plume_case.py: the mean profile of Fig. 7's regime in Fig. 1b's RH bin, plumes at
    style.EPS_SHOWN and at the two ends of the sweep (style.BRACKET_EPS: too weak in maroon, too strong in
    warm gray), with arrowheads showing ascent;
(b) the pooled plume tops of fig_ztop_pooled.py: for the same five ε, in the same colors, the share of
    unstable T + wet-bulb events whose plume tops out in each 20 hPa bin.
The two panels share the pressure axis, so a top in (a) can be read straight across into (b). Both panels
are drawn by those scripts' panel() helpers, so the separate figures and this one cannot drift apart.
(a) reads the regime composites (casc files on Casper, or --schematic-cache anywhere); --column draws
Fig. 1b's single model-level column instead (needs <out>/example_profiles_cache.npz).
    python figures/fig_plume_tops.py [--years 1980-2019] [--schematic-cache F] [--column] [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figdata as FD
import fig_plume_case as PC
import fig_ztop_pooled as ZP
import style

REPO = Path(__file__).resolve().parents[1]
WEIGHTED = False


def draw(one, label, pooled, out_dir, brackets=True, window=(400.0, 600.0)):
    """one, label: the case column (fig_plume_case.case); pooled: fig_ztop_pooled.pooled() output."""
    style.apply()
    eps, h, cat, pe, years = pooled
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.4, 5.4), sharey=True,
                                     gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.08})
    hand_a, facts_a = PC.panel(ax_a, one, label)
    hand_b, facts_b = ZP.panel(ax_b, eps, h, cat, pe, brackets=style.BRACKET_EPS if brackets else None,
                               window=window)
    bottom = max(float(one["p0"][0]) + 15, 1000.0)
    ax_a.set_ylim(bottom, 100)                                     # shared: (b) follows
    ax_b.set_ylabel("")
    ax_b.tick_params(labelleft=False)
    for ax, hand in ((ax_a, hand_a), (ax_b, hand_b)):
        ax.legend(handles=hand, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1,
                  frameon=False, handlelength=2.0, labelspacing=0.3)
    ax_a.set_title("(a) mean profile like Fig. 1b", fontsize=9.5, loc="left", fontweight="bold")
    ax_b.set_title("(b) all unstable events", fontsize=9.5, loc="left", fontweight="bold")
    facts = {f"(a) {k}": v for k, v in facts_a.items()}
    facts.update({f"(b) {k}": v for k, v in facts_b.items()})
    facts["(b) years"] = years
    path = FD.save(fig, out_dir, "plume_tops")
    plt.close(fig)
    return {"path": path, "title": "Plume tops (old Figs 6 + 7)", "facts": facts,
            "script": "figures/fig_plume_tops.py"}


def make(out_dir, years=None, weighted=WEIGHTED, cache=None, brackets=True, source="composite",
         schematic_cache=None):
    one, label = PC.case(out_dir, cache=cache, source=source, years=years, schematic_cache=schematic_cache)
    window = tuple(FD.load("analysis")["ztop_window_hPa"])
    return [draw(one, label, ZP.pooled(years, weighted), out_dir, brackets, window)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", help="e.g. 1980-2019 (default: all figdata files)")
    ap.add_argument("--cache", help="Fig. 1 cache (default <out>/example_profiles_cache.npz)")
    ap.add_argument("--no-brackets", dest="brackets", action="store_false")
    ap.add_argument("--column", dest="source", action="store_const", const="column", default="composite")
    ap.add_argument("--schematic-cache", help="composite cache of fig_plume_schematic (works without Casper)")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs"))
    a = ap.parse_args()
    for item in make(Path(a.out), FD.years_arg(a.years), cache=a.cache, brackets=a.brackets, source=a.source,
                     schematic_cache=a.schematic_cache):
        print(f"Wrote {item['path']}")
        for k, v in item["facts"].items():
            print(f"  {k}: {v}")
