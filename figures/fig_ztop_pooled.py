"""Plume tops, pooled (paper figure): where entraining plumes top out, T and wet-bulb extremes together.

From stage f's plume tops summed in stage g (strike_ptop, strike_category), pooled over years and over
both event types. One curve per ε in SHOWN (0.1 plus style.EPS_SHOWN), colored as in the plume case study
(fig_plume_case.py). Each curve is the share of unstable events (buoyant at ε = 0) whose plume tops out
in each 20 hPa bin, so its area is the share that tops out anywhere; the rest are diluted out. The legend
gives the share topping in the 400–600 hPa window and the share diluted out. With --brackets, the
smallest and largest ε of the sweep are added as thin maroon curves (outside the plausible range), as in the
case study. No dashed lines and no fills (decluttered 2026-11-10); --no-brackets drops the maroon curves.

Not latitude weighted, like the other stage f figures and the shares quoted in the text.
    python figures/fig_ztop_pooled.py [--years 1980-2019] [--brackets]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figdata as FD
import style

REPO = Path(__file__).resolve().parents[1]
WEIGHTED = False
SHOWN = (0.1,) + tuple(style.EPS_SHOWN)              # ε drawn in color: the case-study three plus 0.1
P_COARSEN = 4                                        # 5 → 20 hPa (10 hPa curves were jagged)
DILUTED, PASS = 5, 1                                 # stage f category codes
BRACKET_STYLE = {r: dict(color=c, ls=style.BRACKET_LS) for r, c in style.BRACKET_COLOR.items()}


def quantile_p(h, pc, q):
    c = np.cumsum(h)
    return float(np.interp(q * c[-1], c, pc)) if c[-1] else np.nan


def pooled(years, weighted=False):
    """Pooled T + M: (ε, top counts (ε, p), category counts (ε, code), p edges, year label)."""
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    d0 = data["T"]
    if "strike_ptop" not in d0.hist:
        raise FileNotFoundError("no strike_* products: run stage f, then stage g")
    suf = "_w" if weighted else ""
    h = sum(d.hist["strike_ptop" + suf][d.subbands.index("all")] for d in data.values())
    cat = sum(d.hist["strike_category"][d.subbands.index("all")] for d in data.values())
    return d0.epsilon, h, cat, d0.edges["ptop"], d0.label()


def panel(ax, eps, h, cat, pe, shown=None, brackets=None, window=(400.0, 600.0)):
    """Draw on ax; returns (legend handles, facts). fig_plume_tops.py uses it beside the plume case.
    eps (E,), h (E, P) top counts, cat (E, codes) category counts, pe (P+1,) pressure edges.
    shown: ε to draw in color (default SHOWN); brackets: (weak, strong) ε in maroon, or None."""
    shown = tuple(SHOWN if shown is None else shown)
    lo, hi = window
    total, stable, share = FD.unstable_counts(cat)
    pc = FD.centers(pe)
    k_of = lambda e: int(np.argmin(abs(eps - e)))
    for e in shown + tuple(b for b in (brackets or ()) if b is not None):
        if abs(eps[k_of(e)] - e) > 1e-6:
            raise ValueError(f"ε = {e} not in the stage f sweep {list(eps)}")

    style.paper_axes(ax)
    style.band(ax, lo, hi, edges=False, label=f"{lo:g}–{hi:g} hPa")

    handles, facts, xmax = [], {}, 0.0
    curves = [(e, None) for e in shown]
    if brackets:
        curves = [(brackets[0], "weak")] + curves + [(brackets[1], "strong")]
    for e, role in curves:
        k = k_of(e)
        hc, pce = FD.coarsen(h[k], pe, P_COARSEN)
        y = FD.centers(pce)
        x = 100 * hc / total[k] if total[k] else np.zeros_like(hc, float)
        xmax = max(xmax, np.nanmax(x))
        in_win = 100 * h[k][(pc >= lo) & (pc <= hi)].sum() / total[k]
        dil = 100 * share[DILUTED][k]
        med = quantile_p(h[k], pc, 0.5)
        if role:
            st = BRACKET_STYLE[role]
            col, ls, lw = st["color"], st["ls"], 1.2
            ax.plot(x, y, color=col, ls=ls, lw=lw, zorder=3)
        else:
            col, ls, lw = style.EPS_COLOR[e], "-", 2.0
            ax.plot(x, y, color=col, lw=lw, zorder=4)
        ax.plot(1.0, med, marker="<", color=col, ms=6, transform=ax.get_yaxis_transform(), clip_on=False,
                zorder=5)
        handles.append(Line2D([0], [0], color=col, lw=lw, ls=ls,
                              label=f"$\\varepsilon = {e:g}$: {in_win:.0f}% in band, {dil:.0f}% diluted"))
        facts[f"ε={e:g}"] = (f"median top {med:.0f} hPa; {in_win:.0f}% of unstable events top in {lo:g}–{hi:g} hPa; "
                             f"{dil:.0f}% diluted out")
    ax.set(ylim=(1000, 100), xlim=(0, 1.08 * xmax), xlabel=f"unstable events per {5 * P_COARSEN} hPa [%]",
           ylabel="plume top [hPa]")
    ax.set_yticks(np.arange(100, 1001, 100))
    facts["unstable events"] = f"{int(total[0]):,} of {int(total[0] + stable[0]):,} (T + M pooled)"
    facts["note"] = "triangles on the right axis mark each ε's median top (of plumes that top out)"
    return handles, facts


def draw(eps, h, cat, pe, out_dir, shown=None, brackets=None, window=(400.0, 600.0)):
    style.apply()
    fig, ax = plt.subplots(figsize=(3.8, 4.6))
    handles, facts = panel(ax, eps, h, cat, pe, shown, brackets, window)
    ax.legend(handles=handles, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
              frameon=False, handlelength=2.2, labelspacing=0.35)
    fig.tight_layout()
    path = FD.save(fig, out_dir, "ztop_pooled")
    plt.close(fig)
    return {"path": path, "title": "Plume tops, pooled", "facts": facts, "script": "figures/fig_ztop_pooled.py"}


def make(out_dir, years=None, weighted=WEIGHTED, brackets=True):
    eps, h, cat, pe, label = pooled(years, weighted)
    window = tuple(FD.load("analysis")["ztop_window_hPa"])
    item = draw(eps, h, cat, pe, out_dir, brackets=(eps[0], eps[-1]) if brackets else None, window=window)
    item["facts"]["years"] = label
    return [item]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", help="e.g. 1980-2019 (default: all figdata files)")
    ap.add_argument("--no-brackets", dest="brackets", action="store_false",
                    help="drop the smallest and largest ε of the sweep (maroon)")
    ap.add_argument("--brackets", action="store_true", help="(default; kept so old commands still run)")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs"))
    a = ap.parse_args()
    for item in make(Path(a.out), FD.years_arg(a.years), brackets=a.brackets):
        print(f"Wrote {item['path']}")
        for k, v in item["facts"].items():
            print(f"  {k}: {v}")
