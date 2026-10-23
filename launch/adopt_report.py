"""Write the adopt report: a Markdown page with verdicts, check explanations and figures."""
import collections
import csv
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .adopt import CHECKS, ICON, PASS, WARN, FAIL

VERDICT_ICON = {"adopt": "✅", "adopt-with-correction": "🔧", "adopt-with-note": "⚠️",
                "rebuild": "🔁", "broken": "❌", "missing": "◌"}
VERDICT_MEANS = {
    "adopt": "passes every check; reuse as is",
    "adopt-with-correction": "reuse, with a known correction applied on read (e.g. E07)",
    "adopt-with-note": "probably fine, but look at the warning first",
    "rebuild": "made with settings the project no longer uses; archive and rebuild",
    "broken": "unreadable, empty or crashed; archive and rebuild",
    "missing": "never written; build",
}
COLOR = {PASS: "#2e7d32", WARN: "#f9a825", FAIL: "#c62828"}


def _fig_sizes(sizes, out):
    """One panel per stage: file size per file, median line, small files in red."""
    files = []
    for stage, rows in sizes.items():
        s = np.array([b for _, b, _ in rows]) / 1e6
        if s.size < 2:
            continue
        med = np.median(s)
        fig, ax = plt.subplots(figsize=(6, 2.8))
        colors = [COLOR[WARN] if x < 0.5 * med else "#90a4ae" for x in s]
        ax.bar(range(s.size), s, color=colors, width=1.0)
        ax.axhline(med, color="k", lw=1, label="median")
        ax.axhline(0.5 * med, color=COLOR[WARN], lw=1, ls="--", label="50% of median")
        ax.set_yscale("log")
        ax.set(xlabel="file", ylabel="MB (log)", title=f"Stage {stage}: file sizes")
        if s.size <= 40:
            ax.set_xticks(range(s.size), [lbl for lbl, _, _ in rows], rotation=60, fontsize=6)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        name = f"size_{stage}.png"
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        files.append((stage, name))
    return files


def _fig_range(evidence, out):
    if "range" not in evidence:
        return None
    from .adopt import ANNMAX
    fig, axes = plt.subplots(1, len(evidence["range"]), figsize=(6, 2.4), squeeze=False)
    for ax, (event, vals) in zip(axes[0], sorted(evidence["range"].items())):
        v = np.concatenate(vals)
        lo, hi = ANNMAX[event][2]
        ax.hist(v, bins=60, color="#78909c")
        ax.axvspan(lo, hi, color=COLOR[PASS], alpha=0.12, label="allowed")
        ax.set(title=f"Annual max {ANNMAX[event][1]}", xlabel="K" if event == "T" else "kJ/kg")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "range_b.png", dpi=110)
    plt.close(fig)
    return "range_b.png"


def _fig_e07(evidence, out):
    """z_LCL_2m against surface height. Above sea level → points sit above the 1:1 line."""
    if "e07" not in evidence:
        return None
    zs = np.concatenate([a for a, _ in evidence["e07"]])
    z = np.concatenate([b for _, b in evidence["e07"]])
    idx = np.random.default_rng(0).choice(zs.size, min(zs.size, 20000), replace=False)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.scatter(zs[idx], z[idx], s=2, alpha=0.3, color="#5c6bc0")
    top = np.nanmax(zs[idx]) if idx.size else 1
    ax.plot([0, top], [0, top], "k--", lw=1, label="z_LCL = z_s (LCL at the ground)")
    ax.set(xlabel="surface height z_s [m]", ylabel="stored z_LCL_2m [m]",
           title="E07: is z_LCL measured from sea level?")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "e07_d.png", dpi=110)
    plt.close(fig)
    return "e07_d.png"


def write_report(results, sizes, evidence, out):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "adopt_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "file", "verdict", "path"] + list(CHECKS))
        for r in results:
            w.writerow([r.stage, r.label, r.verdict, r.path] +
                       [f"{r.checks[c][0]}: {r.checks[c][1]}" if c in r.checks else ""
                        for c in CHECKS])

    size_figs = _fig_sizes(sizes, out)
    range_fig, e07_fig = _fig_range(evidence, out), _fig_e07(evidence, out)
    by_stage = collections.defaultdict(list)
    for r in results:
        by_stage[r.stage].append(r)

    L = [f"# Adopt report · {datetime.now():%Y-%m-%d %H:%M}",
         "Read-only check of existing pipeline outputs. Nothing on GLADE was changed.", "",
         "## Verdict per stage", "",
         "| Stage | Files | " + " | ".join(f"{VERDICT_ICON[v]} {v}" for v in VERDICT_ICON) + " |",
         "|---|---|" + "---|" * len(VERDICT_ICON)]
    for stage, rs in by_stage.items():
        c = collections.Counter(r.verdict for r in rs)
        L.append(f"| {stage} | {len(rs)} | " + " | ".join(str(c[v] or "") for v in VERDICT_ICON) + " |")
    L += ["", "**What the verdicts mean**", ""]
    L += [f"- {VERDICT_ICON[v]} **{v}**: {m}" for v, m in VERDICT_MEANS.items()]

    L += ["", "## The checks", ""]
    for cid, (title, why) in CHECKS.items():
        st = collections.Counter(r.checks[cid][0] for r in results if cid in r.checks)
        if not st:
            continue
        tally = " · ".join(f"{ICON[s]} {n}" for s, n in st.items())
        L += [f"### {title}", why, "", f"Result: {tally}", ""]
        if cid == "size":
            L += [f"![sizes {s}]({n})" for s, n in size_figs] + [""]
        if cid == "range" and range_fig:
            L += [f"![range]({range_fig})", ""]
        if cid == "e07" and e07_fig:
            L += [f"![e07]({e07_fig})",
                  "Points on the dashed line would be an LCL at ground level; legacy points "
                  "all sit at or above z_s and rise with it, which is the sea-level offset.", ""]

    problems = [r for r in results if r.verdict not in ("adopt",)]
    L += ["## Files that need a look", ""]
    if not problems:
        L.append("None.")
    else:
        L += ["| Stage | File | Verdict | Why |", "|---|---|---|---|"]
        for r in problems[:300]:
            why = "; ".join(f"{c}: {d or s}" for c, (s, d) in r.checks.items() if s in (WARN, FAIL))
            L.append(f"| {r.stage} | {r.label} | {VERDICT_ICON[r.verdict]} {r.verdict} | {why} |")
        if len(problems) > 300:
            L.append(f"\n…and {len(problems) - 300} more in adopt_report.csv")
    (out / "adopt_report.md").write_text("\n".join(L) + "\n")
    return out / "adopt_report.md"
