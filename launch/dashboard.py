"""Pieces of dashboard.ipynb. Everything reads runs/status/status.json; nothing submits jobs.

The pipeline map is drawn with matplotlib (no JavaScript, so it works in any JupyterLab)
and the same map is written as Mermaid to runs/status/Status-Map.md for the Obsidian vault.
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

from .registry import STAGES
from .status import worst, year_grid

REPO = Path(__file__).resolve().parents[1]
STATUS = REPO / "runs" / "status" / "status.json"
COLOR = {"done": "#a5d6a7", "running": "#fff176", "queued": "#fff9c4", "failed": "#ef9a9a",
         "stale": "#ffcc80", "legacy": "#ffe0b2", "missing": "#eceff1"}
ORDER = ["done", "running", "queued", "failed", "stale", "legacy", "missing"]
# Where each stage sits on the map (column, row)
POS = {"a": (0, 1), "b": (1, 1), "c0": (2, 1), "c1": (3, 1), "c2": (4, 1),
       "d": (5, 1), "e": (6, 1.6), "f": (6, 0.4)}


def refresh(years=None):
    """Run `launch status` (one qstat call) and load the result."""
    cmd = [sys.executable, "-m", "launch", "status"] + (["--years", years] if years else [])
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"launch status failed:\n{r.stderr[-2000:]}")
    return load()


def load():
    return json.loads(STATUS.read_text())


def draw_map(data, ax=None):
    """Pipeline boxes coloured by each stage's worst state, with done/total inside."""
    ax = ax or plt.subplots(figsize=(11, 2.8))[1]
    for key, (x, y) in POS.items():
        st = data["stages"].get(key)
        state = worst(st) if st else "missing"
        frac = st["done"] / st["total"] if st and st["total"] else 0
        frac_old = st["legacy"] / st["total"] if st and st["total"] else 0
        ax.add_patch(FancyBboxPatch((x - 0.4, y - 0.3), 0.8, 0.6, boxstyle="round,pad=0.02",
                                    fc=COLOR[state], ec="#37474f",
                                    lw=2.5 if state == "failed" else 1,
                                    ls="--" if state == "missing" else "-"))
        ax.add_patch(FancyBboxPatch((x - 0.35, y - 0.25), 0.7 * frac, 0.06, boxstyle="square,pad=0",
                                    fc="#2e7d32", ec="none"))
        ax.add_patch(FancyBboxPatch((x - 0.35 + 0.7 * frac, y - 0.25), 0.7 * frac_old, 0.06,
                                    boxstyle="square,pad=0", fc="#ef6c00", ec="none"))
        label = f"{key} {STAGES[key].name}"
        sub = f"{st['done']}/{st['total']} done" if st else "not built"
        if st and st["legacy"]:
            ax.text(x, y - 0.17, f"+{st['legacy']} legacy", ha="center", va="center", fontsize=6.5,
                    color="#bf360c")
        ax.text(x, y + 0.08, label, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x, y - 0.04, sub, ha="center", va="center", fontsize=8)
    for key, stage in STAGES.items():
        for dep in stage.depends_on:
            (x0, y0), (x1, y1) = POS[dep], POS[key]
            ax.annotate("", (x1 - 0.4, y1), (x0 + 0.4, y0),
                        arrowprops=dict(arrowstyle="->", color="#78909c",
                                        connectionstyle="arc3,rad=0.25" if abs(x1 - x0) > 1 else None))
    for i, s in enumerate(ORDER):
        ax.add_patch(FancyBboxPatch((i * 0.95 - 0.4, -0.33), 0.18, 0.14, boxstyle="square,pad=0",
                                    fc=COLOR[s], ec="#90a4ae"))
        ax.text(i * 0.95 - 0.17, -0.26, s, va="center", fontsize=7)
    ax.set(xlim=(-0.6, 6.6), ylim=(-0.5, 2.1), aspect="equal")
    ax.axis("off")
    ax.set_title(f"Pipeline · code {data['git']} · config {data['config']} · {data['generated']}",
                 fontsize=9, loc="left")
    return ax


def heatmap(data, stage, ax=None):
    """Years × (event, month) grid of job states for one stage."""
    st = data["stages"][stage]
    cols = sorted({(j["event"] or "", j["month"] or 0) for j in st["jobs"]})
    years = sorted({j["year"] for j in st["jobs"] if j["year"] is not None}) or [None]
    grid = np.full((len(years), len(cols)), ORDER.index("missing"))
    for j in st["jobs"]:
        grid[years.index(j["year"]), cols.index((j["event"] or "", j["month"] or 0))] = ORDER.index(j["state"])
    ax = ax or plt.subplots(figsize=(min(12, 2 + 0.25 * len(cols)), 1 + 0.18 * len(years)))[1]
    ax.imshow(grid, cmap=ListedColormap([COLOR[s] for s in ORDER]), vmin=0, vmax=len(ORDER) - 1,
              aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(years)), [y or "all" for y in years], fontsize=7)
    ax.set_xticks(range(len(cols)), [f"{e}{m:02d}" if m else e or "·" for e, m in cols],
                  fontsize=6, rotation=90 if len(cols) > 4 else 0)
    ax.set_title(f"Stage {stage} {st['name']}: {st['done']}/{st['total']} done", fontsize=9, loc="left")
    return ax


def failed_jobs(data):
    return [j for st in data["stages"].values() for j in st["jobs"] if j["state"] == "failed"]


def log_tail(path, n=30):
    p = Path(path) if path else None
    if not p or not p.exists():
        return "(no log)"
    return "\n".join(p.read_text(errors="ignore").splitlines()[-n:])


def dry_run(stages="a b c0 c1 c2 d e", years=None):
    cmd = [sys.executable, "-m", "launch", "run", "--stages", *stages.split(), "--dry-run"]
    if years:
        cmd += ["--years", years]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    return r.stdout + r.stderr


def app(years=None):
    """The whole dashboard as one ipywidgets box with a Refresh button."""
    import ipywidgets as W
    from IPython.display import display, clear_output

    out_head, out_map, out_grid, out_log, out_dry = (W.Output() for _ in range(5))
    bars = W.VBox()
    stage_pick = W.Dropdown(description="Stage")
    failed_pick = W.Select(description="Failed", rows=6, layout=W.Layout(width="60%"))
    state = {}

    def draw(_=None):
        data = refresh(years)
        state["data"] = data
        bars.children = [W.HBox([W.Label(f"{k} {st['name']}", layout=W.Layout(width="110px")),
                                 W.IntProgress(value=st["done"] + st["legacy"], max=max(st["total"], 1),
                                               bar_style="danger" if st["failed"] else
                                               "warning" if st["legacy"] else
                                               "success" if st["done"] == st["total"] else "info"),
                                 W.Label(f"{st['done'] + st['legacy']}/{st['total']} exist  " + "  ".join(
                                     f"{s} {st[s]}" for s in ORDER[1:] if st[s]))])
                         for k, st in data["stages"].items()]
        with out_map:
            clear_output(wait=True)
            draw_map(data)
            plt.show()
        stage_pick.options = list(data["stages"])
        fj = failed_jobs(data)
        failed_pick.options = [(f"{j['tag']}  {j['detail']}", j["log"]) for j in fj] or [("none", None)]
        show_grid()

    def show_grid(_=None):
        with out_grid:
            clear_output(wait=True)
            if stage_pick.value:
                heatmap(state["data"], stage_pick.value)
                plt.show()

    def show_log(change):
        with out_log:
            clear_output(wait=True)
            print(log_tail(change["new"]))

    def show_dry(_):
        with out_dry:
            clear_output(wait=True)
            print(dry_run(years=years))

    refresh_btn = W.Button(description="Refresh", icon="refresh", button_style="primary")
    dry_btn = W.Button(description="Preview next run (dry run)", layout=W.Layout(width="240px"))
    refresh_btn.on_click(draw)
    dry_btn.on_click(show_dry)
    stage_pick.observe(show_grid, "value")
    failed_pick.observe(show_log, "value")
    display(W.VBox([refresh_btn, out_map, bars, W.HTML("<b>Stage × year</b>"), stage_pick, out_grid,
                    W.HTML("<b>Failed jobs</b> (pick one to see its log)"), failed_pick, out_log,
                    dry_btn, out_dry]))
    draw()
