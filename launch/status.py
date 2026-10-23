"""Status board, status.json and the live pipeline map. All read from state.collect()."""
import collections
import json
from datetime import datetime
from pathlib import Path

from tpl_mll.config import config_hash
from .registry import STAGES
from .state import STATES

BAR = 16
GRID_ICON = {"done": "✓", "running": "…", "queued": "…", "failed": "✗", "stale": "~",
             "legacy": "?", "missing": "◌"}
MAP_CLASS = {"done": "done", "running": "run", "queued": "run", "failed": "fail",
             "stale": "stale", "legacy": "stale", "missing": "todo"}


def summary(states, git):
    """The status.json payload."""
    stages = {}
    for s in states:
        st = stages.setdefault(s.job.stage.key, {"name": s.job.stage.name, "total": 0,
                                                 **{k: 0 for k in STATES}, "jobs": []})
        st["total"] += 1
        st[s.state] += 1
        st["jobs"].append({"tag": s.job.tag, "year": s.job.year, "month": s.job.month,
                           "event": s.job.event, "state": s.state, "detail": s.detail,
                           "log": str(s.log) if s.log else None})
    return {"generated": datetime.now().isoformat(timespec="seconds"), "git": git,
            "config": config_hash(), "stages": stages}


def worst(st):
    for k in ("failed", "running", "queued", "stale", "legacy", "missing"):
        if st[k]:
            return k
    return "done"


def board(data):
    L = [f"Paper26 pipeline · code {data['git']} · config {data['config']} · {data['generated']}",
         f"{'stage':12} {'done':>9}  {'progress':{BAR + 5}} " + " ".join(f"{k:>7}" for k in STATES[1:])]
    for key, st in data["stages"].items():
        frac = st["done"] / st["total"] if st["total"] else 0
        bar = "█" * round(frac * BAR) + "░" * (BAR - round(frac * BAR))
        L.append(f"{key + ' ' + st['name']:12} {st['done']:>4}/{st['total']:<4}  {bar} {frac:4.0%} "
                 + " ".join(f"{st[k]:>7}" for k in STATES[1:]))
    failed = [j for st in data["stages"].values() for j in st["jobs"] if j["state"] == "failed"]
    if failed:
        L.append(f"⚠ {len(failed)} failed: " + ", ".join(j["tag"] for j in failed[:8])
                 + (" …" if len(failed) > 8 else "") + "   (status --failed for logs)")
    if any(st["legacy"] for st in data["stages"].values()):
        L.append("? legacy = outputs with no provenance; run `launch adopt` to check and stamp them")
    if any(st["stale"] for st in data["stages"].values()):
        L.append("~ stale = made by older code/config; the next run archives and rebuilds them")
    return "\n".join(L)


def failed_list(data, tail=0):
    L = []
    for st in data["stages"].values():
        for j in st["jobs"]:
            if j["state"] != "failed":
                continue
            L.append(f"{j['tag']:28} {j['detail']:28} {j['log'] or '(no log)'}")
            if tail and j["log"] and Path(j["log"]).exists():
                L += ["    " + x for x in Path(j["log"]).read_text(errors="ignore").splitlines()[-tail:]]
    return "\n".join(L) or "No failed jobs."


def year_grid(data, stage):
    """Years down, events (× months) across: ✓ done … active ✗ failed ~ stale ? legacy ◌ missing."""
    st = data["stages"][stage]
    cols = sorted({(j["event"] or "", j["month"] or 0) for j in st["jobs"]})
    cell = {(j["year"], j["event"] or "", j["month"] or 0): j["state"] for j in st["jobs"]}
    years = sorted({j["year"] for j in st["jobs"] if j["year"] is not None})
    head = "      " + " ".join(f"{e}{m:02d}" if m else e or "·" for e, m in cols)
    L = [f"Stage {stage} {st['name']}", head]
    for y in years or [None]:
        L.append(f"{y or '----'}  " + " ".join(
            f"{GRID_ICON[cell.get((y, e, m), 'missing')]:>{3 if m else 1}}" for e, m in cols))
    return "\n".join(L)


def mermaid(data):
    """Pipeline flowchart with each stage coloured by its worst state and its done count."""
    L = ["flowchart LR",
         "  classDef done fill:#c8e6c9,stroke:#2e7d32,color:#000",
         "  classDef run fill:#fff59d,stroke:#f9a825,color:#000",
         "  classDef fail fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:#000",
         "  classDef stale fill:#ffe0b2,stroke:#ef6c00,color:#000",
         "  classDef todo fill:#eceff1,stroke:#90a4ae,stroke-dasharray:5 4,color:#455a64"]
    for key, st in data["stages"].items():
        L.append(f'  {key}["{key} {st["name"]}<br/>{st["done"]}/{st["total"]}"]:::{MAP_CLASS[worst(st)]}')
    for key in data["stages"]:
        for dep in STAGES[key].depends_on:
            if dep in data["stages"]:
                L.append(f"  {dep} --> {key}")
    return "\n".join(L)


def write(data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "status.json").write_text(json.dumps(data, indent=1))
    (out_dir / "Status-Map.md").write_text(
        f"# Pipeline status · {data['generated']}\n\n```mermaid\n{mermaid(data)}\n```\n\n"
        "🟩 done · 🟨 running/queued · 🟥 failed · 🟧 stale/legacy · ⬜ not started\n")
    return out_dir / "status.json"
