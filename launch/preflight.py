"""Preflight: decide what a run would do, then (after a yes) archive and submit.

Plan per job (a rebuild also rebuilds what is downstream of it in the run):
  missing/failed → build · done/queued/running → skip · stale → archive + rebuild
legacy (no provenance) → hold, unless rebuild_legacy=True. Run `launch adopt` on those first.
"""
import collections
import json
import subprocess
from datetime import datetime
from pathlib import Path

from tpl_mll.config import config_hash
from . import archive, hpc, provenance
from .pbs import git_hash, write_all
from .registry import STAGES
from .state import collect

REPO = Path(__file__).resolve().parents[1]
WATCHED = ["src", "pipeline", "config", "launch"]
ACTION = {"missing": "build", "failed": "build", "stale": "archive+build",
          "done": "skip", "queued": "skip", "running": "skip", "legacy": "hold"}


def dirty_files():
    """Uncommitted changes in code that affects results ([] if clean or no git)."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", *WATCHED], cwd=REPO,
                           capture_output=True, text=True, timeout=20)
    except OSError:
        return []
    return [line[3:] for line in r.stdout.splitlines()]


def plan(stage_keys, years=None, events=None, rebuild_legacy=False, **where):
    states = collect(stage_keys, years, events, **where)
    for s in states:
        s.action = ACTION[s.state]
        if s.state == "legacy" and rebuild_legacy:
            s.action = "archive+build"
        if s.state == "missing" and STAGES[s.job.stage.key].legacy_cmd is None:
            s.action = "not built"
    # A rebuilt job makes everything downstream of it in this run out of date too.
    order = list(STAGES)
    rebuilt = []
    for s in sorted(states, key=lambda s: order.index(s.job.stage.key)):
        upstream = [r for r in rebuilt
                    if r.job.stage.key in s.job.stage.depends_on and _matches(s.job, r.job)]
        if upstream and s.action in ("skip", "hold") and s.state not in ("queued", "running"):
            s.action = "archive+build" if s.files and any(f.exists() for f in s.files) else "build"
            s.detail = f"upstream {upstream[0].job.tag} rebuilt"
        if s.action in ("build", "archive+build"):
            rebuilt.append(s)
    return states


def describe(states):
    count = collections.Counter((s.job.stage.key, s.action) for s in states)
    acts = ["build", "archive+build", "skip", "hold", "not built"]
    L = [f"{'stage':10}" + "".join(f"{a:>15}" for a in acts)]
    for key in dict.fromkeys(s.job.stage.key for s in states):
        L.append(f"{key + ' ' + STAGES[key].name:10}"
                 + "".join(f"{count[(key, a)] or '':>15}" for a in acts))
    if any(s.action == "hold" for s in states):
        L.append("hold = legacy outputs with no provenance. Run `launch adopt --apply` to stamp "
                 "the good ones, or pass --rebuild-legacy to archive and rebuild them.")
    return "\n".join(L)


def _matches(job, dep):
    """A job waits for a dependency job when they agree on every dimension both have."""
    return all(getattr(job, k) == getattr(dep, k) for k in ("year", "month", "event")
               if getattr(job, k) is not None and getattr(dep, k) is not None)


def execute(states, run_dir):
    """Archive what is to be replaced, write PBS files and submit with afterok chains."""
    run_dir = Path(run_dir)
    git = git_hash()
    todo = [s for s in states if s.action in ("build", "archive+build")]
    archived = []
    for s in todo:
        if s.action == "archive+build":
            new = provenance.fingerprint(s.job.stage.key)
            for f in s.files:
                if f.exists():
                    archived.append(str(archive.archive(f, s.state, new)))
    written, _ = write_all([s.job for s in todo], run_dir / "pbs")
    script = {p.stem: p for p in written}
    ids = {}
    order = list(STAGES)
    for s in sorted(todo, key=lambda s: order.index(s.job.stage.key)):
        deps = [jid for other, jid in ids.items()
                if other.stage.key in s.job.stage.depends_on and _matches(s.job, other)]
        ids[s.job] = hpc.qsub(script[s.job.tag], deps)
    manifest = {"started": datetime.now().isoformat(timespec="seconds"), "git": git,
                "config": config_hash(), "dirty": dirty_files(),
                "jobs": {j.tag: jid for j, jid in ids.items()}, "archived": archived}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
