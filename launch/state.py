"""Work out the state of every job from output files, provenance attrs, logs and qstat.

States: done · queued · running · failed · stale · legacy · missing
- done:    all outputs exist and were made (or adopted) by the current stage fingerprint
- stale:   outputs exist but were made by other code or config → archive + rebuild
- legacy:  outputs exist with no provenance → run `launch adopt` before deciding
- failed:  log without '=== DONE', or only some outputs written
"""
from dataclasses import dataclass
from pathlib import Path

from tpl_mll.config import load
from . import hpc, provenance
from .adopt import expected_files, _legacy_log, SCRIPT
from .registry import expand

STATES = ["done", "running", "queued", "failed", "stale", "legacy", "missing"]


@dataclass
class JobState:
    job: object
    state: str
    detail: str = ""
    log: Path = None
    files: list = None


def _log_state(path):
    if path is None or not path.exists():
        return None
    return "done" if "=== DONE" in path.read_text(errors="ignore") else "failed"


def classify(job, work, log_dir, years, queue):
    files = [p for _, p, _ in expected_files(job, work, years)]
    new_log = Path(log_dir) / f"{job.tag}.log"
    log = new_log if new_log.exists() else (
        _legacy_log(job, log_dir) if job.stage.key in SCRIPT else None)
    if job.tag in queue:
        return JobState(job, "running" if queue[job.tag] == "R" else "queued", "", log, files)
    have = [p.exists() for p in files]
    if not files or not any(have):
        failed = _log_state(log) == "failed"
        return JobState(job, "failed" if failed else "missing",
                        "log has no DONE" if failed else "", log, files)
    if not all(have):
        return JobState(job, "failed", f"{sum(have)}/{len(have)} files written", log, files)
    stamp = provenance.read(files[0])
    if not stamp:
        return JobState(job, "legacy", "no provenance; run adopt", log, files)
    if stamp.get("code") != provenance.fingerprint(job.stage.key):
        return JobState(job, "stale", f"made by {stamp.get('code')}", log, files)
    return JobState(job, "done", stamp.get("how", ""), log, files)


def collect(stage_keys, years=None, events=None, work=None, log_dir=None):
    paths = load("paths")
    work = Path(work or paths["work"])
    log_dir = Path(log_dir or paths["log_dir"])
    years = tuple(years or load("analysis")["years"])
    queue = hpc.active()
    return [classify(j, work, log_dir, years, queue) for j in expand(stage_keys, years, events)]
