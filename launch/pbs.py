"""Render one PBS script per Job. Same body as the old launcher, plus provenance env vars."""
import subprocess
from pathlib import Path

from tpl_mll.config import load, config_hash

REPO = Path(__file__).resolve().parents[1]

TEMPLATE = """#!/bin/bash -l
#PBS -N {tag}
#PBS -A {project}
#PBS -q {queue}
#PBS -l walltime={walltime}
#PBS -l {select}
#PBS -j oe
#PBS -o {log_dir}/{tag}.log

set -e
module load conda
conda activate {conda_env}

export TMPDIR=${{SCRATCH}}/temp
mkdir -p ${{TMPDIR}}
export PAPER26_GIT={git}
export PAPER26_CONFIG={config}

echo "=== START {tag} host=$(hostname) time=$(date) ==="
{command}
cd {repo} && {python} -m launch stamp {stamp_args}
echo "=== DONE {tag} time=$(date) ==="
"""


def git_hash():
    """Short hash of the repo HEAD, or 'nogit' if git is unavailable."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, cwd=Path(__file__).parent, timeout=10)
        return r.stdout.strip() or "nogit"
    except (OSError, subprocess.TimeoutExpired):
        return "nogit"


def render(job, git=None):
    """Return the PBS text for a job, or None if the stage has no command yet."""
    hpc = load("hpc")
    cmd = job.command(hpc["python"])
    if cmd is None:
        return None
    res = hpc["stages"][job.stage.key]
    return TEMPLATE.format(
        tag=job.tag, project=hpc["project"], queue=hpc["queue"],
        walltime=res["walltime"], select=res["select"],
        log_dir=load("paths")["log_dir"].rstrip("/"), conda_env=hpc["conda_env"],
        git=git or git_hash(), config=config_hash(), command=cmd,
        repo=str(REPO).replace(" ", "\\ "), python=hpc["python"], stamp_args=stamp_args(job))


def stamp_args(job):
    """Arguments for `launch stamp` that identify this job's outputs."""
    s = f"--stage {job.stage.key}"
    for k in ("year", "month", "event"):
        if getattr(job, k) is not None:
            s += f" --{k} {getattr(job, k)}"
    return s


def write_all(jobs, out_dir):
    """Write PBS files into out_dir. Returns (written paths, jobs with no command)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    git = git_hash()
    written, missing = [], []
    for job in jobs:
        text = render(job, git)
        if text is None:
            missing.append(job)
            continue
        path = out_dir / f"{job.tag}.pbs"
        path.write_text(text)
        written.append(path)
    return written, missing
