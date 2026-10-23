"""The only module that talks to PBS. Everything else goes through qsub() and active().

Setting PAPER26_FAKE_PBS=<folder> swaps in a fake scheduler that records submissions in
that folder, so the launch bay can be tested and demoed on a laptop.
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def _fake():
    d = os.environ.get("PAPER26_FAKE_PBS")
    return Path(d) if d else None


def qsub(script, after=()):
    """Submit a PBS script, optionally after other job ids succeed. Returns the job id."""
    fake = _fake()
    name = Path(script).stem
    if fake:
        fake.mkdir(parents=True, exist_ok=True)
        q = fake / "queue.json"
        jobs = json.loads(q.read_text()) if q.exists() else {}
        jid = f"{len(jobs) + 1}.fake"
        jobs[jid] = {"name": name, "state": "Q", "after": list(after)}
        q.write_text(json.dumps(jobs, indent=1))
        return jid
    cmd = ["qsub"]
    if after:
        cmd += ["-W", "depend=afterok:" + ":".join(after)]
    r = subprocess.run(cmd + [str(script)], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"qsub failed for {name}: {r.stderr.strip()}")
    return r.stdout.strip()


def active():
    """{job name: "Q" | "R" | "H"} for this user's jobs that are queued, held or running."""
    fake = _fake()
    if fake:
        q = fake / "queue.json"
        jobs = json.loads(q.read_text()) if q.exists() else {}
        return {j["name"]: j["state"] for j in jobs.values() if j["state"] in "QRH"}
    user = os.environ.get("USER", "")
    try:
        r = subprocess.run(["qstat", "-f", "-F", "json", "-u", user], capture_output=True,
                           text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"warning: could not ask PBS about running jobs ({exc}); treating none as active",
              file=sys.stderr)
        return {}
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    jobs = json.loads(r.stdout).get("Jobs", {})
    return {j["Job_Name"]: j["job_state"] for j in jobs.values()
            if j.get("Job_Owner", "").startswith(user + "@") and j["job_state"] in "QRH"}
