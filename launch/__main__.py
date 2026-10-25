"""Launch bay CLI.

    python -m launch list                                   # stages
    python -m launch adopt --stages b d e --years 1980-2019 # read-only check of existing outputs
    python -m launch adopt --demo                           # toy good/bad files, shows every check
    python -m launch adopt --sample all --apply             # stamp the files that pass
    python -m launch run --stages a b c0 c1 c2 d e          # preflight → plan → y → archive + submit
    python -m launch run --stages d --years 1985 --dry-run  # plan + PBS files only
    python -m launch resume                                 # rerun whatever is not done
    python -m launch status [--failed] [--grid d]           # progress board; writes runs/status/
    python -m launch archive --list | --restore FILE        # what was replaced, and undo
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from .registry import STAGES

REPO = Path(__file__).resolve().parents[1]
BUILT = [k for k, s in STAGES.items() if s.legacy_cmd or s.port]


def _years(text):
    if not text:
        return None
    a, _, b = text.partition("-")
    return int(a), int(b or a)


def _stages(keys):
    for key in keys:
        if key not in STAGES:
            sys.exit(f"unknown stage {key!r}; choose from {', '.join(STAGES)}")
    return keys


def cmd_list(_):
    for s in STAGES.values():
        status = "ported" if s.port else "legacy script" if s.legacy_cmd else "not built"
        deps = ",".join(s.depends_on) or "-"
        print(f"{s.key:3} {s.name:9} grain={'×'.join(s.grain):18} after={deps:6} {status}")


def cmd_adopt(args):
    if args.demo:
        from .adopt_demo import run as demo
        report = demo(REPO / "runs" / "adopt_demo")
    else:
        from .adopt import run
        from .pbs import git_hash
        stages = _stages(args.stages or [k for k in BUILT])
        sample = None if args.sample == "all" else int(args.sample)
        out = REPO / "runs" / f"adopt_{datetime.now():%Y%m%d_%H%M%S}"
        report = run(stages, _years(args.years), args.events, out, sample=sample,
                     apply=args.apply, git=git_hash())
    print(f"Report: {report}")


def cmd_run(args, stages=None):
    from . import preflight
    stages = stages or _stages(args.stages)
    dirty = preflight.dirty_files()
    if dirty and not args.allow_dirty and not args.dry_run:
        sys.exit("Uncommitted changes in code that affects results:\n  " + "\n  ".join(dirty)
                 + "\nCommit them first, or pass --allow-dirty (recorded in the manifest).")
    states = preflight.plan(stages, _years(args.years), args.events,
                            rebuild_legacy=args.rebuild_legacy)
    print(preflight.describe(states))
    todo = [s for s in states if s.action in ("build", "archive+build")]
    run_dir = REPO / "runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    if args.dry_run:
        from .pbs import write_all
        written, _ = write_all([s.job for s in todo], run_dir / "pbs")
        print(f"Dry run: {len(written)} PBS files in {run_dir / 'pbs'}. Nothing archived or submitted.")
        return
    if not todo:
        print("Nothing to do.")
        return
    n_arch = sum(s.action == "archive+build" for s in todo)
    if not args.yes and input(f"Submit {len(todo)} jobs and archive outputs of {n_arch}? [y/N] ").lower() != "y":
        print("Cancelled.")
        return
    m = preflight.execute(states, run_dir)
    print(f"Submitted {len(m['jobs'])} jobs, archived {len(m['archived'])} files. "
          f"Manifest: {run_dir / 'manifest.json'}")


def cmd_resume(args):
    args.stages = BUILT
    cmd_run(args, BUILT)


def cmd_status(args):
    from . import state, status
    from .pbs import git_hash
    data = status.summary(state.collect(_stages(args.stages or list(STAGES)),
                                        _years(args.years)), git_hash())
    path = status.write(data, REPO / "runs" / "status")
    if args.failed:
        print(status.failed_list(data, args.tail))
    elif args.grid:
        print(status.year_grid(data, args.grid))
    else:
        print(status.board(data))
    print(f"(wrote {path} and Status-Map.md)")


def cmd_stamp(args):
    """Called at the end of each PBS job: stamp that job's outputs with provenance."""
    from . import provenance
    from .adopt import expected_files
    from .registry import Job
    from tpl_mll.config import load
    from .pbs import git_hash
    job = Job(STAGES[args.stage], args.year, args.month, args.event,
              load("analysis")["buffer_hours"])
    work = Path(load("paths")["work"])
    years = tuple(load("analysis")["years"])
    n = 0
    for _, path, _ in expected_files(job, work, years):
        if path.exists():
            provenance.stamp(path, args.stage, git_hash())
            n += 1
    print(f"stamped {n} files for {job.tag}")


def cmd_archive(args):
    from . import archive
    from tpl_mll.config import load
    if args.restore:
        print(f"Restored to {archive.restore(args.restore)}")
        return
    rows = archive.listing([load("paths")["work"]])
    for p, size in rows:
        print(f"{size / 1e6:9.1f} MB  {p}")
    print(f"{len(rows)} archived files, {sum(s for _, s in rows) / 1e9:.2f} GB")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m launch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show stages").set_defaults(func=cmd_list)

    a = sub.add_parser("adopt", help="read-only check of existing outputs")
    a.add_argument("--stages", nargs="+")
    a.add_argument("--years")
    a.add_argument("--events", nargs="+", choices=["T", "M"])
    a.add_argument("--sample", default="50", help="files opened per stage, or 'all'")
    a.add_argument("--apply", action="store_true", help="stamp files that pass (needs --sample all)")
    a.add_argument("--demo", action="store_true", help="run on toy good/bad files")
    a.set_defaults(func=cmd_adopt)

    for name, func, help_ in (("run", cmd_run, "preflight, then archive + submit"),
                              ("resume", cmd_resume, "rerun everything not done")):
        r = sub.add_parser(name, help=help_)
        if name == "run":
            r.add_argument("--stages", nargs="+", required=True)
        r.add_argument("--years", help="e.g. 1980-2019 or 1985")
        r.add_argument("--events", nargs="+", choices=["T", "M"])
        r.add_argument("--dry-run", action="store_true")
        r.add_argument("--yes", action="store_true", help="don't ask before submitting")
        r.add_argument("--allow-dirty", action="store_true")
        r.add_argument("--rebuild-legacy", action="store_true",
                       help="archive + rebuild outputs with no provenance")
        r.set_defaults(func=func)

    s = sub.add_parser("status", help="progress board")
    s.add_argument("--stages", nargs="+")
    s.add_argument("--years")
    s.add_argument("--failed", action="store_true")
    s.add_argument("--tail", type=int, default=0, help="with --failed: show last N log lines")
    s.add_argument("--grid", metavar="STAGE")
    s.set_defaults(func=cmd_status)

    t = sub.add_parser("stamp", help="(used by PBS jobs) stamp a job's outputs")
    t.add_argument("--stage", required=True)
    t.add_argument("--year", type=int)
    t.add_argument("--month", type=int)
    t.add_argument("--event")
    t.set_defaults(func=cmd_stamp)

    r = sub.add_parser("archive", help="list or restore archived outputs")
    r.add_argument("--list", action="store_true")
    r.add_argument("--restore", metavar="FILE")
    r.set_defaults(func=cmd_archive)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
