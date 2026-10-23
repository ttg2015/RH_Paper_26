"""Launch bay CLI.

    python -m launch run --stages d e --years 1980-1981 --dry-run
    python -m launch list
    python -m launch adopt --stages b d e --years 1980-2019   # read-only check of existing outputs
    python -m launch adopt --demo                             # toy good/bad files, shows every check

Only --dry-run is implemented so far: it writes PBS files and submits nothing.
Preflight, archive, submit and status come next (see pipeline-map/Planned/Launch Bay).
"""
import argparse
import collections
import sys
from datetime import datetime
from pathlib import Path

from .registry import STAGES, expand
from .pbs import write_all

REPO = Path(__file__).resolve().parents[1]


def _years(text):
    a, _, b = text.partition("-")
    return int(a), int(b or a)


def cmd_list(_):
    for s in STAGES.values():
        status = "legacy script" if s.legacy_cmd else "not built"
        deps = ",".join(s.depends_on) or "-"
        print(f"{s.key:3} {s.name:9} grain={'×'.join(s.grain):18} after={deps:6} {status}")


def cmd_run(args):
    for key in args.stages:
        if key not in STAGES:
            sys.exit(f"unknown stage {key!r}; choose from {', '.join(STAGES)}")
    jobs = expand(args.stages, _years(args.years) if args.years else None, args.events)
    if not args.dry_run:
        sys.exit("Submitting is not built yet (needs preflight + archive). Use --dry-run.")
    out = REPO / "runs" / f"dryrun_{datetime.now():%Y%m%d_%H%M%S}"
    written, missing = write_all(jobs, out)
    counts = collections.Counter(j.stage.key for j in jobs)
    print(f"Dry run: {len(written)} PBS files in {out}")
    for key in args.stages:
        print(f"  {key:3} {STAGES[key].name:9} {counts[key]:5} jobs")
    if missing:
        skipped = sorted({j.stage.key for j in missing})
        print(f"  skipped {len(missing)} jobs: stage {', '.join(skipped)} not built yet")


def cmd_adopt(args):
    if args.demo:
        from .adopt_demo import run as demo
        report = demo(REPO / "runs" / "adopt_demo")
    else:
        from .adopt import run
        stages = args.stages or [k for k in STAGES if k != "f"]
        sample = None if args.sample == "all" else int(args.sample)
        out = REPO / "runs" / f"adopt_{datetime.now():%Y%m%d_%H%M%S}"
        report = run(stages, _years(args.years) if args.years else None, args.events, out,
                     sample=sample)
    print(f"Report: {report}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m launch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show stages").set_defaults(func=cmd_list)
    r = sub.add_parser("run", help="build PBS jobs for stages")
    r.add_argument("--stages", nargs="+", required=True)
    r.add_argument("--years", help="e.g. 1980-2019 or 1985")
    r.add_argument("--events", nargs="+", choices=["T", "M"])
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("adopt", help="read-only check of existing outputs")
    a.add_argument("--stages", nargs="+")
    a.add_argument("--years")
    a.add_argument("--events", nargs="+", choices=["T", "M"])
    a.add_argument("--sample", default="50", help="files opened per stage, or 'all'")
    a.add_argument("--demo", action="store_true", help="run on toy good/bad files")
    a.set_defaults(func=cmd_adopt)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
