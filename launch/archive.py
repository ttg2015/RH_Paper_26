"""Move replaced outputs into _old/ instead of overwriting them. Nothing is ever deleted.

A replaced file goes to <its folder>/_old/<YYYY-MM-DD>_<old code>/<name>, and one line is
added to <its folder>/_old/ARCHIVE_LOG.md.
"""
import shutil
from datetime import date, datetime
from pathlib import Path

from . import provenance

LOG_HEADER = "| when | file | old code | new code | reason |\n|---|---|---|---|---|\n"


def archive(path, reason, new_code=""):
    """Move one file into _old/ and log it. Returns the new location."""
    path = Path(path)
    old_code = provenance.read(path).get("code", "legacy")
    dest_dir = path.parent / "_old" / f"{date.today():%Y-%m-%d}_{old_code}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    n = 1
    while dest.exists():                  # same file archived twice in one day
        dest = dest_dir / f"{path.stem}.{n}{path.suffix}"
        n += 1
    shutil.move(str(path), dest)
    log = path.parent / "_old" / "ARCHIVE_LOG.md"
    if not log.exists():
        log.write_text("# Archive log\n\n" + LOG_HEADER)
    with open(log, "a") as f:
        f.write(f"| {datetime.now():%Y-%m-%d %H:%M} | {path.name} | {old_code} | "
                f"{new_code} | {reason} |\n")
    return dest


def listing(roots):
    """[(archived file, size in bytes)] under every _old/ below the given folders."""
    out = []
    for root in roots:
        for old in Path(root).rglob("_old"):
            out += [(p, p.stat().st_size) for p in old.rglob("*") if p.is_file()
                    and p.name != "ARCHIVE_LOG.md"]
    return sorted(out)


def restore(archived):
    """Move an archived file back, archiving whatever is there now first."""
    archived = Path(archived)
    target = archived.parents[2] / archived.name.split(".")[0]
    target = target.with_suffix(archived.suffix)
    if target.exists():
        archive(target, f"replaced by restore of {archived.parent.name}")
    shutil.move(str(archived), target)
    return target
