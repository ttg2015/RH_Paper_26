"""Which code and config made an output file, stored as netCDF global attributes.

A stage's *fingerprint* hashes everything that can change its numbers: the shared
package (src/tpl_mll), config/analysis.yaml, and the stage's own script. Committing a README
change therefore does not make outputs stale, but editing thermo.py or analysis.yaml does.
"""
import hashlib
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import netCDF4

from tpl_mll.config import CONFIG_DIR, load

REPO = Path(__file__).resolve().parents[1]
PREFIX = "paper26_"


@lru_cache(maxsize=None)
def fingerprint(stage_key):
    from .adopt import SCRIPT
    h = hashlib.sha1()
    # analysis.yaml only: paths.yaml and hpc.yaml (walltimes, queues) can't change results
    for p in sorted((REPO / "src" / "tpl_mll").glob("*.py")) + [CONFIG_DIR / "analysis.yaml"]:
        h.update(p.read_bytes())
    port = REPO / "pipeline" / f"{stage_key}.py"
    script = port if port.exists() else Path(load("paths")["legacy_scripts"]) / SCRIPT.get(stage_key, "")
    h.update(script.read_bytes() if script.is_file() else script.name.encode())
    return h.hexdigest()[:10]


def read(path):
    """Provenance attrs of a file as a dict without the prefix ({} if none or unreadable)."""
    try:
        with netCDF4.Dataset(path) as nc:
            return {k[len(PREFIX):]: nc.getncattr(k) for k in nc.ncattrs() if k.startswith(PREFIX)}
    except (OSError, RuntimeError):
        return {}


def stamp(path, stage_key, git, how="built", note=""):
    """Write provenance attrs into an existing netCDF file (in place, attrs only)."""
    attrs = {"stage": stage_key, "code": fingerprint(stage_key), "git": git, "how": how,
             "stamped": datetime.now().isoformat(timespec="seconds")}
    if note:
        attrs["note"] = note
    with netCDF4.Dataset(path, "a") as nc:
        for k, v in attrs.items():
            nc.setncattr(PREFIX + k, v)
    return attrs
