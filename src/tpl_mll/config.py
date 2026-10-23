"""Load the YAML files in config/ once and hand them out as plain dicts."""
from functools import lru_cache
from pathlib import Path
import hashlib

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@lru_cache(maxsize=None)
def load(name):
    """Return config/<name>.yaml as a dict, e.g. load("analysis")."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def config_hash():
    """Short hash of all config files, stored in outputs for provenance."""
    h = hashlib.sha1()
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        h.update(p.read_bytes())
    return h.hexdigest()[:8]
