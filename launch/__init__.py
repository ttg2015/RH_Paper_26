"""Launch bay: send pipeline stages to PBS on NCAR."""
import sys
from pathlib import Path

# Let `python -m launch` work from the repo root without `pip install -e .`
_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
