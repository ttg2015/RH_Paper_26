"""The one analysis mask (E10): latitude band, land, low terrain. Settings in config/analysis.yaml."""
from pathlib import Path

import numpy as np
import xarray as xr

from .config import load

G = 9.8


def _invariant(key, lat, lon):
    """ERA5 invariant field on the given lat/lon, matched by label (handles N→S storage)."""
    paths = load("paths")
    with xr.open_dataset(Path(paths["era5"]["root"]) / paths["era5"][key]) as ds:
        da = ds[list(ds.data_vars)[0]]
        if "time" in da.dims:
            da = da.isel(time=0)
        return da.reindex(latitude=lat, longitude=lon, method="nearest").values


def surface_height_km(lat, lon):
    return _invariant("invariant_z", lat, lon) / G / 1000.0


def analysis_mask(lat, lon):
    """(lat, lon) bool: inside the band, land fraction > lsm_min, terrain below zs_max_km."""
    cfg = load("analysis")
    lo, hi = cfg["lat_band"]
    band = (np.asarray(lat) >= lo) & (np.asarray(lat) <= hi)
    lsm = _invariant("invariant_lsm", lat, lon)
    zs = surface_height_km(lat, lon)
    return band[:, None] & (lsm > cfg["mask"]["lsm_min"]) & (zs < cfg["mask"]["zs_max_km"])
