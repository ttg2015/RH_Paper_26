"""Load stage g figure data: sum years, pick sub-band and weighting, coarsen bins.

All stored quantities are counts or sums, so years add and coarsening is summing neighbours.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tpl_mll.config import load      # noqa: E402,F401


def folder():
    p = load("paths")
    return Path(p["work"]) / p["outputs"]["figdata"]


def available(event=None):
    """[(year, event)] with a figdata file."""
    out = []
    for f in sorted(folder().glob("figdata_*_b06_*.nc")):
        _, e, _, y = f.stem.split("_")
        if event in (None, e):
            out.append((int(y), e))
    return out


def years_arg(text):
    """'1980-2019' or '1985' → (y0, y1); None → everything."""
    if not text:
        return None
    a, _, b = text.partition("-")
    return int(a), int(b or a)


class FigData:
    """Years of one event summed together."""

    def __init__(self, event, years=None):
        sel = [(y, e) for y, e in available(event) if years is None or years[0] <= y <= years[1]]
        if not sel:
            raise FileNotFoundError(f"no figdata for event {event} in {years or 'any year'} under {folder()}")
        self.event, self.years = event, [y for y, _ in sel]
        first = xr.load_dataset(folder() / f"figdata_{event}_b06_{sel[0][0]}.nc")
        self.attrs, self.edges = dict(first.attrs), {k[6:]: first[k].values for k in first.data_vars
                                                    if k.startswith("edges_")}
        hist_vars = [k for k in first.data_vars if k.startswith(("h1_", "h2_", "casc_", "lb_", "strike_", "hit_"))]
        map_vars = [k for k in first.data_vars if k.startswith("map_")]
        self.hist = {k: first[k].values.copy() for k in hist_vars}
        self.var_attrs = {k: dict(first[k].attrs) for k in hist_vars + map_vars}
        self.subbands = list(first["subband"].values)
        self.epsilon = first["epsilon"].values if "epsilon" in first.coords else None
        self.cases = list(first["case"].values) if "case" in first.coords else []
        self.levels = first["level"].values if "level" in first.coords else None
        self.dp_levels = first["dp"].values if "dp" in first.coords else None
        self.lat, self.lon = first["latitude"].values, first["longitude"].values
        self.in_mask = first["in_mask"].values.astype(bool)
        self.map_sum = {k: np.nan_to_num(first[k].values.astype("float64")) for k in map_vars}
        self.map_n = {k: np.isfinite(first[k].values).astype("float64") for k in map_vars}
        # load_dataset reads and closes each file: two open handles on one file can crash HDF5
        for y, _ in sel[1:]:
            ds = xr.load_dataset(folder() / f"figdata_{event}_b06_{y}.nc")
            for k in hist_vars:
                if k in ds:                    # e.g. strike_* only once stage f has run for that year
                    self.hist[k] += ds[k].values
            for k in map_vars:
                if k not in ds:
                    continue
                v = ds[k].values
                self.map_sum[k] += np.nan_to_num(v)
                self.map_n[k] += np.isfinite(v)

    def h(self, name, subband="all", weighted=False):
        """Counts summed over years for one product, sub-band and weighting."""
        return self.hist[name + ("_w" if weighted else "")][self.subbands.index(subband)]

    def edges_of(self, name, axis="x"):
        a = self.var_attrs[name]
        key = a.get("edges") if name.startswith("h1_") else a[f"{axis}_edges"]
        return self.edges[key.replace("edges_", "")]

    def map_mean(self, q, masked=True):
        k = f"map_{q}"
        with np.errstate(invalid="ignore", divide="ignore"):
            m = self.map_sum[k] / self.map_n[k]
        return np.where(self.in_mask, m, np.nan) if masked else m

    def label(self):
        y = self.years
        return f"{y[0]}–{y[-1]}" if len(y) > 1 else str(y[0])


def coarsen(counts, edges, factor, axis=0):
    """Sum every `factor` neighbouring bins along axis; drops a ragged tail."""
    n = counts.shape[axis] // factor * factor
    c = np.take(counts, np.arange(n), axis=axis)
    shape = list(c.shape)
    shape[axis:axis + 1] = [n // factor, factor]
    return c.reshape(shape).sum(axis=axis + 1), edges[: n + 1: factor]


def centers(edges):
    return 0.5 * (edges[1:] + edges[:-1])


def binned_mean(h2, x_edges, y_edges):
    """Mean of y in each x column of a 2-D count array (NaN where empty)."""
    yc = centers(y_edges)
    n = h2.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (h2 * yc).sum(axis=1) / n, n


def zero_crossing(x, y):
    """x where y first changes sign from + to −, by linear interpolation (E09)."""
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    s = np.nonzero((y[:-1] > 0) & (y[1:] <= 0))[0]
    if not len(s):
        return np.nan
    i = s[0]
    return x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1])


def cli(make, doc):
    """Shared command line for figure scripts: --years, --weighted/--unweighted, --out."""
    import argparse
    ap = argparse.ArgumentParser(description=doc.split("\n")[0])
    ap.add_argument("--years", help="e.g. 1980-2019 (default: all figdata files)")
    ap.add_argument("--unweighted", action="store_true", help="plain counts instead of cos(lat) weights")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs"))
    a = ap.parse_args()
    for item in make(Path(a.out), years_arg(a.years), weighted=not a.unweighted):
        print(f"Wrote {item['path']}")
        for k, v in item["facts"].items():
            print(f"  {k}: {v}")


def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    return out_dir / f"{name}.png"


def crop(h2, ex, ey, xlim, ylim):
    """Sub-array of a 2-D histogram inside x/y limits, with its edges."""
    i0, i1 = np.searchsorted(ex, xlim[0]), np.searchsorted(ex, xlim[1])
    j0, j1 = np.searchsorted(ey, ylim[0]), np.searchsorted(ey, ylim[1])
    return h2[i0:i1, j0:j1], ex[i0:i1 + 1], ey[j0:j1 + 1]


def smooth(h2, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(h2, sigma=sigma)


def mass_levels(h2, fractions):
    """Density levels whose enclosed region holds each fraction of the total (e.g. 0.5, 0.75)."""
    v = np.sort(h2.ravel())[::-1]
    c = np.cumsum(v)
    return {f: v[min(np.searchsorted(c, f * c[-1]), len(v) - 1)] for f in fractions}


def composite(d, case, rh_lo, rh_hi):
    """Cascade-box composite over an RH range: mean h, h*, Z per level, plus h_LPL and p_LPL."""
    e = d.edges["rh_cascade"]
    c = d.cases.index(case)
    k = centers(e)
    sel = (k > rh_lo) & (k < rh_hi)
    w = d.hist["casc_weight"][c, sel].sum(0)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for v in ("MSE", "MSE_sat", "Z", "T"):
            if f"casc_sum_{v}" in d.hist:
                out[v] = d.hist[f"casc_sum_{v}"][c, sel].sum(0) / w
        ws = d.hist["casc_weight_scalar"][c, sel].sum() if "casc_weight_scalar" in d.hist else np.nan
        for v in ("h_LPL", "p_LPL"):
            if f"casc_sum_{v}" in d.hist:
                out[v] = d.hist[f"casc_sum_{v}"][c, sel].sum() / ws
    out["n"] = d.hist["casc_count"][c, sel].sum()
    return out
