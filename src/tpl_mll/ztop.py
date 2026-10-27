"""Stage f physics: is the entraining plume's top in the mid troposphere?

For each event and each ε: lift a plume from the LPL, find its top, put the event in a
category, flag double crossings and entrainment that raises the plume, and split the
excess at the LCL. Then invert: for each event, the ε window that puts the top in band.
"""
import numpy as np

from .plume import G, P_TOP, crossings, excess, fine_grid, plume, to_fine

CATEGORIES = {0: "outside mask", 1: "pass", 2: "too shallow", 3: "too deep", 4: "stable",
              5: "diluted out"}
# stable: never buoyant even undiluted (ε = 0), the same events at every ε
# diluted out: buoyant at ε = 0 but not at this ε
EPS_MAX = 3.0          # 1/km, upper end of the ε-window search
BISECT_STEPS = 18


class Columns:
    """Environment of n events on the fine grid, built once and reused for every ε."""

    def __init__(self, levels, h, h_sat, phi, h0, p0):
        """levels (L,) ascending hPa; h, h_sat, phi (n, L) in kJ/kg; h0, p0 (n,) at the LPL."""
        self.p = fine_grid(p0)
        self.h = to_fine(levels, h, self.p)
        self.h_sat = to_fine(levels, h_sat, self.p)
        self.z = to_fine(levels, phi, self.p) / G          # km above sea level
        self.h0, self.p0 = h0, p0

    def top(self, eps):
        h_p = plume(self.h, self.z, self.h0, eps)
        p_first, p_high, n_layers, buoyant, at_top = crossings(h_p - self.h_sat, self.p)
        return h_p, p_first, p_high, n_layers, buoyant, at_top

    def effective_top(self, eps):
        """Top pressure that increases with ε: P_TOP if still buoyant there, p0 if never buoyant."""
        _, _, p_high, _, buoyant, at_top = self.top(eps)
        return np.where(at_top, P_TOP, np.where(buoyant, p_high, self.p0))


def categorise(p_high, buoyant, at_top, window, buoyant0):
    """buoyant0: buoyant anywhere at ε = 0 (separates stable from diluted out)."""
    lo, hi = window                                        # e.g. 400, 600 hPa
    cat = np.full(p_high.shape, 1, dtype=np.int8)
    cat[p_high > hi] = 2
    cat[(p_high < lo) | at_top] = 3
    cat[~buoyant] = 5
    cat[~buoyant0] = 4
    return cat


def sweep(cols, eps_list, p_lcl, window):
    """Results for every ε as a dict of (n_eps, n) arrays."""
    keys = ["p_top_first", "p_top", "z_top", "category", "multi_layer", "non_monotonic",
            "E_total", "E_below", "E_above"]
    out = {k: [] for k in keys}
    buoyant0 = cols.top(0.0)[4]
    for eps in eps_list:
        h_p, p_first, p_high, n_layers, buoyant, at_top = cols.top(eps)
        p_top = np.where(at_top | ~buoyant, np.nan, p_high)
        k = np.argmin(np.abs(np.log(cols.p) - np.log(np.nan_to_num(p_top, nan=1.0))[:, None]), 1)
        z_top = np.where(np.isfinite(p_top), cols.z[np.arange(len(k)), k], np.nan)
        below_top = cols.p >= np.nan_to_num(p_top, nan=np.inf)[:, None]
        non_mono = np.any((cols.h > h_p) & below_top, 1)
        tot, below, above = excess(h_p, cols.h, cols.z, cols.p, np.nan_to_num(p_top, nan=np.inf),
                                   p_lcl, eps)
        for key, val in zip(keys, [p_first, p_top, z_top, categorise(p_high, buoyant, at_top, window, buoyant0),
                                   n_layers > 1, non_mono, tot, below, above]):
            out[key].append(np.where(np.isfinite(p_top), val, np.nan) if key.startswith("E_") else val)
    return {k: np.array(v) for k, v in out.items()}


def eps_for_top(cols, target):
    """ε [1/km] that puts the top at `target` hPa (bisection; top pressure rises with ε).

    NaN: the top is lower than target even undiluted (ε = 0); inf: still higher at EPS_MAX.
    """
    lo = np.zeros(len(cols.p0))
    hi = np.full(len(cols.p0), EPS_MAX)
    reach0 = cols.effective_top(0.0) <= target
    beyond = cols.effective_top(EPS_MAX) < target
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        deeper = cols.effective_top(mid) < target
        lo = np.where(deeper, mid, lo)
        hi = np.where(deeper, hi, mid)
    eps = 0.5 * (lo + hi)
    return np.where(~reach0, np.nan, np.where(beyond, np.inf, eps))
