"""Entraining plume from the lowest pressure level (LPL), convective top, excess integral.

Ported from Paper26/Ztop_Distribution.ipynb ([16] Vertical_Profile_Cloud2_xr, [46]
Ztop_Finder_xr, [55] Excess_Integral_xr). Changes from the prototype:
- the plume starts at the LPL with h_LPL (not at min(SP, 1000 hPa) with h_2m)
- integration is in height from the profile's own geopotential, so the hydrostatic
  step (and the T vs T_v question, audit caveat 12) drops out
- every crossing is found, so double crossings are flagged instead of hidden
- all numpy and vectorised over events; inputs are (n events, levels)

Plume equation: dh_p/dz = -ε (h_p - h_env), h_p(z_LPL) = h_LPL, so
    h_p(z) = h_LPL - ε e^{-ε(z - z0)} ∫_{z0}^{z} (h_LPL - h_env) e^{ε(z' - z0)} dz'
Units: h in kJ/kg, z in km, ε in 1/km.
"""
import numpy as np

N_FINE = 160          # points on the fine grid from the LPL up to P_TOP
P_TOP = 100.0         # hPa, top of the integration
G = 9.8               # geopotential (kJ/kg) / G = height (km)


def fine_grid(p_start, n=N_FINE, p_top=P_TOP):
    """(n_events, n) pressures, log-spaced from each event's start pressure up to p_top."""
    frac = np.linspace(0.0, 1.0, n)
    return p_start[:, None] * (p_top / p_start[:, None]) ** frac


def to_fine(levels, field, p_fine):
    """Linear-in-log(p) interpolation of field (n, L) on ascending levels (L,) to p_fine (n, k)."""
    lnl, lnp = np.log(levels), np.log(p_fine)
    i = np.clip(np.searchsorted(lnl, lnp) - 1, 0, len(levels) - 2)
    w = (lnp - lnl[i]) / (lnl[i + 1] - lnl[i])
    rows = np.arange(field.shape[0])[:, None]
    return field[rows, i] * (1 - w) + field[rows, i + 1] * w


def _cumtrapz(y, x):
    out = np.zeros_like(y)
    out[:, 1:] = np.cumsum(0.5 * (y[:, 1:] + y[:, :-1]) * np.diff(x, axis=1), axis=1)
    return out


def plume(h_env, z, h0, eps):
    """Plume MSE on the same (n, k) grid as h_env and z (km); column 0 is the start.
    eps is one value or one per event (n,)."""
    dz = z - z[:, :1]
    eps = np.broadcast_to(np.asarray(eps, dtype=float), h0.shape)[:, None]
    integral = _cumtrapz((h0[:, None] - h_env) * np.exp(eps * dz), z)
    return h0[:, None] - eps * np.exp(-eps * dz) * integral


def crossings(diff, p):
    """Downward crossings of diff = h_p - h* going up the column (buoyant → not).

    Returns (p_first, p_highest, n_layers, ever_buoyant, buoyant_at_top):
    p_first   first downward crossing above the first buoyant point (the EL of the lowest layer)
    p_highest highest downward crossing (the EL of the highest layer)
    n_layers  number of buoyant layers that end below P_TOP
    """
    pos = diff > 0
    down = pos[:, :-1] & ~pos[:, 1:]                     # k → k+1 goes from + to −
    k = np.arange(down.shape[1])
    # interpolate the crossing pressure in log(p)
    d0, d1 = diff[:, :-1], diff[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(down, d0 / (d0 - d1), np.nan)
    lnp = np.log(p[:, :-1]) + w * (np.log(p[:, 1:]) - np.log(p[:, :-1]))
    first_k = np.where(down.any(1), np.argmax(down, 1), -1)
    last_k = np.where(down.any(1), down.shape[1] - 1 - np.argmax(down[:, ::-1], 1), -1)
    rows = np.arange(diff.shape[0])
    p_first = np.where(first_k >= 0, np.exp(lnp[rows, np.maximum(first_k, 0)]), np.nan)
    p_highest = np.where(last_k >= 0, np.exp(lnp[rows, np.maximum(last_k, 0)]), np.nan)
    return p_first, p_highest, down.sum(1), pos.any(1), pos[:, -1]


def excess(h_p, h_env, z, p, p_top, p_lcl, eps):
    """ε ∫ (h_p - h_env) dz from the start to the top, and its parts below / above the LCL."""
    y = h_p - h_env
    def integral(mask):
        yy = np.where(mask, y, 0.0)
        return eps * np.sum(0.5 * (yy[:, 1:] + yy[:, :-1]) * np.diff(z, axis=1), axis=1)
    below_top = p >= p_top[:, None]
    total = integral(below_top)
    below = integral(below_top & (p >= p_lcl[:, None]))
    return total, below, total - below
