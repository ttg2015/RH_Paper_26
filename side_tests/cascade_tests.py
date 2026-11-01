"""Candidate profile measurables for K = −dδ_a/d ln RH. Functions for cascade_tests.ipynb.

Fits run over RH 0.30–0.60 in ln RH with a free intercept, so K < 0 when the excess grows with RH and
C = −dz_LCL/d ln RH > 0, as in δ_a = K ln(RH0/RH). Each candidate says K comes from something readable
off the profiles. Levels stay in pressure; height above ground is only used for slopes.
"""
from pathlib import Path

import numpy as np

G_KM = 9.8                                                    # Z in kJ/kg → km
CP, LV = 1.0047, 2500.8                                       # kJ/kg/K, kJ/kg
WINDOW = (0.30, 0.60)
MIN_N = 100                                                   # events per RH bin
MIN_BINS = 4                                                  # populated bins in the window
EPS = 0.2                                                     # km⁻¹
TOP_HPA = 500.0


def load_all(folder, events=("T", "M"), years=None):
    """{event: summed arrays} from casc_{E}_{Y}.npz."""
    out = {}
    for e in events:
        tot, n = {}, 0
        for p in sorted(Path(folder).glob(f"casc_{e}_*.npz")):
            if years and not years[0] <= int(p.stem.split("_")[2]) <= years[1]:
                continue
            z = np.load(p)
            for k in z.files:
                tot[k] = z[k] if k in ("levels", "rh_edges", "names") else tot.get(k, 0) + z[k]
            n += 1
        if n:
            tot["n_files"] = n
            out[e] = tot
    return out


def group(a, name, b=None):
    """One group as mean values per RH bin. Pass b to pool two events (same group name only)."""
    g = list(a["names"]).index(name)
    def s(k):
        x = a[k][g]
        if b is not None:
            x = x + b[k][list(b["names"]).index(name)]
        return x
    with np.errstate(invalid="ignore", divide="ignore"):
        c = {"n": s("n"), "levels": a["levels"], "rh_centers": 0.5 * (a["rh_edges"][1:] + a["rh_edges"][:-1])}
        for v in ("h_LPL", "h_star500", "h500", "zLCL_LPL", "p_LCL_LPL", "p_LPL", "RH_2m", "zs_km"):
            c[v] = s(f"s_{v}") / s(f"w_{v}")
        for v in ("MSE", "MSE_sat", "Z", "T"):
            c[v] = s(f"p_{v}") / s(f"pw_{v}")
    c["zLCL"] = c["zLCL_LPL"] / 1000.0
    c["z"] = c["Z"] / G_KM - c["zs_km"][:, None]              # km above the mean surface of the bin
    c["delta_a"] = c["h_LPL"] - c["h_star500"]
    return c


def interp_p(levels, y, p):
    """y on levels at pressure p, linear in ln p (NaN outside the finite levels)."""
    m = np.isfinite(y)
    if m.sum() < 2 or not np.isfinite(p):
        return np.nan
    o = np.argsort(levels[m])
    return np.interp(np.log(p), np.log(levels[m][o]), y[m][o], left=np.nan, right=np.nan)


def pw3(p, z, h, p_min, min_ml=3, min_s=3, min_u=2, skip=2, min_d=0.30):
    """Constrained continuous three-segment fit, from Cascade_Summary (section 2).
    Slopes G_bl (mixed layer) > G_h (sickle) < G_up (flattening), joints at kink and break."""
    out = dict(G_bl=np.nan, G_h=np.nan, G_up=np.nan, z_kink=np.nan, p_kink=np.nan, h_kink=np.nan,
               z_break=np.nan, p_break=np.nan, h_break=np.nan, rmse=np.nan)
    m = np.isfinite(z) & np.isfinite(h) & np.isfinite(p) & (p >= p_min)
    if m.sum() < min_ml + min_s + min_u:
        return out
    o = np.argsort(z[m])
    zz, hh, pp = z[m][o], h[m][o], p[m][o]
    g0 = np.diff(hh) / np.diff(zz)
    i0 = 0
    while i0 < min(skip, g0.size) and g0[i0] > 0:            # leading levels where h rises with z
        i0 += 1
    zz, hh, pp = zz[i0:], hh[i0:], pp[i0:]
    n = zz.size
    if n < min_ml + min_s + min_u:
        return out
    best = None
    for j in range(min_ml, n - min_s - min_u + 2):
        for k in range(j + min_s - 1, n - min_u + 1):
            X = np.column_stack([np.ones(n), zz, np.clip(zz - zz[j - 1], 0, None), np.clip(zz - zz[k - 1], 0, None)])
            c = np.linalg.lstsq(X, hh, rcond=None)[0]
            g_bl, g_h, g_up = c[1], c[1] + c[2], c[1] + c[2] + c[3]
            if not (g_bl < 0.0 and g_bl > g_h and g_h < g_up) or zz[j - 1] - zz[0] < min_d:
                continue
            sse = np.sum((hh - X @ c) ** 2)
            if best is None or sse < best[0]:
                best = (sse, j, k, c)
    if best is None:
        return out
    sse, j, k, c = best
    zk, zb, g_h = zz[j - 1], zz[k - 1], c[1] + c[2]
    hk = c[0] + c[1] * zk
    out.update(G_bl=c[1], G_h=g_h, G_up=c[1] + c[2] + c[3], z_kink=zk, p_kink=pp[j - 1], h_kink=hk,
               z_break=zb, p_break=pp[k - 1], h_break=hk + g_h * (zb - zk), rmse=np.sqrt(sse / n))
    return out


def plume_top(lev, z, he, h0, p0, eps, top=TOP_HPA, npts=200):
    """Plume of Eq. plume through one composite, from p0 to `top`: h_c at top and the area (h0 − h_c)/ε."""
    m = np.isfinite(z) & np.isfinite(he) & (lev <= p0) & (lev >= top)
    if m.sum() < 3:
        return np.nan
    o = np.argsort(z[m])
    zz, hh = z[m][o], he[m][o]
    grid = np.linspace(zz[0], zz[-1], npts)
    e = np.interp(grid, zz, hh)
    dz = grid[1] - grid[0]
    a = np.exp(-eps * dz)
    hc = h0
    for i in range(1, npts):                                 # exact over a step, trapezoid in h_e
        hc = a * hc + (1 - a) * 0.5 * (e[i] + e[i - 1])
    return hc


def eps_required(lev, z, he, h0, p0, target, lo=1e-3, hi=3.0):
    """ε for which the plume arrives at 500 hPa with h_c = h*500 (NaN if h0 is not above h*500)."""
    if not h0 > target:
        return np.nan
    f = lambda e: plume_top(lev, z, he, h0, p0, e) - target
    a, b = f(lo), f(hi)
    if not (np.isfinite(a) and np.isfinite(b)) or a * b > 0:
        return np.nan
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if a * v <= 0:
            hi = mid
        else:
            lo, a = mid, v
    return 0.5 * (lo + hi)


def measures(c, min_n=MIN_N, ref_rh=0.625):
    """Every quantity per RH bin for one group. The fixed profile is the moistest near the top of the window:
    its kink is lowest, so its sickle covers the LCL of every drier bin."""
    lev, nb = c["levels"].astype(float), len(c["n"])
    keys = ("h_at_LCL", "slope_LCL", "slope_min", "G_bl", "G_h", "G_up", "z_kink", "p_kink", "h_kink", "p_break",
            "h_ref_at_LCL", "dil_500", "eps_req", "p_min", "fit_rmse")
    m = {k: np.full(nb, np.nan) for k in keys}
    use = c["n"] >= min_n
    if not use.any():
        return m
    idx = np.nonzero(use)[0]
    r = idx[np.argmin(abs(c["rh_centers"][idx] - ref_rh))]
    for i in idx:
        h, z = c["MSE"][i], c["z"][i]
        pl = c["p_LCL_LPL"][i]
        m["h_at_LCL"][i] = interp_p(lev, h, pl)
        m["h_ref_at_LCL"][i] = interp_p(lev, c["MSE"][r], pl)
        zl = interp_p(lev, z, pl)
        up = pl - 100.0                                       # local slope, 100 hPa above the LCL
        m["slope_LCL"][i] = (interp_p(lev, h, up) - m["h_at_LCL"][i]) / (interp_p(lev, z, up) - zl)
        fin = np.isfinite(h) & (lev >= 400.0) & (lev < pl)
        if fin.any():
            kmin = np.nonzero(fin)[0][np.argmin(h[fin])]
            m["p_min"][i] = lev[kmin]
            m["slope_min"][i] = (h[kmin] - m["h_at_LCL"][i]) / (z[kmin] - zl)
            f = pw3(lev, z, h, lev[kmin])
            for k, kk in (("G_bl", "G_bl"), ("G_h", "G_h"), ("G_up", "G_up"), ("z_kink", "z_kink"), ("p_kink", "p_kink"),
                          ("h_kink", "h_kink"), ("p_break", "p_break"), ("fit_rmse", "rmse")):
                m[k][i] = f[kk]
        p0 = min(c["p_LPL"][i], np.nanmax(lev[np.isfinite(h)]))
        hc = plume_top(lev, z, h, c["h_LPL"][i], p0, EPS)
        m["dil_500"][i] = c["h_LPL"][i] - hc
        m["eps_req"][i] = eps_required(lev, z, h, c["h_LPL"][i], p0, c["h_star500"][i])
    return m


def kfit(c, y, min_n=MIN_N, window=WINDOW):
    """K = −dy/d ln RH by least squares over the window, weighted by events (NaN if too few bins)."""
    x = c["RH_2m"]
    ok = (c["n"] >= min_n) & np.isfinite(y) & np.isfinite(x) & (x >= window[0]) & (x <= window[1])
    if ok.sum() < MIN_BINS:
        return np.nan
    return -np.polyfit(np.log(x[ok]), y[ok], 1, w=np.sqrt(c["n"][ok]))[0]


def wmean(c, y, min_n=MIN_N, window=WINDOW):
    x = c["RH_2m"]
    ok = (c["n"] >= min_n) & np.isfinite(y) & np.isfinite(x) & (x >= window[0]) & (x <= window[1])
    return np.average(y[ok], weights=c["n"][ok]) if ok.sum() >= MIN_BINS else np.nan


def candidates(c, m=None):
    """K target, the exact pieces, and each candidate K for one group."""
    m = m or measures(c)
    C = kfit(c, c["zLCL"])
    C_kink = kfit(c, m["z_kink"])
    out = {"K": kfit(c, c["delta_a"]), "K_hLPL": kfit(c, c["h_LPL"]), "K_hstar": kfit(c, c["h_star500"]), "C": C, "C_kink": C_kink,
           "Gamma_req": np.nan,
           "n": c["n"][(c["RH_2m"] >= WINDOW[0]) & (c["RH_2m"] <= WINDOW[1])].sum()}
    out["Gamma_req"] = out["K_hLPL"] / C
    out["G_h"], out["G_bl"] = wmean(c, m["G_h"]), wmean(c, m["G_bl"])
    out["coupling"] = C_kink / C
    # exact: K_hLPL = K_at_LCL + K_gap_LCL, and K_at_LCL = K_slide + K_shift
    out["K_at_LCL"] = kfit(c, m["h_at_LCL"])
    out["K_gap_LCL"] = kfit(c, c["h_LPL"] - m["h_at_LCL"])
    out["K_slide"] = kfit(c, m["h_ref_at_LCL"])
    out["K_shift"] = out["K_at_LCL"] - out["K_slide"]
    out["K_at_kink"] = kfit(c, m["h_kink"])
    out["K_gap_kink"] = kfit(c, c["h_LPL"] - m["h_kink"])
    out["K_sickle"] = out["G_h"] * C                          # A: the slope drawn on the profile, with the LCL
    out["K_sickle_kink"] = out["G_h"] * C_kink                # A, with the depth the kink actually moves
    out["K_tilt"] = (out["G_h"] - out["G_bl"]) * C            # h_LPL = h_kink − G_bl·D: constant tilt gives A − G_bl
    out["K_tilt_kink"] = (out["G_h"] - out["G_bl"]) * C_kink
    out["K_fit_all"] = out["K_sickle_kink"] + kfit(c, -m["G_bl"] * m["z_kink"])  # sickle plus the fitted gap, no h_LPL
    out["K_local"] = wmean(c, m["slope_LCL"]) * C
    out["K_bulk"] = wmean(c, m["slope_min"]) * C
    out["K_dil"] = kfit(c, m["dil_500"])                      # dilution of the ε plume on the way to 500 hPa
    out["eps_req"] = wmean(c, m["eps_req"])
    return out


CANDS = {"K_sickle": "sickle slope × C (A)", "K_sickle_kink": "sickle slope × kink C", "K_tilt": "(sickle − ML slope) × C",
         "K_tilt_kink": "(sickle − ML slope) × kink C", "K_fit_all": "sickle × kink C + fitted gap", "K_local": "local slope above LCL × C",
         "K_bulk": "bulk slope LCL→h_min × C", "K_at_LCL": "profile energy at the LCL", "K_slide": "fixed profile, moving LCL",
         "K_at_kink": "energy at the fitted kink", "K_dil": f"plume dilution at ε = {EPS:g}", "K_hLPL": "h_LPL itself (B)"}


def score(rows, key, target="K"):
    """Slope through the origin, R², RMSE and mean ratio of a candidate against the target across groups."""
    x = np.array([r[key] for r in rows], float)
    y = np.array([r[target] for r in rows], float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(n=int(ok.sum()), r2=np.nan, rmse=np.nan, ratio=np.nan)
    x, y = x[ok], y[ok]
    r = np.corrcoef(x, y)[0, 1]
    return dict(n=int(ok.sum()), r2=r * r, rmse=float(np.sqrt(np.mean((x - y) ** 2))), ratio=float((x @ y) / (y @ y)))
