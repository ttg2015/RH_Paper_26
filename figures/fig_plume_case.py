"""Plume Case (paper figure): one event's column with entraining plumes at a few ε.

One annual-maximum column at the event hour (lead 0), read from the stage c2/d outputs. Plumes start at
the lowest pressure level (LPL) with the mixed-layer MSE h_ML and are drawn up to their top, the highest
crossing of h*. ε = 0 is the undilute parcel. The ε values and colors are style.EPS_SHOWN / EPS_COLOR, the
same as in the pooled plume-top figure, so the two read together.

Default: the grid point nearest Beijing in 1986, wet-bulb (MSE) extreme, as in Fig. 1b. With --pick, the
event nearest the target whose plumes give a clean example: δ_a > 0, and at every shown ε one buoyant
layer with a top between 250 and 800 hPa that descends as ε grows.

    python figures/fig_plume_case.py [--year 1986] [--event M] [--lat 39.9 --lon 116.4] [--pick]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figdata as FD
import style
from tpl_mll import ztop

REPO = Path(__file__).resolve().parents[1]
EPS = (0.0,) + tuple(style.EPS_SHOWN)
BEIJING = (39.9, 116.4)
PICK = dict(radius_deg=6.0, p_top_range=(250.0, 800.0))


def _open(year, event, buffer=6):
    import xarray as xr
    paths = FD.load("paths")
    work = Path(paths["work"])
    prof = xr.open_dataset(work / paths["outputs"]["profiles"] / f"profiles_{event}_b{buffer:02d}_{year}.nc")
    der = xr.open_dataset(work / paths["outputs"]["derived"] / f"derived_{event}_b{buffer:02d}_{year}.nc")
    return prof.sel(lead=0).isel(year=0), der.sel(lead=0).isel(year=0)


def _grid(da):
    return da.transpose("latitude", "longitude", ...).values


def columns(year, event, buffer=6):
    """Every finite column of the year as flat arrays, with lat/lon and the analysis mask."""
    from tpl_mll.masks import analysis_mask
    prof, der = _open(year, event, buffer)
    lat, lon = prof["latitude"].values, prof["longitude"].values
    ok = np.isfinite(_grid(der["MSE_LPL"]))
    i, j = np.nonzero(ok)
    take = lambda da: _grid(da)[i, j]
    return dict(levels=prof["level"].values.astype(float), h=take(prof["MSE"]), h_sat=take(prof["MSE_sat"]),
                phi=take(prof["Z"]), h0=take(der["MSE_LPL"]), p0=take(der["LPL_pressure"]),
                p_lcl=take(der["p_LCL_LPL"]), rh=take(der["RH_2m"]), lat=lat[i], lon=lon[j],
                in_mask=analysis_mask(lat, lon)[i, j])


def distance_deg(lat, lon, target):
    dlon = (lon - target[1] + 180.0) % 360.0 - 180.0
    return np.hypot(lat - target[0], dlon * np.cos(np.deg2rad(target[0])))


def tops(col, eps=EPS):
    """Plume on the fine grid and its top for each ε, for a dict of (n, ...) columns."""
    cols = ztop.Columns(col["levels"], col["h"], col["h_sat"], col["phi"], col["h0"], col["p0"])
    out = {}
    for e in eps:
        h_p, _, p_high, n_layers, buoyant, at_top = cols.top(e)
        out[e] = dict(h_p=h_p, p_top=p_high, n_layers=n_layers, buoyant=buoyant, at_top=at_top)
    return cols, out


def h_star_500(col):
    k = int(np.argmin(abs(col["levels"] - 500.0)))
    return col["h_sat"][..., k]


def choose(col, target, pick):
    """Index of the column to draw: nearest to target, or nearest clean example with --pick."""
    d = distance_deg(col["lat"], col["lon"], target)
    if not pick:
        return int(np.argmin(d))
    near = np.nonzero((d <= PICK["radius_deg"]) & col["in_mask"] & (col["h0"] > h_star_500(col)))[0]
    if not len(near):
        raise ValueError(f"no unstable in-mask events within {PICK['radius_deg']}° of {target}")
    sub = {k: (v if k == "levels" else v[near]) for k, v in col.items()}
    _, res = tops(sub, style.EPS_SHOWN)
    lo, hi = PICK["p_top_range"]
    good = np.ones(len(near), bool)
    prev = np.zeros(len(near))
    for e in style.EPS_SHOWN:
        r = res[e]
        p = r["p_top"]
        good &= r["buoyant"] & ~r["at_top"] & (r["n_layers"] == 1) & (p >= lo) & (p <= hi) & (p > prev + 20)
        prev = np.where(np.isfinite(p), p, prev)
    if not good.any():
        raise ValueError("no event near the target passes the --pick criteria; widen PICK or move the target")
    k = near[good][np.argmin(d[near][good])]
    return int(k)


def draw(one, out_dir, label):
    """one: a single column as (1, ...) arrays plus scalars; returns the item dict for make_all."""
    style.apply()
    cols, res = tops(one)
    p = cols.p[0]
    h_env, h_sat = cols.h[0], cols.h_sat[0]
    h0, p0, p_lcl = float(one["h0"][0]), float(one["p0"][0]), float(one["p_lcl"][0])
    hs500 = float(h_star_500(one)[0])

    fig, ax = plt.subplots(figsize=(3.8, 5.0))
    style.paper_axes(ax)
    ax.axhspan(600, 400, color=style.WINDOW_COLOR, alpha=0.08, lw=0, zorder=0)
    ax.plot(h_env, p, color="k", lw=2.0, zorder=3)
    ax.plot(h_sat, p, color="k", lw=1.6, ls="--", zorder=3)
    facts = {"event": label, "h_ML": f"{h0:.1f} kJ/kg at {p0:.0f} hPa",
             "δ_a = h_ML − h*500": f"{h0 - hs500:+.1f} kJ/kg", "LCL of LPL air": f"{p_lcl:.0f} hPa"}
    handles = [Line2D([0], [0], color="k", lw=2.0, label="environment $h$"),
               Line2D([0], [0], color="k", lw=1.6, ls="--", label="saturated $h^*$")]
    for e in EPS:
        r = res[e]
        h_p = r["h_p"][0]
        col = style.EPS_COLOR[e]
        lw, ls = (1.4, ":") if e == 0 else (2.0, "-")
        if r["at_top"][0]:                                      # still buoyant at 100 hPa
            top, end, where = np.nan, p.min(), "above 100 hPa"
        elif r["buoyant"][0]:
            top = end = float(r["p_top"][0])
            where = f"top {top:.0f} hPa"
        else:                                                   # diluted out: stop at closest approach to h*
            gap = np.where(p <= p_lcl, h_sat - h_p, np.inf)
            top, end, where = np.nan, float(p[int(np.argmin(gap))]), "diluted out"
        ax.plot(np.where(p >= end, h_p, np.nan), p, color=col, lw=lw, ls=ls, zorder=4)
        x_end = np.interp(np.log(end), np.log(p[::-1]), h_p[::-1])
        if e and np.isfinite(top):
            ax.plot(x_end, end, "o", color=col, ms=5.5, mec="white", mew=0.8, zorder=5)
        elif e:
            ax.plot(x_end, end, "x", color=col, ms=5.5, mew=1.6, zorder=5)
        name = "undilute ($\\varepsilon = 0$)" if e == 0 else f"$\\varepsilon = {e:g}$ km$^{{-1}}$"
        mk = None if e == 0 else ("o" if np.isfinite(top) else "x")
        handles.append(Line2D([0], [0], color=col, lw=lw, ls=ls, marker=mk, ms=4.5, label=f"{name}: {where}"))
        facts[f"ε={e:g}"] = where
    if h0 > hs500:                                              # δ_a = h_ML − h*500, as in Fig. 1
        ax.annotate("", xy=(h0, 500), xytext=(hs500, 500),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=0.9, shrinkA=0, shrinkB=0), zorder=6)
        ax.text(0.5 * (h0 + hs500), 492, r"$\delta_a$", ha="center", va="bottom", fontsize=8.5, zorder=6)
    ax.plot(h0, p0, "s", color="k", ms=4.5, zorder=6)
    ax.annotate("$h_{ML}$", (h0, p0), xytext=(4, -2), textcoords="offset points", fontsize=8, va="top")
    ax.axhline(p_lcl, color="#6B7280", lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(0.98, p_lcl, "LCL", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7,
            color="#4B5563")
    ax.text(0.98, 425, "400–600 hPa", transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7, color=style.WINDOW_COLOR)
    upper = p >= 150
    lo_x = np.nanmin(h_env[upper]) - 2
    hi_x = max(h0, hs500) + 6
    ax.set(xlim=(lo_x, hi_x), ylim=(min(1000.0, p0 + 15), 150), xlabel=r"MSE (kJ kg$^{-1}$)",
           ylabel="Pressure (hPa)")
    ax.set_yticks(np.arange(200, 1001, 100))
    ax.legend(handles=handles, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=False, handlelength=2.0, columnspacing=1.0, labelspacing=0.35)
    fig.tight_layout()
    return {"path": FD.save(fig, out_dir, "plume_case"), "title": "Plume Case", "facts": facts,
            "script": "figures/fig_plume_case.py"}


def make(out_dir, years=None, weighted=False, year=1986, event="M", target=BEIJING, pick=False):
    col = columns(year, event)
    k = choose(col, target, pick)
    one = {key: (v if key == "levels" else v[k:k + 1]) for key, v in col.items()}
    lat, lon = float(one["lat"][0]), float(one["lon"][0])
    label = (f"{'wet-bulb' if event == 'M' else 'temperature'} extreme {year}, {lat:.2f}°N {lon:.2f}°E, "
             f"RH {float(one['rh'][0]):.2f}{'' if one['in_mask'][0] else ' (outside analysis mask)'}")
    return [draw(one, out_dir, label)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=1986)
    ap.add_argument("--event", choices=["T", "M"], default="M")
    ap.add_argument("--lat", type=float, default=BEIJING[0])
    ap.add_argument("--lon", type=float, default=BEIJING[1])
    ap.add_argument("--pick", action="store_true", help="nearest clean example instead of the nearest point")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs"))
    a = ap.parse_args()
    for item in make(Path(a.out), year=a.year, event=a.event, target=(a.lat, a.lon), pick=a.pick):
        print(f"Wrote {item['path']}")
        for k, v in item["facts"].items():
            print(f"  {k}: {v}")
