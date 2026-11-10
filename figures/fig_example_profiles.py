"""Example Profiles (paper Fig. 1): the temperature and the wet-bulb extreme of one grid point, with the excesses.

Ports the hand-made Fig. 1 into the pipeline. Both annual maxima (T and M) of one year at the grid point
nearest the target (default Beijing, 1986), at the event hour (lead 0), from the stage c2 profiles and
stage d derived files, as fig_plume_case.py reads them. Each panel shows
    h (solid) above the lowest pressure level (LPL), joined to the 2-m value h_2m at the surface pressure;
    h* (dashed); the mixed-layer value h_ML (square, at the LPL) and h_2m (circle);
    a vertical line at h*500, and the excesses as labeled arrows:
        δ_a = h_ML − h*500,  δ_S = h_2m − h_ML,  δ = h_2m − h*500 (the definitions of stage g).
The exact lat/lon go in the figure title, as the reviewer asked.

Each run from the pipeline files writes the two columns and their numbers to <out>/example_profiles_cache.npz
(or --dump FILE). fig_wbt_increase.py reads it for its Beijing dots (make_all runs this figure first), and
--cache FILE redraws this figure anywhere, no Casper files needed.

    python figures/fig_example_profiles.py [--year 1986] [--lat 39.9 --lon 116.4] [--dump F | --cache F]
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

REPO = Path(__file__).resolve().parents[1]
BEIJING = (39.9, 116.4)
CACHE_NAME = "example_profiles_cache.npz"                    # written next to the figures (out_dir)
X_FROM_HPA = 300.0                                            # x-limits from levels at or below this pressure
KEYS = ("levels", "h", "h_sat", "h_ml", "p_lpl", "h_2m", "p_sfc", "T2m", "q2m", "rh", "p_lcl", "lat", "lon")


def column(year, event, target, buffer=6):
    """One event's column nearest `target` as a dict of plain numbers and 1-D arrays."""
    import fig_plume_case as P
    prof, der = P._open(year, event, buffer)
    lat, lon = prof["latitude"].values, prof["longitude"].values
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    d = P.distance_deg(la, lo, target)
    d[~np.isfinite(P._grid(der["MSE_LPL"]))] = np.inf
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    g = lambda da: P._grid(da)[i, j]
    opt = lambda v: float(g(prof[v])) if v in prof else np.nan          # 2-m T and q: only facts and Fig. 2 dots
    out = {"levels": prof["level"].values.astype(float), "h": g(prof["MSE"]), "h_sat": g(prof["MSE_sat"]),
           "h_ml": float(g(der["MSE_LPL"])), "p_lpl": float(g(der["LPL_pressure"])), "h_2m": float(g(prof["MSE2m"])),
           "p_sfc": float(g(prof["SP"])) / 100.0, "T2m": opt("T2m"), "q2m": opt("q2m"),
           "rh": float(g(der["RH_2m"])), "p_lcl": float(g(der["p_LCL_LPL"])), "lat": float(lat[i]),
           "lon": float(lon[j])}
    for tname in ("time", "valid_time"):                          # the event date, if the file keeps it
        if tname in prof:
            try:
                out["date"] = str(np.datetime_as_string(np.asarray(prof[tname].values).ravel()[0], unit="D"))
            except (TypeError, ValueError, IndexError):
                pass
            break
    return out


def h_star_500(col):
    lev, hs = np.asarray(col["levels"], float), np.asarray(col["h_sat"], float)
    ok = np.isfinite(hs)
    o = np.argsort(lev[ok])                                       # np.interp needs ascending x
    return float(np.interp(np.log(500.0), np.log(lev[ok][o]), hs[ok][o]))


def excesses(col):
    hs = h_star_500(col)
    return {"h*500": hs, "delta_a": col["h_ml"] - hs, "delta_S": col["h_2m"] - col["h_ml"],
            "delta": col["h_2m"] - hs}


def from_pipeline(year, target):
    return {e: column(year, e, target) for e in ("T", "M")}


def dump(path, cols, year):
    out = {"year": np.array(year)}
    for e, c in cols.items():
        for k, v in c.items():
            out[f"{e}_{k}"] = np.asarray(v)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


def from_cache(path):
    z = np.load(path)
    cols = {}
    for e in ("T", "M"):
        cols[e] = {k[2:]: (z[k] if z[k].ndim else z[k].item()) for k in z.files if k.startswith(e + "_")}
    return cols, int(z["year"])


def arrow(ax, x0, x1, y, text, color="k"):
    """Double-headed arrow from x0 to x1 at pressure y, labeled just right of its right end."""
    if abs(x1 - x0) >= 0.05:
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.0, shrinkA=0, shrinkB=0,
                                    mutation_scale=7), zorder=7)
    ax.text(max(x0, x1) + 0.6, y, text, ha="left", va="center", fontsize=8.5, color=color, zorder=7)


def panel(ax, col, event, letter):
    lev = col["levels"]
    ex = excesses(col)
    hs = ex["h*500"]
    style.paper_axes(ax)
    up = (lev <= col["p_lpl"]) & np.isfinite(col["h"])
    o = np.argsort(lev[up])
    ax.plot(col["h"][up][o], lev[up][o], color="k", lw=2.0, zorder=3)
    ax.plot([col["h_ml"], col["h_2m"]], [col["p_lpl"], col["p_sfc"]], color="k", lw=1.2, zorder=3)  # surface layer
    sat = np.isfinite(col["h_sat"]) & (lev <= col["p_lpl"])
    ax.plot(col["h_sat"][sat], lev[sat], color="k", lw=1.6, ls="--", zorder=3)
    ax.axvline(hs, color=style.WINDOW_COLOR, lw=0.9, zorder=1)
    ax.plot(hs, 500.0, "D", color=style.WINDOW_COLOR, ms=4.5, zorder=5)
    ax.plot(col["h_ml"], col["p_lpl"], "s", color="k", ms=5, zorder=6)
    ax.plot(col["h_2m"], col["p_sfc"], "o", color=style.EVENT_COLOR[event], mec="k", mew=0.6, ms=6.5, zorder=6)
    # the excesses on three lines above the surface: δ_S, δ_a, then their sum δ
    base = col["p_lpl"]
    y_s, y_a, y_d = base - 60.0, base - 115.0, base - 170.0
    arrow(ax, col["h_ml"], col["h_2m"], y_s, r"$\delta_S$")
    arrow(ax, hs, col["h_ml"], y_a, r"$\delta_a$")
    arrow(ax, hs, col["h_2m"], y_d, r"$\delta$")
    for x, y0, p_end in ((col["h_ml"], y_a, col["p_lpl"]), (col["h_2m"], y_d, col["p_sfc"])):
        ax.plot([x, x], [y0, p_end], color="0.6", lw=0.6, ls=":", zorder=2)
    trop = lev >= X_FROM_HPA                                   # x-range from the troposphere only
    x_all = np.r_[col["h"][up & trop], col["h_2m"], hs, col["h_ml"]]
    ax.set(xlim=(np.nanmin(x_all) - 3, np.nanmax(x_all) + 6), ylim=(max(col["p_sfc"], col["p_lpl"]) + 10, 200),
           xlabel="MSE " + style.KJ)
    ax.set_yticks(np.arange(200, 1001, 100))
    date = f", {col['date']}" if col.get("date") else ""
    ax.set_title(f"{style.EVENT_SHORT[event]} extreme{date}", fontsize=10, color=style.EVENT_COLOR[event])
    style.panel_letter(ax, letter)
    return ex


def draw(cols, year, out_dir):
    style.apply()
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.6), sharey=True)
    facts = {}
    for k, e in enumerate(("T", "M")):
        ex = panel(axs[k], cols[e], e, k)
        c = cols[e]
        facts[f"{e} event"] = (f"{c['lat']:.2f}°N {c['lon']:.2f}°E{', ' + c['date'] if c.get('date') else ''}; "
                               f"h_2m {c['h_2m']:.1f}, h_ML {c['h_ml']:.1f} at {c['p_lpl']:.0f} hPa, h*500 {ex['h*500']:.1f}; "
                               f"δ {ex['delta']:+.1f}, δ_S {ex['delta_S']:+.1f}, δ_a {ex['delta_a']:+.1f} kJ/kg; "
                               f"RH {100 * c['rh']:.0f}%, T2m {c['T2m'] - 273.15:.1f} °C")
    axs[0].set_ylabel("Pressure [hPa]")
    c = cols["T"]
    fig.suptitle(f"{year}, grid point {c['lat']:.2f}°N, {c['lon']:.2f}°E", fontsize=10, y=0.995)
    axs[1].legend(handles=[Line2D([0], [0], color="k", lw=2.0, label="$h$"),
                           Line2D([0], [0], color="k", lw=1.6, ls="--", label="$h^*$"),
                           Line2D([0], [0], ls="", marker="o", mfc="w", mec="k", label="$h_{2m}$"),
                           Line2D([0], [0], ls="", marker="s", color="k", label="$h_{ML}$ (LPL)"),
                           Line2D([0], [0], color=style.WINDOW_COLOR, marker="D", ms=4, label=r"$h^*_{500}$")],
                  loc="upper right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    path = FD.save(fig, out_dir, "example_profiles")
    plt.close(fig)
    facts["note"] = "δ = h_2m − h*500, δ_S = h_2m − h_ML, δ_a = h_ML − h*500 (stage g definitions)"
    return {"path": path, "title": "Example Profiles (Fig. 1)", "facts": facts,
            "script": "figures/fig_example_profiles.py"}


def load(year=1986, target=BEIJING, cache=None, dump_to=None):
    """(cols, year) from the cache if given, else from the pipeline files (dumping the cache to dump_to)."""
    if cache:
        return from_cache(cache)
    cols = from_pipeline(year, target)
    if dump_to:
        dump(dump_to, cols, year)
    return cols, year


def make(out_dir, years=None, weighted=False, year=1986, target=BEIJING, cache=None, dump_to=None):
    cols, year = load(year, target, cache, dump_to or Path(out_dir) / CACHE_NAME)
    return [draw(cols, year, out_dir)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=1986)
    ap.add_argument("--lat", type=float, default=BEIJING[0])
    ap.add_argument("--lon", type=float, default=BEIJING[1])
    ap.add_argument("--cache", help="draw from a cache written with --dump")
    ap.add_argument("--dump", help=f"where to write the cache (default <out>/{CACHE_NAME})")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs"))
    a = ap.parse_args()
    for item in make(Path(a.out), year=a.year, target=(a.lat, a.lon), cache=a.cache, dump_to=a.dump):
        print(f"Wrote {item['path']}")
        for k, v in item["facts"].items():
            print(f"  {k}: {v}")
