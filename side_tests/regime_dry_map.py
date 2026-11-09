"""Where do the driest bins of a cascade regime come from? Do they collocate?

Selects the events behind the two driest 5 % RH bins (RH < 0.10) of one regime of plume_top_vs_rh.py
(default: tropics with a moister free troposphere, h*500 339–342, h500 332–335), with the same mask
and terrain cut as cascade_composites.py, for both event types and every year. Saves the events and
maps where they are.

Reads   Profiles/profiles_{E}_b06_{Y}.nc (MSE and MSE_sat at 500 hPa only), Profiles/derived/derived_{E}_b06_{Y}.nc
Writes  <work>/side_tests/dry_map/{events.npz, dry_map.png, dry_map.md}

    python side_tests/regime_dry_map.py [--years 1980-2019] [--regime "tropics, moister FT"] [--rh-max 0.10]
    python side_tests/regime_dry_map.py --plot-only        # redraw from events.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "pipeline"), str(REPO / "figures"), str(REPO / "side_tests")]

RH_SPLIT = 0.05                                               # the two 5 % bins
BOX_DEG = 1.0                                                 # map resolution for counts
KEYS = ("lat", "lon", "year", "event", "rh", "h_LPL", "h_star500", "h500", "p_LCL_LPL", "p_LPL")


def regime_box(name):
    from plume_top_vs_rh import REGIMES
    if name not in REGIMES:
        raise SystemExit(f"unknown regime {name!r}; choose from {list(REGIMES)}")
    return REGIMES[name]


def events(year, event, hs, h, rh_max, buffer=6):
    """Events of one year and type inside the box with RH < rh_max, as a dict of 1-D arrays."""
    import xarray as xr
    import g
    paths = g.load("paths")
    work, out = Path(paths["work"]), paths["outputs"]
    cfg = g.load("figdata")
    lead = cfg["lead"]
    with xr.open_dataset(work / out["profiles"] / f"profiles_{event}_b{buffer:02d}_{year}.nc") as p:
        p5 = p[["MSE", "MSE_sat"]].sel(lead=lead, level=500).isel(year=0).load()
    with xr.open_dataset(work / out["derived"] / f"derived_{event}_b{buffer:02d}_{year}.nc") as d:
        der = d[["MSE_LPL", "RH_2m", "p_LCL_LPL", "LPL_pressure"]].sel(lead=lead).isel(year=0).load()
    grid = lambda da: da.transpose("latitude", "longitude").values.astype("float64")
    lat, lon = p5["latitude"].values, p5["longitude"].values
    hs500, h500 = grid(p5["MSE_sat"]), grid(p5["MSE"])
    rh, h_lpl = grid(der["RH_2m"]), grid(der["MSE_LPL"])
    # same mask as cascade_composites.py: figure mask, cascade terrain cut, finite h_LPL and h*500,
    # and the grid cell's half-open edges [lo, hi)
    mask = g.figure_mask(lat, lon, cfg) & (g.surface_height_km(lat, lon) < cfg["cascade"]["zs_max_km"])
    sel = (mask & np.isfinite(h_lpl) & np.isfinite(hs500) & (rh >= 0) & (rh < rh_max)
           & (hs500 >= hs[0]) & (hs500 < hs[1]) & (h500 >= h[0]) & (h500 < h[1]))
    i, j = np.nonzero(sel)
    n = len(i)
    return {"lat": lat[i], "lon": lon[j], "year": np.full(n, year), "event": np.full(n, event),
            "rh": rh[i, j], "h_LPL": h_lpl[i, j], "h_star500": hs500[i, j], "h500": h500[i, j],
            "p_LCL_LPL": grid(der["p_LCL_LPL"])[i, j], "p_LPL": grid(der["LPL_pressure"])[i, j]}


def collect(years, hs, h, rh_max):
    parts = []
    for y in range(years[0], years[1] + 1):
        for e in ("T", "M"):
            ev = events(y, e, hs, h, rh_max)
            print(f"{y} {e}: {len(ev['lat'])} events", flush=True)
            parts.append(ev)
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def summary(ev, name, hs, h, rh_max):
    """Markdown: totals, years, event types, and the 5° boxes holding most events."""
    lon = (ev["lon"] + 180.0) % 360.0 - 180.0
    L = [f"# Driest bins of '{name}' (h*500 {hs[0]:g}–{hs[1]:g}, h500 {h[0]:g}–{h[1]:g}, RH < {rh_max:g})", ""]
    for lo, hi in ((0.0, RH_SPLIT), (RH_SPLIT, rh_max)):
        m = (ev["rh"] >= lo) & (ev["rh"] < hi)
        cells = len(set(zip(ev["lat"][m].round(2), lon[m].round(2))))
        years = np.unique(ev["year"][m])
        L += [f"- RH {lo:.2f}–{hi:.2f}: {m.sum():,} events in {cells:,} grid cells over {len(years)} years "
              f"({(ev['event'][m] == 'T').sum():,} T, {(ev['event'][m] == 'M').sum():,} M); "
              f"mean LCL {np.nanmean(ev['p_LCL_LPL'][m]):.0f} hPa, mean LPL {np.nanmean(ev['p_LPL'][m]):.0f} hPa"]
    bl, bo = np.floor(ev["lat"] / 5) * 5, np.floor(lon / 5) * 5
    keys, counts = np.unique(np.stack([bl, bo], 1), axis=0, return_counts=True)
    top = np.argsort(counts)[::-1][:12]
    total = len(ev["lat"])
    L += ["", "| 5° box (lat, lon) | events | share | cumulative | years | T / M |", "|---|---|---|---|---|---|"]
    cum = 0
    for k in top:
        m = (bl == keys[k, 0]) & (bo == keys[k, 1])
        cum += counts[k]
        L.append(f"| {keys[k, 0]:+.0f}…{keys[k, 0] + 5:+.0f}, {keys[k, 1]:+.0f}…{keys[k, 1] + 5:+.0f} | {counts[k]:,} | "
                 f"{100 * counts[k] / total:.0f}% | {100 * cum / total:.0f}% | {len(np.unique(ev['year'][m]))} | "
                 f"{(ev['event'][m] == 'T').sum():,} / {(ev['event'][m] == 'M').sum():,} |")
    yrs, yc = np.unique(ev["year"], return_counts=True)
    L += ["", "Events per year: " + ", ".join(f"{y}: {c}" for y, c in zip(yrs, yc))]
    return "\n".join(L) + "\n"


def figure(ev, path, name, rh_max):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import maps
    import style
    style.apply()
    lon = (ev["lon"] + 180.0) % 360.0 - 180.0
    le = np.arange(-60, 75.01, BOX_DEG)
    oe = np.arange(-180, 180.01, BOX_DEG)
    fig = plt.figure(figsize=(15, 8.6))
    subplot_kw = {"projection": maps.ccrs.Robinson()} if maps.HAVE_CARTOPY else {}
    bins = ((0.0, RH_SPLIT), (RH_SPLIT, rh_max))
    counts = [np.histogram2d(ev["lat"][(ev["rh"] >= a) & (ev["rh"] < b)], lon[(ev["rh"] >= a) & (ev["rh"] < b)],
                             [le, oe])[0] for a, b in bins]
    vmax = max(c.max() for c in counts) or 1
    for k, ((a, b), c) in enumerate(zip(bins, counts)):
        ax = fig.add_subplot(2, 2, k + 1, **subplot_kw)
        im = maps.draw(ax, oe, le, np.ma.masked_equal(c, 0), band=None, labels=(k == 0), cmap="viridis",
                       norm=LogNorm(1, vmax))
        ax.set_title(f"RH {a:.2f}–{b:.2f}: {int(c.sum()):,} events", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, label=f"events per {BOX_DEG:g}° cell, 40 years")
    ax = style.paper_axes(fig.add_subplot(2, 2, 3))
    yrs = np.arange(ev["year"].min(), ev["year"].max() + 1) if len(ev["year"]) else np.array([0])
    for e, col in (("T", style.EVENT_COLOR["T"]), ("M", style.EVENT_COLOR["M"])):
        m = ev["event"] == e
        ax.bar(yrs + (-0.2 if e == "T" else 0.2), [np.sum(m & (ev["year"] == y)) for y in yrs], width=0.4,
               color=col, label=style.EVENT_LABEL[e])
    ax.set(xlabel="year", ylabel="events", title="Events per year (both bins)")
    ax.legend(frameon=False, fontsize=8)
    ax = style.paper_axes(fig.add_subplot(2, 2, 4))
    ax.hist(ev["p_LCL_LPL"], bins=np.arange(300, 1001, 20), color="#6B7280")
    ax.set(xlabel="LCL of mixed-layer air (hPa)", ylabel="events", title="Each event's LCL (both bins)")
    ax.invert_xaxis()
    fig.suptitle(f"Driest bins (RH < {rh_max:g}) of the '{name}' regime: where they come from", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", default="1980-2019")
    ap.add_argument("--regime", default="tropics, moister FT")
    ap.add_argument("--rh-max", type=float, default=0.10)
    ap.add_argument("--out", help="folder (default <work>/side_tests/dry_map)")
    ap.add_argument("--plot-only", action="store_true", help="redraw from an existing events.npz")
    a = ap.parse_args(argv)
    hs, h = regime_box(a.regime)
    if a.out:
        out = Path(a.out)
    else:
        from tpl_mll.config import load
        out = Path(load("paths")["work"]) / "side_tests" / "dry_map"
    out.mkdir(parents=True, exist_ok=True)
    if a.plot_only:
        z = np.load(out / "events.npz")
        ev = {k: z[k] for k in KEYS}
    else:
        ev = collect(tuple(int(y) for y in a.years.split("-")), hs, h, a.rh_max)
        np.savez_compressed(out / "events.npz", **ev)
    text = summary(ev, a.regime, hs, h, a.rh_max)
    (out / "dry_map.md").write_text(text)
    figure(ev, out / "dry_map.png", a.regime, a.rh_max)
    print(text)
    print(f"Wrote {out / 'dry_map.md'}, dry_map.png, events.npz")


if __name__ == "__main__":
    main()
