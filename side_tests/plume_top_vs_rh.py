"""Does the plume top rise with surface RH when the free troposphere is held fixed? (advisor's pathway ii)

The pooled data say no: at ε = 0.2 the median plume top stays near 500 hPa at every RH (old Fig. 6d). But
pooling mixes free tropospheres. Here we hold the free troposphere fixed: for three regimes in the
(h*500, h500) plane, with T and wet-bulb extremes pooled, we lift a plume through the mean profile of each
5 % RH bin and record its top (highest crossing of h*) at each ε in style.EPS_SHOWN.

Regimes are unions of the 3 kJ/kg grid cells of cascade_composites.py, chosen to sit on the draft's
cascade boxes: middle latitudes, tropics with a drier free troposphere (h*500 − h500 ≈ 10–11), and tropics
with a moister one (≈ 7). A plume through a mean profile is not the mean of the events' plume tops; this
asks whether the composite cascade, the object of the RH argument, shows the effect.

Reads   <work>/side_tests/casc/casc_{E}_{Y}.npz (cascade_composites.py)
Writes  <work>/side_tests/plume_top_vs_rh.md and .png

    python side_tests/plume_top_vs_rh.py [--years 1980-2019] [--casc DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "figures"), str(REPO / "side_tests")]
import cascade_tests as T                                     # noqa: E402
import cascade_composites as CC                               # noqa: E402
import style                                                  # noqa: E402
from tpl_mll import ztop                                      # noqa: E402

# (h*500 range, h500 range) in kJ/kg, on the cascade_composites grid
# shading in the height row is the entrainment depth, LCL to top
REGIMES = {
    "middle latitudes": ((327.0, 330.0), (320.0, 323.0)),
    "tropics, drier FT": ((336.0, 339.0), (326.0, 329.0)),
    "tropics, moister FT": ((339.0, 342.0), (332.0, 335.0)),
}
EPS = (0.0,) + tuple(style.EPS_SHOWN)
MIN_N = T.MIN_N
WINDOW = T.WINDOW


def cells(hs_range, h_range):
    """Grid-cell names of cascade_composites.py inside the given ranges."""
    out = []
    for i in range(len(CC.GRID_HS) - 1):
        for j in range(len(CC.GRID_H) - 1):
            if (hs_range[0] <= CC.GRID_HS[i] and CC.GRID_HS[i + 1] <= hs_range[1]
                    and h_range[0] <= CC.GRID_H[j] and CC.GRID_H[j + 1] <= h_range[1]):
                out.append(f"g{i:02d}_{j:02d}")
    if not out:
        raise ValueError(f"no grid cells inside h* {hs_range}, h {h_range}")
    return out


def pooled(arrays, names):
    """Composite per RH bin over several groups and event types, in the layout of cascade_tests.group."""
    keys = [k for k in arrays[0] if k == "n" or k.startswith(("s_", "w_", "p_", "pw_"))]
    tot = {}
    for a in arrays:
        idx = [list(a["names"]).index(n) for n in names]
        for k in keys:
            tot[k] = tot.get(k, 0) + a[k][idx].sum(0)
    a0 = arrays[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = {"n": tot["n"], "levels": a0["levels"], "rh_centers": 0.5 * (a0["rh_edges"][1:] + a0["rh_edges"][:-1])}
        for v in CC.SCALARS:
            c[v] = tot[f"s_{v}"] / tot[f"w_{v}"]
        for v in CC.PROFILES:
            c[v] = tot[f"p_{v}"] / tot[f"pw_{v}"]
    c["delta_a"] = c["h_LPL"] - c["h_star500"]
    return c


def plume_tops(c, eps=EPS, min_n=MIN_N):
    """{ε: top pressure per RH bin}; NaN where diluted out, 100 where still buoyant at the column top."""
    lev = np.asarray(c["levels"], float)
    out = {e: np.full(len(c["n"]), np.nan) for e in eps}
    for i in range(len(c["n"])):
        if c["n"][i] < min_n or not np.isfinite(c["h_LPL"][i]):
            continue
        ok = np.isfinite(c["MSE"][i]) & np.isfinite(c["MSE_sat"][i]) & np.isfinite(c["Z"][i])
        lv = lev[ok]
        o = np.argsort(lv)                                    # ascending pressure, TOA → surface
        p0 = min(c["p_LPL"][i], lv.max())
        cols = ztop.Columns(lv[o], c["MSE"][i][ok][o][None], c["MSE_sat"][i][ok][o][None],
                            c["Z"][i][ok][o][None], np.array([c["h_LPL"][i]]), np.array([p0]))
        for e in eps:
            _, _, p_high, _, buoyant, at_top = cols.top(e)
            out[e][i] = ztop.P_TOP if at_top[0] else (p_high[0] if buoyant[0] else np.nan)
    return out


def heights(c, tops):
    """Top heights (km above the bin's mean surface) for each ε, from the composite geopotential."""
    lev = np.asarray(c["levels"], float)
    out = {}
    for e, p in tops.items():
        z = np.full(len(p), np.nan)
        for i in range(len(p)):
            if np.isfinite(p[i]) and p[i] > ztop.P_TOP:
                z[i] = T.interp_p(lev, c["Z"][i], p[i]) / T.G_KM - c["zs_km"][i]
        out[e] = z
    return out


def slope(x, y, n, window=WINDOW, min_n=MIN_N):
    """Event-weighted slope dy/dx over the RH window (NaN with fewer than 3 usable bins)."""
    ok = np.isfinite(y) & (n >= min_n) & (x >= window[0]) & (x <= window[1])
    if ok.sum() < 3:
        return np.nan
    return np.polyfit(x[ok], y[ok], 1, w=np.sqrt(n[ok]))[0]


def report(name, c, tops, L):
    x = c["rh_centers"]
    ok = c["n"] >= MIN_N
    zt = heights(c, tops)
    z_lcl = c["zLCL_LPL"] / 1000.0
    L += [f"### {name}", "", "| RH | n | δ_a | p_LCL | z_LCL km | " + " | ".join(f"top ε={e:g}" for e in EPS)
          + " | " + " | ".join(f"z_top ε={e:g} km" for e in EPS[1:]) + " |",
          "|---" * (5 + len(EPS) + len(EPS) - 1) + "|"]
    fz = lambda z: "–" if not np.isfinite(z) else f"{z:.2f}"
    fmt = lambda p: "diluted" if not np.isfinite(p) else (">100" if p <= ztop.P_TOP else f"{p:.0f}")
    for i in np.nonzero(ok)[0]:
        L.append(f"| {x[i]:.3f} | {int(c['n'][i]):,} | {c['delta_a'][i]:+.1f} | {c['p_LCL_LPL'][i]:.0f} | "
                 f"{z_lcl[i]:.2f} | " + " | ".join(fmt(tops[e][i]) for e in EPS) + " | "
                 + " | ".join(fz(zt[e][i]) for e in EPS[1:]) + " |")
    sl = {e: slope(x, np.where(tops[e] > ztop.P_TOP, tops[e], np.nan), c["n"]) for e in EPS[1:]}
    L += ["", f"Over RH {WINDOW[0]:.2f}–{WINDOW[1]:.2f}: d(top)/d(RH) = "
          + ", ".join(f"{sl[e]:+.0f} hPa per unit RH at ε = {e:g}" for e in sl)
          + " (negative = top rises as RH rises).",
          "In height: d(z_top)/d(RH) = " + ", ".join(f"{slope(x, zt[e], c['n']):+.2f} km" for e in EPS[1:])
          + f" per unit RH; d(z_LCL)/d(RH) = {slope(x, z_lcl, c['n']):+.2f} km; entrainment depth z_top − z_LCL: "
          + ", ".join(f"{slope(x, zt[e] - z_lcl, c['n']):+.2f} km" for e in EPS[1:]) + " per unit RH (ε = "
          + ", ".join(f"{e:g}" for e in EPS[1:]) + ").", ""]
    return sl


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style.apply()
    fig, grid = plt.subplots(2, len(results), figsize=(3.3 * len(results), 6.8), sharey="row", squeeze=False)
    axes = grid[0]
    for k, (ax, (name, (c, tops))) in enumerate(zip(axes, results.items())):
        style.paper_axes(ax)
        ax.axhspan(600, 400, color=style.WINDOW_COLOR, alpha=0.08, lw=0)
        ok = c["n"] >= MIN_N
        x = c["rh_centers"]
        for e in EPS[1:]:
            y = np.where(ok & (tops[e] > ztop.P_TOP), tops[e], np.nan)
            ax.plot(x, y, "-o", color=style.EPS_COLOR[e], ms=3.5, lw=1.8, label=f"ε = {e:g}")
        ax.plot(x, np.where(ok, c["p_LCL_LPL"], np.nan), color="#6B7280", lw=1.2, ls=(0, (1, 2)), label="LCL")
        ax.axvspan(*WINDOW, color="0.85", alpha=0.35, lw=0, zorder=0)
        ax.set(xlim=(0, 1), ylim=(1000, 150), xlabel="surface RH", title=name)
        style.panel_letter(ax, k)
    axes[0].set_ylabel("plume top of the composite (hPa)")
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)
    z_max = 0.0                                                      # one height scale that fits every panel
    for c, tops in results.values():
        ok = c["n"] >= MIN_N
        for z in heights(c, tops).values():
            z_max = max(z_max, np.nanmax(np.where(ok, z, np.nan), initial=0.0))
        z_max = max(z_max, np.nanmax(np.where(ok, c["zLCL_LPL"] / 1000.0, np.nan), initial=0.0))
    for k, (ax, (name, (c, tops))) in enumerate(zip(grid[1], results.items())):  # the same in height
        style.paper_axes(ax)
        ok = c["n"] >= MIN_N
        x = c["rh_centers"]
        zt = heights(c, tops)
        z_lcl = np.where(ok, c["zLCL_LPL"] / 1000.0, np.nan)
        for e in EPS[1:]:
            ax.plot(x, np.where(ok, zt[e], np.nan), "-o", color=style.EPS_COLOR[e], ms=3.5, lw=1.8)
            ax.fill_between(x, z_lcl, np.where(ok, zt[e], np.nan), color=style.EPS_COLOR[e], alpha=0.10, lw=0)
        ax.plot(x, z_lcl, color="#6B7280", lw=1.2, ls=(0, (1, 2)))
        ax.axvspan(*WINDOW, color="0.85", alpha=0.35, lw=0, zorder=0)
        ax.set(xlim=(0, 1), ylim=(0, 1.08 * z_max), xlabel="surface RH")
        style.panel_letter(ax, k + len(results))
    grid[1][0].set_ylabel("height above ground (km)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", default="1980-2019")
    ap.add_argument("--casc", help="folder of casc_*.npz (default <work>/side_tests/casc)")
    args = ap.parse_args(argv)
    years = tuple(int(y) for y in args.years.split("-"))
    if args.casc:
        folder = Path(args.casc)
    else:
        import figdata as FD
        folder = Path(FD.load("paths")["work"]) / "side_tests" / "casc"
    A = T.load_all(folder, years=years)
    L = [f"# Plume top vs surface RH, free troposphere held fixed · {args.years}", "",
         "T and wet-bulb extremes pooled; plume through the mean profile of each 5 % RH bin, from the LPL "
         "with the bin's mean h_LPL. 'diluted' = never buoyant at that ε.", ""]
    results = {}
    for name, (hs, h) in REGIMES.items():
        names = cells(hs, h)
        c = pooled([A["T"], A["M"]], names)
        tops = plume_tops(c)
        L.insert(len(L), f"<!-- {name}: cells {', '.join(names)} -->")
        report(f"{name} (h*500 {hs[0]:g}–{hs[1]:g}, h500 {h[0]:g}–{h[1]:g})", c, tops, L)
        results[name] = (c, tops)
    out = folder.parent
    figure(results, out / "plume_top_vs_rh.png")
    text = "\n".join(L) + "\n"
    (out / "plume_top_vs_rh.md").write_text(text)
    print(text)
    print(f"Wrote {out / 'plume_top_vs_rh.md'} and .png")


if __name__ == "__main__":
    main()
