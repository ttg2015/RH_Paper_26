"""Quick-look checks and demo figures for stage f (Ztop plausibility). Used by stage_f_checks.ipynb.

Everything reads pipeline outputs only (Ztop/, Profiles/, Profiles/derived/) and writes nothing.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tpl_mll.config import load                  # noqa: E402
from tpl_mll import ztop                          # noqa: E402
from tpl_mll.masks import analysis_mask           # noqa: E402

CAT = {1: ("pass", "#43a047"), 2: ("too shallow", "#fb8c00"), 3: ("too deep", "#5e35b1"),
       4: ("no buoyancy", "#b0bec5")}


def _dir(key):
    p = load("paths")
    return Path(p["work"]) / p["outputs"][key]


def open_ztop(year, event):
    return xr.open_dataset(_dir("ztop") / f"ztop_{event}_b06_{year}.nc")


def available():
    """[(year, event)] with a ztop file."""
    out = []
    for f in sorted(_dir("ztop").glob("ztop_*_b06_*.nc")):
        _, e, _, y = f.stem.split("_")
        out.append((int(y), e))
    return out


def fractions(datasets, weighted=False):
    """(ε, {category: fraction}) pooled over datasets; cos(lat) weights if weighted (E02)."""
    eps = datasets[0]["epsilon"].values
    tot = {c: np.zeros(len(eps)) for c in CAT}
    for ds in datasets:
        cat = ds["category"]
        w = np.cos(np.deg2rad(ds["latitude"])) if weighted else xr.ones_like(ds["latitude"])
        w = w.broadcast_like(cat.isel(epsilon=0))
        for c in CAT:
            tot[c] += ((cat == c) * w).sum(["latitude", "longitude"]).values
    n = sum(tot.values())
    return eps, {c: tot[c] / n for c in CAT}


def fig_fractions(by_event, weighted=False):
    """Figure A prototype: stacked category fractions vs ε, plus pass / buoyant."""
    fig, axes = plt.subplots(1, len(by_event), figsize=(5 * len(by_event), 3.6), squeeze=False)
    for ax, (event, dss) in zip(axes[0], by_event.items()):
        eps, fr = fractions(dss, weighted)
        ax.stackplot(eps, *[100 * fr[c] for c in CAT], labels=[CAT[c][0] for c in CAT],
                     colors=[CAT[c][1] for c in CAT], alpha=0.85)
        buoy = 1 - fr[4]
        ax.plot(eps, 100 * fr[1] / np.where(buoy > 0, buoy, np.nan), "k-o", ms=3,
                label="pass / buoyant")
        ax.set(xlabel="ε [1/km]", ylabel="% of events", ylim=(0, 100),
               title=f"{event} events · {len(dss)} year(s){' · cos(lat) weighted' if weighted else ''}")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def table(by_event):
    rows = []
    for event, dss in by_event.items():
        eps, fr = fractions(dss)
        for i, e in enumerate(eps):
            buoy = 1 - fr[4][i]
            rows.append(f"{event}  ε={e:.2f}  " + "  ".join(f"{CAT[c][0]} {100*fr[c][i]:4.0f}%" for c in CAT)
                        + f"   pass/buoyant {100*fr[1][i]/buoy if buoy else float('nan'):4.0f}%")
    return "\n".join(rows)


def fig_map(ds, eps=0.2):
    """Where each category happens at one ε."""
    from matplotlib.colors import ListedColormap
    cat = ds["category"].sel(epsilon=eps, method="nearest")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(ds["longitude"], ds["latitude"], cat, shading="nearest",
                  cmap=ListedColormap([CAT[c][1] for c in CAT]), vmin=0.5, vmax=4.5)
    for c in CAT:
        ax.plot([], [], "s", color=CAT[c][1], label=CAT[c][0])
    ax.legend(fontsize=7, ncol=4, loc="lower left")
    ax.set(xlabel="longitude", ylabel="latitude",
           title=f"Stage f category at ε = {float(cat.epsilon):.2f} · {ds.attrs['event_type']} {ds.attrs['year']}")
    return fig


def fig_windows(ds):
    """ε window per event: where the top sits at 400 hPa (lo) and at 600 hPa (hi)."""
    lo, hi = ds["eps_top_lo"].values.ravel(), ds["eps_top_hi"].values.ravel()
    ok = np.isfinite(lo) & np.isfinite(hi)
    width = (hi - lo)[ok]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    bins = np.linspace(0, 1, 51)
    axes[0].hist(lo[ok], bins, alpha=0.6, label="ε with top at 400 hPa")
    axes[0].hist(hi[ok], bins, alpha=0.6, label="ε with top at 600 hPa")
    axes[0].set(xlabel="ε [1/km]", ylabel="events", title="Per-event ε window edges")
    axes[0].legend(fontsize=7)
    axes[1].hist(width, np.linspace(0, 0.5, 51), color="#78909c")
    collapsed = np.mean(width < 0.01)
    axes[1].set(xlabel="window width [1/km]", title=f"Window width · {collapsed:.0%} collapse (< 0.01)")
    fig.tight_layout()
    inside = np.isfinite(ds["category"].isel(epsilon=0).values.ravel())
    print(f"events: {inside.sum()} · with a window: {ok.sum()} · "
          f"never reach 400 hPa even undiluted: {(inside & np.isnan(lo)).sum()} · "
          f"still above 400 hPa at ε = {ztop.EPS_MAX}: {np.isinf(lo).sum()}")
    return fig


def fig_flags(ds):
    """How often the top is layered (two buoyant layers) or entrainment raises the plume."""
    eps = ds["epsilon"].values
    inside = np.isfinite(ds["category"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for k, label in (("multi_layer", "multi-layer (two crossings)"),
                     ("non_monotonic", "entrainment raises plume")):
        frac = (ds[k] > 0).where(inside).sum(["latitude", "longitude"]) / inside.sum(["latitude", "longitude"])
        ax.plot(eps, 100 * frac, "-o", ms=3, label=label)
    ax.set(xlabel="ε [1/km]", ylabel="% of events", title="Flags")
    ax.legend(fontsize=7)
    return fig


def delta_check(year, event):
    """'No buoyancy' at small ε should ≈ the δ = h_LPL − h*500 < 0 fraction."""
    prof = xr.open_dataset(_dir("profiles") / f"profiles_{event}_b06_{year}.nc").sel(lead=0).isel(year=0)
    der = xr.open_dataset(_dir("derived") / f"derived_{event}_b06_{year}.nc").sel(lead=0).isel(year=0)
    m = analysis_mask(prof["latitude"].values, prof["longitude"].values)
    d = (der["MSE_LPL"] - prof["MSE_sat"].sel(level=500)).transpose("latitude", "longitude").values[m]
    d = d[np.isfinite(d)]
    z = open_ztop(year, event)
    nb = ((z["category"] == 4).sum(["latitude", "longitude"]) / np.isfinite(z["category"]).sum(["latitude", "longitude"])).values
    return (f"{event} {year}: {d.size} events · δ<0 {100*np.mean(d < 0):.0f}% · "
            f"no buoyancy at ε={float(z.epsilon[0]):.2f}: {100*nb[0]:.0f}%")


def fig_examples(year, event, eps_show=(0.1, 0.2, 0.3), n=2, seed=0):
    """Sounding gallery: random events from each category at ε = 0.2, with their plumes.

    Each panel: environment h (solid), h* (dashed), plumes for several ε (coloured),
    the 400–600 hPa window (shaded), the LCL of LPL air (dotted) and the plume start (dot).
    """
    z = open_ztop(year, event)
    cat = z["category"].sel(epsilon=0.2, method="nearest").values
    prof = xr.open_dataset(_dir("profiles") / f"profiles_{event}_b06_{year}.nc").sel(lead=0).isel(year=0)
    der = xr.open_dataset(_dir("derived") / f"derived_{event}_b06_{year}.nc").sel(lead=0).isel(year=0)
    rng = np.random.default_rng(seed)
    picks = []
    for c in CAT:
        ii, jj = np.nonzero(cat == c)
        for k in rng.choice(len(ii), min(n, len(ii)), replace=False) if len(ii) else []:
            picks.append((c, ii[k], jj[k]))
    fig, axes = plt.subplots(n, len(CAT), figsize=(3.2 * len(CAT), 3.6 * n), squeeze=False, sharey=True)
    levels = prof["level"].values.astype(float)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(eps_show)))
    lo, hi = load("analysis")["ztop_window_hPa"]
    for col, c in enumerate(CAT):
        mine = [p for p in picks if p[0] == c]
        for row in range(n):
            ax = axes[row][col]
            if row >= len(mine):
                ax.axis("off")
                if row == 0:
                    ax.text(0.5, 0.5, f"no {CAT[c][0]} events", ha="center", transform=ax.transAxes,
                            color=CAT[c][1])
                continue
            ax.axhspan(lo, hi, color="#e8f5e9")
            _, i, j = mine[row]
            col_sel = dict(latitude=i, longitude=j)
            h = prof["MSE"].isel(col_sel).values
            hs = prof["MSE_sat"].isel(col_sel).values
            phi = prof["Z"].isel(col_sel).values
            h0 = float(der["MSE_LPL"].isel(col_sel))
            p0 = float(der["LPL_pressure"].isel(col_sel))
            plcl = float(der["p_LCL_LPL"].isel(col_sel))
            cols = ztop.Columns(levels, h[None], hs[None], phi[None], np.array([h0]), np.array([p0]))
            below = levels <= p0
            ax.plot(h[below], levels[below], "k-", lw=1.2, label="h env")
            ax.plot(hs[below], levels[below], "k--", lw=1.2, label="h* env")
            for e, cc in zip(eps_show, colors):
                hp, _, p_top, *_ = cols.top(e)
                hp = hp[0]
                ax.plot(hp, cols.p[0], color=cc, lw=1.5, label=f"plume ε={e}")
                if np.isfinite(p_top[0]):
                    ax.plot(np.interp(np.log(p_top[0]), np.log(cols.p[0][::-1]), hp[::-1]), p_top[0], "o",
                            color=cc, ms=5)
            ax.axhline(plcl, color="#6d4c41", ls=":", lw=1)
            ax.plot(h0, p0, "ko", ms=4)
            ax.set(ylim=(1050, 100), xlim=(np.nanmin(h[below]) - 5, np.nanmax(hs[below & (levels >= 150)]) + 5))
            ax.set_title(f"{CAT[c][0]} · {float(prof.latitude[i]):.1f}°, {float(prof.longitude[j]):.1f}°",
                         fontsize=8, color=CAT[c][1])
            if col == 0:
                ax.set_ylabel("hPa")
            ax.set_xlabel("kJ/kg", fontsize=8)
    axes[0][0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"{event} {year}: example soundings by category at ε = 0.2 (green band = 400–600 hPa, "
                 "dots = plume tops, dotted = LCL)", fontsize=9)
    fig.tight_layout()
    return fig
