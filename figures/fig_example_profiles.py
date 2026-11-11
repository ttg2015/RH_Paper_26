"""Example Profiles (paper Fig. 1): one point's 1986 temperature and wet-bulb extremes on ERA5 model levels.

The look of the original hand-made figure (Model_Level_MSE_Profile_Comparison.ipynb, cell [21]): two panels,
T extreme left and wet-bulb (MSE) extreme right, h solid and h* dashed with inline labels, a dashed vertical
line at h*500, a dotted one at h_ML, the 2-m value as a dot at the surface, the mixed and surface layers
shaded. What changed from the notebook:
  * event hours come from the pipeline's annual maxima (AnnMax_{T2m,MSE2m}_{YEAR}.nc), not hard-coded;
  * h_2m is the ERA5 2-m field the paper uses (stage c2 MSE2m), not the lowest model level (E05);
  * h_ML is model-level h interpolated (in ln p) to the lowest pressure level above the surface (LPL),
    the paper's definition; h*500 is interpolated to 500 hPa, not the nearest model level;
  * geopotential is integrated on half levels with virtual temperature (IFS scheme), h and h* use the
    pipeline's constants (tpl_mll.thermo);
  * the layers are diagnosed, not hand-set: mixed-layer top = the lower of the LCL of LPL air (pipeline
    p_LCL_LPL) and the θv parcel top (first level with θv > θv(lowest level) + THV_EXCESS); the surface layer
    is the lowest SL_FRACTION of that depth (the usual 10 % convention);
  * the excesses δ_S and δ_a are marked, so the text can refer to them (LABEL_EXCESS);
  * the exact lat/lon go in the title.

Model levels: RDA d633006 e5.oper.an.ml, 6-hour files per variable (T, Q, SP) on the N320 Gaussian grid,
137 levels with hybrid coefficients a/b; surface geopotential from the d633006 invariant file. The point is
the model-grid point nearest the target; the pipeline values come from the nearest 0.25° point.

Each run on Casper writes <out>/example_profiles_cache.npz, which fig_wbt_increase.py reads for its dots
and which --cache redraws anywhere.
    python figures/fig_example_profiles.py [--year 1986] [--lat 38.75 --lon 116.25] [--cache F] [--out DIR]
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

REPO = Path(__file__).resolve().parents[1]
TARGET = (38.75, 116.25)            # the original figure's point (on the 0.25° grid)
YEAR = 1986
CACHE_NAME = "example_profiles_cache.npz"
RDA = "/glade/campaign/collections/rda/data"
ML_DIR = RDA + "/d633006/e5.oper.an.ml"
ML_INVARIANT = RDA + "/d633006/e5.oper.invariant/e5.oper.invariant.128_129_z.regn320sc.2016010100_2016010100.nc"
RD = 287.06                         # J/kg/K (IFS value, for the geopotential only)
THV_EXCESS = 0.5                    # K, θv parcel method for the mixed-layer top
SL_FRACTION = 0.10                  # surface layer = lowest 10 % of the mixed layer
LABEL_EXCESS = True
EVENT_TITLE = {"T": "Temperature-extreme event", "M": "Wet-bulb-extreme event"}
ML_SHADE, SL_SHADE = "#DCEAF6", "#C9CED3"
H500_COLOR, HML_COLOR, H2M_COLOR = "#7F1D1D", "#4A7FB0", "#2E8B57"


# --- model levels ------------------------------------------------------------------------------------
def ml_paths(t, ml_dir=ML_DIR):
    """The 6-hour model-level files holding hour t, and t's index in them."""
    t = np.datetime64(t, "h").astype(object)
    start = (t.hour // 6) * 6
    stamp = f"{t:%Y%m%d}{start:02d}_{t:%Y%m%d}{start + 5:02d}"
    d = f"{ml_dir}/{t:%Y%m}"
    return ({"T": f"{d}/e5.oper.an.ml.0_5_0_0_0_t.regn320sc.{stamp}.nc",
             "Q": f"{d}/e5.oper.an.ml.0_5_0_1_0_q.regn320sc.{stamp}.nc",
             "SP": f"{d}/e5.oper.an.ml.128_134_sp.regn320sc.{stamp}.nc"}, t.hour - start)


def invariant_path(path=ML_INVARIANT):
    if Path(path).exists():
        return path
    hits = sorted(glob.glob(str(Path(path).parent) + "/**/*128_129_z.regn320sc*.nc", recursive=True))
    if not hits:
        raise FileNotFoundError(f"no model-grid surface geopotential near {path}")
    return hits[0]


def geopotential(T, q, p_half, phi_s):
    """Full-level geopotential [m²/s²] by the IFS hydrostatic scheme; arrays ordered top → bottom.
    T, q: (L,); p_half: (L+1,) Pa with p_half[0] = top; phi_s: surface geopotential."""
    tv = T * (1.0 + 0.609133 * q)
    L = T.size
    phi_full, phi_h = np.empty(L), phi_s
    for k in range(L - 1, -1, -1):
        if p_half[k] <= 0.0:                                  # model top: p = 0 above the first level
            alpha, dlnp = np.log(2.0), 0.0
        else:
            dlnp = np.log(p_half[k + 1] / p_half[k])
            alpha = 1.0 - p_half[k] / (p_half[k + 1] - p_half[k]) * dlnp
        phi_full[k] = phi_h + alpha * RD * tv[k]
        phi_h = phi_h + RD * tv[k] * dlnp
    return phi_full


def ml_column(t, lat, lon, ml_dir=ML_DIR, invariant=None):
    """One model-level sounding: p [hPa], T, q, Φ, h, h* (top → bottom), p_sfc, and the grid point used."""
    import xarray as xr
    from tpl_mll import thermo
    paths, it = ml_paths(t, ml_dir)
    sel = dict(latitude=lat, longitude=lon % 360.0, method="nearest")
    ds_t = xr.open_dataset(paths["T"]).sel(**sel).isel(time=it)
    q = xr.open_dataset(paths["Q"]).sel(**sel).isel(time=it)["Q"].values.astype(float)
    sp = float(xr.open_dataset(paths["SP"]).sel(**sel).isel(time=it)["SP"])
    zs = xr.open_dataset(invariant or invariant_path()).sel(**sel)["Z"]
    phi_s = float(zs.isel(time=0) if "time" in zs.dims else zs)
    T = ds_t["T"].values.astype(float)
    a, b = ds_t["a_model"].values, ds_t["b_model"].values
    ah, bh = ds_t["a_half"].values, ds_t["b_half"].values
    p_full, p_half = a + b * sp, ah + bh * sp
    if p_full[0] > p_full[-1]:                                # make sure it runs top → bottom
        T, q, p_full, p_half = T[::-1], q[::-1], p_full[::-1], p_half[::-1]
    phi = geopotential(T, q, p_half, phi_s)
    return {"p_ml": p_full / 100.0, "T_ml": T, "q_ml": q, "phi_ml": phi,
            "h_ml_prof": thermo.mse(T, q, phi / 1000.0), "hs_ml_prof": thermo.mse_sat(T, p_full, phi / 1000.0),
            "p_sfc_ml": sp / 100.0, "phi_s": phi_s, "lat_ml": float(ds_t["latitude"]),
            "lon_ml": float(ds_t["longitude"])}


# --- pipeline values at the 0.25° point --------------------------------------------------------------
def event_time(year, event, target):
    """Hour of the annual maximum at the 0.25° point nearest target (stage b AnnMax files)."""
    import xarray as xr
    paths = FD.load("paths")
    key, var = ("annmax_T2m", "T2m") if event == "T" else ("annmax_MSE2m", "MSE2m")
    f = Path(paths["work"]) / paths["outputs"][key] / f"AnnMax_{var}_{year}.nc"
    ds = xr.open_dataset(f).sel(latitude=target[0], longitude=target[1] % 360.0, method="nearest")
    return np.datetime64(ds["event_time"].values, "h")


def pipeline_point(year, event, target):
    """h_2m, h_LPL, the LPL, p_sfc, T2m, q2m, RH, p_LCL of LPL air at the 0.25° point (stage c2/d, lead 0)."""
    import fig_plume_case as P
    prof, der = P._open(year, event)
    prof = prof.sel(latitude=target[0], longitude=target[1] % 360.0, method="nearest")
    der = der.sel(latitude=target[0], longitude=target[1] % 360.0, method="nearest")
    v = lambda ds, k: float(ds[k]) if k in ds else np.nan
    return {"h_2m": v(prof, "MSE2m"), "h_LPL_pl": v(der, "MSE_LPL"), "p_lpl": v(der, "LPL_pressure"),
            "p_sfc": v(prof, "SP") / 100.0, "T2m": v(prof, "T2m"), "q2m": v(prof, "q2m"), "rh": v(der, "RH_2m"),
            "p_lcl": v(der, "p_LCL_LPL"), "lat": float(prof["latitude"]), "lon": float(prof["longitude"])}


def column(year, event, target, ml_dir=ML_DIR):
    t = event_time(year, event, target)
    col = pipeline_point(year, event, target)
    col.update(ml_column(t, *target, ml_dir=ml_dir))
    col["time"] = str(t)
    return finish(col)


def interp_p(p, y, p0):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    o = np.argsort(p[ok])
    return float(np.interp(np.log(p0), np.log(p[ok][o]), y[ok][o]))


def finish(col):
    """Derived numbers: h_ML at the LPL and h*500 from the model levels; the layer tops."""
    p, h, hs = col["p_ml"], col["h_ml_prof"], col["hs_ml_prof"]
    col["h_ml"] = interp_p(p, h, col["p_lpl"])
    col["h500"] = interp_p(p, hs, 500.0)
    col["levels"], col["h"], col["h_sat"] = p, h, hs                     # names fig_wbt_increase.py reads
    thv = col["T_ml"] * (1.0 + 0.608 * col["q_ml"]) * (1000.0 / p) ** 0.2857
    above = np.nonzero(thv[::-1] > thv[-1] + THV_EXCESS)[0]             # from the bottom up
    p_parcel = p[::-1][above[0]] if above.size else p.min()
    p_top = max(p_parcel, col["p_lcl"]) if np.isfinite(col["p_lcl"]) else p_parcel
    col["p_ml_top"] = float(p_top)
    col["p_sl_top"] = float(col["p_sfc"] - SL_FRACTION * (col["p_sfc"] - p_top))
    return col


def excesses(col):
    hs = col["h500"] if "h500" in col else interp_p(col["levels"], col["h_sat"], 500.0)
    return {"h*500": hs, "delta_a": col["h_ml"] - hs, "delta_S": col["h_2m"] - col["h_ml"], "delta": col["h_2m"] - hs}


def h_star_500(col):
    return excesses(col)["h*500"]


# --- cache -------------------------------------------------------------------------------------------
def dump(path, cols, year):
    out = {"year": np.array(year)}
    for e, c in cols.items():
        for k, v in c.items():
            out[f"{e}_{k}"] = np.asarray(v)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


def from_cache(path):
    z = np.load(path)
    cols = {e: {k[2:]: (z[k] if z[k].ndim else z[k].item()) for k in z.files if k.startswith(e + "_")}
            for e in ("T", "M")}
    return cols, int(z["year"])


def load(year=YEAR, target=TARGET, cache=None, dump_to=None):
    if cache:
        return from_cache(cache)
    cols = {e: column(year, e, target) for e in ("T", "M")}
    if dump_to:
        dump(dump_to, cols, year)
    return cols, year


# --- figure ------------------------------------------------------------------------------------------
def harrow(ax, x0, x1, y, text, color="k"):
    if abs(x1 - x0) >= 0.05:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), zorder=7,
                    arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.0, shrinkA=0, shrinkB=0,
                                    mutation_scale=7))
    ax.text(0.5 * (x0 + x1), y - 4, text, ha="center", va="bottom", fontsize=9, color=color, zorder=7)


def panel(ax, col, event, k):
    p, h, hs = col["p_ml"], col["h_ml_prof"], col["hs_ml_prof"]
    ex = excesses(col)
    c = style.EVENT_COLOR[event]
    xlo = np.nanmin(h[p >= 300.0]) - 1.5
    xhi = max(np.nanmax(h[p >= 850.0]), col["h_2m"], ex["h*500"], col["h_ml"]) + 3.0
    p_bot = max(col["p_sfc"], col["p_sfc_ml"])
    # layers
    ax.axhspan(col["p_ml_top"], p_bot, color=ML_SHADE, lw=0, zorder=0)
    ax.axhspan(col["p_sl_top"], p_bot, color=SL_SHADE, lw=0, zorder=0.5)
    if col["p_sl_top"] - col["p_ml_top"] >= 30.0:              # label a layer only where it has room
        ax.text(xlo + 0.03 * (xhi - xlo), 0.5 * (col["p_ml_top"] + col["p_sl_top"]), "Mixed layer",
                color="#4A90C8", fontsize=8.5, fontweight="bold", va="center", zorder=2)
    if p_bot - col["p_sl_top"] >= 12.0:
        ax.text(xlo + 0.03 * (xhi - xlo), 0.5 * (col["p_sl_top"] + p_bot), "Surface layer", color="#6B7280",
                fontsize=8, fontweight="bold", va="center", zorder=2)
    # profiles
    ax.plot(h, p, color=c, lw=2.2, zorder=4)
    above_lcl = p <= (col["p_lcl"] if np.isfinite(col["p_lcl"]) else p_bot)
    ax.plot(np.where(above_lcl, hs, np.nan), p, color=c, lw=2.0, ls="--", zorder=4)
    # reference values
    ax.axvline(ex["h*500"], color=H500_COLOR, lw=1.5, ls="--", zorder=3)
    ax.plot(ex["h*500"], 500.0, "o", color=H500_COLOR, ms=4.5, zorder=6)
    ax.text(ex["h*500"] - 0.4, 492, r"$h^*_{500}$", color=H500_COLOR, ha="right", va="bottom", fontsize=10, zorder=6)
    ax.axvline(col["h_ml"], color=HML_COLOR, lw=1.3, ls=":", zorder=3)
    ax.text(col["h_ml"] + 0.4, col["p_ml_top"] - 12, r"$h_{ML}$", color=HML_COLOR, ha="left", va="bottom",
            fontsize=10, zorder=6)
    ax.plot(col["h_2m"], col["p_sfc"], "o", color=H2M_COLOR, mec="white", mew=0.8, ms=7, zorder=8, clip_on=False)
    ax.text(col["h_2m"] + 0.3, col["p_sfc"] - 8, "2 m", color=H2M_COLOR, fontsize=9.5, fontweight="bold",
            ha="left", va="bottom", zorder=8)
    # inline curve labels, as in the original
    i_h = int(np.nanargmin(np.abs(p - 600.0)))
    ax.text(h[i_h] - 0.6, 600.0, "$h$", color=c, fontsize=12, ha="right", va="center", zorder=6)
    j = np.nonzero(above_lcl & (p >= 400.0) & (p <= 700.0) & np.isfinite(hs) & (hs < xhi - 1))[0]
    if j.size:
        jj = j[np.argmax(p[j])]
        ax.text(hs[jj] + 0.6, p[jj], "$h^*$", color=c, fontsize=12, ha="left", va="center", zorder=6)
    # the excesses
    if LABEL_EXCESS:
        harrow(ax, col["h_ml"], col["h_2m"], col["p_sl_top"] - 6, r"$\delta_S$")
        harrow(ax, ex["h*500"], col["h_ml"], 0.5 * (col["p_ml_top"] + col["p_sl_top"]) - 4, r"$\delta_a$")
    ax.set(xlim=(xlo, xhi), ylim=(p_bot, 100), xlabel="MSE " + style.KJ)
    ax.set_yticks(np.arange(100, 1001, 100))
    when = np.datetime64(col["time"], "h").astype(object) if col.get("time") else None
    ax.set_title(f"{EVENT_TITLE[event]}" + (f"\n{when:%Y-%m-%d %H}Z" if when else ""), fontsize=10.5,
                 fontweight="bold")
    style.panel_letter(ax, k)
    return ex


def draw(cols, year, out_dir):
    style.apply()
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 6.4), sharey=True)
    facts = {}
    for k, e in enumerate(("T", "M")):
        ex = panel(axs[k], cols[e], e, k)
        c = cols[e]
        facts[f"{e} event"] = (f"{c.get('time', '?')}; model point {c['lat_ml']:.2f}°N {c['lon_ml']:.2f}°E "
                               f"(pipeline point {c['lat']:.2f}°N {c['lon']:.2f}°E); h_2m {c['h_2m']:.1f}, "
                               f"h_ML {c['h_ml']:.1f} at {c['p_lpl']:.0f} hPa (pressure-level product "
                               f"{c['h_LPL_pl']:.1f}), h*500 {ex['h*500']:.1f}; δ {ex['delta']:+.1f}, "
                               f"δ_S {ex['delta_S']:+.1f}, δ_a {ex['delta_a']:+.1f} kJ/kg; RH {100 * c['rh']:.0f}%, "
                               f"T2m {c['T2m'] - 273.15:.1f} °C; ML top {c['p_ml_top']:.0f} hPa (LCL "
                               f"{c['p_lcl']:.0f}); lowest model level minus h_2m "
                               f"{np.asarray(c['h_ml_prof'])[-1] - c['h_2m']:+.1f} kJ/kg")
    axs[0].set_ylabel("Pressure [hPa]")
    c = cols["T"]
    fig.suptitle(f"Annual temperature and wet-bulb extremes at {c['lat']:.2f}°N, {c['lon']:.2f}°E, {year}",
                 fontsize=11.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    path = FD.save(fig, out_dir, "example_profiles")
    plt.close(fig)
    facts["note"] = ("δ = h_2m − h*500, δ_S = h_2m − h_ML, δ_a = h_ML − h*500; h_2m from the ERA5 2-m fields "
                     "(stage c2), the profile from the 137 model levels")
    return {"path": path, "title": "Example Profiles (Fig. 1, model levels)", "facts": facts,
            "script": "figures/fig_example_profiles.py"}


def make(out_dir, years=None, weighted=False, year=YEAR, target=TARGET, cache=None, dump_to=None):
    cols, year = load(year, target, cache, dump_to or Path(out_dir) / CACHE_NAME)
    return [draw(cols, year, out_dir)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--year", type=int, default=YEAR)
    ap.add_argument("--lat", type=float, default=TARGET[0])
    ap.add_argument("--lon", type=float, default=TARGET[1])
    ap.add_argument("--cache", help="redraw from a cache written by an earlier run")
    ap.add_argument("--dump", help=f"where to write the cache (default <out>/{CACHE_NAME})")
    ap.add_argument("--out", default=str(REPO / "runs" / "figs"))
    a = ap.parse_args()
    for item in make(Path(a.out), year=a.year, target=(a.lat, a.lon), cache=a.cache, dump_to=a.dump):
        print(f"Wrote {item['path']}")
        for k, v in item["facts"].items():
            print(f"  {k}: {v}")
