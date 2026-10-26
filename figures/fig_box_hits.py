"""Where do events in each free-tropospheric energy box happen? Years hit per land cell.

Ports of Paper_Plots-2 cells 165 (h500_hstar500_hitcount_tp_ml_view.pdf: cascade TPL box, tropics
and midlatitude views) and 166–169 (tpl_mll_hit_locations.pdf: appendix TPL/MLL boxes, global).
Each event type uses its own box (old cell 165 applied the T box to both).
    python figures/fig_box_hits.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import style

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:            # pragma: no cover
    ccrs = None

CMAP = plt.get_cmap("YlOrRd").copy()
CMAP.set_bad((1, 1, 1, 0))


def years_hit(d, name):
    v = d.hist.get(f"hit_{name}")
    return None if v is None else np.where(v > 0, v, np.nan)


def panel(ax, d, field, vmax, title, extent=None):
    if ccrs is None:
        pcm = ax.pcolormesh(d.lon, d.lat, field, cmap=CMAP, vmin=0, vmax=vmax, shading="auto")
        if extent:
            ax.set_ylim(*extent)
    else:
        if extent:
            ax.set_extent([-180, 180, *extent], crs=ccrs.PlateCarree())
        else:
            ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.4", zorder=2)
        pcm = ax.pcolormesh(d.lon, d.lat, np.ma.masked_invalid(field), transform=ccrs.PlateCarree(),
                            cmap=CMAP, vmin=0, vmax=vmax, shading="auto", zorder=3)
    ax.set_title(title, fontsize=11)
    return pcm


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    sub = FD.load("analysis")["lat_subbands"]
    nyr = len(data["T"].years)
    kw = {"subplot_kw": {"projection": ccrs.PlateCarree()}} if ccrs else {}
    items = []
    # (1) tropics vs midlatitude view of each event type's cascade TPL box
    fig, ax = plt.subplots(2, 2, figsize=(16, 9), squeeze=False, **kw)
    for i, e in enumerate(("T", "M")):
        f = years_hit(data[e], f"{e}_TPL")
        for j, (band, ext) in enumerate((("tropics", sub["TP"]), ("midlatitudes", sub["ML"]))):
            pcm = panel(ax[i][j], data[e], f, nyr, f"{style.EVENT_LABEL[e]} · TPL box · {band}", ext)
    fig.subplots_adjust(right=0.9, wspace=0.08, hspace=0.25)
    fig.colorbar(pcm, cax=fig.add_axes([0.92, 0.15, 0.015, 0.7]), label="years hit")
    fig.suptitle("h500/h*500 box hit count (cascade TPL boxes)", y=0.98)
    items.append({"path": FD.save(fig, out_dir, "box_hits_tp_ml"), "title": "Box hit count · TPL box",
                  "facts": {"years": data["T"].label(), "boxes": "cascade T_TPL / M_TPL (figdata.yaml)"},
                  "script": "figures/fig_box_hits.py"})
    plt.close(fig)
    # (2) appendix TPL / MLL boxes, global
    fig, ax = plt.subplots(2, 2, figsize=(16, 9), squeeze=False, **kw)
    for i, e in enumerate(("T", "M")):
        for j, b in enumerate(("TPL", "MLL")):
            pcm = panel(ax[i][j], data[e], years_hit(data[e], f"{e}_{b}_app"), nyr, f"{style.EVENT_LABEL[e]} · {b}")
    fig.subplots_adjust(right=0.9, wspace=0.08, hspace=0.25)
    fig.colorbar(pcm, cax=fig.add_axes([0.92, 0.15, 0.015, 0.7]), label="years hit")
    fig.suptitle("Where TPL and MLL events strike (T vs WBT extremes)", y=0.98)
    items.append({"path": FD.save(fig, out_dir, "box_hits_locations"), "title": "Box hit locations · appendix boxes",
                  "facts": {"years": data["T"].label(), "boxes": "hit_boxes in figdata.yaml"},
                  "script": "figures/fig_box_hits.py"})
    plt.close(fig)
    return items


if __name__ == "__main__":
    FD.cli(make, __doc__)
