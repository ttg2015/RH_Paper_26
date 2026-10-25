"""Global map axes for the paper figures: Robinson projection with coastlines, light land and
ocean shading and the analysis latitude band drawn as dashed lines (old Paper_Plots-2 cell 50).
Falls back to plain lat-lon axes if cartopy is missing, so figures still build anywhere."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:                     # pragma: no cover
    HAVE_CARTOPY = False


def grid(nrows, ncols, figsize):
    if HAVE_CARTOPY:
        return plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                            subplot_kw={"projection": ccrs.Robinson()},
                            gridspec_kw=dict(hspace=0.15, wspace=0.05))
    return plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True, sharey=True)


def draw(ax, lon, lat, values, band=None, labels=True, **kw):
    """pcolormesh of a lat-lon field plus coastlines, land/ocean shading, gridlines and band lines."""
    if not HAVE_CARTOPY:
        im = ax.pcolormesh(lon, lat, values, shading="auto", **kw)
        for b in band or ():
            ax.axhline(b, color="k", ls="--", lw=1)
        return im
    data_crs = ccrs.PlateCarree()
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.2)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", alpha=0.3)
    im = ax.pcolormesh(lon, lat, values, transform=data_crs, shading="auto", **kw)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="k")
    for b in band or ():
        ax.plot(np.linspace(-180, 180, 361), np.full(361, b), transform=data_crs,
                color="k", ls="--", lw=1.2)
    gl = ax.gridlines(crs=data_crs, draw_labels=labels, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 30))
    gl.xlabel_style = gl.ylabel_style = {"size": 8}
    return im
