"""Excess Maps (paper figure): multi-year mean δ_FT and δ_S during h2m and T2m extremes.

Port of Paper_Plots-2 cell 50 (circulation_anomalies_no_lsm.png): Robinson projection, coastlines,
land/ocean shading, ocean cells shown, cells with SP ≤ 500 hPa blank, ±30 kJ/kg. New: the analysis
latitude band is drawn as dashed lines, so data outside it are visible but marked.
    python figures/fig_excess_maps.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import maps
import style

ROWS = [("delta_FT", r"$\delta_{a}$"), ("delta_S", r"$\delta_S$")]
COLS = [("T", style.EVENT_SHORT["T"]), ("M", style.EVENT_SHORT["M"])]   # temperature first, as elsewhere
LIM = 30


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    band = FD.load("analysis")["lat_band"]
    fig, axes = maps.grid(2, 2, (13, 6.6))
    facts = {}
    for i, (q, qlab) in enumerate(ROWS):
        for j, (e, elab) in enumerate(COLS):
            d = data[e]
            m = d.map_mean(q, masked=False)
            im = maps.draw(axes[i][j], d.lon, d.lat, m, band=band, labels=(j == 0),
                           cmap=style.DIVERGING_CMAP, vmin=-LIM, vmax=LIM)
            axes[i][j].set_title(f"{qlab}, {elab} extremes", fontsize=11, pad=4)
            style.panel_letter(axes[i][j], 2 * i + j, x=0.0, y=1.06)
            land = d.in_mask
            facts[f"mean {q} {e} (land in band, cells)"] = f"{np.nanmean(m[land]):.2f} kJ/kg"
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical", shrink=0.8, pad=0.02, extend="both")
    cb.set_label("excess " + style.KJ)                    # one shared scale for all four maps
    facts.update(years=data["T"].label(), cells="all with SP > 500 hPa (ocean included)",
                 band_lines=f"{band[0]}° and {band[1]}°", projection="Robinson" if maps.HAVE_CARTOPY else "lat-lon (no cartopy)")
    return [{"path": FD.save(fig, out_dir, "excess_maps"), "title": "Excess Maps", "facts": facts,
             "script": "figures/fig_excess_maps.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
