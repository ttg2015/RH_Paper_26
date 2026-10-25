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

ROWS = [("delta_FT", r"$\mathbf{\delta}_{\mathbf{FT}}$"), ("delta_S", r"$\mathbf{\delta_S}$")]
COLS = [("M", r"$\mathbf{h_{2m}}$"), ("T", r"$\mathbf{T_{2m}}$")]
LIM = 30


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    band = FD.load("analysis")["lat_band"]
    fig, axes = maps.grid(2, 2, (22, 11))
    facts = {}
    for i, (q, qlab) in enumerate(ROWS):
        for j, (e, elab) in enumerate(COLS):
            d = data[e]
            m = d.map_mean(q, masked=False)
            im = maps.draw(axes[i][j], d.lon, d.lat, m, band=band, labels=(j == 0),
                           cmap="RdBu_r", vmin=-LIM, vmax=LIM)
            axes[i][j].set_title(f"{qlab} during {elab} extremes", fontsize=20, fontweight="bold", pad=8)
            cb = fig.colorbar(im, ax=axes[i][j], orientation="vertical", shrink=0.85, pad=0.02)
            cb.set_label("kJ/kg", fontsize=14)
            land = d.in_mask
            facts[f"mean {q} {e} (land in band, cells)"] = f"{np.nanmean(m[land]):.2f} kJ/kg"
    fig.suptitle(f"Excess Energy Contributions · {data['T'].label()}", fontsize=24, fontweight="bold", y=1.0)
    facts.update(years=data["T"].label(), cells="all with SP > 500 hPa (ocean included)",
                 band_lines=f"{band[0]}° and {band[1]}°", projection="Robinson" if maps.HAVE_CARTOPY else "lat-lon (no cartopy)")
    return [{"path": FD.save(fig, out_dir, "excess_maps"), "title": "Excess Maps", "facts": facts,
             "script": "figures/fig_excess_maps.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
