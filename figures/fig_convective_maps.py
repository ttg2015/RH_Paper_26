"""Where the free-tropospheric excess comes from: δ_FT, δ_below and δ_above maps (unstable events).

Port of Paper_Plots-2 cell 126 (convective_maps_T/M.png) on Robinson maps. For each land cell in
the band, the mean over years with δ_FT > 0 of: δ_FT, δ_below = h_LPL − h_plume(p_LCL) and
δ_above = δ_FT − δ_below (plume from the LPL, ε from config/figdata.yaml, default 0.2 km⁻¹).
    python figures/fig_convective_maps.py [--years 1980-2019]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figdata as FD
import maps
import style

ROWS = [("delta_FT", r"$\delta_{FT}$"), ("delta_below", r"$\delta_{below}$"), ("delta_above", r"$\delta_{above}$")]
CMAP = {"T": "Reds", "M": "Blues"}


def unstable_means(d):
    sb, nb = d.map_sum["map_delta_below"], d.map_n["map_delta_below"]
    sa = d.map_sum["map_delta_above"]
    with np.errstate(invalid="ignore", divide="ignore"):
        below, above = sb / nb, sa / nb
    return {"delta_FT": below + above, "delta_below": below, "delta_above": above}, nb


def make(out_dir, years=None, weighted=True):
    style.apply()
    data = {e: FD.FigData(e, years) for e in ("T", "M")}
    band = FD.load("analysis")["lat_band"]
    eps = FD.load("figdata")["convective"]["eps"]
    fig, axes = maps.grid(3, 2, (18, 14))
    facts = {}
    for j, e in enumerate(("T", "M")):
        m, n = unstable_means(data[e])
        for i, (q, lab) in enumerate(ROWS):
            vmax = np.nanpercentile(m[q], 98) if np.isfinite(m[q]).any() else 1
            im = maps.draw(axes[i][j], data[e].lon, data[e].lat, m[q], band=band, labels=(j == 0),
                           cmap=CMAP[e], vmin=0, vmax=vmax)
            axes[i][j].set_title(f"{lab} · {style.EVENT_LABEL[e]}", fontsize=13)
            fig.colorbar(im, ax=axes[i][j], shrink=0.8, pad=0.02, label="kJ/kg")
            facts[f"mean {q} {e} (unstable)"] = f"{np.nanmean(m[q]):.2f} kJ/kg"
    fig.suptitle(f"Convective split of δ_FT (unstable events, ε = {eps:g} km⁻¹) · {data['T'].label()}", fontsize=15)
    facts.update(eps=f"{eps:g} km⁻¹", years=data["T"].label())
    return [{"path": FD.save(fig, out_dir, "convective_maps"), "title": "Convective maps: δ_below and δ_above",
             "facts": facts, "script": "figures/fig_convective_maps.py"}]


if __name__ == "__main__":
    FD.cli(make, __doc__)
