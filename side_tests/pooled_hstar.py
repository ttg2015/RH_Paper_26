"""Does h*500 stay fixed with RH in the pooled data? Section 7 assumes it, but Task 5 only showed it inside boxes.

Splits the pooled fit exactly: δ_a = h_LPL − h*500, so K_δ = K_hLPL − K_h*500 when each is fitted as
an anomaly from its value at RH0, with the same window and form as fig_excess_vs_rh_entrainment.
Also fits C from the mean LCL, and reports T and M alone, since a pooled drift can come from the T/M mix.

Reads   Profiles/profiles_{E}_b06_{Y}.nc, Profiles/derived/derived_{E}_b06_{Y}.nc
Writes  <work>/side_tests/pooled_hstar.md

    python side_tests/pooled_hstar.py [--years 1980-2019]

Figure mask (same events as the fit), lead from figdata.yaml, unweighted.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "pipeline"), str(REPO / "figures")]
import figdata as FD                                          # noqa: E402
import g                                                      # noqa: E402

EDGES = np.round(np.arange(0.0, 1.0501, 0.05), 6)
WINDOW = (0.30, 0.60)
MIN_EVENTS = 200
C_DRAFT = 2.0                                                 # km


def one(year, event, cfg):
    """Per RH bin: n, Σh_LPL, Σh*500, Σz_LCL (LPL air, km above ground)."""
    paths = g.load("paths")
    work = Path(paths["work"])
    out = paths["outputs"]
    grid = lambda da: da.transpose("latitude", "longitude").values.astype("float64")
    with xr.open_dataset(work / out["profiles"] / f"profiles_{event}_b06_{year}.nc") as p:
        hs = p["MSE_sat"].sel(level=500, lead=cfg["lead"]).isel(year=0)
        lat, lon = p["latitude"].values, p["longitude"].values
        hs = grid(hs)
    with xr.open_dataset(work / out["derived"] / f"derived_{event}_b06_{year}.nc") as d:
        der = d[["MSE_LPL", "RH_2m", "z_LCL_LPL", "Z_LPL"]].sel(lead=cfg["lead"]).isel(year=0).load()
    z = grid(der["z_LCL_LPL"])
    if der["z_LCL_LPL"].attrs.get("reference") != "above_ground":          # legacy d (E07)
        zs_m = g.surface_height_km(lat, lon) * 1000.0
        z = g.z_lcl_above_ground_legacy(z, zs_m, grid(der["Z_LPL"]) / 9.8 * 1000.0)
    h, rh = grid(der["MSE_LPL"]), grid(der["RH_2m"])
    ok = g.figure_mask(lat, lon, cfg) & np.isfinite(h) & np.isfinite(hs) & np.isfinite(rh)
    k = np.digitize(rh[ok], EDGES) - 1
    use = (k >= 0) & (k < len(EDGES) - 1)
    nb = len(EDGES) - 1
    zok = np.nan_to_num(z[ok][use])
    return np.stack([np.bincount(k[use], minlength=nb).astype(float),
                     np.bincount(k[use], weights=h[ok][use], minlength=nb),
                     np.bincount(k[use], weights=hs[ok][use], minlength=nb),
                     np.bincount(k[use], weights=zok / 1000.0, minlength=nb),
                     np.bincount(k[use], weights=np.isfinite(z[ok][use]).astype(float), minlength=nb)], 1)


def means(acc):
    n = acc[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"h": acc[:, 1] / n, "hs": acc[:, 2] / n, "z": acc[:, 3] / acc[:, 4]}
    for v in out.values():
        v[n < MIN_EVENTS] = np.nan
    return out, n


def fit(x, y, n, rh0, form):
    """K in y − y(RH0) = K·g(RH), through the point at RH0, as fig_excess_vs_rh_entrainment.fit_k."""
    ok = np.isfinite(y) & (x >= WINDOW[0]) & (x <= WINDOW[1])
    if ok.sum() < 3 or not np.isfinite(rh0):
        return np.nan
    f = np.isfinite(y)
    y0 = np.interp(rh0, x[f], y[f])
    gg = np.log(rh0 / x[ok]) if form == "log" else 1 - x[ok] / rh0
    return (n[ok] * gg * (y[ok] - y0)).sum() / (n[ok] * gg * gg).sum()


def section(name, acc, rh0, L):
    x = FD.centers(EDGES)
    m, n = means(acc)
    L += [f"## {name}", "", "| RH | n | h_LPL | h*500 | δ_a | z_LCL [km] |", "|---|---|---|---|---|---|"]
    for i in range(len(x)):
        if np.isfinite(m["h"][i]):
            L.append(f"| {EDGES[i]:.2f}–{EDGES[i + 1]:.2f} | {int(n[i]):,} | {m['h'][i]:.2f} | {m['hs'][i]:.2f} | "
                     f"{m['h'][i] - m['hs'][i]:.2f} | {m['z'][i]:.2f} |")
    w = (x >= WINDOW[0]) & (x <= WINDOW[1]) & np.isfinite(m["hs"])
    if w.sum() < 3:
        L += ["", "Too few populated bins in the window.", ""]
        return
    i0, i1 = np.nonzero(w)[0][[0, -1]]
    L += ["", f"Across the window ({x[i0]:.3f} → {x[i1]:.3f}): h*500 changes by **{m['hs'][i1] - m['hs'][i0]:+.2f}**, "
              f"h_LPL by **{m['h'][i1] - m['h'][i0]:+.2f}** kJ/kg, z_LCL by {m['z'][i1] - m['z'][i0]:+.2f} km.", "",
          f"Fits with RH0 = {rh0:.3f} (K in kJ/kg; K_δ = K_hLPL − K_h*500):", "",
          "| form | K_δ | K_hLPL | K_h*500 | K_h*500 / K_hLPL | fitted C [km] | Γ_h = K_hLPL / C [kJ/kg/km] |",
          "|---|---|---|---|---|---|---|"]
    for form in ("log", "lin"):
        kh, ks = fit(x, m["h"], n, rh0, form), fit(x, m["hs"], n, rh0, form)
        c = fit(x, m["z"], n, rh0, form)
        L.append(f"| {form} | {kh - ks:.2f} | {kh:.2f} | {ks:.2f} | {ks / kh:+.0%} | {c:.2f} | {kh / c:.2f} |")
    L.append("")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1980-2019")
    a = ap.parse_args(argv)
    y0, y1 = (int(v) for v in a.years.split("-"))
    cfg = g.load("figdata")
    acc = {"T": 0, "M": 0}
    years = {"T": [], "M": []}
    for year, e in FD.available():
        if y0 <= year <= y1:
            acc[e] = acc[e] + one(year, e, cfg)
            years[e].append(year)
            print(year, e, flush=True)
    data = {e: FD.FigData(e, (y0, y1)) for e in ("T", "M")}
    rh0 = FD.rh0(data)
    pooled = acc["T"] + acc["M"]
    L = ["# Is h*500 fixed with RH in the pooled data?", "",
         f"T {len(years['T'])} years, M {len(years['M'])} years. Figure mask, unweighted. RH bins 0.05, bins with n ≥ {MIN_EVENTS}. "
         f"Fit window {WINDOW[0]:.2f}–{WINDOW[1]:.2f}. RH0 = {rh0:.3f} (pooled, FD.rh0) for all three sections.", "",
         f"K_δ here should be close to the fit in facts.json; the draft takes C = {C_DRAFT:g} km. "
         "If K_h*500 is small next to K_hLPL, the fixed-h*500 assumption holds for the pooled fit.", ""]
    section("Pooled T + M", pooled, rh0, L)
    section("Temperature extremes alone", acc["T"], rh0, L)
    section("Wet-bulb extremes alone", acc["M"], rh0, L)
    text = "\n".join(L) + "\n"
    out = Path(g.load("paths")["work"]) / "side_tests" / "pooled_hstar.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
