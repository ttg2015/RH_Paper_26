"""Stage g: figure data. Reduce one year × event of profiles + derived into small, additive arrays.

Everything written is a count or a sum on fine bins, so years combine by summation and
figures can coarsen bins, pick sub-bands and switch area weighting without rerunning this.

  δ_FT    = h_LPL − h*500        δ_S = h_2m − h_LPL        δ_total = h_2m − h*500
  RH_2m   from stage d (q / q_sat)
  zLCL_2m LCL of 2 m air above ground [m]; legacy d files store it above sea level (E07)
          and are corrected here on read.

Reads:  Profiles/profiles_{E}_b{BUF}_{YEAR}.nc (c2), Profiles/derived/derived_{E}_b{BUF}_{YEAR}.nc (d)
Writes: FigData/figdata_{E}_b{BUF}_{YEAR}.nc
Bins:   config/figdata.yaml
Usage:  python pipeline/g.py YEAR EVENT [--buffer 6]
Design: pipeline-map/Planned/Figures & FigData.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpl_mll.config import load                                  # noqa: E402
from tpl_mll.corrections import z_lcl_above_ground_legacy        # noqa: E402
from tpl_mll.masks import _invariant, surface_height_km          # noqa: E402
from tpl_mll import ztop                                          # noqa: E402

UNITS = {"delta_FT": "kJ kg-1", "delta_S": "kJ kg-1", "delta_total": "kJ kg-1",
         "delta_below": "kJ kg-1", "delta_above": "kJ kg-1",
         "RH_2m": "1", "zLCL_2m": "m above ground", "zLCL_LPL": "m above ground",
         "h_star500": "kJ kg-1", "h_2m": "kJ kg-1", "h_LPL": "kJ kg-1", "h500": "kJ kg-1"}
FEEDS = {"h1": "Fig - Excess Histograms", "h2_rh_dFT": "Fig - Excess vs RH",
         "h2_rh_dS": "Excess vs RH (surface layer)", "h2_rh_dtotal": "Excess histograms by RH (dry vs moist)", "h2_zlcl_dFT": "Fig - LCL-Binned Decomposition",
         "map": "Fig - Excess Maps", "cascade": "Fig - RH Cascades",
         "h2_hs500_h2m": "Surface layer correction (old TempThing_overlay_right)",
         "h2_hs500_hlpl": "Surface layer correction (old TempThing_overlay_right)",
         "h2_hs500_h500": "RH conditioning demo panel (a); box hit-count figure",
         "h2_rh_zlcl2m_unstable": "Fig - LCL-Binned Decomposition (panel a)",
         "lb": "Fig - LCL-Binned Decomposition (panels b, c)"}


def edges(spec):
    start, stop, step = spec
    return np.round(np.arange(start, stop + step / 2, step), 6)


def fields(year, event, buffer, lead):
    """All quantities on the full (lat, lon) grid at the event hour."""
    paths = load("paths")
    work = Path(paths["work"])
    out = paths["outputs"]
    with xr.open_dataset(work / out["profiles"] / f"profiles_{event}_b{buffer:02d}_{year}.nc") as p:
        prof = p[["MSE2m", "MSE", "MSE_sat", "T", "Z", "SP"]].sel(lead=lead).isel(year=0).load()
    with xr.open_dataset(work / out["derived"] / f"derived_{event}_b{buffer:02d}_{year}.nc") as d:
        der = d[["MSE_LPL", "RH_2m", "z_LCL_2m", "z_LCL_LPL", "Z_LPL", "LPL_pressure", "p_LCL_LPL"]
                ].sel(lead=lead).isel(year=0).load()
    grid = lambda da: da.transpose("latitude", "longitude").values.astype("float64")
    lat, lon = prof["latitude"].values, prof["longitude"].values
    h_lpl = grid(der["MSE_LPL"])
    h_2m = grid(prof["MSE2m"])
    hs500 = grid(prof["MSE_sat"].sel(level=500))
    z = grid(der["z_LCL_2m"])
    z_lpl = grid(der["z_LCL_LPL"])
    if der["z_LCL_2m"].attrs.get("reference") != "above_ground":      # legacy d (E07)
        zs_m = surface_height_km(lat, lon) * 1000.0
        z = z_lcl_above_ground_legacy(z, zs_m)
        z_lpl = z_lcl_above_ground_legacy(z_lpl, zs_m, grid(der["Z_LPL"]) / 9.8 * 1000.0)
    f = {"delta_FT": h_lpl - hs500, "delta_S": h_2m - h_lpl, "delta_total": h_2m - hs500,
         "RH_2m": grid(der["RH_2m"]), "zLCL_2m": z,
         "h500": grid(prof["MSE"].sel(level=500)), "h_star500": hs500,
         "h_2m": h_2m, "h_LPL": h_lpl, "zLCL_LPL": z_lpl, "p_LPL": grid(der["LPL_pressure"]),
         "p_LCL_LPL": grid(der["p_LCL_LPL"]), "SP_hPa": grid(prof["SP"]) / 100.0}
    levels = prof["level"].values.astype(float)
    above = prof["level"] <= prof["SP"] / 100.0                     # drop below-ground levels
    profiles = {v: prof[v].where(above).transpose("latitude", "longitude", "level").values
                for v in ("MSE", "MSE_sat", "T", "Z")}
    # raw (unmasked) columns for the plume: it starts at the LPL, so it never uses levels below it
    profiles["raw"] = {v: prof[v].transpose("latitude", "longitude", "level").values.astype("float64")
                       for v in ("MSE", "MSE_sat", "Z")}
    return f, lat, lon, levels, profiles


def convective(f, levels, raw, mask, eps):
    """Old Paper_Plots-2 cell 124 split, for unstable events: plume from the LPL with h_LPL,
    δ_below = h_LPL − h_plume(p_LCL of LPL air), δ_above = δ_FT − δ_below. NaN elsewhere."""
    dB = np.full(mask.shape, np.nan)
    act = mask & (f["delta_FT"] > 0) & np.isfinite(f["p_LCL_LPL"]) & np.isfinite(f["h_LPL"])
    i, j = np.nonzero(act)
    for s in range(0, len(i), 40000):
        ii, jj = i[s:s + 40000], j[s:s + 40000]
        p0 = np.minimum(f["p_LPL"][ii, jj], levels.max())
        cols = ztop.Columns(levels, raw["MSE"][ii, jj], raw["MSE_sat"][ii, jj], raw["Z"][ii, jj],
                            f["h_LPL"][ii, jj], p0)
        hp = ztop.plume(cols.h, cols.z, cols.h0, eps)
        plcl = f["p_LCL_LPL"][ii, jj]
        lnp = np.log(cols.p)                                        # descending pressure → ascending index
        k = np.clip((lnp > np.log(plcl)[:, None]).sum(1) - 1, 0, lnp.shape[1] - 2)
        r = np.arange(len(k))
        w = (np.log(plcl) - lnp[r, k]) / (lnp[r, k + 1] - lnp[r, k])
        hp_lcl = hp[r, k] * (1 - w) + hp[r, k + 1] * w
        ok = (plcl <= p0) & (plcl >= cols.p[:, -1])
        dB[ii, jj] = np.where(ok, f["h_LPL"][ii, jj] - hp_lcl, np.nan)
    return dB, f["delta_FT"] - dB


def figure_mask(lat, lon, cfg):
    """Latitude band + land; terrain cut only if figdata.yaml sets zs_max_km."""
    lo, hi = load("analysis")["lat_band"]
    band = (lat >= lo) & (lat <= hi)
    m = band[:, None] & (_invariant("invariant_lsm", lat, lon) > cfg["mask"]["lsm_min"])
    if cfg["mask"].get("zs_max_km") is not None:
        m &= surface_height_km(lat, lon) < cfg["mask"]["zs_max_km"]
    return m


def cascade(f, levels, profiles, mask, weight, cfg):
    """Sums of each profile over events in (box, RH bin), plus weights per level and event counts."""
    c = cfg["cascade"]
    e = edges(c["rh_bins"])
    nb = len(e) - 1
    boxes = c["boxes"]
    sums = {v: np.zeros((len(boxes), nb, len(levels))) for v in c["profile_vars"]}
    wsum = np.zeros((len(boxes), nb, len(levels)))
    count = np.zeros((len(boxes), nb))
    scal = {v: np.zeros((len(boxes), nb)) for v in ("h_LPL", "p_LPL")}
    swt = np.zeros((len(boxes), nb))
    rh_idx = np.digitize(f["RH_2m"], e) - 1
    for b, (name, box) in enumerate(boxes.items()):
        inbox = (mask & (f["h_star500"] > box["h_star"][0]) & (f["h_star500"] < box["h_star"][1])
                 & (f["h500"] > box["h"][0]) & (f["h500"] < box["h"][1]) & (rh_idx >= 0) & (rh_idx < nb))
        i, j = np.nonzero(inbox)
        k = rh_idx[i, j]
        np.add.at(count[b], k, 1)
        w = weight[i, j][:, None]
        for v in scal:
            np.add.at(scal[v][b], k, np.nan_to_num(f[v][i, j]) * weight[i, j])
        np.add.at(swt[b], k, weight[i, j])
        for v in c["profile_vars"]:
            x = profiles[v][i, j]                                     # (n, level), NaN below ground
            np.add.at(sums[v][b], k, np.nan_to_num(x) * w)
            if v == c["profile_vars"][0]:
                np.add.at(wsum[b], k, np.isfinite(x) * w)
    return e, sums, wsum, count, scal, swt


def strike(year, event, buffer, mask, weight, sub_sel, f, cfg):
    """Plume-top distributions from stage f's Ztop file (None if it doesn't exist yet)."""
    paths = load("paths")
    zpath = Path(paths["work"]) / paths["outputs"]["ztop"] / f"ztop_{event}_b{buffer:02d}_{year}.nc"
    if not zpath.exists():
        return None
    with xr.open_dataset(zpath) as z:
        z = z[["p_top", "p_top_first", "category"]].load()
    eps = z["epsilon"].values
    pe = edges(cfg["strike"]["p_bins"])
    out = xr.Dataset(coords={"epsilon": eps})
    out["edges_ptop"] = ("edge_ptop", pe)
    ptop = z["p_top"].transpose("epsilon", "latitude", "longitude").values
    pfirst = z["p_top_first"].transpose("epsilon", "latitude", "longitude").values
    cat = z["category"].transpose("epsilon", "latitude", "longitude").values
    names = list(sub_sel)
    for var, arr in (("strike_ptop", ptop), ("strike_ptop_first", pfirst)):
        for wname, suf in ((False, ""), (True, "_w")):
            h = np.zeros((len(names), len(eps), len(pe) - 1))
            for s, name in enumerate(names):
                for k in range(len(eps)):
                    m = sub_sel[name] & np.isfinite(arr[k])
                    h[s, k] = np.histogram(arr[k][m], pe, weights=weight[m] if wname else None)[0]
            out[var + suf] = (("subband", "epsilon", "bin_ptop"), h)
            out[var + suf].attrs.update(edges="edges_ptop", units="count" if not wname else "cos(lat)-weighted count",
                                        feeds="Fig - Plume strike points")
    ncat = np.zeros((len(names), len(eps), 5))
    for s, name in enumerate(names):
        for k in range(len(eps)):
            m = sub_sel[name] & np.isfinite(cat[k])
            ncat[s, k] = np.bincount(cat[k][m].astype(int), minlength=5)[:5]
    out["strike_category"] = (("subband", "epsilon", "category_code"), ncat)
    # RH-resolved versions (all of the band), for splitting dry vs moist at plot time
    re_ = edges(cfg["strike"]["rh_bins"])
    out["edges_rh_strike"] = ("edge_rh_strike", re_)
    rh = f["RH_2m"]
    base = sub_sel["all"] & np.isfinite(rh)
    for wname, suf in ((False, ""), (True, "_w")):
        hp = np.zeros((len(eps), len(re_) - 1, len(pe) - 1))
        hc = np.zeros((len(eps), len(re_) - 1, 5))
        for k in range(len(eps)):
            m = base & np.isfinite(ptop[k])
            hp[k] = np.histogram2d(rh[m], ptop[k][m], [re_, pe], weights=weight[m] if wname else None)[0]
            m = base & np.isfinite(cat[k])
            idx = np.clip(np.digitize(rh[m], re_) - 1, 0, len(re_) - 2)
            np.add.at(hc[k], (idx, cat[k][m].astype(int)), weight[m] if wname else 1.0)
        out["strike_ptop_rh" + suf] = (("epsilon", "bin_rh_strike", "bin_ptop"), hp)
        out["strike_category_rh" + suf] = (("epsilon", "bin_rh_strike", "category_code"), hc)
    out["strike_ptop_rh"].attrs.update(x_edges="edges_rh_strike", y_edges="edges_ptop")
    out["strike_category"].attrs.update(note="counts per category code: 1 pass, 2 too shallow, 3 too deep, 4 no buoyancy")
    k = int(np.argmin(abs(eps - cfg["strike"]["eps_detail"])))
    m = sub_sel["all"] & np.isfinite(ptop[k]) & np.isfinite(f["RH_2m"])
    out["strike_rh_ptop"] = (("bin_rh", "bin_ptop"),
                             np.histogram2d(f["RH_2m"][m], ptop[k][m], [edges(cfg["bins"]["rh"]), pe])[0])
    out["strike_rh_ptop"].attrs.update(x_edges="edges_rh", y_edges="edges_ptop", eps_km=float(eps[k]))
    out["map_strike_ptop"] = (("latitude", "longitude"), np.where(mask, ptop[k], np.nan).astype("float32"))
    out["map_strike_ptop"].attrs.update(units="hPa", eps_km=float(eps[k]))
    out.attrs["strike_eps_detail"] = float(eps[k])
    return out


def run(year, event, buffer):
    cfg = load("figdata")
    acfg = load("analysis")
    f, lat, lon, levels, profiles = fields(year, event, buffer, cfg["lead"])
    mask = figure_mask(lat, lon, cfg)
    sub = {"all": acfg["lat_band"], **acfg["lat_subbands"]}
    lat2 = np.broadcast_to(lat[:, None], mask.shape)
    weight = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)
    E = {k: edges(v) for k, v in cfg["bins"].items()}

    ds = xr.Dataset(coords={"subband": list(sub)})
    for k, e in E.items():
        ds[f"edges_{k}"] = (f"edge_{k}", e)
    sel = {}
    for name, (lo, hi) in sub.items():
        sel[name] = mask & (lat2 >= lo) & (lat2 <= hi)

    for q in cfg["hist1d"]:
        for wname, wsuffix in ((None, ""), ("cos", "_w")):
            rows = []
            for name in sub:
                m = sel[name] & np.isfinite(f[q])
                rows.append(np.histogram(f[q][m], E["delta"],
                                         weights=weight[m] if wname else None)[0])
            ds[f"h1_{q}{wsuffix}"] = (("subband", "bin_delta"), np.array(rows, dtype="float64"))
            ds[f"h1_{q}{wsuffix}"].attrs.update(units="count" if not wname else "cos(lat)-weighted count",
                                                quantity=q, quantity_units=UNITS[q], edges="edges_delta",
                                                feeds=FEEDS["h1"])

    for name2, (qx, bx, qy, by) in cfg["hist2d"].items():
        for wname, wsuffix in ((None, ""), ("cos", "_w")):
            rows = []
            for name in sub:
                m = sel[name] & np.isfinite(f[qx]) & np.isfinite(f[qy])
                rows.append(np.histogram2d(f[qx][m], f[qy][m], [E[bx], E[by]],
                                           weights=weight[m] if wname else None)[0])
            ds[f"{name2}{wsuffix}"] = (("subband", f"bin_{bx}", f"bin_{by}"), np.array(rows))
            ds[f"{name2}{wsuffix}"].attrs.update(
                units="count" if not wname else "cos(lat)-weighted count", x=qx, y=qy,
                x_units=UNITS[qx], y_units=UNITS[qy], x_edges=f"edges_{bx}", y_edges=f"edges_{by}",
                feeds=FEEDS[name2])

    # Convective split (old cells 124/137): unstable events in the figure mask
    conv = cfg["convective"]
    dB, dA = convective(f, levels, profiles["raw"], mask, conv["eps"])
    f["delta_below"], f["delta_above"] = dB, dA
    unst = mask & np.isfinite(dB)
    m2 = unst & np.isfinite(f["RH_2m"]) & np.isfinite(f["zLCL_2m"])
    ds["h2_rh_zlcl2m_unstable"] = (("bin_rh", "bin_zlcl"),
                                   np.histogram2d(f["RH_2m"][m2], f["zLCL_2m"][m2], [E["rh"], E["zlcl"]])[0])
    ds["h2_rh_zlcl2m_unstable"].attrs.update(x="RH_2m", y="zLCL_2m", x_edges="edges_rh", y_edges="edges_zlcl",
                                              units="count", feeds=FEEDS["h2_rh_zlcl2m_unstable"])
    qs = ["delta_FT", "delta_below", "delta_above"]
    for coord, bkey in (("zLCL_LPL", "zlcl"), ("RH_2m", "rh")):
        m = unst & np.isfinite(f[coord])
        idx = np.digitize(f[coord][m], E[bkey]) - 1
        ok = (idx >= 0) & (idx < len(E[bkey]) - 1)
        stats = np.zeros((len(E[bkey]) - 1, len(qs), 3))           # n, sum, sum of squares
        for k, q in enumerate(qs):
            v = f[q][m][ok]
            np.add.at(stats[:, k, 0], idx[ok], 1.0)
            np.add.at(stats[:, k, 1], idx[ok], v)
            np.add.at(stats[:, k, 2], idx[ok], v * v)
        ds[f"lb_{coord}"] = ((f"bin_{bkey}", "lb_quantity", "lb_stat"), stats)
        ds[f"lb_{coord}"].attrs.update(edges=f"edges_{bkey}", units="kJ kg-1", feeds=FEEDS["lb"],
                                       note="unstable events (delta_FT > 0); stats n, sum, sum of squares",
                                       eps_km=conv["eps"])
    ds = ds.assign_coords(lb_quantity=qs, lb_stat=["n", "sum", "sumsq"])

    # Maps: all cells (ocean too) where SP > map_sp_min_hPa; band and land drawn at plot time
    ds = ds.assign_coords(latitude=lat, longitude=lon)
    ds["in_mask"] = (("latitude", "longitude"), mask.astype("int8"))
    ok_sp = f["SP_hPa"] > cfg["map_sp_min_hPa"]
    for q in cfg["maps"] + ["delta_below", "delta_above"]:
        ds[f"map_{q}"] = (("latitude", "longitude"), np.where(ok_sp, f[q], np.nan).astype("float32"))
        ds[f"map_{q}"].attrs.update(units=UNITS[q], feeds=FEEDS["map"],
                                    note="one value per cell per year; sum over years and divide by the count "
                                         f"of finite values; NaN where SP <= {cfg['map_sp_min_hPa']} hPa")

    # Cascade composites (only boxes for this event type carry events, by construction)
    ce, csum, cw, cn, cscal, cswt = cascade(f, levels, profiles, mask, weight, cfg)
    names = list(cfg["cascade"]["boxes"])
    own = [b.startswith(event + "_") for b in names]
    ds = ds.assign_coords(case=names, level=levels)
    ds["edges_rh_cascade"] = ("edge_rh_cascade", ce)
    for v, s in csum.items():
        ds[f"casc_sum_{v}"] = (("case", "bin_rh_cascade", "level"), np.where(np.array(own)[:, None, None], s, 0.0))
        ds[f"casc_sum_{v}"].attrs.update(note="cos(lat)-weighted sum over events; divide by casc_weight",
                                         feeds=FEEDS["cascade"])
    ds["casc_weight"] = (("case", "bin_rh_cascade", "level"), np.where(np.array(own)[:, None, None], cw, 0.0))
    ds["casc_count"] = (("case", "bin_rh_cascade"), np.where(np.array(own)[:, None], cn, 0.0))
    for v, s in cscal.items():
        ds[f"casc_sum_{v}"] = (("case", "bin_rh_cascade"), np.where(np.array(own)[:, None], s, 0.0))
    ds["casc_weight_scalar"] = (("case", "bin_rh_cascade"), np.where(np.array(own)[:, None], cswt, 0.0))
    ds["casc_count"].attrs.update(units="events", note="unweighted events per box and RH bin")

    # Box hit flags per cell (sum over years = years hit), land in band, terrain kept
    boxes_all = {**cfg["cascade"]["boxes"], **cfg["hit_boxes"]}
    for bname, box in boxes_all.items():
        if not bname.startswith(event + "_"):
            continue
        inbox = (mask & (f["h_star500"] > box["h_star"][0]) & (f["h_star500"] < box["h_star"][1])
                 & (f["h500"] > box["h"][0]) & (f["h500"] < box["h"][1]))
        ds[f"hit_{bname}"] = (("latitude", "longitude"), inbox.astype("int8"))
        ds[f"hit_{bname}"].attrs.update(note="1 if this year's event is in the box", box=str(box))

    st = strike(year, event, buffer, mask, weight, sel, f, cfg)
    if st is not None:
        ds = xr.merge([ds, st.drop_vars(["latitude", "longitude"], errors="ignore")
                       .assign_coords(latitude=lat, longitude=lon)], combine_attrs="override")
    else:
        print("  no Ztop file yet: strike-point products skipped (run stage f first)", flush=True)

    n = {name: int(sel[name].sum()) for name in sub}
    ds.attrs.update(year=year, event_type=event, buffer_hours=buffer, lead=cfg["lead"],
                    lat_band=list(acfg["lat_band"]), mask_lsm_min=acfg["mask"]["lsm_min"],
                    mask_zs_max_km=str(cfg["mask"].get("zs_max_km")),
                    definitions="delta_FT = MSE_LPL - MSE_sat(500); delta_S = MSE2m - MSE_LPL; "
                                "delta_total = MSE2m - MSE_sat(500)",
                    zlcl_reference="above ground (legacy d corrected on read, E07)",
                    n_events=str(n))
    out = Path(load("paths")["work"]) / load("paths")["outputs"]["figdata"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"figdata_{event}_b{buffer:02d}_{year}.nc"
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"{year} {event}: events per sub-band {n}", flush=True)
    for q in cfg["hist1d"]:
        v = f[q][sel["all"] & np.isfinite(f[q])]
        print(f"  {q}: mean {v.mean():.2f}  median {np.median(v):.2f}  >0 {np.mean(v > 0):.0%}", flush=True)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("event", choices=["T", "M"])
    ap.add_argument("--buffer", type=int, default=6)
    a = ap.parse_args()
    run(a.year, a.event, a.buffer)
