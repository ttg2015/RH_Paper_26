"""Dense, realistic-looking figure data for previewing and testing the figure scripts.

Writes FigData files with the same layout as pipeline/g.py, but filled from ~100k sampled
events per year and event type (δ_FT ≈ 20·ln(0.34/RH) + noise; T events dry, M events moist),
plus smooth cascade composites. Not physics: only for checking how figures draw.
"""
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _edges(spec):
    a, b, s = spec
    return np.round(np.arange(a, b + s / 2, s), 6)


def write(folder, years=(1985, 1986), n=100_000, seed=0):
    cfg = yaml.safe_load(open(ROOT / "config/figdata.yaml"))
    E = {k: _edges(v) for k, v in cfg["bins"].items()}
    ce = _edges(cfg["cascade"]["rh_bins"])
    levels = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
                       850, 900, 950, 1000], float)
    lat = np.arange(-50, 65.1, 2.5)
    lon = np.arange(0, 360, 2.5)
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for year in years:
        for e in ("T", "M"):
            rh = np.clip(rng.beta(2.0, 4.5, n) if e == "T" else rng.beta(5.0, 3.5, n), 0.02, 1.0)
            lat_e = rng.uniform(-50, 65, n)
            dft = 20 * np.log(0.34 / rh) * (1 - 0.5 * rh) + rng.normal(0, 3, n) - (4 if e == "T" else 0)
            dft = -dft
            ds_ = rng.normal(3 if e == "T" else 1.5, 1.5, n)
            z = np.clip(125 * 30 * (1 - rh) ** 1.3 + rng.normal(0, 150, n), 0, None)
            w = np.cos(np.deg2rad(lat_e))
            sub = {"all": (-50, 65), "TP": (-20, 20), "ST": (20, 35), "ML": (35, 65)}
            ds = xr.Dataset(coords={"subband": list(sub)})
            for k, v in E.items():
                ds[f"edges_{k}"] = (f"edge_{k}", v)
            q = {"delta_FT": dft, "delta_S": ds_, "delta_total": dft + ds_, "RH_2m": rh, "zLCL_2m": z}
            for name in cfg["hist1d"]:
                for wt, suf in ((None, ""), (w, "_w")):
                    rows = [np.histogram(q[name][(lat_e >= a) & (lat_e <= b)], E["delta"],
                                         weights=None if wt is None else wt[(lat_e >= a) & (lat_e <= b)])[0]
                            for a, b in sub.values()]
                    ds[f"h1_{name}{suf}"] = (("subband", "bin_delta"), np.array(rows, float))
                    ds[f"h1_{name}{suf}"].attrs.update(edges="edges_delta")
            for name, (qx, bx, qy, by) in cfg["hist2d"].items():
                if qx not in q or qy not in q:
                    continue                                  # energy planes filled below
                for wt, suf in ((None, ""), (w, "_w")):
                    rows = [np.histogram2d(q[qx][(lat_e >= a) & (lat_e <= b)], q[qy][(lat_e >= a) & (lat_e <= b)],
                                           [E[bx], E[by]],
                                           weights=None if wt is None else wt[(lat_e >= a) & (lat_e <= b)])[0]
                            for a, b in sub.values()]
                    ds[f"{name}{suf}"] = (("subband", f"bin_{bx}", f"bin_{by}"), np.array(rows))
                    ds[f"{name}{suf}"].attrs.update(x_edges=f"edges_{bx}", y_edges=f"edges_{by}")
            # surface-layer planes, LCL-binned stats (unstable), RH × z_LCL density
            hs500 = rng.normal(338 if e == "M" else 332, 6, n)
            h_lpl = hs500 + dft
            h2m = h_lpl + ds_
            for name, (xq, yq) in {"h2_hs500_h2m": (hs500, h2m), "h2_hs500_hlpl": (hs500, h_lpl)}.items():
                for wt, suf in ((None, ""), (w, "_w")):
                    rows = [np.histogram2d(xq[(lat_e >= a) & (lat_e <= b)], yq[(lat_e >= a) & (lat_e <= b)],
                                           [E["energy"], E["energy"]],
                                           weights=None if wt is None else wt[(lat_e >= a) & (lat_e <= b)])[0]
                            for a, b in sub.values()]
                    ds[f"{name}{suf}"] = (("subband", "bin_energy", "bin_energy_y"), np.array(rows))
                    ds[f"{name}{suf}"].attrs.update(x_edges="edges_energy", y_edges="edges_energy")
            u = dft > 0
            dB = np.clip(0.15 * dft + 0.0008 * z, 0, None)
            dA = dft - dB
            ds["h2_rh_zlcl2m_unstable"] = (("bin_rh", "bin_zlcl"), np.histogram2d(rh[u], z[u], [E["rh"], E["zlcl"]])[0])
            ds["h2_rh_zlcl2m_unstable"].attrs.update(x_edges="edges_rh", y_edges="edges_zlcl")
            for coord, bkey, cv in (("zLCL_LPL", "zlcl", z), ("RH_2m", "rh", rh)):
                idx = np.digitize(cv[u], E[bkey]) - 1
                ok = (idx >= 0) & (idx < len(E[bkey]) - 1)
                st = np.zeros((len(E[bkey]) - 1, 3, 3))
                for k, v in enumerate((dft[u], dB[u], dA[u])):
                    np.add.at(st[:, k, 0], idx[ok], 1.0)
                    np.add.at(st[:, k, 1], idx[ok], v[ok])
                    np.add.at(st[:, k, 2], idx[ok], v[ok] ** 2)
                ds[f"lb_{coord}"] = ((f"bin_{bkey}", "lb_quantity", "lb_stat"), st)
            # maps
            ds = ds.assign_coords(latitude=lat, longitude=lon)
            ds["in_mask"] = (("latitude", "longitude"), (rng.random((len(lat), len(lon))) < 0.3).astype("int8"))
            L, O = np.meshgrid(lon, lat)
            for name in cfg["maps"]:
                base = (8 if e == "M" else -6) * np.cos(np.deg2rad(O)) + 3 * np.sin(np.deg2rad(L) * 2)
                ds[f"map_{name}"] = (("latitude", "longitude"), (base + rng.normal(0, 2, base.shape)).astype("float32"))
            fte = ds["map_delta_FT"].values
            ds["map_delta_below"] = (("latitude", "longitude"), np.where(fte > 0, 0.2 * fte, np.nan).astype("float32"))
            ds["map_delta_above"] = (("latitude", "longitude"), np.where(fte > 0, 0.8 * fte, np.nan).astype("float32"))
            # cascade: profiles drop from h_ML through a sickle to a minimum near 600 hPa
            boxes = list(cfg["cascade"]["boxes"])
            nb = len(ce) - 1
            rhc = 0.5 * (ce[1:] + ce[:-1])
            zl = 7.4 * np.log(1000 / levels)
            sums = {v: np.zeros((4, nb, len(levels))) for v in cfg["cascade"]["profile_vars"]}
            cw = np.zeros((4, nb, len(levels)))
            cn = np.zeros((4, nb))
            for b, name in enumerate(boxes):
                if not name.startswith(e + "_"):
                    continue
                trop = "TPL" in name
                cnt = 4000 * np.exp(-((rhc - (0.6 if e == "M" else 0.3)) / 0.18) ** 2)
                for k, r in enumerate(rhc):
                    hml = (335 if trop else 322) + 25 * r
                    top = 1.6 - 1.2 * r
                    hmin = 333 if trop else 318
                    h = np.where(zl < top, hml, hmin + (hml - hmin) * np.exp(-(zl - top) / 1.2)) + 0.4 * np.maximum(zl - 8, 0) ** 2
                    hs = (342 if trop else 328) + 0.5 * (zl - 5.5) ** 2 + (1 - r) * 8 * np.exp(-zl)
                    sums["MSE"][b, k] = h * cnt[k]
                    sums["MSE_sat"][b, k] = hs * cnt[k]
                    sums["T"][b, k] = (300 - 6.5 * zl) * cnt[k]
                    sums["Z"][b, k] = 9.8 * zl * cnt[k]
                    cw[b, k] = cnt[k]
                    cn[b, k] = cnt[k]
            ds = ds.assign_coords(case=boxes, level=levels)
            ds["edges_rh_cascade"] = ("edge_rh_cascade", ce)
            for v, s in sums.items():
                ds[f"casc_sum_{v}"] = (("case", "bin_rh_cascade", "level"), s)
            ds["casc_weight"] = (("case", "bin_rh_cascade", "level"), cw)
            ds["casc_count"] = (("case", "bin_rh_cascade"), cn)
            hl = np.zeros_like(cn)
            for b, name in enumerate(boxes):
                hl[b] = ((335 if "TPL" in name else 322) + 25 * rhc) * cn[b]
            ds["casc_sum_h_LPL"] = (("case", "bin_rh_cascade"), hl)
            ds["casc_sum_p_LPL"] = (("case", "bin_rh_cascade"), 1000.0 * cn)
            ds["casc_weight_scalar"] = (("case", "bin_rh_cascade"), cn)
            dpg = _edges(cfg["cascade"]["dp_grid"])
            pdp = 1000.0 - dpg
            ds = ds.assign_coords(dp=dpg)
            for v in ("MSE", "MSE_sat"):
                s = np.zeros((4, nb, len(dpg)))
                for b in range(4):
                    for kk in range(nb):
                        prof = sums[v][b, kk] / max(cn[b, kk], 1e-9)
                        s[b, kk] = np.interp(np.log(pdp), np.log(levels), prof) * cn[b, kk]
                ds[f"casc_dp_sum_{v}"] = (("case", "bin_rh_cascade", "dp"), s)
                ds[f"casc_dp_weight_{v}"] = (("case", "bin_rh_cascade", "dp"), np.repeat(cn[..., None], len(dpg), 2))
            ds["h2_hs500_h500"] = (("subband", "bin_energy", "bin_energy_y"),
                                   np.stack([np.histogram2d(rng.normal(337, 5, n), rng.normal(327, 5, n),
                                                            [E["energy"], E["energy"]])[0]] * 4))
            ds["h2_hs500_h500"].attrs.update(x_edges="edges_energy", y_edges="edges_energy")
            ds["h2_hs500_h500_w"] = ds["h2_hs500_h500"]
            ds.attrs.update(year=year, event_type=e, n_events=str({"all": n}))
            ds.to_netcdf(folder / f"figdata_{e}_b06_{year}.nc")
