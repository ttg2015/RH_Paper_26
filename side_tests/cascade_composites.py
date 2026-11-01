"""Composite profiles by RH in many free-tropospheric boxes, to look for a profile measurable that matches K.

Groups for each event type: the four cascade boxes, every box of a grid in (h*500, h500), and "all"
(every event, no box). Each group gets additive sums per RH bin and pressure level.

Reads   Profiles/profiles_{E}_b06_{Y}.nc, Profiles/derived/derived_{E}_b06_{Y}.nc (through pipeline/g.py)
Writes  <work>/side_tests/casc/casc_{E}_{Y}.npz

    python side_tests/cascade_composites.py YEAR EVENT      # or qsub side_tests/run_cascade_composites.pbs

Figure mask plus the cascade terrain cut, lead from figdata.yaml, fixed pressure levels, unweighted.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "pipeline")]
import g                                                      # noqa: E402

RH_EDGES = np.round(np.arange(0.0, 1.0501, 0.05), 6)
GRID_WIDTH = 3.0                                              # kJ/kg, as the cascade boxes
GRID_HS = np.arange(318.0, 348.1, GRID_WIDTH)                 # h*500 edges
GRID_H = np.arange(308.0, 341.1, GRID_WIDTH)                  # h500 edges
SCALARS = ("h_LPL", "h_star500", "h500", "zLCL_LPL", "p_LCL_LPL", "p_LPL", "RH_2m", "zs_km")
PROFILES = ("MSE", "MSE_sat", "Z", "T")


def group_names(event, boxes):
    casc = [k for k in boxes if k.startswith(event + "_")]
    grid = [f"g{i:02d}_{j:02d}" for i in range(len(GRID_HS) - 1) for j in range(len(GRID_H) - 1)]
    return casc, grid, casc + grid + ["all"]


def accumulate(f, profiles, mask, zs_km, event, boxes):
    """Sums over events per (group, RH bin): n; scalar sums and counts; level sums and counts."""
    casc, grid, names = group_names(event, boxes)
    ng, nb, nl = len(names), len(RH_EDGES) - 1, next(iter(profiles.values())).shape[-1]
    out = {"n": np.zeros((ng, nb))}
    for v in SCALARS:
        out[f"s_{v}"], out[f"w_{v}"] = np.zeros((ng, nb)), np.zeros((ng, nb))
    for v in PROFILES:
        out[f"p_{v}"], out[f"pw_{v}"] = np.zeros((ng, nb, nl)), np.zeros((ng, nb, nl))
    field = dict(f, zs_km=zs_km)
    k = np.digitize(f["RH_2m"], RH_EDGES) - 1
    base = mask & (k >= 0) & (k < nb) & np.isfinite(f["h_LPL"]) & np.isfinite(f["h_star500"])
    n_grid = len(GRID_HS) - 1, len(GRID_H) - 1
    ih = np.digitize(f["h_star500"], GRID_HS) - 1
    jh = np.digitize(f["h500"], GRID_H) - 1
    in_grid = (ih >= 0) & (ih < n_grid[0]) & (jh >= 0) & (jh < n_grid[1])
    gid = len(casc) + ih * n_grid[1] + jh

    def add(g_index, sel):
        i, j = np.nonzero(sel)
        gi = g_index[i, j] if isinstance(g_index, np.ndarray) else np.full(len(i), g_index)
        kk = k[i, j]
        np.add.at(out["n"], (gi, kk), 1)
        for v in SCALARS:
            x = field[v][i, j]
            np.add.at(out[f"s_{v}"], (gi, kk), np.nan_to_num(x))
            np.add.at(out[f"w_{v}"], (gi, kk), np.isfinite(x))
        for v in PROFILES:
            x = profiles[v][i, j]
            np.add.at(out[f"p_{v}"], (gi, kk), np.nan_to_num(x))
            np.add.at(out[f"pw_{v}"], (gi, kk), np.isfinite(x))

    for b, name in enumerate(casc):
        box = boxes[name]
        add(b, base & (f["h_star500"] > box["h_star"][0]) & (f["h_star500"] < box["h_star"][1])
            & (f["h500"] > box["h"][0]) & (f["h500"] < box["h"][1]))
    add(gid, base & in_grid)
    add(ng - 1, base)
    return out, names


def run(year, event, out_dir, force=False):
    path = Path(out_dir) / f"casc_{event}_{year}.npz"
    if path.exists() and not force:
        print(path.name, "exists, skipped", flush=True)
        return
    cfg = g.load("figdata")
    f, lat, lon, levels, profiles = g.fields(year, event, 6, cfg["lead"])
    zs_km = g.surface_height_km(lat, lon)
    mask = g.figure_mask(lat, lon, cfg) & (zs_km < cfg["cascade"]["zs_max_km"])
    res, names = accumulate(f, {v: profiles[v] for v in PROFILES}, mask, zs_km, event, cfg["cascade"]["boxes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, levels=levels, rh_edges=RH_EDGES, names=np.array(names), **res)
    print(f"{path.name}: {int(res['n'][-1].sum()):,} events", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("year", type=int)
    ap.add_argument("event", choices=["T", "M"])
    ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    run(a.year, a.event, a.out or Path(g.load("paths")["work"]) / "side_tests" / "casc", a.force)


if __name__ == "__main__":
    main()
