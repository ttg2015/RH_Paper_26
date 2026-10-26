"""Stage g and the figure scripts, end to end on synthetic soundings (2 years × T, M)."""
import importlib.util, os, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "figures"), str(ROOT / "tests")]

import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr
import yaml

from tpl_mll import config, thermo
from test_ztop import LEVELS, sounding


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem + "_stage", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(root, years=(1985, 1986)):
    lat = np.arange(-60.0, 70.1, 10.0)
    lon = np.arange(0.0, 360.0, 30.0)
    ny, nx = len(lat), len(lon)
    work = root / "work"
    (work / "Profiles/derived").mkdir(parents=True)
    zs = np.zeros((ny, nx)); zs[:, 3] = 400.0                       # a 400 m plateau column (E07)
    for k, year in enumerate(years):
        for e, (t0, rh0) in {"T": (308, 0.25), "M": (302, 0.6)}.items():
            rng = np.random.default_rng(k * 10 + (e == "M"))
            T_s = t0 + 4 * rng.standard_normal((ny, nx))
            rh_s = np.clip(rh0 + 0.15 * rng.standard_normal((ny, nx)), 0.05, 0.98)
            P = {v: np.zeros((1, 1, ny, nx, len(LEVELS)), "float32") for v in ("MSE", "MSE_sat", "Z", "T")}
            S = {v: np.zeros((1, 1, ny, nx), "float32") for v in ("MSE2m", "SP")}
            D = {v: np.zeros((1, 1, ny, nx), "float32") for v in
                 ("MSE_LPL", "LPL_pressure", "p_LCL_LPL", "RH_2m", "z_LCL_2m", "z_LCL_LPL", "Z_LPL")}
            for a in range(ny):
                for b in range(nx):
                    h, hs, phi, T, q = sounding(T_s[a, b], rh_s[a, b])
                    P["MSE"][0, 0, a, b], P["MSE_sat"][0, 0, a, b], P["Z"][0, 0, a, b] = h, hs, phi
                    P["T"][0, 0, a, b] = T
                    S["MSE2m"][0, 0, a, b] = h[-1] + 2.0                  # a warm surface layer
                    S["SP"][0, 0, a, b] = 100000.0 - 4500.0 * (zs[a, b] > 0)   # plateau: 955 hPa
                    D["MSE_LPL"][0, 0, a, b] = h[-1]
                    D["LPL_pressure"][0, 0, a, b] = 1000.0
                    D["p_LCL_LPL"][0, 0, a, b] = thermo.p_lcl(T[-1], rh_s[a, b], 1000.0)
                    D["RH_2m"][0, 0, a, b] = rh_s[a, b]
                    D["z_LCL_2m"][0, 0, a, b] = zs[a, b] + thermo.dz_lcl(T[-1], rh_s[a, b])   # legacy: ASL
                    D["z_LCL_LPL"][0, 0, a, b] = D["z_LCL_2m"][0, 0, a, b]
                    D["Z_LPL"][0, 0, a, b] = zs[a, b] * 9.8 / 1000.0 + phi[-1]
            dims = ("year", "lead", "latitude", "longitude")
            co = {"year": [year], "lead": [0], "latitude": lat, "longitude": lon}
            xr.Dataset({**{v: (dims + ("level",), x) for v, x in P.items()},
                        **{v: (dims, x) for v, x in S.items()}},
                       coords={**co, "level": LEVELS}).to_netcdf(work / f"Profiles/profiles_{e}_b06_{year}.nc")
            xr.Dataset({v: (dims, x) for v, x in D.items()}, coords=co).to_netcdf(
                work / f"Profiles/derived/derived_{e}_b06_{year}.nc")
    inv = root / "era5"; inv.mkdir()
    nlat = lat[::-1]
    land = np.ones((ny, nx)); land[:, 0] = 0
    for name, var, val in (("lsm.nc", "LSM", land), ("z.nc", "Z", zs * 9.8)):
        xr.Dataset({var: (("time", "latitude", "longitude"), val[::-1][None])},
                   coords={"time": [0], "latitude": nlat, "longitude": lon}).to_netcdf(inv / name)
    paths = yaml.safe_load(open(config.CONFIG_DIR / "paths.yaml"))
    paths.update(work=str(work) + "/", era5={**paths["era5"], "root": str(inv) + "/",
                                             "invariant_z": "z.nc", "invariant_lsm": "lsm.nc"})
    (root / "paths.yaml").write_text(yaml.safe_dump(paths))
    return work, years


class TestFigData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.work, self.years = build(root)
        os.environ["PAPER26_PATHS"] = str(root / "paths.yaml")
        config.load.cache_clear()
        g, f = _module(ROOT / "pipeline/g.py"), _module(ROOT / "pipeline/f.py")
        for y in self.years:
            for e in ("T", "M"):
                f.run(y, e, 6, 100)            # f first: stage g reads the Ztop files
                g.run(y, e, 6)
        self.root = root

    def tearDown(self):
        os.environ.pop("PAPER26_PATHS")
        config.load.cache_clear()
        self.tmp.cleanup()

    def test_counts_add_up_and_e07_is_corrected(self):
        import figdata as FD
        ds = xr.load_dataset(self.work / "FigData/figdata_M_b06_1985.nc")
        n_all = eval(ds.attrs["n_events"])["all"]
        self.assertEqual(int(ds["h1_delta_FT"].sel(subband="all").sum()), n_all)
        self.assertEqual(int(ds["h2_rh_dFT"].sel(subband="all").sum()), n_all)
        d = FD.FigData("M")                                   # both years summed
        self.assertEqual(d.h("h1_delta_FT").sum(), 2 * n_all)
        # z_LCL above ground: the 400 m plateau column must not be shifted up by 400 m
        z = FD.FigData("M", (1985, 1985))
        h = z.h("h2_zlcl_dFT").sum(axis=1)
        zc = FD.centers(z.edges_of("h2_zlcl_dFT", "x"))
        self.assertLess((h * zc).sum() / h.sum(), 2500)
        # coarsening keeps totals
        c, e = FD.coarsen(d.h("h1_delta_FT"), d.edges_of("h1_delta_FT"), 4)
        self.assertEqual(c.sum(), d.h("h1_delta_FT")[: len(e) * 4 - 4].sum())

    def test_make_all_builds_every_figure(self):
        import make_all
        out = self.root / "figs"
        sys.argv = ["make_all.py", "--out", str(out)]
        make_all.main()
        index = (out / "index.html").read_text()
        # the toy grid has no T_TPL events at RH 45–55 %, so only the plume schematic may report a failure
        self.assertEqual(index.count("failed</h2>"), index.count("fig_plume_schematic failed</h2>"))
        for name in ("excess_histograms", "excess_histograms_unweighted", "weighting_check", "surface_layer",
                     "excess_maps", "excess_vs_rh", "excess_vs_rh_entrainment", "lcl_binned",
                     "convective_maps", "cascade", "excess_vs_zlcl", "wbt_increase", "ztop_fractions",
                     "delta_s_ft_vs_rh", "rh_demo", "box_hits_tp_ml", "box_hits_locations",
                     "strike_points", "strike_map", "cascade_surface"):
            self.assertTrue((out / f"{name}.png").exists(), name)
            self.assertTrue((out / f"{name}.pdf").exists(), name)


if __name__ == "__main__":
    unittest.main()
