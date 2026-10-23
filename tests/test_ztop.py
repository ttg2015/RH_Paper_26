"""Stage f on synthetic soundings built with the real thermo code."""
import os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import xarray as xr
import yaml

from tpl_mll import config, thermo, ztop
from tpl_mll.plume import plume

LEVELS = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
                   350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
                   925, 950, 975, 1000], dtype=float)


def sounding(T_s, rh_s, rh_ft=0.3):
    """Hot surface, 7 K/km lapse to an isothermal 210 K stratosphere; RH falls from rh_s to rh_ft."""
    z = 7.4 * np.log(1000.0 / LEVELS)                       # km, rough
    T = np.maximum(T_s - 7.0 * z, 210.0)
    rh = rh_ft + (rh_s - rh_ft) * np.exp(-z / 1.5)
    q = rh * thermo.q_sat(T, LEVELS * 100)
    phi = thermo.g * z                                      # kJ/kg (g · km)
    return thermo.mse(T, q, phi), thermo.mse_sat(T, LEVELS * 100, phi), phi, T, q


class TestPlumePhysics(unittest.TestCase):
    def test_undiluted_plume_keeps_its_energy(self):
        h, hs, phi, *_ = sounding(310, 0.5)
        z = np.tile(phi / 9.8, (1, 1))
        hp = plume(h[None], z, np.array([350.0]), 0.0)
        self.assertTrue(np.allclose(hp, 350.0))

    def test_entraining_plume_relaxes_to_environment(self):
        z = np.linspace(0, 10, 200)[None]
        env = np.full_like(z, 330.0)
        hp = plume(env, z, np.array([350.0]), 0.5)
        expect = 330 + 20 * np.exp(-0.5 * z)                 # analytic solution
        self.assertTrue(np.allclose(hp, expect, atol=0.05))

    def test_top_lowers_as_eps_grows(self):
        h, hs, phi, *_ = sounding(305, 0.55)
        c = ztop.Columns(LEVELS, h[None], hs[None], phi[None], h[-1:].copy(), np.array([1000.0]))
        tops = [c.top(e)[2][0] for e in (0.0, 0.1, 0.2)]
        finite = [t for t in tops if np.isfinite(t)]
        self.assertEqual(finite, sorted(finite))             # pressure rises with ε
        lo, hi = ztop.eps_for_top(c, 400.0)[0], ztop.eps_for_top(c, 600.0)[0]
        self.assertLessEqual(lo, hi)

    def test_double_crossing_is_flagged(self):
        h, hs, phi, *_ = sounding(305, 0.6)
        hs = hs.copy()
        hs[(LEVELS >= 450) & (LEVELS <= 500)] += 25         # a warm layer (inversion) mid-column
        c = ztop.Columns(LEVELS, h[None], hs[None], phi[None], h[-1:].copy(), np.array([1000.0]))
        _, p_first, p_high, n_layers, *_ = c.top(0.0)
        self.assertGreaterEqual(n_layers[0], 2)
        self.assertGreater(p_first[0], p_high[0])


class TestStageF(unittest.TestCase):
    """Fake c2/d files on a small grid → pipeline/f.py → ztop file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        lat = np.arange(-60.0, 70.1, 10.0)                  # S → N like the pipeline
        lon = np.arange(0.0, 360.0, 30.0)
        ny, nx = len(lat), len(lon)
        rng = np.random.default_rng(0)
        T_s = 300 + 15 * rng.random((ny, nx))
        rh_s = 0.2 + 0.6 * rng.random((ny, nx))
        prof = {k: np.zeros((1, 1, ny, nx, len(LEVELS)), "float32") for k in ("MSE", "MSE_sat", "Z")}
        der = {k: np.zeros((1, 1, ny, nx), "float32") for k in ("MSE_LPL", "LPL_pressure", "p_LCL_LPL")}
        for a in range(ny):
            for b in range(nx):
                h, hs, phi, T, q = sounding(T_s[a, b], rh_s[a, b])
                prof["MSE"][0, 0, a, b], prof["MSE_sat"][0, 0, a, b], prof["Z"][0, 0, a, b] = h, hs, phi
                der["MSE_LPL"][0, 0, a, b] = h[-1]
                der["LPL_pressure"][0, 0, a, b] = 1000.0
                der["p_LCL_LPL"][0, 0, a, b] = thermo.p_lcl(T[-1], rh_s[a, b], 1000.0)
        dims = ("year", "lead", "latitude", "longitude")
        co = {"year": [1985], "lead": [0], "latitude": lat, "longitude": lon}
        work = root / "work"
        (work / "Profiles/derived").mkdir(parents=True)
        xr.Dataset({k: (dims + ("level",), v) for k, v in prof.items()},
                   coords={**co, "level": LEVELS}).to_netcdf(work / "Profiles/profiles_M_b06_1985.nc")
        xr.Dataset({k: (dims, v) for k, v in der.items()}, coords=co).to_netcdf(
            work / "Profiles/derived/derived_M_b06_1985.nc")
        inv = root / "era5"
        inv.mkdir()
        nlat = lat[::-1]                                    # invariant files are N → S
        land = np.ones((ny, nx)); land[:, 0] = 0.0           # one ocean column
        xr.Dataset({"LSM": (("time", "latitude", "longitude"), land[None])},
                   coords={"time": [0], "latitude": nlat, "longitude": lon}).to_netcdf(inv / "lsm.nc")
        xr.Dataset({"Z": (("time", "latitude", "longitude"), np.zeros((1, ny, nx)))},
                   coords={"time": [0], "latitude": nlat, "longitude": lon}).to_netcdf(inv / "z.nc")
        paths = yaml.safe_load(open(config.CONFIG_DIR / "paths.yaml"))
        paths.update(work=str(work) + "/", era5={**paths["era5"], "root": str(inv) + "/",
                                                 "invariant_z": "z.nc", "invariant_lsm": "lsm.nc"})
        (root / "paths.yaml").write_text(yaml.safe_dump(paths))
        os.environ["PAPER26_PATHS"] = str(root / "paths.yaml")
        config.load.cache_clear()
        self.work, self.lat, self.lon = work, lat, lon

    def tearDown(self):
        os.environ.pop("PAPER26_PATHS")
        config.load.cache_clear()
        self.tmp.cleanup()

    def test_stage_f_writes_plausible_tops(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("stage_f", Path(__file__).resolve().parents[1] / "pipeline/f.py")
        f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
        f.run(1985, "M", 6, chunk=50)
        ds = xr.open_dataset(self.work / "Ztop/ztop_M_b06_1985.nc")
        cat = ds["category"]
        # masked out: outside −50..65 and the ocean column
        self.assertTrue(np.isnan(cat.sel(latitude=-60)).all())
        self.assertTrue(np.isnan(cat.isel(longitude=0)).all())
        inside = cat.sel(latitude=slice(-50, 60)).isel(longitude=slice(1, None))
        self.assertFalse(np.isnan(inside).any())
        # undiluted plumes go highest; tops never move up as ε grows
        p = ds["p_top"].values
        ok = np.isfinite(p[:-1]) & np.isfinite(p[1:])
        self.assertTrue(np.all(p[1:][ok] >= p[:-1][ok] - 1e-3))
        # categories cover more than one outcome on this varied grid, and passes exist
        self.assertGreater(len(np.unique(inside.values)), 1)
        self.assertTrue((inside == 1).any())
        # excess split adds up
        tot, below, above = (ds[k].values for k in ("E_total", "E_below", "E_above"))
        m = np.isfinite(tot)
        self.assertTrue(np.allclose(below[m] + above[m], tot[m], atol=1e-3))
        lo, hi = ds["eps_top_lo"].values, ds["eps_top_hi"].values
        both = np.isfinite(lo) & np.isfinite(hi)
        self.assertTrue(np.all(lo[both] <= hi[both] + 1e-6))
        ds.close()


if __name__ == "__main__":
    unittest.main()
