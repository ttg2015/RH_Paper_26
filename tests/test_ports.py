"""Stage d and e ports against the legacy scripts, on the same synthetic files.

The legacy scripts are run unchanged (with a stand-in Locations_of_files module pointing at a temp
folder). The ports must reproduce them exactly, except where a fix is supposed to change things:
  d: z_LCL above ground (E07), and the Romps c_vv change (p_LCL, T_LCL move slightly)
  e: latitude band and boxes from config (identical when the config matches the legacy values)
Legacy scripts come from legacy/ in the repo or the legacy_scripts folder in paths.yaml; the
comparison tests are skipped if neither exists.
"""
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import numpy as np
import xarray as xr
import yaml

from tpl_mll import config, thermo
from test_ztop import LEVELS, sounding


def legacy_dir():
    for p in (ROOT / "legacy", Path(yaml.safe_load(open(ROOT / "config/paths.yaml"))["legacy_scripts"])):
        if (p / "d_Add_Derived.py").exists() and (p / "e_RH_bins.py").exists():
            return p
    return None


LEGACY = legacy_dir()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(root):
    """Profiles + invariant files on a 10° grid: one 400 m plateau column, one SP > 1000 hPa column."""
    lat = np.arange(-60.0, 70.1, 10.0)
    lon = np.arange(0.0, 360.0, 30.0)
    ny, nx, nl = len(lat), len(lon), len(LEVELS)
    zs = np.zeros((ny, nx))
    zs[:, 3] = 400.0
    leads = [0, 6]
    rng = np.random.default_rng(3)
    V = {v: np.zeros((1, 2, ny, nx, nl), "float32") for v in ("T", "Q", "Z", "q_sat", "MSE", "MSE_sat")}
    S = {v: np.zeros((1, 2, ny, nx), "float32") for v in ("T2m", "q2m", "SP", "MSE2m")}
    for L in range(2):
        for a in range(ny):
            for b in range(nx):
                Ts = 300 + 10 * rng.random() + 2 * L
                rh = 0.15 + 0.7 * rng.random()
                h, hs, phi, T, q = sounding(Ts, rh)
                sp = 95500.0 if zs[a, b] else (101200.0 if b == 5 else 100000.0 - 300 * rng.random())
                V["T"][0, L, a, b], V["Q"][0, L, a, b], V["Z"][0, L, a, b] = T, q, phi + zs[a, b] * 9.8 / 1000
                V["q_sat"][0, L, a, b] = thermo.q_sat(T, LEVELS * 100)
                V["MSE"][0, L, a, b] = thermo.mse(T, q, V["Z"][0, L, a, b])
                V["MSE_sat"][0, L, a, b] = thermo.mse_sat(T, LEVELS * 100, V["Z"][0, L, a, b])
                S["T2m"][0, L, a, b] = Ts + 1.5
                S["q2m"][0, L, a, b] = q[-1] * 1.05
                S["SP"][0, L, a, b] = sp
                S["MSE2m"][0, L, a, b] = thermo.mse(Ts + 1.5, q[-1] * 1.05, zs[a, b] * 9.8 / 1000)
    dims = ("year", "lead", "latitude", "longitude")
    work = root / "work"
    (work / "Profiles/derived").mkdir(parents=True)
    xr.Dataset({**{v: (dims + ("level",), x) for v, x in V.items()}, **{v: (dims, x) for v, x in S.items()}},
               coords={"year": [1985], "lead": leads, "latitude": lat, "longitude": lon, "level": LEVELS}
               ).to_netcdf(work / "Profiles/profiles_T_b06_1985.nc")
    # identical M file so stage e sees both event types
    (work / "Profiles/profiles_M_b06_1985.nc").write_bytes((work / "Profiles/profiles_T_b06_1985.nc").read_bytes())
    inv = root / "era5"
    inv.mkdir()
    land = np.ones((ny, nx))
    land[:, 0] = 0.0
    for name, var, val in (("lsm.nc", "LSM", land), ("z.nc", "Z", zs * 9.8)):
        xr.Dataset({var: (("time", "latitude", "longitude"), val[::-1][None])},
                   coords={"time": [0], "latitude": lat[::-1], "longitude": lon}).to_netcdf(inv / name)
    paths = yaml.safe_load(open(config.CONFIG_DIR / "paths.yaml"))
    paths.update(work=str(work) + "/", era5={**paths["era5"], "root": str(inv) + "/",
                                             "invariant_z": "z.nc", "invariant_lsm": "lsm.nc"})
    (root / "paths.yaml").write_text(yaml.safe_dump(paths))
    return work, inv, zs, lat, lon


class Ports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work, self.inv, self.zs, self.lat, self.lon = build(self.root)
        os.environ["PAPER26_PATHS"] = str(self.root / "paths.yaml")
        config.load.cache_clear()
        self.d = module(ROOT / "pipeline/d.py", "stage_d")
        self.e = module(ROOT / "pipeline/e.py", "stage_e")

    def tearDown(self):
        for k in ("PAPER26_PATHS", "PAPER26_ANALYSIS"):
            os.environ.pop(k, None)
        config.load.cache_clear()
        for m in ("Locations_of_files", "z_Thermodynamics"):
            sys.modules.pop(m, None)
        self.tmp.cleanup()

    def run_legacy(self, script, argv):
        """Run a legacy script's main() with a stand-in Locations_of_files."""
        loc = types.ModuleType("Locations_of_files")
        loc.Profiles_dir = str(self.work / "Profiles") + "/"
        loc.gph_file = str(self.inv / "z.nc")
        loc.mask_file = str(self.inv / "lsm.nc")
        loc.work = str(self.work) + "/"
        sys.modules["Locations_of_files"] = loc
        sys.path.insert(0, str(LEGACY))
        try:
            m = module(LEGACY / script, "legacy_" + script[:-3])
            old = sys.argv
            sys.argv = [script] + argv
            from contextlib import redirect_stdout
            import io
            with redirect_stdout(io.StringIO()):
                m.main()
            sys.argv = old
        finally:
            sys.path.remove(str(LEGACY))

    # ---------------------------------------------------------------- stage d
    def test_d_physics(self):
        self.d.run(1985, "T", 6)
        ds = xr.load_dataset(self.work / "Profiles/derived/derived_T_b06_1985.nc")
        self.assertEqual(ds["z_LCL_2m"].attrs["reference"], "above_ground")
        # LPL: deepest level ≤ SP, 1000 hPa where SP > 1000 hPa
        lpl = ds["LPL_pressure"].isel(year=0, lead=0)
        self.assertTrue((lpl.isel(longitude=3) == 950).all())       # plateau, SP = 955 hPa
        self.assertTrue((lpl.isel(longitude=5) == 1000).all())      # SP = 1012 hPa
        # heights above ground: never negative, plateau not shifted up by 400 m
        z2 = ds["z_LCL_2m"].values
        self.assertTrue(np.nanmin(z2) >= 0)
        self.assertLess(abs(np.nanmean(z2[..., 3]) - np.nanmean(z2[..., 2])), 300)
        # LPL parcel: its LCL is at least the LPL's own height above ground
        zl = ds["z_LCL_LPL"].values
        lpl_agl = ds["Z_LPL"].values / 9.8 * 1000 - self.zs[None, None]
        self.assertTrue(np.all(zl >= np.maximum(lpl_agl, 0) - 1e-3))

    @unittest.skipUnless(LEGACY, "legacy scripts not available")
    def test_d_matches_legacy_except_fixes(self):
        self.run_legacy("d_Add_Derived.py", ["1985", "T", "--buffer", "6"])
        path = self.work / "Profiles/derived/derived_T_b06_1985.nc"
        old = xr.load_dataset(path)
        path.unlink()
        self.d.run(1985, "T", 6)
        new = xr.load_dataset(path)
        same = ["LPL_pressure", "T_LPL", "Q_LPL", "Z_LPL", "MSE_LPL", "MSE_sat_LPL", "RH_2m", "RH_LPL"]
        for v in same:
            np.testing.assert_allclose(new[v], old[v], rtol=1e-6, err_msg=v)
        for v in ("T_LCL_2m", "T_LCL_LPL"):                          # c_vv 1318 → 1418
            np.testing.assert_allclose(new[v], old[v], atol=0.15, err_msg=v)
        for v in ("p_LCL_2m", "p_LCL_LPL"):
            np.testing.assert_allclose(new[v], old[v], atol=0.5, err_msg=v)
        zs = self.zs[None, None]
        lpl_agl = np.maximum(old["Z_LPL"].values / 9.8 * 1000 - zs, 0)
        # E07: legacy = z_s + Δz; port = Δz (2 m) and LPL height above ground + Δz (LPL)
        np.testing.assert_allclose(new["z_LCL_2m"], old["z_LCL_2m"].values - zs, atol=20)
        np.testing.assert_allclose(new["z_LCL_LPL"], old["z_LCL_LPL"].values - zs + lpl_agl, atol=20)

    # ---------------------------------------------------------------- stage e
    def analysis_with_band(self, band):
        cfg = yaml.safe_load(open(config.CONFIG_DIR / "analysis.yaml"))
        cfg["lat_band"] = band
        p = self.root / "analysis.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        os.environ["PAPER26_ANALYSIS"] = str(p)
        config.load.cache_clear()

    def derived_both(self):
        for et in ("T", "M"):
            self.d.run(1985, et, 6)

    @unittest.skipUnless(LEGACY, "legacy scripts not available")
    def test_e_matches_legacy_with_legacy_band(self):
        self.derived_both()
        self.run_legacy("e_RH_bins.py", ["1985", "--buffer", "6"])
        path = self.work / "rh_binned/rh_binned_b06_1985.nc"
        old = xr.load_dataset(path)
        path.unlink()
        self.analysis_with_band([-55.0, 55.0])
        self.e.run(1985, 6)
        new = xr.load_dataset(path)
        self.assertEqual(list(new["case"].values), list(old["case"].values))
        np.testing.assert_array_equal(new["n_events"], old["n_events"])
        self.assertGreater(int(old["n_events"].sum()), 0)
        for v in [k for k in old.data_vars if k.startswith(("sum_", "box_"))]:
            np.testing.assert_allclose(new[v], old[v], rtol=1e-6, err_msg=v)

    def test_e_uses_config_band(self):
        self.derived_both()
        self.e.run(1985, 6)
        ds = xr.load_dataset(self.work / "rh_binned/rh_binned_b06_1985.nc")
        band = config.load("analysis")["lat_band"]
        self.assertEqual((ds.attrs["lat_min"], ds.attrs["lat_max"]), tuple(band))
        n = int(ds["n_events"].sum())
        self.analysis_with_band([-55.0, 55.0])
        self.e.run(1985, 6)
        n55 = int(xr.load_dataset(self.work / "rh_binned/rh_binned_b06_1985.nc")["n_events"].sum())
        self.assertGreaterEqual(n, n55)                             # −50…65 adds the 60°N row


if __name__ == "__main__":
    unittest.main()
