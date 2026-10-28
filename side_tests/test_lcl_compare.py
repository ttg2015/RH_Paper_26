"""Local check of lcl_compare.summarise() on synthetic soundings built with the real thermo code.

    python test_lcl_compare.py [--repo PATH]       (no HPC data needed)
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("PAPER26_REPO", HERE.parent))
if "--repo" in sys.argv:
    REPO = Path(sys.argv[sys.argv.index("--repo") + 1])
    del sys.argv[sys.argv.index("--repo"):sys.argv.index("--repo") + 2]
for sub in ("src", "pipeline"):
    sys.path.insert(0, str(REPO / sub))
sys.path.insert(0, str(HERE))

import g                                   # noqa: E402  pipeline/g.py
import lcl_compare as L                    # noqa: E402
from tpl_mll import thermo                 # noqa: E402

LEVELS = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
                   350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
                   925, 950, 975, 1000], dtype=float)
I500, I975 = int(np.nonzero(LEVELS == 500)[0][0]), int(np.nonzero(LEVELS == 975)[0][0])


def sounding(T_s, rh_s, rh_ft=0.3):
    z = 7.4 * np.log(1000.0 / LEVELS)                       # km
    T = np.maximum(T_s - 7.0 * z, 210.0)
    rh = rh_ft + (rh_s - rh_ft) * np.exp(-z / 1.5)
    q = rh * thermo.q_sat(T, LEVELS * 100)
    phi = thermo.g * z                                      # kJ/kg
    return T, q, phi


def grid(n_lat=6, n_lon=8, t2m_offset=0.0, q2m_scale=1.0, sp_hpa=985.0):
    """Events on a small grid. Surface at sp_hpa; LPL = 975 hPa; 2 m air = LPL air + offsets."""
    rng = np.random.default_rng(0)
    n = n_lat * n_lon
    Ts, rhs = rng.uniform(300, 318, n), rng.uniform(0.25, 0.9, n)
    MSE, MSEs, Z, T975, q975 = [], [], [], [], []
    for T_s, rh_s in zip(Ts, rhs):
        T, q, phi = sounding(T_s, rh_s)
        MSE.append(thermo.mse(T, q, phi))
        MSEs.append(thermo.mse_sat(T, LEVELS * 100, phi))
        Z.append(phi)
        T975.append(T[I975]), q975.append(q[I975])
    r = lambda v: np.array(v).reshape(n_lat, n_lon, -1)
    raw = {"MSE": r(MSE), "MSE_sat": r(MSEs), "Z": r(Z)}
    T975, q975 = np.array(T975).reshape(n_lat, n_lon), np.array(q975).reshape(n_lat, n_lon)
    zs_m = 7.4 * np.log(1000.0 / sp_hpa) * 1000.0
    lpl_pa = 975.0 * 100.0
    lpl_agl = np.maximum(raw["Z"][..., I975] * 1000.0 / thermo.g - zs_m, 0.0)
    T2, q2 = T975 + t2m_offset, q975 * q2m_scale
    rh2, rhL = thermo.rh(q2, T2, sp_hpa * 100.0), thermo.rh(q975, T975, lpl_pa)
    p2 = thermo.p_lcl(T2, rh2, sp_hpa * 100.0) / 100.0
    pL = thermo.p_lcl(T975, rhL, lpl_pa) / 100.0
    z2 = thermo.z_lcl(T2, rh2, 0.0)
    zL = thermo.z_lcl(T975, rhL, lpl_agl)
    h_lpl = raw["MSE"][..., I975]
    f = {"delta_FT": h_lpl - raw["MSE_sat"][..., I500], "zLCL_2m": z2, "zLCL_LPL": zL, "RH_2m": rh2,
         "p_LCL_LPL": pL, "p_LPL": np.full_like(z2, 975.0), "h_LPL": h_lpl, "SP_hPa": np.full_like(z2, sp_hpa)}
    x = {"p_LCL_2m": p2, "RH_LPL": rhL, "lpl_agl": lpl_agl}
    return f, x, raw, np.ones(z2.shape, bool)


class TestSummarise(unittest.TestCase):
    def test_identical_parcels_give_zero_difference(self):
        # 2 m air equal to LPL air and LPL at the ground: both LCLs must coincide
        f, x, raw, mask = grid(sp_hpa=975.0)
        res = L.summarise(f, x, mask, LEVELS, raw, 0.2, g.convective)
        self.assertGreater(res["s_unstable"][0], 0)
        self.assertEqual(res["s_sumabs_diff"][0], 0.0)
        self.assertEqual(res["s_drop_lpl_only"][0] + res["s_drop_2m_only"][0], 0)
        self.assertAlmostEqual(L.pct(res["h_dz"], L.DZ_EDGES, 50), 0.0, delta=10.0)
        for k in ("lcl_below_own_start", "p_z_order_disagree", "lpl_below_ground", "nonfinite_p_lcl"):
            self.assertEqual(res[f"c_{k}"][0], 0, k)

    def test_warmer_2m_air_has_higher_own_level_lcl(self):
        # same moisture but 1.5 K warmer at 2 m -> lower RH_2m -> higher LCL above its own level
        f, x, raw, mask = grid(t2m_offset=1.5)
        res = L.summarise(f, x, mask, LEVELS, raw, 0.2, g.convective)
        self.assertLess(L.pct(res["h_own"], L.DZ_EDGES, 50), 0.0)
        self.assertEqual(res["c_rh_lpl_gt_rh_2m"][0], res["n_valid"][0])

    def test_additive_and_reportable(self):
        f, x, raw, mask = grid(t2m_offset=1.0)
        a = L.summarise(f, x, mask, LEVELS, raw, 0.2, g.convective)
        with tempfile.TemporaryDirectory() as d:
            for ev, yr in (("T", 1980), ("M", 1980), ("T", 1981)):
                np.savez_compressed(Path(d) / f"lcl_{ev}_{yr}.npz", **a)
            L.report(d, quiet=True)
            text = (Path(d) / "lcl_report.md").read_text()
        self.assertIn("INSUFFICIENT DATA", text)       # 48 synthetic events: no bin reaches n = 500
        self.assertIn("T+M", text)

    def test_binned_means_use_the_same_events(self):
        f, x, raw, mask = grid(t2m_offset=1.0)
        res = L.summarise(f, x, mask, LEVELS, raw, 0.2, g.convective)
        self.assertEqual(res["b_rh_n"].sum(), res["s_both"][0])
        self.assertEqual(res["b_zl_n"].sum(), res["s_both"][0])


if __name__ == "__main__":
    unittest.main()
