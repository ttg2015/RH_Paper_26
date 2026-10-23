import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from tpl_mll import thermo
from tpl_mll.corrections import z_lcl_above_ground_legacy


class TestThermo(unittest.TestCase):
    def test_saturated_parcel_lcl_at_parcel(self):
        self.assertAlmostEqual(float(thermo.T_lcl(300.0, 1.0)), 300.0, places=3)
        self.assertAlmostEqual(float(thermo.dz_lcl(300.0, 1.0)), 0.0, places=1)

    def test_lcl_matches_rule_of_thumb(self):
        # Espy: z_LCL ≈ 125 m per K of dew-point depression
        T, Td, p = 303.15, 288.15, 1e5
        RH = thermo.e_sat(Td) / thermo.e_sat(T)
        dz = float(thermo.dz_lcl(T, RH))
        self.assertLess(abs(dz - 125 * (T - Td)) / dz, 0.08)

    def test_mse_magnitude(self):
        h = thermo.mse(300.0, 0.015, 0.0)
        self.assertAlmostEqual(h, 1.004709 * 300 + 2500.8 * 0.015, places=6)

    def test_q_sat_reasonable(self):
        self.assertAlmostEqual(float(thermo.q_sat(303.15, 1e5)), 0.0272, delta=0.001)

    def test_e07_correction(self):
        zs, z_lpl, dz = 300.0, 450.0, 1200.0
        self.assertEqual(z_lcl_above_ground_legacy(zs + dz, zs), dz)
        self.assertEqual(z_lcl_above_ground_legacy(zs + dz, zs, z_lpl), dz + 150.0)


if __name__ == "__main__":
    unittest.main()
