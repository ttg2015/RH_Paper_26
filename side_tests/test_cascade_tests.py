"""Known answers for cascade_tests.py: composites built from a profile with set slopes."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade_tests as T

LEV = np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300.])
Z_LEV = 7.4 * np.log(1000.0 / LEV)                            # km above a surface at 1000 hPa
RH = 0.025 + 0.05 * np.arange(21)
C_SET, GH, GBL, H_KINK0, Z0 = 2.0, -8.0, -1.0, 340.0, 1.0


def p_of_z(z):
    return 1000.0 * np.exp(-z / 7.4)


def make(gh=GH, gbl=GBL, shift=0.0):
    """Sliding picture: fixed sickle, kink exactly at the LCL, tilted layer of slope gbl below it."""
    n = 2000.0 * np.ones(21)
    c = {"n": n, "levels": LEV, "rh_centers": RH, "RH_2m": RH, "zs_km": np.zeros(21), "h500": np.full(21, 320.0),
         "h_star500": np.full(21, 330.0)}
    zl = np.clip(Z0 - C_SET * np.log(RH / 0.45), 0.3, 4.5)
    c["zLCL"] = zl
    c["zLCL_LPL"] = zl * 1000.0
    c["p_LCL_LPL"] = p_of_z(zl)
    c["p_LPL"] = np.full(21, 1000.0)
    c["z"] = np.tile(Z_LEV, (21, 1))
    h = np.zeros((21, len(LEV)))
    for i in range(21):
        s = shift * np.log(RH[i] / 0.45)                      # whole profile moves with RH at fixed pressure
        hk = H_KINK0 + gh * (zl[i] - Z0)
        zb = 4.0                                              # the break
        h[i] = np.where(Z_LEV < zl[i], hk + gbl * (Z_LEV - zl[i]), hk + gh * (np.minimum(Z_LEV, zb) - zl[i]) + 1.0 * np.maximum(Z_LEV - zb, 0)) + s
    c["MSE"] = h
    c["Z"] = Z_LEV[None] * T.G_KM * np.ones((21, 1))
    c["MSE_sat"] = np.full_like(h, 330.0)
    c["T"] = np.full_like(h, 290.0)
    c["h_LPL"] = h[:, 0]
    c["delta_a"] = c["h_LPL"] - c["h_star500"]
    return c


class Known(unittest.TestCase):
    def test_sliding_slope(self):
        """Fixed sickle, kink at the LCL: at-LCL and slide terms both give Γ_h·C, the shift is zero."""
        r = T.candidates(make())
        self.assertAlmostEqual(r["C"], C_SET, delta=0.02)
        self.assertAlmostEqual(r["K_at_LCL"], GH * C_SET, delta=0.5)
        self.assertAlmostEqual(r["K_slide"], GH * C_SET, delta=0.5)
        self.assertAlmostEqual(r["K_shift"], 0.0, delta=0.6)      # interpolating across the corner at the LCL

    def test_gap_and_sum(self):
        """K_hLPL = K_at_LCL + K_gap exactly, and a tilted layer shows up in the gap."""
        r = T.candidates(make())
        self.assertAlmostEqual(r["K_hLPL"], r["K_at_LCL"] + r["K_gap_LCL"], places=8)
        self.assertAlmostEqual(r["K_hLPL"], (GH - GBL) * C_SET, delta=0.2)   # constant tilt: B = A − Γ_BL
        self.assertAlmostEqual(r["K_tilt"], r["K_hLPL"], delta=2.0)

    def test_shift_detected(self):
        r = T.candidates(make(shift=3.0))
        self.assertAlmostEqual(r["K_shift"], -3.0, delta=0.6)      # K = −d/d ln RH

    def test_three_segments(self):
        m = T.measures(make())
        i = np.argmin(abs(RH - 0.425))
        self.assertAlmostEqual(m["G_h"][i], GH, delta=1.5)
        self.assertAlmostEqual(m["G_bl"][i], GBL, delta=1.5)

    def test_plume(self):
        """Uniform environment: h_c relaxes as exp(−ε z); ε_required inverts it."""
        z = Z_LEV.copy()
        he = np.full_like(z, 320.0)
        got = T.plume_top(LEV, z, he, 340.0, 1000.0, 0.2)
        self.assertAlmostEqual(got, 320.0 + 20.0 * np.exp(-0.2 * 7.4 * np.log(2)), places=3)
        self.assertAlmostEqual(T.eps_required(LEV, z, he, 340.0, 1000.0, got), 0.2, places=3)

    def test_closed_form_matches_plume(self):
        """h_c = (1 − f) h0 + f h̄_e reproduces the marched plume, with f = 1 − exp(−εL), on a real-shaped profile."""
        c = make()
        i = int(np.argmin(abs(RH - 0.425)))
        for eps in (0.05, 0.2, 0.4):
            r = T.closed_form(LEV, c["z"][i], c["MSE"][i], 1000.0, eps, c["p_LCL_LPL"][i], 330.0)
            march = T.plume_top(LEV, c["z"][i], c["MSE"][i], 340.0, 1000.0, eps)
            self.assertAlmostEqual(march, (1 - r["f"]) * 340.0 + r["f"] * r["hbar"], places=8)
            self.assertAlmostEqual(r["f"], 1 - np.exp(-eps * r["L"]), places=8)
            self.assertAlmostEqual(r["below"] + r["above"], r["f"] * (330.0 - r["hbar"]), places=8)

    def test_closed_form_inverts_eps_req(self):
        """At ε = ε_req the formula returns δ_a exactly: the algebra, not a result."""
        c = make()
        i = int(np.argmin(abs(RH - 0.625)))
        h0, hs = c["h_LPL"][i], 330.0
        eps = T.eps_required(LEV, c["z"][i], c["MSE"][i], h0, 1000.0, hs)
        r = T.closed_form(LEV, c["z"][i], c["MSE"][i], 1000.0, eps)
        self.assertAlmostEqual(r["f"] / (1 - r["f"]) * (hs - r["hbar"]), h0 - hs, places=4)

    def test_closed_candidates_run(self):
        r = T.closed_candidates(make())
        self.assertTrue(np.isfinite(r["K_closed_0.2"]))
        self.assertLess(abs(r["K_below_0.2"]), 30.0)                # small parts, not the raw energy


if __name__ == "__main__":
    unittest.main()
