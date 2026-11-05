"""Does the zero-buoyancy entraining-plume closure of Romps (2016) predict δ_a in the cascade composites?

If the lower free troposphere is neutral to an entraining plume, the saturated entraining MSE
h* − f·L q* (f = a/(1+a), Romps 2016 Eq. 6) is constant with height above cloud base. A plume that starts
saturated at the LCL with the mixed-layer humidity then gives, with no fitting,

    δ_a = h_LPL − h*500 = f · (L q_ML − L q*500).

Two independent estimates of f per RH bin:
    f_excess   δ_a / (L q_ML − L q*500)                     from the excess
    f_profile  slope of h* − h*500 on L q* − L q*500        from the environment, LCL to 500 hPa
Romps' tropical value a = 0.2 gives f = 0.17 (ε ≈ 0.2 /km mid-troposphere). Agreement of the two f's, and
with 0.17, says the excess is what an entraining-plume-neutral column holds.

Reads   <work>/side_tests/casc/casc_{E}_{Y}.npz (cascade_composites.py)

    python side_tests/romps_check.py [--years 1980-2019] [--casc DIR] [--a 0.2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "figures"), str(REPO / "side_tests")]
import cascade_tests as T                                     # noqa: E402

CP, LV, G_KM = T.CP, T.LV, T.G_KM


def qsat(t_k, p_hpa):
    """Saturation specific humidity over liquid (Bolton 1980), kg/kg."""
    es = 6.112 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))
    return 0.622 * es / (p_hpa - 0.378 * es)


def lq_ml(h, z_km, p_hpa):
    """L q of mixed-layer air, from saturation at its LCL: solve h = cp T + g z + L q*(T, p) for T."""
    t = np.full_like(h, 290.0)
    for _ in range(50):
        f = CP * t + G_KM * z_km + LV * qsat(t, p_hpa) - h
        dq = (qsat(t + 0.01, p_hpa) - qsat(t, p_hpa)) / 0.01
        t = t - f / (CP + LV * dq)
    return LV * qsat(t, p_hpa)


def per_bin(c):
    lev = np.asarray(c["levels"], float)
    k5 = int(np.argmin(abs(lev - 500.0)))
    lq = c["MSE_sat"] - CP * c["T"] - c["Z"]                    # L q* on levels, kJ/kg
    lq500 = lq[:, k5]
    lqm = lq_ml(c["h_LPL"], c["zs_km"] + c["zLCL"], c["p_LCL_LPL"])
    x = lqm - lq500
    f_ex = c["delta_a"] / x
    f_pr = np.full(len(x), np.nan)
    for i in range(len(x)):
        sel = (lev < c["p_LCL_LPL"][i]) & (lev >= 500.0) & np.isfinite(c["MSE_sat"][i]) & np.isfinite(lq[i])
        if sel.sum() >= 3:
            dy, dx = c["MSE_sat"][i, sel] - c["MSE_sat"][i, k5], lq[i, sel] - lq500[i]
            f_pr[i] = (dx @ dy) / (dx @ dx)                       # slope through the 500 hPa point
    return x, f_ex, f_pr


def report(name, c, a, L):
    x, f_ex, f_pr = per_bin(c)
    f0 = a / (1 + a)
    ok = (c["n"] >= T.MIN_N) & np.isfinite(x)
    win = ok & (c["RH_2m"] >= T.WINDOW[0]) & (c["RH_2m"] <= T.WINDOW[1])
    L += [f"### {name}", "", "| RH | n | δ_a | L(q_ML − q*500) | pred (a={:g}) | f_excess | f_profile |".format(a),
          "|---|---|---|---|---|---|---|"]
    for i in np.nonzero(ok)[0]:
        L.append(f"| {c['RH_2m'][i]:.2f} | {int(c['n'][i]):,} | {c['delta_a'][i]:.1f} | {x[i]:.1f} | "
                 f"{f0 * x[i]:.1f} | {f_ex[i]:.2f} | {f_pr[i]:.2f} |")
    if win.sum() >= T.MIN_BINS:
        w = c["n"][win]
        fe = np.average(f_ex[win], weights=w)
        fp = np.nanmedian(f_pr[win])
        slope = np.polyfit(x[win], c["delta_a"][win], 1, w=np.sqrt(w))
        L += ["", f"Window {T.WINDOW}: f_excess {fe:.2f} (a = {fe / (1 - fe):.2f}), f_profile {fp:.2f} "
              f"(a = {fp / (1 - fp):.2f}), slope of δ_a on L(q_ML − q*500) {slope[0]:.2f} with intercept "
              f"{slope[1]:+.1f}; Romps a = {a:g} gives f = {f0:.2f}."]
    L.append("")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", default="1980-2019")
    ap.add_argument("--casc", help="folder of casc_*.npz (default <work>/side_tests/casc)")
    ap.add_argument("--a", type=float, default=0.2)
    args = ap.parse_args(argv)
    years = tuple(int(y) for y in args.years.split("-"))
    if args.casc:
        folder = Path(args.casc)
    else:
        import figdata as FD
        folder = Path(FD.load("paths")["work"]) / "side_tests" / "casc"
    A = T.load_all(folder, years=years)
    L = [f"# Romps (2016) closure check · {args.years}", ""]
    for e, a in A.items():
        names = [n for n in a["names"] if n.startswith(e + "_")] + ["all"]
        for n in names:
            report(f"{e} {n}", T.group(a, n), args.a, L)
    if "T" in A and "M" in A:
        report("pooled all", T.group(A["T"], "all", A["M"]), args.a, L)
    text = "\n".join(L) + "\n"
    (folder.parent / "romps_check.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
