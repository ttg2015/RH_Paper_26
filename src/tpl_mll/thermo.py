"""Thermodynamics shared by every stage and figure.

Ported from Paper26_Cleaned/Scripts/z_Thermodynamics.py. Changes from the legacy module:
- one set of constants (cp = 1.004709, Lv = 2500.8 kJ/kg), read from config/analysis.yaml (E13)
- Romps c_vv = 1418 J/(kg K) as in Romps (2017); legacy used 1318
- LCL height is returned above the parcel's own level; callers add a base height
  explicitly, so "above ground" vs "above sea level" can't be mixed up (E07)
"""
import numpy as np
from scipy.special import lambertw

from .config import load

_C = load("analysis")["constants"]
cp = _C["cp"]      # kJ/(kg K)
Lv = _C["Lv"]      # kJ/kg
g = _C["g"]        # m/s^2
Rd = _C["Rd"]      # kJ/(kg K)
eps = _C["eps"]    # Rd/Rv


# --- Moisture -----------------------------------------------------------------
def e_sat(T):
    """Saturation vapour pressure over water [Pa] (IFS/Buck form). T in K."""
    return 611.21 * np.exp(17.502 * (T - 273.16) / (T - 32.19))


def e2q(e, p):
    """Specific humidity [kg/kg] from vapour pressure e and pressure p (same units)."""
    return eps * e / (p - (1 - eps) * e)


def q_sat(T, p):
    """Saturation specific humidity [kg/kg]. T in K, p in Pa."""
    return e2q(e_sat(T), p)


def q_from_dewpoint(Td, p):
    """Specific humidity [kg/kg] from dew point [K] and pressure [Pa]."""
    return e2q(e_sat(Td), p)


def rh(q, T, p):
    """Relative humidity as q / q_sat (the legacy pipeline definition). p in Pa."""
    return q / q_sat(T, p)


# --- Moist static energy (kJ/kg) ----------------------------------------------
def mse(T, q, phi):
    """h = cp T + Lv q + Φ. T in K, q in kg/kg, Φ geopotential in kJ/kg."""
    return cp * T + Lv * q + phi


def mse_sat(T, p, phi):
    """h* = cp T + Lv q_sat + Φ. p in Pa, Φ in kJ/kg."""
    return cp * T + Lv * q_sat(T, p) + phi


# --- Romps (2017) exact LCL, SI units ------------------------------------------
_R_cpm = 1005.0     # J/(kg K)
_R_Rd = 287.0       # J/(kg K)
_R_g = 9.8          # m/s^2
_R_E0v = 2.3740e6   # J/kg
_R_cvv = 1418.0     # J/(kg K)
_R_cvl = 4119.0     # J/(kg K)
_R_rv = 461.0       # J/(kg K)
_R_T3 = 273.16      # K
_R_cpv = _R_cvv + _R_rv
_R_a = _R_cpm / _R_Rd + (_R_cvl - _R_cpv) / _R_rv


def _romps_c(T):
    b = -(_R_E0v - (_R_cvv - _R_cvl) * _R_T3) / (_R_rv * T)
    return b / _R_a


def T_lcl(T, RH):
    """LCL temperature [K] of a parcel with temperature T [K] and RH in (0, 1]."""
    RH = np.clip(RH, 1e-3, 1.0)
    c = _romps_c(T)
    w = lambertw(RH ** (1.0 / _R_a) * c * np.exp(c), k=-1)
    return np.real(c * T / w)


def p_lcl(T, RH, p):
    """LCL pressure, same units as the parcel pressure p."""
    return p * (T_lcl(T, RH) / T) ** (_R_cpm / _R_Rd)


def dz_lcl(T, RH):
    """LCL height [m] above the parcel's own level."""
    return _R_cpm / _R_g * (T - T_lcl(T, RH))


def z_lcl(T, RH, z_base):
    """LCL height [m] = z_base + dz_lcl. Pass the parcel's height in the frame you want:
    0 for above the parcel, z_LPL - zs for above ground, z_LPL for above sea level."""
    return z_base + dz_lcl(T, RH)
