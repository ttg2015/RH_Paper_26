"""Corrections applied to legacy outputs so they can be adopted instead of rebuilt."""


def z_lcl_above_ground_legacy(z_lcl_stored, zs, z_base=None):
    """Convert a legacy d_Add_Derived z_LCL (E07) to height above ground [m].

    Legacy files store zs + dz for both parcels, where zs is surface elevation [m].
    - 2 m parcel: pass z_base=None; result is z_lcl_stored - zs (exact).
    - LPL parcel: pass z_base = height of the LPL above sea level [m]
      (Z_LPL geopotential / g, check its units first); result is
      dz + (z_base - zs), i.e. measured from the ground, not from the LPL.
    """
    dz = z_lcl_stored - zs
    if z_base is None:
        return dz
    return dz + (z_base - zs)
