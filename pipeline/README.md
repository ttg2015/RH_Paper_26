# Pipeline stages

| Stage | Legacy script | Ported | Notes |
|---|---|---|---|
| a surface | a_Surface_Fields.py | no | |
| b annmax | b_Ann_Max_Finder.py | no | |
| c0 mask | c0_build_mask.py | no | |
| c1 sweep | c1_sweep_month.py | no | |
| c2 stitch | c2_stitch_year.py | no | |
| d derived | d_Add_Derived.py | no | E07: use `thermo.z_lcl` with an explicit base height; E08: LPL with LPL |
| e rh_bins | e_RH_bins.py | no | E01 band, E12 boxes from config |
| f ztop | — (prototype: Paper26/Ztop_Distribution.ipynb) | **yes**: `f.py` | plume from LPL, ε sweep, both crossing rules, ε window; physics in `src/tpl_mll/plume.py`, `ztop.py` |
