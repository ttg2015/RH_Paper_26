# File map

Where the outputs are, and where the SAM engine is. Written 2026-10-23.

Local paths (on a Mac) were listed directly. Cluster paths (GLADE, Torch) were read from config files and notebooks, not checked on the clusters, so confirm each with `ls` before relying on it.

## Three repos

| Repo | What | Local folder (author's Mac) |
|---|---|---|
| `RH_Paper_26` (this repo) | Paper 2026 ERA5 pipeline and launch bay | `Paper_2026/paper26/` |
| `SAM_HEAT` | Clean SAM launcher and analysis for Torch | `SAM_HEAT/` |
| `SAM_CODE_MESSY` | Earlier SAM notebooks, in the order the work happened (private) | `SAM_CODE/` |

Not in any repo: the SAM Fortran source, the legacy stage a–e scripts, and the `pipeline-map` notes (see [SETUP_AND_DATA.md](SETUP_AND_DATA.md)).

---

## 1. The SAM engine

SAM is Marat Khairoutdinov's model. The FWRCE build is Robert van der Drift's modified version (adds `r_stomata` land resistance and SST relaxation), patched here to build on Torch.

### Use these first (`SAM_HEAT` repo)

| What | Where | Notes |
|---|---|---|
| **Driver (launch, advance, checks)** | `samheat/` | The clean, tested replacement for the `SAM_CODE_MESSY` notebooks. Start with its `README.md`. |
| Paths, modules, Slurm account | `samheat/paths.py` | The only file to edit for a new user or site. |
| Namelist (prm) rendering | `samheat/presets.py`, `namelist.py` | The prm is generated and verified, never hand-edited. Unknown keys raise an error. |
| Stage logic (ocean, stage 1, stage 2) | `samheat/stages.py`, `advance.py`, `convergence.py` | `advance()` has hard stops (`max_submits`, `max_days`, `STOP` file). |
| Slurm interface | `samheat/slurm.py` | The only module that calls Slurm. A fake Slurm for tests is `tests/fake_hpc.py`. |
| Run staging and `OUT_*` layout | `samheat/staging.py` | Creates `OUT_2D OUT_3D OUT_MOMENTS OUT_MOVIES OUT_STAT RESTART`. |
| Case files (grid, sounding, reference prm) | `case/FWRCE/` | `grd` (64 levels), `snd_vdd300K`, `prm_upstream_reference`. |
| Notebooks to run in order | `notebooks/00_Build_and_Smoke.ipynb`, `01_Launch_Control.ipynb`, `02_Repro_vdD2025.ipynb` | Smoke test, launch control (edit the LEVERS cell), reproduction of van der Drift & O'Gorman (2025). |
| Heat-extreme analysis notebooks | `notebooks/10_Heat_Stats.ipynb` through `13_Explore.ipynb` | |

### Fortran source (the model itself; not in git)

| What | Where on the author's Mac |
|---|---|
| Patched FWRCE source tree | `SAM_CODE/model_source/SAM_FWRCE_build/SAM_SRC/` |
| Patched build files | `.../SAM_FWRCE_build/Build`, `Makefile`. Stock originals: `Build.orig`, and `SAM_CODE_MESSY/02_build_fwrce_model/stock_sam_6.11.8/` |
| Land and ocean patches | `.../SAM_SRC/simple_ocean.f90`, `.../SAM_SRC/SLM/soil_proc.f90` |
| Domain decomposition | `.../SAM_SRC/domain.f90`. `nsubdomains_x * nsubdomains_y` must equal `tasks_per_node` (32 = 4 x 8). |
| How the build was done | `SAM_CODE_MESSY/02_build_fwrce_model/Modified_SAM_Build.ipynb`. Intel ifx and OpenMPI, four patches applied by notebook. |
| Original van der Drift source and analysis | `SAM_HEAT/exports/vdd_zenodo/` (`SAM_SRC`, `SAM_RUN_SCRIPTS`, `SAM_OUTPUT_SCRIPTS`, `ANALYSIS`, his `his_*.json/.npy` reference numbers). Not in git. |
| Executable name | `SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM` |

### Superseded (history only)

`SAM_CODE_MESSY/04_land_rv_sweep/Land_RV_Sweep_v3.ipynb` produced the 12-run v4 dataset. Earlier versions v1 and v2 had bugs, listed in that repo's `README.md`. `08_wind_sweep/` is the wind-sweep launcher.

---

## 2. Where the outputs are

### 2a. SAM runs on NYU Torch (Slurm)

Model output stays on Torch and is not in git.

| What | Where (from `paths.py` and the notebooks) |
|---|---|
| Torch root | `/scratch/ttg2015/Torch/ExtremeHeat/SCM/` |
| Build and executable | `.../SCM/SAM_FWRCE_build/` |
| Stock SAM and converters (`stat2nc`, `bin3D2nc`, `2Dbin2nc`) | `.../SCM/SAM6.11.8/UTIL/` |
| Run data and radiation files | `.../SCM/SAM_Test/` |
| **New SAM_HEAT experiments** | `.../SCM/SAM_HEAT_runs/<experiment name>/`. Each has `OUT_2D OUT_3D OUT_STAT RESTART`, `config.json` (all settings plus git hash) and `ledger.json` (every action). |
| Ocean reference run (older) | `.../SCM/FWRCE_ocean_300K_002/` |
| Reference sounding (older) | `.../SCM/reference_profiles/snd_300K` |
| v4 land sweep runs (older) | `.../SCM/iter_300K_RV<r_v>_v4/`, with `.../SCM/land_sweep_state_v4.json` as the sweep tracker |
| van der Drift upstream scripts | `.../SCM/precip_extremes_RH_RCE/` |

### 2b. SAM results kept on the author's Mac (not in git)

| What | Where |
|---|---|
| Mixed-layer top by 15 methods (v4) | `SAM_CODE/05_analysis_mixed_layer/ml_top_by_method_v4.csv` |
| MSE-deficit decomposition (v4) | `SAM_CODE/05_analysis_mixed_layer/mse_decomposition_v4_buoy_min_flux10_top.csv` |
| Surface and 2 m summary (v4) | `SAM_CODE/surface_2m_summary_v4.csv` |
| Written reports | `SAM_HEAT/exports/` (`rv_spectrum_report.md`, `blh_rh_reasoning.md`, `cloud_precip_definitions.md`, `cloud_3d_visualization.md`, `mean_wind_literature.md`) |

### 2c. Paper 2026 ERA5 pipeline on NCAR GLADE

From [`config/paths.yaml`](../config/paths.yaml). These are the files to move for isentropic work.

| What | Where |
|---|---|
| Output root | `/glade/derecho/scratch/tylerg/ExtremeHeat/Paper26_Cleaned/` |
| Surface fields (stage a) | `.../T2m/`, `Q2m/`, `SP/`, `MSE2m/` |
| Annual maxima (stage b) | `.../AnnMax/T2m/`, `AnnMax/MSE2m/` |
| Profiles (stages c0, c1, c2) | `.../Profiles/` (`intermediate/`, `masks/`) |
| Derived fields: `MSE_LPL`, `RH_2m`, `z_LCL`, `p_LCL` (stage d) | `.../Profiles/derived/` |
| RH-binned data (stage e) | `.../rh_binned/` |
| Convective-top results (stage f) | `.../Convective/`, `.../Ztop/` |
| Replaced outputs | `<folder>/_old/<date>_<code>/`, listed in `_old/ARCHIVE_LOG.md` |
| PBS scripts and logs | `.../pbs/`, `.../logs/` |
| ERA5 inputs (read-only) | `/glade/campaign/collections/rda/data/d633000/` (pressure levels, surface) and `d633006/` (model levels) |

Stage-by-stage file naming is in `pipeline-map/Stages/` (not in git). For example, stage d writes `derived_{...}_b06_{YEAR}`.
