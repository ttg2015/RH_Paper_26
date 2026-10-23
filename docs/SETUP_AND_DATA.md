# Setup and data

For a collaborator who has their own account on NYU Torch and wants to run isentropic analysis. Written 2026-10-23.

Where each file lives is in [FILE_MAP.md](FILE_MAP.md).

## What is and is not in git

| Piece | In git? | What to do |
|---|---|---|
| `paper26` code (`src/`, `launch/`, `pipeline/`, `config/`, `tests/`) | Yes (this repo) | Clone it. |
| SAM driver (`samheat/`) | Yes, repo `SAM_HEAT` | Clone it. |
| SAM Fortran source (`SAM_SRC`, `Build`, `Makefile`) | **No**, gitignored on purpose | SAM is Marat Khairoutdinov's model and is not ours to redistribute. Get stock SAM 6.11.8 and van der Drift's FWRCE source from their authors, then ask Tyler for the four patches (`Build`, `Makefile`, `simple_ocean.f90`, `soil_proc.f90`). Committing them as `.patch` files is a to-do. |
| Stage a–e scripts | **No** (`legacy/` is gitignored) | Ask Tyler. Until they are ported to `pipeline/`, only stage f is reproducible from this repo. |
| Error notes (`pipeline-map/`, E01–E15) | **No** | Ask Tyler for a copy. The notes are plain markdown. Read `Error-Board.md` before trusting any stage a–e output. |
| Data (`*.nc`, `*.npy`, run output) | No, by design | See "Data" below. |

## Setup

### 1. This repo (`paper26`)

```bash
git clone <RH_Paper_26 url> paper26 && cd paper26
python -m unittest discover -s tests -t .        # should pass anywhere
```

The `launch` command talks to PBS, which is NCAR's scheduler. Torch uses Slurm. **A Slurm backend does not exist yet.** Until it does, on Torch you can:

- run tests and read the code,
- run the launch bay against the fake scheduler: `PAPER26_FAKE_PBS=<folder> python -m launch run --stages f --dry-run`,
- write your own isentropic stage and run it with `sbatch`.

Replace the site values in `config/paths.yaml` and `config/hpc.yaml` with your own scratch paths, project and queue. They currently hold the author's NCAR values.

### 2. SAM (`SAM_HEAT`)

Follow that repo's `README.md`. In short: clone it, run `git config core.hooksPath .githooks`, then edit `samheat/paths.py` (the only place for paths, modules and Slurm account) to point at your own build and scratch space, and run `notebooks/00_Build_and_Smoke.ipynb` first. It needs a built SAM executable, which needs the source (see the table above).

### 3. Python environment

Neither repo has a pinned environment file yet. `paper26/pyproject.toml` needs `numpy`, `scipy`, `xarray`, `netCDF4`, `pyyaml`. On NCAR the author uses `module load conda && conda activate npl`. On Torch, create your own environment with the same packages, then `pip install -e . --no-deps`.

### 4. Claude Code on Torch

Run Claude Code on a login node, or on your laptop over ssh. Login nodes usually have outbound internet and compute nodes usually don't. Do not run heavy work on the login node: submit it with `sbatch`. You need your own Anthropic account. Do not use anyone else's credentials or allocation.

Rules the launch bay already enforces, and that any session should follow:

- dry run first (`--dry-run`),
- nothing is overwritten or deleted: replaced files move to `_old/` and are logged,
- no uncommitted changes in `src/`, `pipeline/`, `config/` or `launch/` unless `--allow-dirty`.

## Data

Raw ERA5 stays on NCAR. Nothing here is tracked in git.

**Recommended route:** Tyler runs stages a–e on NCAR as usual. The intermediate products you need move to Torch with Globus (NCAR GLADE to Torch scratch). The isentropic analysis then becomes a new stage `g` that reads those files, so you do not need to port a–e or copy ERA5.

Candidate files to move, from `config/paths.yaml`:

- `Profiles/derived/` (`MSE_LPL`, `RH_2m`, `z_LCL`, `p_LCL`),
- `Profiles/` (the stitched year profiles),
- `rh_binned/`,
- `Ztop/` and `Convective/`, if you need stage f results.

**Not decided yet:** which variables, years and events the isentropic analysis needs (for example pressure-level T, q and geopotential, or surface pressure). That decides the exact transfer list. If you already have ERA5 on Torch, you can build your own inputs instead. The NCAR RDA ERA5 collection (`d633000`) can also be downloaded from RDA.

Isentropic analysis does not exist in this repo yet.
