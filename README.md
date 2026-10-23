# paper26

Code for Paper 2026: the surface energy excess during heat extremes. ERA5 goes in; the paper's figures come out.
The pipeline map, the error notes and the design plans are in the Obsidian vault at `../pipeline-map`.

## Layout
| Folder | What |
|---|---|
| `config/` | `paths.yaml` (GLADE locations), `analysis.yaml` (band, masks, constants, ε sweep), `hpc.yaml` (PBS) |
| `src/tpl_mll/` | Shared physics: `thermo.py` (q_sat, MSE, Romps LCL), `corrections.py` (fixes for legacy outputs) |
| `pipeline/` | Stages a → f, ported one at a time from `legacy/` |
| `launch/` | The launch bay: `python -m launch ...` |
| `figures/` | One notebook per paper figure, reading only pipeline outputs |
| `tests/` | `python3 -m unittest discover -s tests -t .` |
| `legacy/` | Copies of the old scripts (not in git) |

## Launch bay
```bash
python -m launch list                                         # stages and their status
python -m launch run --stages d e --years 1980-2019 --dry-run # write PBS files to runs/, submit nothing
```
Not built yet: `adopt` (check existing outputs), preflight, archive to `_old/`, submit with `afterok`, `status`, `dashboard.ipynb`.
Until a stage is ported, the launch bay runs its legacy script on GLADE with the old launcher's arguments.

## Setup on NCAR
```bash
module load conda && conda activate npl
pip install -e . --no-deps
```
