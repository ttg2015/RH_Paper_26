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
| `docs/` | `FILE_MAP.md` (where outputs and the SAM engine are) and `SETUP_AND_DATA.md` (setup for a collaborator, what is not in git) |
| `figures/` | One notebook per paper figure, reading only pipeline outputs |
| `tests/` | `python3 -m unittest discover -s tests -t .` |
| `legacy/` | Copies of the old scripts (not in git) |

## Launch bay
```bash
python -m launch list                                   # stages
python -m launch adopt --stages b c2 d e                # read-only check of existing outputs → runs/adopt_*/adopt_report.md
python -m launch adopt --demo                           # toy files with planted faults; shows what each check catches
python -m launch adopt --sample all --apply             # stamp files that pass, so they count as done
python -m launch run --stages a b c0 c1 c2 d e --dry-run  # plan + PBS files, nothing submitted
python -m launch run --stages d e                       # preflight → plan → y → archive + submit
python -m launch resume                                 # rerun whatever is not done
python -m launch status [--failed --tail 20] [--grid c1]  # progress board
python -m launch archive --list | --restore FILE        # what was replaced, and undo
```
Open `dashboard.ipynb` in JupyterHub for the same status with progress bars, a pipeline map, a stage × year grid and failed-job logs.

**How a run decides what to do.** Every output file carries `paper26_*` attributes saying which code fingerprint made it
(hash of `src/tpl_mll`, `config/` and the stage's script). Per job:

| State | Meaning | `run` does |
|---|---|---|
| done | outputs made or adopted by the current fingerprint | skip |
| queued / running | already in PBS | skip |
| missing / failed | no outputs, partial outputs or a log without `=== DONE` | build |
| stale | outputs from other code or config | move to `_old/`, rebuild |
| legacy | outputs with no attributes (made before the launch bay) | hold; run `adopt` first, or `--rebuild-legacy` |

A rebuilt job also rebuilds what is downstream of it in the same run. Jobs are chained with PBS `afterok`, so the whole
run is submitted at once. Nothing is overwritten or deleted: replaced files move to `<folder>/_old/<date>_<code>/` and are
logged in `_old/ARCHIVE_LOG.md`. `run` refuses uncommitted changes in `src/ pipeline/ config/ launch/` unless `--allow-dirty`.

Until a stage is ported to `pipeline/`, the launch bay runs its legacy script on GLADE with the old launcher's arguments.
Set `PAPER26_FAKE_PBS=<folder>` to try everything on a laptop with a fake scheduler.

## Setup on NCAR
```bash
module load conda && conda activate npl
pip install -e . --no-deps
```
