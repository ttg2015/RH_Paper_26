# Side tests (paper-channel test plan, 2026-10-28)

Read-only checks on the pipeline outputs. Nothing here is in a stage fingerprint (only `src/`,
`config/analysis.yaml` and `pipeline/` are), so running or editing these never makes a stage stale.
Outputs go to `<work>/side_tests/` on scratch.

| Task | Script | Needs |
|---|---|---|
| 0 | unit tests + `launch status` (in the PBS job) | — |
| 1 LCL of LPL vs 2 m air | `lcl_compare.py` (reviewed copy of the side-draft one) | stage d, g code |
| 2 paper numbers from figure data | `t2_figdata_numbers.py` | stage g files |
| draft | every bold placeholder in Draft 3: `draft_numbers.py` | stage g, Tasks 1, 3, 4 outputs |
| pooled h*500 vs RH (Section 7 assumption) | `pooled_hstar.py` | stage d files, stage g files |
| profile measurable for K (Section 7) | `cascade_composites.py` + `run_cascade_composites.pbs`, then `cascade_tests.ipynb` | stage d files |
| 3 stage f categories vs δ_a | `t345_events.py` | stage f rerun with stable / diluted-out codes |
| 4 η and closure, h* uniformity | `t345_events.py` | stage f files |
| 5 per-box cascade slope Γ_h | `t345_events.py` | stage d files |
| 6 alternative mixed layers, terrain, 400/600 hPa, plume null (2026-11-02 review) | `review_tests.py` + `run_review_tests.pbs` | stage d files |
| blue numbers the figure facts don't give | `blue_checks.py` (login node) | stage g files |
| h_ML spread inside the null cells; latitude and h*500 per RH bin | `cell_spread.py` (login node or qcmd) | profiles + derived, one level |
| 7 model vs pressure levels | `t7_model_levels.py` **parked: not in the job, never run** | — |
| surface jump: h_2m vs lowest model level, and Task 7 on the same events (2026-11-10) | `surface_jump.py` (`plan` on login node) + `run_surface_jump.pbs` | stages b, c2, d; RDA d633006 model levels |

## Commands (on Casper, repo root)

After `python -m launch run --stages f g --years 1980-2019` has finished:

    qsub side_tests/run_side_tests.pbs

Then open `side_tests/side_tests.ipynb` on JupyterHub (it reads these outputs; `RUN = True` recomputes chosen years), or read
`/glade/derecho/scratch/tylerg/ExtremeHeat/Paper26_Cleaned/side_tests/{t0_status.txt, lcl/lcl_report.md, t2_report.md, t345/t345_report.md}`.

Task 2 alone is quick and can run on the login node: `python side_tests/t2_figdata_numbers.py`.
