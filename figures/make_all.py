"""Make every figure, then write runs/figs/index.html: one page with each figure and its facts.

    python figures/make_all.py                    # all years found, cos(lat) weighted
    python figures/make_all.py --years 1980-1989 --unweighted --out runs/figs_80s

Figures read only figure data (stage g FigData/, stage f Ztop/), so this takes seconds and
can be rerun freely while tweaking the artwork. A figure that fails is reported on the page
and the rest still get made.
"""
import argparse
import html
import importlib
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figdata as FD          # noqa: E402

FIGURES = ["fig_excess_hist", "fig_excess_vs_rh", "fig_excess_vs_rh_entrainment", "fig_cascade",
           "fig_excess_maps", "fig_excess_vs_zlcl",
           "fig_ztop_fractions"]


def git_hash():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True)
    return r.stdout.strip() or "nogit"


def page(items, failures, opts):
    cards = []
    for it in items:
        facts = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
                        for k, v in it["facts"].items())
        name = Path(it["path"]).name
        cards.append(f"""<section><h2>{html.escape(it['title'])}</h2>
<a href="{name}"><img src="{name}" alt="{html.escape(it['title'])}"></a>
<table>{facts}</table>
<p class="src">Made by <code>{it['script']}</code> · <a href="{Path(name).with_suffix('.pdf').name}">PDF</a></p></section>""")
    for mod, err in failures:
        cards.append(f"<section class='fail'><h2>{mod} failed</h2><pre>{html.escape(err)}</pre></section>")
    return f"""<!doctype html><meta charset="utf-8"><title>Paper26 figures</title>
<style>
:root{{--bg:#fff;--fg:#1f2933;--muted:#62727b;--card:#f6f8f9;--line:#dde3e7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111517;--fg:#e4e9ec;--muted:#93a1a8;--card:#1b2124;--line:#2c3539}}}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:1100px;margin:auto;padding:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:18px 0}}
img{{max-width:100%;background:#fff;border-radius:6px}} table{{border-collapse:collapse;font-size:13px;margin-top:8px}}
th{{text-align:left;color:var(--muted);font-weight:500;padding:2px 14px 2px 0}} .src{{color:var(--muted);font-size:13px}}
.fail{{border-color:#c62828}} pre{{white-space:pre-wrap;font-size:12px}}
</style>
<h1>Paper26 figures</h1>
<p class="src">Made {datetime.now():%Y-%m-%d %H:%M} · code {git_hash()} · years {opts['years']} ·
{'cos(lat) weighted' if opts['weighted'] else 'unweighted'}</p>
{''.join(cards)}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years")
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "figs"))
    ap.add_argument("--only", nargs="+", help="subset of figure modules")
    a = ap.parse_args()
    out = Path(a.out)
    items, failures = [], []
    for mod in a.only or FIGURES:
        try:
            m = importlib.import_module(mod)
            got = m.make(out, FD.years_arg(a.years), weighted=not a.unweighted)
            items += got
            print(f"✓ {mod}")
        except Exception:
            failures.append((mod, traceback.format_exc(limit=3)))
            print(f"✗ {mod}: {traceback.format_exc(limit=1).strip().splitlines()[-1]}")
    index = out / "index.html"
    index.write_text(page(items, failures, {"years": a.years or "all available", "weighted": not a.unweighted}))
    print(f"{len(items)} figures, {len(failures)} failed → {index}")


if __name__ == "__main__":
    main()
