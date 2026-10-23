"""Which stages exist, what they run, and how many jobs each one makes.

Until a stage is ported to pipeline/, its command points at the legacy script on GLADE,
with exactly the arguments the old !_Launcher used, so the PBS files can be diffed.
"""
from dataclasses import dataclass, field
from itertools import product

from tpl_mll.config import load


@dataclass(frozen=True)
class Stage:
    key: str
    name: str
    grain: tuple                 # subset of ("year", "month", "event")
    depends_on: tuple = ()
    legacy_cmd: str = None       # format string; None = not built yet
    port: str = None             # pipeline/ module once ported


STAGES = {s.key: s for s in [
    Stage("a", "surface", ("year", "month"),
          legacy_cmd="a_Surface_Fields.py {year} {month}"),
    Stage("b", "annmax", ("year", "event"), ("a",),
          legacy_cmd="b_Ann_Max_Finder.py {year} {event}"),
    Stage("c0", "mask", ("event",), ("b",),
          legacy_cmd="c0_build_mask.py {event} --buffer {buffer}"),
    Stage("c1", "sweep", ("year", "month", "event"), ("c0", "a"),
          legacy_cmd="c1_sweep_month.py {year} {month} {event} --buffer {buffer}"),
    Stage("c2", "stitch", ("year", "event"), ("c1",),
          legacy_cmd="c2_stitch_year.py {year} {event} --buffer {buffer}"),
    Stage("d", "derived", ("year", "event"), ("c2",),
          legacy_cmd="d_Add_Derived.py {year} {event} --buffer {buffer}"),
    Stage("e", "rh_bins", ("year",), ("d",),
          legacy_cmd="e_RH_bins.py {year} --buffer {buffer}"),
    Stage("f", "ztop", ("year", "event"), ("d",)),
]}


@dataclass(frozen=True)
class Job:
    stage: Stage
    year: int = None
    month: int = None
    event: str = None
    buffer: int = 6

    @property
    def tag(self):
        s = self.stage.key
        if self.event is not None:
            s += f"_{self.event}"
        s += f"_b{self.buffer:02d}"
        if self.year is not None:
            s += f"_{self.year}"
        if self.month is not None:
            s += f"_{self.month:02d}"
        return s

    def command(self, python="python"):
        paths = load("paths")
        if self.stage.legacy_cmd is None:
            return None
        script = paths["legacy_scripts"].replace(" ", "\\ ")
        args = self.stage.legacy_cmd.format(
            year=self.year, month=self.month, event=self.event, buffer=self.buffer)
        return f"{python} {script}/{args}"


def expand(stage_keys, years=None, events=None, months=range(1, 13)):
    """List every Job for the given stages, years and events."""
    cfg = load("analysis")
    y0, y1 = years or cfg["years"]
    events = events or cfg["events"]
    buffer = cfg["buffer_hours"]
    axes = {"year": range(y0, y1 + 1), "month": months, "event": events}
    jobs = []
    for key in stage_keys:
        stage = STAGES[key]
        dims = [d for d in ("year", "month", "event") if d in stage.grain]
        for combo in product(*(axes[d] for d in dims)):
            jobs.append(Job(stage, buffer=buffer, **dict(zip(dims, combo))))
    return jobs
