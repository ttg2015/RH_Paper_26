"""End to end on the toy tree from adopt_demo, with the fake PBS scheduler."""
import json, os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netCDF4
import numpy as np

import launch  # noqa: F401  (adds src/ to the path)
from launch import adopt, adopt_demo, archive, preflight, provenance, state, status

YEARS = (1980, 1981)


class TestLaunchFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.work, self.logs, self.inv = adopt_demo.build(root, np.random.default_rng(1))
        self.where = dict(work=self.work, log_dir=self.logs)
        os.environ["PAPER26_FAKE_PBS"] = str(root / "pbs")
        self.root = root

    def tearDown(self):
        os.environ.pop("PAPER26_FAKE_PBS")
        self.tmp.cleanup()

    def states(self):
        return {s.job.tag: s for s in state.collect(["b", "d", "e"], YEARS, **self.where)}

    def adopt(self):
        adopt.run(["b", "d", "e"], YEARS, ["T", "M"], self.root / "report", sample=None,
                  apply=True, zs_file=self.inv, script_dir=self.root / "none", **self.where)

    def test_legacy_until_adopted_then_done(self):
        st = self.states()
        self.assertEqual(st["d_M_b06_1981"].state, "missing")
        self.assertEqual(st["b_T_b06_1980"].state, "legacy")
        self.adopt()
        st = self.states()
        self.assertEqual(st["b_T_b06_1980"].state, "done")
        self.assertEqual(st["d_T_b06_1980"].state, "done")          # adopted with E07 note
        self.assertEqual(st["b_T_b06_1981"].state, "legacy")        # °C file not stamped
        self.assertEqual(st["e_b06_1981"].state, "done")
        self.assertIn("E07", provenance.read(st["d_T_b06_1980"].files[0])["note"])

    def test_stale_when_code_changes(self):
        self.adopt()
        f = self.states()["e_b06_1981"].files[0]
        with netCDF4.Dataset(f, "a") as nc:
            nc.setncattr("paper26_code", "oldcode")
        self.assertEqual(self.states()["e_b06_1981"].state, "stale")

    def test_plan_holds_legacy_and_submit_archives_and_chains(self):
        self.adopt()
        plan = {s.job.tag: s for s in preflight.plan(["b", "d", "e"], YEARS, **self.where)}
        self.assertEqual(plan["d_M_b06_1981"].action, "build")
        self.assertEqual(plan["b_T_b06_1981"].action, "hold")
        self.assertEqual(plan["b_T_b06_1980"].action, "skip")

        states = preflight.plan(["b", "d", "e"], YEARS, rebuild_legacy=True, **self.where)
        old = {s.job.tag: s for s in states}["b_T_b06_1981"].files[0]
        m = preflight.execute(states, self.root / "run")
        self.assertFalse(old.exists())                               # moved, not deleted
        archived = archive.listing([self.work])
        self.assertIn(old.name, [p.name for p, _ in archived])
        self.assertIn("b_T_b06_1981", m["jobs"])
        queue = json.loads((self.root / "pbs" / "queue.json").read_text())
        by_name = {j["name"]: j for j in queue.values()}
        e_after = set(by_name["e_b06_1981"]["after"])                # e waits for d of its year
        self.assertTrue({m["jobs"][t] for t in m["jobs"] if t.startswith("d_") and t.endswith("1981")} <= e_after)
        # queued jobs show as queued; done ones are left alone
        st = self.states()
        self.assertEqual(st["b_T_b06_1981"].state, "queued")
        self.assertEqual(st["b_T_b06_1980"].state, "done")

    def test_restore_brings_file_back(self):
        f = self.states()["e_b06_1980"].files[0]
        moved = archive.archive(f, "test")
        self.assertFalse(f.exists())
        archive.restore(moved)
        self.assertTrue(f.exists())

    def test_status_outputs(self):
        self.adopt()
        data = status.summary(state.collect(["b", "d", "e"], YEARS, **self.where), "abc")
        self.assertEqual(data["stages"]["b"]["done"], 2)
        out = status.write(data, self.root / "status")
        self.assertTrue((out.parent / "Status-Map.md").exists())
        self.assertIn("b annmax", status.board(data))
        self.assertIn("d --> e", status.mermaid(data))
        self.assertIn("1981", status.year_grid(data, "d"))
        import matplotlib; matplotlib.use("Agg")
        from launch import dashboard
        dashboard.draw_map(data)
        dashboard.heatmap(data, "d")


if __name__ == "__main__":
    unittest.main()
