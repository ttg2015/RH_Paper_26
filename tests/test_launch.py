import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import launch  # noqa: F401  (adds src/ to the path)
from launch.registry import expand
from launch.pbs import write_all


class TestLaunch(unittest.TestCase):
    def test_job_counts_match_old_launcher(self):
        jobs = expand(["a", "b", "c1", "d", "e"], years=(1980, 2019))
        n = lambda k: sum(j.stage.key == k for j in jobs)
        self.assertEqual((n("a"), n("b"), n("c1"), n("d"), n("e")), (480, 80, 960, 80, 40))

    def test_dry_run_matches_old_command(self):
        job = expand(["c2"], years=(1985, 1985), events=["T"])[0]     # c2 still runs the legacy script
        with tempfile.TemporaryDirectory() as tmp:
            written, missing = write_all([job], tmp)
            text = written[0].read_text()
        self.assertIn("Extreme\\ Heat/Paper26_Cleaned/Scripts/c2_stitch_year.py 1985 T --buffer 6", text)
        self.assertIn("#PBS -A UNYU0020", text)
        self.assertIn("=== DONE c2_T_b06_1985", text)

    def test_ported_d_runs_from_repo(self):
        job = expand(["d"], years=(1985, 1985), events=["T"])[0]
        self.assertTrue(job.command().endswith("pipeline/d.py 1985 T --buffer 6"))

    def test_ported_stage_runs_from_repo(self):
        job = expand(["f"], years=(1985, 1985), events=["M"])[0]
        self.assertTrue(job.command().endswith("pipeline/f.py 1985 M --buffer 6"))


if __name__ == "__main__":
    unittest.main()
