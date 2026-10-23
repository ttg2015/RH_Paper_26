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
        job = expand(["d"], years=(1985, 1985), events=["T"])[0]
        with tempfile.TemporaryDirectory() as tmp:
            written, missing = write_all([job], tmp)
            text = written[0].read_text()
        self.assertIn("Extreme\\ Heat/Paper26_Cleaned/Scripts/d_Add_Derived.py 1985 T --buffer 6", text)
        self.assertIn("#PBS -A UNYU0020", text)
        self.assertIn("=== DONE d_T_b06_1985", text)

    def test_unbuilt_stage_is_skipped(self):
        jobs = expand(["f"], years=(1985, 1985))
        with tempfile.TemporaryDirectory() as tmp:
            written, missing = write_all(jobs, tmp)
        self.assertEqual((len(written), len(missing)), (0, 2))


if __name__ == "__main__":
    unittest.main()
