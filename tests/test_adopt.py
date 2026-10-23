import csv, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import launch  # noqa: F401  (adds src/ to the path)
from launch import adopt_demo


class TestAdopt(unittest.TestCase):
    def test_every_planted_fault_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = adopt_demo.run(Path(tmp))
            with open(report.with_suffix(".csv")) as f:
                rows = {(r["stage"], r["file"]): r for r in csv.DictReader(f)}
        for (stage, label), (what, cid) in adopt_demo.PLANTED.items():
            status = rows[(stage, label)][cid].split(":")[0]
            self.assertIn(status, ("fail", "warn"), f"{what} not caught by {cid}")
        self.assertEqual(rows[("b", "MSE2m 1980")]["verdict"], "adopt")
        self.assertEqual(rows[("e", "1981")]["verdict"], "adopt")


if __name__ == "__main__":
    unittest.main()
