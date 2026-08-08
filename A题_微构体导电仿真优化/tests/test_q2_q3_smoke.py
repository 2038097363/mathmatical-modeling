from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


q2_solver = _load_module("q2_solver", ROOT / "问题" / "问题2" / "src" / "solve.py")
q3_solver = _load_module("q3_solver", ROOT / "问题" / "问题3" / "src" / "solve.py")


class Q2Q3PipelineSmokeTests(unittest.TestCase):
    def test_q2_writes_a_reusable_threshold_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            q2_args = q2_solver.parse_args(
                [
                    "--output-dir",
                    str(base / "q2"),
                    "--boundary-mode",
                    "A",
                    "--max-count",
                    "2",
                    "--trials",
                    "3",
                    "--counts",
                    "1",
                    "2",
                    "--seed",
                    "31",
                    "--workers",
                    "1",
                    "--batch-size",
                    "1",
                    "--no-resume",
                ]
            )
            q2_result = q2_solver.run(q2_args)
            artifact = Path(q2_result["threshold_artifact"])
            self.assertTrue(artifact.is_file())
            self.assertEqual(len(q2_result["probability_records"]), 2)

    def test_q3_defaults_are_formal_but_tests_can_override_trial_count(self) -> None:
        defaults = q3_solver.parse_args([])
        self.assertEqual(defaults.confirmation_trials, 50_000)
        self.assertEqual(defaults.output_dir, ROOT / "问题" / "问题3" / "results")
        self.assertNotIn("results-smoke", str(defaults.threshold_artifact))
        smoke = q3_solver.parse_args(["--confirmation-trials", "3"])
        self.assertEqual(smoke.confirmation_trials, 3)

    def test_formal_sources_do_not_import_prototypes(self) -> None:
        sources = [
            ROOT / "公共代码" / "microstructure_sim.py",
            ROOT / "问题" / "问题2" / "src" / "solve.py",
            ROOT / "问题" / "问题3" / "src" / "solve.py",
        ]
        for source in sources:
            text = source.read_text(encoding="utf-8")
            self.assertNotIn("boundary_sim", text)
            self.assertNotIn("exact_sim", text)
            self.assertNotIn("periodic_distance_bounds(", text)


if __name__ == "__main__":
    unittest.main()
