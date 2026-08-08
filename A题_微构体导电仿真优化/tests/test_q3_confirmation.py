from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from microstructure_sim import SimulationConfig  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


q3_solver = _load_module(
    "q3_confirmation_solver", ROOT / "问题" / "问题3" / "src" / "solve.py"
)


def _write_threshold_artifact(
    path: Path, config: SimulationConfig, samples: list[int]
) -> None:
    payload = {
        "kind": "microstructure_threshold_samples",
        "configuration_fingerprint": config.fingerprint,
        "configuration": config.to_dict(),
        "records": [
            {"trial_id": trial_id, "first_connection_index": value}
            for trial_id, value in enumerate(samples)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class Q3ConfirmationDecisionTests(unittest.TestCase):
    def test_bonferroni_cp_confirms_adjacent_pair(self) -> None:
        result = q3_solver.analyze_confirmation_samples(
            np.full(1000, 3, dtype=np.int64),
            candidates=[2, 3, 4],
            max_count=4,
        )
        self.assertEqual(result["bonferroni_statement_count"], 6)
        self.assertAlmostEqual(
            result["per_bound_confidence"], 1.0 - 0.05 / 6.0
        )
        self.assertEqual(result["decision"]["result_status"], "confirmed_minimum")
        self.assertEqual(result["decision"]["confirmed_minimum_integer"], 3)
        self.assertEqual(result["decision"]["confirmed_predecessor_integer"], 2)

    def test_unresolved_predecessor_prevents_minimum_claim(self) -> None:
        samples = np.asarray([3] * 900 + [4] * 100, dtype=np.int64)
        result = q3_solver.analyze_confirmation_samples(
            samples,
            candidates=[2, 3, 4],
            max_count=4,
        )
        self.assertEqual(
            result["decision"]["result_status"], "minimum_not_confirmed"
        )
        self.assertIsNone(result["decision"]["confirmed_minimum_integer"])
        self.assertEqual(
            result["decision"]["lowest_statistically_feasible_integer"], 4
        )
        self.assertEqual(result["decision"]["unresolved_integer_interval"], [3, 3])

    def test_predecessor_must_be_in_frozen_set(self) -> None:
        result = q3_solver.analyze_confirmation_samples(
            np.full(1000, 3, dtype=np.int64),
            candidates=[3],
            max_count=3,
        )
        self.assertEqual(
            result["decision"]["result_status"], "minimum_not_confirmed"
        )
        self.assertEqual(
            result["decision"]["lowest_statistically_feasible_integer"], 3
        )

    def test_requested_precision_can_be_confirmed_without_unique_integer(self) -> None:
        samples = np.asarray([612] * 500 + [614] * 500, dtype=np.int64)
        result = q3_solver.analyze_confirmation_samples(
            samples,
            candidates=[611, 612, 613, 614],
            max_count=614,
            target=0.5,
        )
        decision = result["decision"]
        self.assertEqual(decision["result_status"], "minimum_not_confirmed")
        self.assertEqual(decision["minimum_integer_bracket"], [612, 614])
        self.assertTrue(decision["reported_precision_confirmed"])
        self.assertEqual(decision["reported_volume_fraction_formatted"], "0.87%")


class Q3ConfirmationFreezeTests(unittest.TestCase):
    def test_freeze_records_independent_stream_and_bonferroni_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exploration.json"
            config = SimulationConfig(
                max_count=6,
                trial_count=5,
                boundary_mode="D",
                master_seed=31,
                stream_id=2,
            )
            _write_threshold_artifact(path, config, [2, 3, 3, 4, 7])
            freeze, confirmation = q3_solver.build_confirmation_freeze(
                path,
                target=0.6,
                candidate_radius=1,
                confirmation_trials=7,
                confirmation_stream_id=3,
            )
            self.assertEqual(freeze["candidate_freeze"]["candidates"], [2, 3, 4])
            self.assertEqual(
                freeze["candidate_freeze"]["exploration_empirical_threshold"], 3
            )
            self.assertEqual(
                freeze["confirmation_protocol"]["bonferroni_statement_count"], 6
            )
            self.assertTrue(freeze["confirmation_protocol"]["stream_ids_distinct"])
            self.assertEqual(confirmation.stream_id, 3)
            self.assertEqual(confirmation.trial_count, 7)
            self.assertEqual(confirmation.max_count, 4)
            self.assertEqual(confirmation.boundary_mode.value, "D")

            with self.assertRaisesRegex(ValueError, "stream_id"):
                q3_solver.build_confirmation_freeze(
                    path,
                    target=0.6,
                    confirmation_trials=7,
                    confirmation_stream_id=2,
                )

    def test_freeze_rejects_non_primary_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "exploration.json"
            config = SimulationConfig(
                max_count=3, trial_count=2, boundary_mode="A", stream_id=2
            )
            _write_threshold_artifact(path, config, [1, 2])
            with self.assertRaisesRegex(ValueError, "D 边界"):
                q3_solver.build_confirmation_freeze(
                    path, target=0.5, confirmation_trials=3
                )

    def test_existing_freeze_cannot_be_silently_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "freeze.json"
            q3_solver._write_or_validate_freeze(path, {"protocol": 1})
            q3_solver._write_or_validate_freeze(path, {"protocol": 1})
            with self.assertRaisesRegex(ValueError, "不一致"):
                q3_solver._write_or_validate_freeze(path, {"protocol": 2})

    def test_three_trial_run_is_resumable_and_uses_new_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            exploration_path = base / "q2" / "threshold_samples.json"
            exploration_config = SimulationConfig(
                max_count=2,
                trial_count=3,
                boundary_mode="D",
                master_seed=41,
                stream_id=2,
            )
            _write_threshold_artifact(exploration_path, exploration_config, [1, 2, 2])
            arguments = [
                "--threshold-artifact",
                str(exploration_path),
                "--output-dir",
                str(base / "q3"),
                "--target",
                "0.5",
                "--candidate-radius",
                "1",
                "--confirmation-trials",
                "3",
                "--confirmation-stream-id",
                "3",
                "--workers",
                "1",
                "--batch-size",
                "1",
            ]
            first = q3_solver.run(q3_solver.parse_args([*arguments, "--no-resume"]))
            second = q3_solver.run(q3_solver.parse_args(arguments))
            self.assertEqual(first["fixed_trial_count"], 3)
            self.assertEqual(first["confirmation_configuration"]["stream_id"], 3)
            self.assertNotEqual(
                first["exploration_configuration_fingerprint"],
                first["confirmation_configuration_fingerprint"],
            )
            self.assertEqual(first["confirmation_sha256"], second["confirmation_sha256"])
            self.assertEqual(first["candidate_records"], second["candidate_records"])
            self.assertTrue(Path(first["freeze_path"]).is_file())
            self.assertTrue(Path(first["summary_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
