from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


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


q4_solver = _load_module(
    "q4_solver", ROOT / "问题" / "问题4" / "src" / "solve.py"
)


def _screening_manifest(
    candidate: tuple[int, int] = (1, 1),
    *,
    target: float = 0.8,
    family_seed: int = 31,
) -> dict:
    n_a, n_b = candidate
    config = q4_solver.MixedSimulationConfig(
        n_a=n_a,
        n_b=n_b,
        trial_count=2,
        master_seed=family_seed,
        stream_id=4,
    )
    record = {
        **q4_solver.design_metrics(n_a, n_b),
        "estimate": target,
        "configuration": config.to_dict(),
        "configuration_fingerprint": config.fingerprint,
    }
    return {
        "kind": "q4_screening_results",
        "target_probability": target,
        "fixed_trial_count": 2,
        "stream_id": 4,
        "maximum_static_graph_design": [n_a, n_b],
        "candidate_selection_rule": "test rule",
        "screening_candidate": record,
    }


def _build_freeze(
    directory: Path,
    *,
    candidate: tuple[int, int] = (1, 1),
    target: float = 0.8,
    familywise_confidence: float = 0.95,
) -> dict:
    screening_json = directory / "q4_screening.json"
    screening_csv = directory / "q4_screening.csv"
    pareto_artifact = directory / "mixed_pareto_frontier_samples.json"
    pareto_artifact.write_text("{}", encoding="utf-8")
    manifest = _screening_manifest(candidate, target=target)
    manifest["pareto_frontier_artifact"] = str(pareto_artifact.resolve())
    manifest["pareto_frontier_artifact_sha256"] = q4_solver._sha256(
        pareto_artifact
    )
    screening_json.write_text(json.dumps(manifest), encoding="utf-8")
    screening_csv.write_text("n_a,n_b\n", encoding="utf-8")
    return q4_solver.build_confirmation_freeze(
        manifest,
        screening_json_path=screening_json,
        screening_csv_path=screening_csv,
        confirmation_trials=7,
        confirmation_stream_id=5,
        familywise_confidence=familywise_confidence,
    )


def _confirmation_record(
    role: str,
    design: tuple[int, int],
    proof_status: str,
) -> dict:
    return {
        "role": role,
        **q4_solver.design_metrics(*design),
        "estimate": 0.5,
        "clopper_pearson_one_sided_upper": 0.7,
        "proof_status": proof_status,
    }


class Q4IntegerDomainTests(unittest.TestCase):
    def test_integer_domain_counts_match_frontier_queries(self) -> None:
        frontiers = [
            ((1, 3), (3, 1)),
            ((0, 4), (2, 2)),
            (),
        ]
        counts = q4_solver.integer_domain_success_counts(frontiers, 3, 4)
        self.assertEqual(counts.shape, (4, 5))
        for n_a in range(4):
            for n_b in range(5):
                expected = sum(
                    any(first <= n_a and second <= n_b for first, second in frontier)
                    for frontier in frontiers
                )
                self.assertEqual(int(counts[n_a, n_b]), expected)

    def test_full_domain_candidate_uses_exact_cost_order(self) -> None:
        counts = q4_solver.integer_domain_success_counts(
            [((0, 4), (1, 0))] * 9 + [()], 1, 4
        )
        selected = q4_solver.minimum_empirically_feasible_design(
            counts, trials=10, target=0.9
        )
        self.assertEqual(selected, (0, 4, 9))
        self.assertLess(
            q4_solver.cost_weight(0, 4), q4_solver.cost_weight(1, 0)
        )

    def test_cp_lower_candidate_rule_requires_screening_margin(self) -> None:
        counts = np.asarray(
            [[1800, 1822, 1823], [1823, 1823, 1823]], dtype=np.int32
        )
        point = q4_solver.minimum_screening_feasible_design(
            counts,
            trials=2000,
            target=0.9,
            confidence=0.95,
            rule="point_estimate",
        )
        conservative = q4_solver.minimum_screening_feasible_design(
            counts,
            trials=2000,
            target=0.9,
            confidence=0.95,
            rule="cp_lower",
        )
        self.assertEqual(point, (0, 0, 1800, 1800))
        self.assertEqual(conservative, (0, 2, 1823, 1823))
        self.assertGreater(
            q4_solver.clopper_pearson_one_sided_bounds(1823, 2000, 0.95)[0],
            0.9,
        )

    def test_exact_integer_cost_and_volume_metrics(self) -> None:
        metrics = q4_solver.design_metrics(1, 1)
        self.assertEqual(metrics["cost_weight"], 631)
        self.assertAlmostEqual(metrics["cost_yuan"], math.pi * 631 / 120_000)
        self.assertAlmostEqual(metrics["a_volume_um3"], 0.0045 * math.pi)
        self.assertAlmostEqual(metrics["b_volume_um3"], 0.032 * math.pi / 3.0)
        self.assertAlmostEqual(
            metrics["total_volume_percent"],
            100.0
            * (metrics["a_volume_um3"] + metrics["b_volume_um3"])
            / 1000.0,
        )

    def test_strictly_cheaper_maximal_frontier_covers_integer_domain(self) -> None:
        candidate_weight = q4_solver.cost_weight(1, 1)
        frontier = q4_solver.cheaper_maximal_frontier(candidate_weight)
        self.assertEqual(frontier, [(0, 9), (1, 0)])

        cheaper = []
        for n_a in range(candidate_weight // q4_solver.A_COST_WEIGHT + 1):
            for n_b in range(candidate_weight // q4_solver.B_COST_WEIGHT + 1):
                if q4_solver.cost_weight(n_a, n_b) < candidate_weight:
                    cheaper.append((n_a, n_b))
                    self.assertTrue(
                        any(
                            frontier_a >= n_a and frontier_b >= n_b
                            for frontier_a, frontier_b in frontier
                        )
                    )
        self.assertEqual(
            len(cheaper),
            q4_solver.count_strictly_cheaper_designs(candidate_weight),
        )
        for first_index, first in enumerate(frontier):
            self.assertLess(q4_solver.cost_weight(*first), candidate_weight)
            for second in frontier[first_index + 1 :]:
                self.assertFalse(
                    first[0] >= second[0] and first[1] >= second[1]
                )
                self.assertFalse(
                    second[0] >= first[0] and second[1] >= first[1]
                )

    def test_equal_cost_designs_are_enumerated_exactly(self) -> None:
        weight = q4_solver.cost_weight(64, 0)
        self.assertEqual(q4_solver.equal_cost_designs(weight), [(0, 567), (64, 0)])
        self.assertEqual(q4_solver.equal_cost_designs(631), [(1, 1)])

    def test_minimum_unexcluded_cost_uses_dominance_coverage(self) -> None:
        candidate_weight = q4_solver.cost_weight(1, 1)
        self.assertEqual(
            q4_solver.minimum_unexcluded_cheaper_design(candidate_weight, []),
            (0, 0),
        )
        self.assertEqual(
            q4_solver.minimum_unexcluded_cheaper_design(
                candidate_weight, [(0, 9)]
            ),
            (1, 0),
        )
        self.assertEqual(
            q4_solver.minimum_unexcluded_cheaper_design(
                candidate_weight, [(1, 0)]
            ),
            (0, 1),
        )
        self.assertIsNone(
            q4_solver.minimum_unexcluded_cheaper_design(
                candidate_weight, [(0, 9), (1, 0)]
            )
        )


class Q4FreezeAndDecisionTests(unittest.TestCase):
    def test_screening_candidate_uses_exact_cost_order(self) -> None:
        records = [
            {"n_a": 1, "n_b": 0, "cost_weight": 567, "estimate": 0.89},
            {"n_a": 0, "n_b": 9, "cost_weight": 576, "estimate": 0.91},
            {"n_a": 1, "n_b": 1, "cost_weight": 631, "estimate": 0.95},
        ]
        selected = q4_solver.select_screening_candidate(records, target=0.9)
        self.assertEqual((selected["n_a"], selected["n_b"]), (0, 9))

    def test_freeze_separates_streams_and_freezes_bonferroni_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            freeze = _build_freeze(base, target=0.8)
            protocol = freeze["confirmation_protocol"]
            self.assertEqual(
                freeze["strictly_cheaper_domain"]["maximal_frontier"],
                [[0, 9], [1, 0]],
            )
            self.assertEqual(protocol["bonferroni_statement_count"], 3)
            self.assertAlmostEqual(protocol["per_statement_alpha"], 0.05 / 3.0)
            self.assertEqual(protocol["screening_stream_id"], 4)
            self.assertEqual(protocol["confirmation_stream_id"], 5)
            self.assertEqual(protocol["target_probability"], 0.8)
            self.assertTrue(protocol["shared_crn_across_candidate_and_frontier"])
            self.assertTrue(protocol["one_static_graph_per_trial"])
            self.assertEqual(protocol["maximum_static_graph_design"], [1, 9])
            config = q4_solver.MixedSimulationConfig.from_dict(
                protocol["configuration"]
            )
            self.assertEqual(config.master_seed, 31)
            self.assertEqual(config.stream_id, 5)
            self.assertEqual(config.trial_count, 7)
            self.assertEqual((config.n_a, config.n_b), (1, 9))
            self.assertEqual(config.fingerprint, protocol["configuration_fingerprint"])
            self.assertTrue(
                all(
                    "configuration" not in frozen
                    for frozen in freeze["confirmation_designs"]
                )
            )

            manifest = _screening_manifest(target=0.8)
            screening_json = base / "q4_screening.json"
            screening_csv = base / "q4_screening.csv"
            with self.assertRaisesRegex(ValueError, "stream_id"):
                q4_solver.build_confirmation_freeze(
                    manifest,
                    screening_json_path=screening_json,
                    screening_csv_path=screening_csv,
                    confirmation_trials=7,
                    confirmation_stream_id=4,
                    familywise_confidence=0.95,
                )

    def test_all_frontier_points_excluded_certifies_global_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            freeze = _build_freeze(
                Path(temporary), familywise_confidence=0.9
            )
            records = [
                _confirmation_record(
                    "candidate", (1, 1), "candidate_statistically_feasible"
                ),
                _confirmation_record(
                    "strictly_cheaper_maximal",
                    (0, 9),
                    "strictly_cheaper_design_excluded",
                ),
                _confirmation_record(
                    "strictly_cheaper_maximal",
                    (1, 0),
                    "strictly_cheaper_design_excluded",
                ),
            ]
            result = q4_solver.analyze_confirmation_records(freeze, records)
            self.assertEqual(
                result["result_status"], "globally_certified_minimum_cost"
            )
            self.assertIn("90%", result["conclusion_label_zh"])
            self.assertEqual(
                result["cost_uncertainty_interval"]["lower_cost_weight"], 631
            )
            self.assertEqual(
                result["cost_uncertainty_interval"]["upper_cost_weight"], 631
            )

    def test_unexcluded_frontier_point_downgrades_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            freeze = _build_freeze(Path(temporary))
            records = [
                _confirmation_record(
                    "candidate", (1, 1), "candidate_statistically_feasible"
                ),
                _confirmation_record(
                    "strictly_cheaper_maximal",
                    (0, 9),
                    "strictly_cheaper_design_excluded",
                ),
                _confirmation_record(
                    "strictly_cheaper_maximal",
                    (1, 0),
                    "strictly_cheaper_design_not_excluded",
                ),
            ]
            result = q4_solver.analyze_confirmation_records(freeze, records)
            self.assertEqual(
                result["result_status"], "lowest_statistically_feasible_cost"
            )
            self.assertFalse(
                result["all_strictly_cheaper_maximal_designs_excluded"]
            )
            self.assertEqual(
                result["cost_uncertainty_interval"]["minimum_not_excluded_design"],
                [1, 0],
            )

    def test_unconfirmed_candidate_suppresses_minimum_cost_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            freeze = _build_freeze(Path(temporary))
            records = [
                _confirmation_record(
                    "candidate", (1, 1), "candidate_not_confirmed"
                ),
                _confirmation_record(
                    "strictly_cheaper_maximal",
                    (0, 9),
                    "strictly_cheaper_design_excluded",
                ),
                _confirmation_record(
                    "strictly_cheaper_maximal",
                    (1, 0),
                    "strictly_cheaper_design_excluded",
                ),
            ]
            result = q4_solver.analyze_confirmation_records(freeze, records)
            self.assertEqual(
                result["result_status"], "screening_candidate_not_confirmed"
            )
            self.assertIsNone(result["reported_design"])
            self.assertIsNone(result["cost_uncertainty_interval"])


class Q4SmallRunTests(unittest.TestCase):
    def test_screen_stage_is_resumable_and_writes_json_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "q4"
            arguments = [
                "--stage",
                "screen",
                "--output-dir",
                str(output),
                "--design",
                "0,0",
                "--design",
                "1,0",
                "--max-screening-designs",
                "2",
                "--screening-trials",
                "2",
                "--screening-batch-size",
                "1",
                "--workers",
                "1",
                "--seed",
                "43",
            ]
            first = q4_solver.run(
                q4_solver.parse_args([*arguments, "--no-resume"])
            )
            second = q4_solver.run(q4_solver.parse_args(arguments))
            self.assertEqual(first["result_status"], "screening_complete")
            self.assertFalse(first["final_evidence_available"])
            self.assertEqual(first["screening_candidate"], None)
            self.assertEqual(first["screening_candidate"], second["screening_candidate"])
            self.assertTrue((output / "q4_screening.json").is_file())
            self.assertTrue((output / "q4_screening.csv").is_file())
            self.assertFalse((output / "q4_confirmation_freeze.json").exists())
            payload = json.loads(
                (output / "q4_screening.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["design_count"], 2)
            self.assertEqual(payload["integer_domain_design_count"], 2)
            self.assertTrue(
                Path(payload["integer_domain_success_counts"]).is_file()
            )
            self.assertTrue(payload["one_static_graph_per_trial"])
            self.assertEqual(payload["maximum_static_graph_design"], [1, 0])
            self.assertEqual(
                len({record["artifact_path"] for record in payload["records"]}), 1
            )
            self.assertTrue(all(row["trials"] == 2 for row in payload["records"]))
            self.assertTrue(all(row["stream_id"] == 4 for row in [
                record["configuration"] for record in payload["records"]
            ]))

    def test_cli_defaults_and_axis_destinations_match_run_contract(self) -> None:
        defaults = q4_solver.parse_args([])
        self.assertEqual(defaults.screening_trials, 2_000)
        self.assertEqual(defaults.confirmation_trials, 50_000)
        self.assertEqual(defaults.screening_stream_id, 4)
        self.assertEqual(defaults.confirmation_stream_id, 5)
        self.assertEqual(defaults.max_n_a, 720)
        self.assertEqual(defaults.max_n_b, 6000)
        self.assertEqual(defaults.step_n_a, 120)
        self.assertEqual(defaults.step_n_b, 1000)
        self.assertEqual(defaults.screening_candidate_rule, "point_estimate")


if __name__ == "__main__":
    unittest.main()
