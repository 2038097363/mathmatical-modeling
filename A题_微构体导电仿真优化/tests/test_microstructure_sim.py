from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from geometry_kernel import Cylinder, capsule_cylinder_distance  # noqa: E402
from microstructure_sim import (  # noqa: E402
    BoundaryMode,
    CylinderFragment,
    SimulationConfig,
    boundary_spec,
    clopper_pearson_interval,
    clopper_pearson_one_sided_bounds,
    draw_isotropic_directions,
    draw_uniform_centers,
    evaluate_fragment_contact,
    first_connection_prefix,
    fragment_trial,
    load_threshold_artifact,
    merge_shard_payloads,
    nominal_volume_percent,
    probability_at_prefix,
    run_simulation,
    run_trial_ids,
    smallest_confidence_threshold,
    smallest_empirical_threshold,
    wilson_interval,
    wilson_one_sided_bounds,
)


class MicrostructureGeometryTests(unittest.TestCase):
    def test_default_boundary_is_primary_d(self) -> None:
        config = SimulationConfig(max_count=1, trial_count=1)
        self.assertEqual(config.boundary_mode, BoundaryMode.D)
        self.assertEqual(boundary_spec(BoundaryMode.D).role, "primary")
        self.assertNotEqual(boundary_spec(BoundaryMode.B).role, "primary")

    def test_seed_sequence_is_prefix_stable(self) -> None:
        short = SimulationConfig(max_count=10, trial_count=2)
        long = SimulationConfig(max_count=20, trial_count=2)
        np.testing.assert_array_equal(
            draw_uniform_centers(short, 1), draw_uniform_centers(long, 1)[:10]
        )
        np.testing.assert_array_equal(
            draw_isotropic_directions(short, 1),
            draw_isotropic_directions(long, 1)[:10],
        )

    def test_isotropic_direction_moments(self) -> None:
        config = SimulationConfig(max_count=40_000, trial_count=1)
        directions = draw_isotropic_directions(config, 0)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=2e-15)
        self.assertTrue(np.all(np.abs(directions.mean(axis=0)) < 0.012))
        self.assertTrue(
            np.all(np.abs(np.square(directions).mean(axis=0) - 1.0 / 3.0) < 0.012)
        )

    def test_d_fragmentation_preserves_centerline_length(self) -> None:
        config = SimulationConfig(max_count=1, trial_count=1, boundary_mode="D")
        centers = np.array([[4500.0, 4000.0, -4500.0]])
        directions = np.array([[1.0, 2.0, -1.0]]) / math.sqrt(6.0)
        fragments = fragment_trial(centers, directions, config)[0]
        total = sum(2.0 * fragment.cylinder.half_length for fragment in fragments)
        self.assertAlmostEqual(total, config.cylinder_length_nm, places=8)
        for fragment in fragments:
            for endpoint in fragment.cylinder.endpoints:
                self.assertTrue(np.all(endpoint >= -config.half_box_nm - 1e-9))
                self.assertTrue(np.all(endpoint <= config.half_box_nm + 1e-9))

    def test_x_crossing_particle_distinguishes_d_b_a(self) -> None:
        centers = np.array([[4900.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        expected = {"D": 2, "B": 2, "A": 1}
        for mode, threshold in expected.items():
            with self.subTest(mode=mode):
                config = SimulationConfig(
                    max_count=1, trial_count=1, boundary_mode=mode
                )
                result = first_connection_prefix(
                    centers, directions, config, include_witness=True
                )
                self.assertEqual(result.first_connection_index, threshold)
        self.assertFalse(boundary_spec(BoundaryMode.D).connect_same_source)
        self.assertTrue(boundary_spec(BoundaryMode.B).connect_same_source)
        self.assertTrue(boundary_spec(BoundaryMode.A).connect_same_source)

    def test_b_uses_axis_clipping_not_three_dimensional_boolean_intersection(self) -> None:
        config = SimulationConfig(max_count=1, trial_count=1, boundary_mode="B")
        centers = np.array([[4900.0, 0.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0]])
        fragments = fragment_trial(centers, directions, config)[0]
        self.assertEqual(len(fragments), 1)
        self.assertAlmostEqual(2.0 * fragments[0].cylinder.half_length, 2600.0)
        self.assertIn(
            "不等于完整三维圆柱",
            boundary_spec(BoundaryMode.B).implementation_limitations[0],
        )

    def test_b_connects_independent_particles_across_y_periodic_face(self) -> None:
        centers = np.array(
            [[-2500.0, 4969.5, 0.0], [2500.0, -4969.5, 0.0]]
        )
        directions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        primary = SimulationConfig(max_count=2, trial_count=1, boundary_mode="B")
        literal = SimulationConfig(max_count=2, trial_count=1, boundary_mode="D")
        primary_result = first_connection_prefix(centers, directions, primary)
        literal_result = first_connection_prefix(centers, directions, literal)
        primary_brute = first_connection_prefix(
            centers, directions, primary, use_spatial_index=False
        )
        self.assertEqual(primary_result.first_connection_index, 2)
        self.assertEqual(primary_brute.first_connection_index, 2)
        self.assertEqual(literal_result.first_connection_index, 3)

    def test_flat_end_narrow_phase_rejects_capsule_false_contact(self) -> None:
        first_cylinder = Cylinder(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            100.0,
            30.0,
        )
        second_cylinder = Cylinder(
            np.array([0.0, 0.0, 261.0]),
            np.array([0.0, 0.0, 1.0]),
            100.0,
            30.0,
        )
        first = CylinderFragment(0, 0, 0.0, 1.0, (0, 0, 0), first_cylinder)
        second = CylinderFragment(1, 0, 0.0, 1.0, (0, 0, 0), second_cylinder)
        self.assertLessEqual(
            capsule_cylinder_distance(first_cylinder, second_cylinder), 1.8
        )
        evaluation = evaluate_fragment_contact(
            first, second, SimulationConfig(max_count=2, trial_count=1)
        )
        self.assertFalse(evaluation.connected)
        self.assertFalse(evaluation.broad_phase_rejected)
        self.assertEqual(evaluation.narrow_phase_calls, 1)
        self.assertGreater(evaluation.lower_nm, 60.9)

    def test_spatial_hash_matches_all_pairs_on_small_trials(self) -> None:
        for mode in BoundaryMode:
            for trial_id in range(2):
                config = SimulationConfig(
                    max_count=14,
                    trial_count=2,
                    boundary_mode=mode,
                    master_seed=20260807,
                )
                centers = draw_uniform_centers(config, trial_id)
                directions = draw_isotropic_directions(config, trial_id)
                fast = first_connection_prefix(centers, directions, config)
                brute = first_connection_prefix(
                    centers, directions, config, use_spatial_index=False
                )
                self.assertEqual(
                    fast.first_connection_index, brute.first_connection_index
                )

    def test_witness_path_is_available(self) -> None:
        config = SimulationConfig(max_count=1, trial_count=1, boundary_mode="A")
        result = first_connection_prefix(
            np.array([[4900.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
            config,
            include_witness=True,
        )
        self.assertEqual(result.diagnostics.witness_nodes[0], "LEFT_ELECTRODE")
        self.assertEqual(result.diagnostics.witness_nodes[-1], "RIGHT_ELECTRODE")
        self.assertIn(
            "same_source_internal", result.diagnostics.witness_edge_types
        )


class MicrostructureStatisticsAndShardTests(unittest.TestCase):
    def test_binomial_intervals_cover_known_values(self) -> None:
        wilson = wilson_interval(50, 100)
        self.assertAlmostEqual(wilson[0], 0.4038315304, places=9)
        self.assertAlmostEqual(wilson[1], 0.5961684696, places=9)
        self.assertAlmostEqual(
            clopper_pearson_interval(0, 10)[1], 0.3084971078, places=9
        )
        self.assertAlmostEqual(
            clopper_pearson_interval(10, 10)[0], 0.6915028922, places=9
        )
        self.assertAlmostEqual(
            wilson_one_sided_bounds(9050, 10_000)[0],
            0.9000669,
            places=7,
        )
        self.assertAlmostEqual(
            clopper_pearson_one_sided_bounds(9050, 10_000)[0],
            0.9000393,
            places=7,
        )

    def test_probability_and_integer_thresholds(self) -> None:
        samples = np.array([2, 3, 3, 4, 6], dtype=np.int64)
        record = probability_at_prefix(samples, 3, max_count=5)
        self.assertEqual(record["successes"], 3)
        self.assertEqual(record["estimate"], 0.6)
        self.assertEqual(smallest_empirical_threshold(samples, 5, 0.6), 3)
        self.assertIsNone(smallest_empirical_threshold(samples, 5, 0.9))
        self.assertIsNone(
            smallest_confidence_threshold(samples, 5, 0.6, interval="wilson")
        )

    def test_q2_integer_volume_conversion(self) -> None:
        expected = {
            354: 0.5004557099,
            424: 0.5994158785,
            495: 0.6997897639,
            707: 0.9994977028,
        }
        for count, percentage in expected.items():
            self.assertAlmostEqual(nominal_volume_percent(count), percentage, places=9)

    def test_merge_sorts_and_rejects_duplicates_or_incompatibility(self) -> None:
        config = SimulationConfig(
            max_count=1, trial_count=2, boundary_mode="A", master_seed=17
        )
        first = run_trial_ids(config, [0])
        second = run_trial_ids(config, [1])
        merged = merge_shard_payloads([second, first], config)
        self.assertEqual(
            [record["trial_id"] for record in merged["records"]], [0, 1]
        )
        with self.assertRaisesRegex(ValueError, "重复"):
            merge_shard_payloads([first, first, second], config)
        incompatible = SimulationConfig(
            max_count=1, trial_count=2, boundary_mode="D", master_seed=17
        )
        with self.assertRaisesRegex(ValueError, "指纹"):
            merge_shard_payloads([first, second], incompatible)

    def test_single_and_multiprocess_runs_have_identical_samples(self) -> None:
        config = SimulationConfig(
            max_count=2, trial_count=4, boundary_mode="A", master_seed=23
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            sequential_path = run_simulation(
                config, base / "sequential", workers=1, batch_size=1
            )
            parallel_path = run_simulation(
                config, base / "parallel", workers=2, batch_size=1
            )
            _, sequential, _ = load_threshold_artifact(sequential_path)
            _, parallel, _ = load_threshold_artifact(parallel_path)
            np.testing.assert_array_equal(sequential, parallel)
            resumed_path = run_simulation(
                config, base / "parallel", workers=2, batch_size=1, resume=True
            )
            payload = json.loads(resumed_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["configuration_fingerprint"], config.fingerprint)

    def test_completed_shard_is_checkpointed_before_later_failure(self) -> None:
        config = SimulationConfig(
            max_count=1, trial_count=2, boundary_mode="A", master_seed=29
        )
        first_payload = run_trial_ids(config, [0])
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "interrupted"
            with patch(
                "microstructure_sim.run_trial_ids",
                side_effect=[first_payload, RuntimeError("injected failure")],
            ):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    run_simulation(
                        config,
                        output_dir,
                        workers=1,
                        batch_size=1,
                        resume=False,
                    )
            self.assertTrue(
                (output_dir / "shards" / "shard_000000_000000.json").is_file()
            )
            self.assertFalse(
                (output_dir / "shards" / "shard_000001_000001.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
