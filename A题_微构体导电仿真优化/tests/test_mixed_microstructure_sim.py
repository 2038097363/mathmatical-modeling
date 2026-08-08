from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from geometry_kernel import Cylinder, Sphere, aabb, distance_bounds  # noqa: E402
from microstructure_sim import (  # noqa: E402
    BoundaryMode,
    SimulationConfig,
    draw_isotropic_directions,
    draw_uniform_centers,
    first_connection_prefix,
)
from mixed_microstructure_sim import (  # noqa: E402
    BOUNDARY_CONTRACT,
    ClippedSphere,
    MixedSimulationConfig,
    MixedTrialGeometry,
    connectivity_samples_at_design,
    evaluate_exact_contact,
    fragment_sphere,
    generate_mixed_trial,
    load_fixed_design_artifact,
    load_pareto_frontier_artifact,
    merge_shard_payloads,
    run_fixed_design_simulation,
    run_pareto_frontier_simulation,
    run_pareto_trial_ids,
    run_trial_ids,
    solve_fixed_design,
    solve_pareto_connectivity,
    _mixed_aabb,
)


class ClippedSphereSupportTests(unittest.TestCase):
    def test_support_matches_independent_slsqp(self) -> None:
        shape = ClippedSphere(
            np.array([1.25, -0.4, 0.6]),
            1.5,
            np.array([-0.5, -1.0, -0.2]),
            np.array([1.0, 0.8, 1.0]),
        )
        directions = (
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 1.0, 0.1]),
            np.array([0.2, -0.7, 1.3]),
            np.array([-0.4, -0.1, -1.0]),
        )
        bounds = list(zip(shape.box_lower, shape.box_upper, strict=True))
        for direction in directions:
            result = minimize(
                lambda point: -float(np.dot(direction, point)),
                0.5 * (shape.box_lower + shape.box_upper),
                method="SLSQP",
                bounds=bounds,
                constraints={
                    "type": "ineq",
                    "fun": lambda point: shape.radius**2
                    - float(np.dot(point - shape.sphere_center, point - shape.sphere_center)),
                },
                options={"ftol": 1e-12, "maxiter": 500},
            )
            self.assertTrue(result.success, result.message)
            support = shape.support(direction)
            self.assertTrue(np.all(support >= shape.box_lower - 1e-12))
            self.assertTrue(np.all(support <= shape.box_upper + 1e-12))
            self.assertLessEqual(
                float(np.linalg.norm(support - shape.sphere_center)),
                shape.radius + 1e-12,
            )
            self.assertAlmostEqual(
                float(np.dot(direction, support)), -float(result.fun), delta=2e-8
            )

    def test_oblique_cap_support_is_not_coordinatewise_ball_clipping(self) -> None:
        shape = ClippedSphere(
            np.zeros(3),
            1.0,
            np.array([-2.0, -2.0, -2.0]),
            np.array([0.5, 2.0, 2.0]),
        )
        support = shape.support(np.array([1.0, 1.0, 0.0]))
        np.testing.assert_allclose(
            support, np.array([0.5, math.sqrt(0.75), 0.0]), atol=2e-12
        )
        np.testing.assert_array_equal(shape.support(np.zeros(3)), shape.center)

    def test_translation_and_generic_aabb_preserve_clipped_shape(self) -> None:
        shape = ClippedSphere(
            np.array([1.2, 0.0, 0.0]),
            1.0,
            np.array([-1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        lower, upper = aabb(shape)
        np.testing.assert_allclose(lower, np.array([0.2, -math.sqrt(0.96), -math.sqrt(0.96)]))
        np.testing.assert_allclose(
            upper, np.array([1.0, math.sqrt(0.96), math.sqrt(0.96)])
        )
        shift = np.array([3.0, -2.0, 4.0])
        moved = shape.translated(shift)
        np.testing.assert_allclose(moved.support([1.0, 2.0, -1.0]), shape.support([1.0, 2.0, -1.0]) + shift)

    def test_closed_form_coordinate_aabb_matches_support_aabb(self) -> None:
        shapes = (
            ClippedSphere(
                np.array([1.2, 0.0, 0.0]),
                1.0,
                np.full(3, -1.0),
                np.full(3, 1.0),
            ),
            ClippedSphere(
                np.array([1.2, 1.1, 0.0]),
                1.0,
                np.full(3, -1.0),
                np.full(3, 1.0),
            ),
            ClippedSphere(
                np.array([1.2, 1.1, 1.05]),
                1.0,
                np.full(3, -1.0),
                np.full(3, 1.0),
            ),
        )
        for shape in shapes:
            reference_lower, reference_upper = aabb(shape)
            exact_lower, exact_upper = _mixed_aabb(shape)
            np.testing.assert_allclose(exact_lower, reference_lower, atol=2e-12)
            np.testing.assert_allclose(exact_upper, reference_upper, atol=2e-12)

    def test_tangent_intersection_keeps_gjk_bounds_valid(self) -> None:
        tangent = ClippedSphere(
            np.zeros(3),
            1.0,
            np.array([1.0, -2.0, -2.0]),
            np.array([2.0, 2.0, 2.0]),
        )
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_equal(tangent.support([-1.0, 1.0, 0.0]), expected)
        point = Sphere(np.array([0.0, 1.0, 0.0]), 0.0)
        bounds = distance_bounds(tangent, point)
        true_distance = math.sqrt(2.0)
        self.assertLessEqual(bounds.lower, true_distance)
        self.assertGreaterEqual(bounds.upper, true_distance)

    def test_empty_intersection_is_rejected_without_geometric_expansion(self) -> None:
        with self.assertRaisesRegex(ValueError, "没有交集"):
            ClippedSphere(
                np.zeros(3),
                1.0,
                np.array([1.0 + 1e-14, -1.0, -1.0]),
                np.array([2.0, 1.0, 1.0]),
            )

    def test_missing_fma_fails_instead_of_silently_weakening_geometry(self) -> None:
        with patch.object(math, "fma", None):
            with self.assertRaisesRegex(RuntimeError, "Python 3.13"):
                ClippedSphere(
                    np.zeros(3),
                    1.0,
                    np.full(3, -1.0),
                    np.full(3, 1.0),
                )

    def test_extreme_direction_scaling_and_large_multiplier_bracket(self) -> None:
        shape = ClippedSphere(
            np.zeros(3),
            1.0,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 2.0, 0.0]),
        )
        support = shape.support(np.array([1.0, 1e-50, 0.0]))
        np.testing.assert_allclose(
            support, np.array([0.5, math.sqrt(0.75), 0.0]), atol=2e-12
        )

        regular = ClippedSphere(
            np.array([0.2, -0.1, 0.3]),
            1.0,
            np.full(3, -0.8),
            np.full(3, 0.7),
        )
        direction = np.array([0.3, -0.4, 0.5])
        reference = regular.support(direction)
        np.testing.assert_allclose(regular.support(1e-300 * direction), reference)
        np.testing.assert_allclose(regular.support(1e300 * direction), reference)

    def test_tangent_cap_does_not_flip_a_gap_above_cutoff(self) -> None:
        radius = 200.0
        cap = ClippedSphere(
            np.zeros(3),
            radius,
            np.array([radius, -400.0, -400.0]),
            np.array([400.0, 400.0, 400.0]),
        )
        gap = 1.800001
        point = Sphere(
            np.array([radius + gap / math.sqrt(2.0), gap / math.sqrt(2.0), 0.0]),
            0.0,
        )
        config = MixedSimulationConfig(0, 0, 1)
        self.assertFalse(evaluate_exact_contact(cap, point, config).connected)


class MixedContactGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MixedSimulationConfig(n_a=0, n_b=0, trial_count=1)
        self.cylinder = Cylinder(
            np.zeros(3), np.array([0.0, 0.0, 1.0]), 2500.0, 30.0
        )

    def _assert_threshold_triplet(self, factory) -> None:
        below, equal, above = (
            evaluate_exact_contact(*factory(delta), self.config).connected
            for delta in (-1e-6, 0.0, 1e-6)
        )
        self.assertTrue(below)
        self.assertTrue(equal)
        self.assertFalse(above)

    def test_aa_threshold_equality_and_both_sides(self) -> None:
        def factory(delta):
            second = Cylinder(
                np.array([61.8 + delta, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                2500.0,
                30.0,
            )
            return self.cylinder, second

        self._assert_threshold_triplet(factory)

    def test_ab_threshold_equality_and_both_sides(self) -> None:
        def factory(delta):
            return self.cylinder, Sphere(
                np.array([231.8 + delta, 0.0, 0.0]), 200.0
            )

        self._assert_threshold_triplet(factory)

    def test_bb_threshold_equality_and_both_sides(self) -> None:
        def factory(delta):
            return (
                Sphere(np.zeros(3), 200.0),
                Sphere(np.array([401.8 + delta, 0.0, 0.0]), 200.0),
            )

        self._assert_threshold_triplet(factory)

    def test_bb_threshold_nextafter_sides(self) -> None:
        threshold = 401.8
        distances = (
            np.nextafter(threshold, -math.inf),
            threshold,
            np.nextafter(threshold, math.inf),
        )
        connected = [
            evaluate_exact_contact(
                Sphere(np.zeros(3), 200.0),
                Sphere(np.array([distance, 0.0, 0.0]), 200.0),
                self.config,
            ).connected
            for distance in distances
        ]
        self.assertEqual(connected, [True, True, True])

    def test_clipped_sphere_uses_generic_gjk_at_contact_threshold(self) -> None:
        cap = ClippedSphere(
            np.array([5100.0, 0.0, 0.0]),
            200.0,
            np.full(3, -5000.0),
            np.full(3, 5000.0),
        )

        below = evaluate_exact_contact(
            cap, Sphere(np.array([4698.200001, 0.0, 0.0]), 200.0), self.config
        )
        above = evaluate_exact_contact(
            cap, Sphere(np.array([4698.199999, 0.0, 0.0]), 200.0), self.config
        )
        self.assertTrue(below.connected)
        self.assertFalse(above.connected)
        self.assertEqual(below.method, "generic_convex_gjk_bounds")

    def test_boundary_contract_is_d_with_exact_b_and_approximate_a(self) -> None:
        self.assertEqual(BOUNDARY_CONTRACT["mode"], "D")
        self.assertIn("no minimum image", BOUNDARY_CONTRACT["periodic_axes"])
        self.assertIn("exact ball-cell", BOUNDARY_CONTRACT["B_geometry"])
        self.assertIn("centerline", BOUNDARY_CONTRACT["limitation"])


class SphereFragmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MixedSimulationConfig(0, 1, 1)

    def test_interior_tangent_face_and_overflow_piece_counts(self) -> None:
        interior = fragment_sphere(np.zeros(3), 0, self.config)
        tangent = fragment_sphere(np.array([4800.0, 0.0, 0.0]), 0, self.config)
        crossed = fragment_sphere(np.array([4900.0, 0.0, 0.0]), 0, self.config)
        corner = fragment_sphere(np.full(3, 4900.0), 0, self.config)
        self.assertEqual(len(interior), 1)
        self.assertIsInstance(interior[0].shape, Sphere)
        self.assertEqual(len(tangent), 1)
        self.assertIsInstance(tangent[0].shape, Sphere)
        self.assertEqual(len(crossed), 2)
        self.assertEqual(
            {fragment.cell_shift for fragment in crossed},
            {(0, 0, 0), (-1, 0, 0)},
        )
        self.assertTrue(all(isinstance(fragment.shape, ClippedSphere) for fragment in crossed))
        self.assertEqual(len(corner), 8)
        self.assertTrue(all(isinstance(fragment.shape, ClippedSphere) for fragment in corner))

    def test_every_mapped_piece_is_inside_base_box_and_has_positive_extent(self) -> None:
        fragments = fragment_sphere(np.full(3, -4900.0), 0, self.config)
        self.assertEqual(len(fragments), 8)
        for fragment in fragments:
            lower, upper = aabb(fragment.shape)
            self.assertTrue(np.all(lower >= -5000.0 - 1e-10))
            self.assertTrue(np.all(upper <= 5000.0 + 1e-10))
            self.assertTrue(np.all(upper > lower))


class MixedGraphTests(unittest.TestCase):
    @staticmethod
    def _empty_a() -> tuple[np.ndarray, np.ndarray]:
        return np.empty((0, 3)), np.empty((0, 3))

    def test_zero_zero_is_not_conductive(self) -> None:
        config = MixedSimulationConfig(0, 0, 1)
        a_centers, a_directions = self._empty_a()
        result = solve_fixed_design(
            MixedTrialGeometry(a_centers, a_directions, np.empty((0, 3))), config
        )
        self.assertFalse(result.conductive)
        self.assertEqual(result.diagnostics.candidate_pairs, 0)

    def test_one_x_overflow_sphere_has_disconnected_relocated_pieces(self) -> None:
        config = MixedSimulationConfig(0, 1, 1)
        a_centers, a_directions = self._empty_a()
        result = solve_fixed_design(
            MixedTrialGeometry(
                a_centers, a_directions, np.array([[4900.0, 0.0, 0.0]])
            ),
            config,
        )
        self.assertFalse(result.conductive)
        self.assertEqual(result.diagnostics.b_fragment_count, 2)
        self.assertEqual(result.diagnostics.clipped_b_fragment_count, 2)
        self.assertEqual(result.diagnostics.electrode_contacts, 2)
        self.assertEqual(result.diagnostics.internal_a_edges, 0)

    def test_same_source_fragments_are_filtered_even_if_geometrically_close(self) -> None:
        config = MixedSimulationConfig(
            0,
            1,
            1,
            box_length_nm=10.0,
            b_radius_nm=4.5,
            contact_cutoff_nm=1.8,
            cell_size_nm=2.0,
        )
        a_centers, a_directions = self._empty_a()
        geometry = MixedTrialGeometry(
            a_centers, a_directions, np.array([[4.9, 0.0, 0.0]])
        )
        fixed = solve_fixed_design(geometry, config)
        pareto = solve_pareto_connectivity(geometry, config)
        self.assertFalse(fixed.conductive)
        self.assertEqual(pareto.connectivity_frontier, ())
        self.assertGreaterEqual(fixed.diagnostics.same_source_skips, 1)
        self.assertGreaterEqual(pareto.diagnostics.same_source_skips, 1)
        self.assertEqual(fixed.diagnostics.bb_contacts, 0)
        self.assertEqual(pareto.diagnostics.bb_contacts, 0)

    def test_opposite_x_particles_do_not_gain_a_fake_periodic_edge(self) -> None:
        centers = np.array([[-4900.0, 0.0, 0.0], [4900.0, 0.0, 0.0]])
        config = MixedSimulationConfig(0, 2, 1)
        a_centers, a_directions = self._empty_a()
        indexed = solve_fixed_design(
            MixedTrialGeometry(a_centers, a_directions, centers), config
        )
        brute = solve_fixed_design(
            MixedTrialGeometry(a_centers, a_directions, centers),
            config,
            use_spatial_index=False,
        )
        self.assertFalse(indexed.conductive)
        self.assertEqual(indexed.conductive, brute.conductive)

    def test_interior_sphere_chain_connects_electrodes(self) -> None:
        x_values = np.linspace(-4800.0, 4800.0, 25)
        centers = np.column_stack(
            (x_values, np.zeros(x_values.size), np.zeros(x_values.size))
        )
        config = MixedSimulationConfig(0, len(centers), 1)
        a_centers, a_directions = self._empty_a()
        result = solve_fixed_design(
            MixedTrialGeometry(a_centers, a_directions, centers), config
        )
        self.assertTrue(result.conductive)
        self.assertEqual(result.diagnostics.b_fragment_count, len(centers))

    def test_rng_streams_are_independent_and_prefix_stable(self) -> None:
        short = MixedSimulationConfig(5, 7, 2, master_seed=99)
        long = MixedSimulationConfig(10, 12, 2, master_seed=99)
        short_geometry = generate_mixed_trial(short, 1)
        long_geometry = generate_mixed_trial(long, 1)
        np.testing.assert_array_equal(
            short_geometry.a_centers, long_geometry.a_centers[:5]
        )
        np.testing.assert_array_equal(
            short_geometry.a_directions, long_geometry.a_directions[:5]
        )
        np.testing.assert_array_equal(
            short_geometry.b_centers, long_geometry.b_centers[:7]
        )
        self.assertFalse(
            np.array_equal(short_geometry.a_centers, short_geometry.b_centers[:5])
        )

    def test_n_b_zero_matches_existing_d_mode_on_same_trials(self) -> None:
        mixed_config = MixedSimulationConfig(
            30, 0, 4, master_seed=20260807, stream_id=6
        )
        base_config = SimulationConfig(
            max_count=30,
            trial_count=4,
            boundary_mode=BoundaryMode.D,
            master_seed=20260807,
            stream_id=6,
        )
        for trial_id in range(4):
            geometry = generate_mixed_trial(mixed_config, trial_id)
            np.testing.assert_array_equal(
                geometry.a_centers, draw_uniform_centers(base_config, trial_id)
            )
            np.testing.assert_array_equal(
                geometry.a_directions,
                draw_isotropic_directions(base_config, trial_id),
            )
            mixed = solve_fixed_design(geometry, mixed_config)
            base = first_connection_prefix(
                geometry.a_centers, geometry.a_directions, base_config
            )
            self.assertEqual(
                mixed.conductive,
                base.first_connection_index <= mixed_config.n_a,
            )
            self.assertEqual(mixed.diagnostics.internal_a_edges, 0)

    def test_spatial_index_matches_all_pairs_on_small_mixed_trials(self) -> None:
        config = MixedSimulationConfig(
            12, 10, 4, master_seed=20260807, stream_id=8
        )
        for trial_id in range(config.trial_count):
            geometry = generate_mixed_trial(config, trial_id)
            indexed = solve_fixed_design(geometry, config)
            brute = solve_fixed_design(
                geometry, config, use_spatial_index=False
            )
            self.assertEqual(indexed.conductive, brute.conductive)


class MixedParetoTests(unittest.TestCase):
    def test_kdtree_full_sphere_graph_matches_all_pairs(self) -> None:
        x_values = np.linspace(-4800.0, 4800.0, 25)
        centers = np.column_stack(
            (x_values, np.zeros(x_values.size), np.zeros(x_values.size))
        )
        config = MixedSimulationConfig(0, len(centers), 1)
        empty = np.empty((0, 3))
        geometry = MixedTrialGeometry(empty, empty, centers)
        indexed = solve_pareto_connectivity(geometry, config)
        brute = solve_pareto_connectivity(
            geometry, config, use_spatial_index=False
        )
        self.assertEqual(indexed.connectivity_frontier, brute.connectivity_frontier)
        self.assertEqual(indexed.diagnostics.bb_contacts, brute.diagnostics.bb_contacts)
        self.assertEqual(
            indexed.pareto_diagnostics.edge_count,
            brute.pareto_diagnostics.edge_count,
        )

    def test_optimized_pareto_matches_all_pairs_on_random_mixed_trials(self) -> None:
        config = MixedSimulationConfig(
            6, 18, 3, master_seed=20260807, stream_id=12
        )
        for trial_id in range(config.trial_count):
            geometry = generate_mixed_trial(config, trial_id)
            indexed = solve_pareto_connectivity(geometry, config)
            brute = solve_pareto_connectivity(
                geometry, config, use_spatial_index=False
            )
            self.assertEqual(
                indexed.connectivity_frontier,
                brute.connectivity_frontier,
                trial_id,
            )
            self.assertEqual(
                indexed.pareto_diagnostics.edge_count,
                brute.pareto_diagnostics.edge_count,
                trial_id,
            )

    def test_pareto_completed_shard_survives_later_failure(self) -> None:
        config = MixedSimulationConfig(0, 0, 2, master_seed=37)
        first_payload = run_pareto_trial_ids(config, [0])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "interrupted_pareto"
            with patch(
                "mixed_microstructure_sim.run_pareto_trial_ids",
                side_effect=[first_payload, RuntimeError("forced pareto failure")],
            ):
                with self.assertRaisesRegex(RuntimeError, "forced pareto failure"):
                    run_pareto_frontier_simulation(
                        config,
                        output,
                        workers=1,
                        batch_size=1,
                        resume=False,
                    )
            self.assertTrue(
                (output / "shards" / "shard_000000_000000.json").is_file()
            )
            self.assertFalse(
                (output / "shards" / "shard_000001_000001.json").exists()
            )

    def test_one_static_graph_matches_every_fixed_design_on_small_grid(self) -> None:
        config = MixedSimulationConfig(2, 2, 1, b_radius_nm=2500.0)
        geometry = MixedTrialGeometry(
            a_centers=np.array([[-2500.0, 0.0, 0.0], [2500.0, 0.0, 0.0]]),
            a_directions=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            b_centers=np.array([[2500.0, 0.0, 0.0], [-2500.0, 0.0, 0.0]]),
        )
        pareto = solve_pareto_connectivity(geometry, config)
        self.assertEqual(
            pareto.connectivity_frontier, ((0, 2), (1, 1), (2, 0))
        )
        for n_a in range(config.n_a + 1):
            for n_b in range(config.n_b + 1):
                fixed_config = replace(config, n_a=n_a, n_b=n_b)
                fixed_geometry = MixedTrialGeometry(
                    geometry.a_centers[:n_a],
                    geometry.a_directions[:n_a],
                    geometry.b_centers[:n_b],
                )
                direct = solve_fixed_design(fixed_geometry, fixed_config).conductive
                predicted = bool(
                    connectivity_samples_at_design(
                        [pareto.connectivity_frontier], n_a, n_b
                    )[0]
                )
                self.assertEqual(direct, predicted, (n_a, n_b))

    def test_pareto_artifact_is_resumable_and_queryable(self) -> None:
        config = MixedSimulationConfig(2, 2, 2, master_seed=29)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pareto"
            first = run_pareto_frontier_simulation(
                config, output, workers=1, batch_size=1, resume=False
            )
            second = run_pareto_frontier_simulation(
                config, output, workers=1, batch_size=1, resume=True
            )
            self.assertEqual(first, second)
            stored_config, frontiers, records = load_pareto_frontier_artifact(second)
            self.assertEqual(stored_config.fingerprint, config.fingerprint)
            self.assertEqual(len(frontiers), config.trial_count)
            self.assertEqual(len(records), config.trial_count)
            samples = connectivity_samples_at_design(frontiers, 2, 2)
            self.assertEqual(samples.shape, (2,))


class MixedShardTests(unittest.TestCase):
    def test_completed_shard_survives_a_later_batch_failure(self) -> None:
        config = MixedSimulationConfig(0, 0, 2, master_seed=13)
        first_payload = run_trial_ids(config, [0])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "interrupted"
            with patch(
                "mixed_microstructure_sim.run_trial_ids",
                side_effect=[first_payload, RuntimeError("forced later failure")],
            ):
                with self.assertRaisesRegex(RuntimeError, "forced later failure"):
                    run_fixed_design_simulation(
                        config,
                        output,
                        workers=1,
                        batch_size=1,
                        resume=False,
                    )
            completed = output / "shards" / "shard_000000_000000.json"
            pending = output / "shards" / "shard_000001_000001.json"
            self.assertTrue(completed.is_file())
            self.assertFalse(pending.exists())
            payload = json.loads(completed.read_text(encoding="utf-8"))
            self.assertEqual(payload["trial_ids"], [0])

    def test_merge_rejects_duplicates_and_incompatible_fingerprints(self) -> None:
        config = MixedSimulationConfig(2, 2, 2, master_seed=17)
        first = run_trial_ids(config, [0])
        second = run_trial_ids(config, [1])
        merged = merge_shard_payloads([second, first], config)
        self.assertEqual(
            [record["trial_id"] for record in merged["records"]], [0, 1]
        )
        with self.assertRaisesRegex(ValueError, "重复"):
            merge_shard_payloads([first, first, second], config)
        incompatible = MixedSimulationConfig(2, 3, 2, master_seed=17)
        with self.assertRaisesRegex(ValueError, "指纹"):
            merge_shard_payloads([first, second], incompatible)

    def test_serial_parallel_and_resume_are_identical(self) -> None:
        config = MixedSimulationConfig(4, 5, 4, master_seed=23)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            sequential_path = run_fixed_design_simulation(
                config, base / "sequential", workers=1, batch_size=1
            )
            parallel_path = run_fixed_design_simulation(
                config, base / "parallel", workers=2, batch_size=1
            )
            _, sequential, _ = load_fixed_design_artifact(sequential_path)
            _, parallel, _ = load_fixed_design_artifact(parallel_path)
            np.testing.assert_array_equal(sequential, parallel)
            resumed_path = run_fixed_design_simulation(
                config,
                base / "parallel",
                workers=2,
                batch_size=1,
                resume=True,
            )
            payload = json.loads(resumed_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["configuration_fingerprint"], config.fingerprint)
            self.assertEqual(payload["trials"], config.trial_count)


if __name__ == "__main__":
    unittest.main()
