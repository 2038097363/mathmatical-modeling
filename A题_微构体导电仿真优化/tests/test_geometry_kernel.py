from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from geometry_kernel import (  # noqa: E402
    Cylinder,
    Sphere,
    capsule_cylinder_distance,
    distance_bounds,
    periodic_distance_bounds,
    shape_plane_distance,
)


class GeometryKernelTests(unittest.TestCase):
    def test_parallel_flat_cylinder_gap(self) -> None:
        first = Cylinder.from_endpoints([0, 0, -2500], [0, 0, 2500], 30)
        second = Cylinder.from_endpoints([61.8, 0, -2500], [61.8, 0, 2500], 30)
        result = distance_bounds(first, second)
        self.assertLessEqual(result.lower, 1.8)
        self.assertGreaterEqual(result.upper, 1.8)
        self.assertLess(result.width, 2e-9)

    def test_capsule_only_supplies_lower_bound(self) -> None:
        first = Cylinder.from_endpoints([0, 0, -2500], [0, 0, 2500], 30)
        second = Cylinder.from_endpoints([0, 0, 2561.8], [0, 0, 7561.8], 30)
        self.assertAlmostEqual(capsule_cylinder_distance(first, second), 1.8, places=10)
        self.assertGreater(distance_bounds(first, second).lower, 1.8)

    def test_perpendicular_intersection(self) -> None:
        first = Cylinder.from_endpoints([-2500, 0, 0], [2500, 0, 0], 30)
        second = Cylinder.from_endpoints([0, -2500, 0], [0, 2500, 0], 30)
        self.assertLessEqual(distance_bounds(first, second).upper, 2e-9)

    def test_cylinder_plane_uses_radius(self) -> None:
        cylinder = Cylinder.from_endpoints([-100, 4980, 0], [100, 4980, 0], 30)
        self.assertEqual(shape_plane_distance(cylinder, [0, 1, 0], 5000), 0.0)

    def test_minimum_image_is_explicit_not_implicit(self) -> None:
        first = Sphere(np.array([-4900.0, 0.0, 0.0]), 100.0)
        second = Sphere(np.array([4900.0, 0.0, 0.0]), 100.0)
        self.assertGreater(distance_bounds(first, second).lower, 9000.0)
        self.assertLessEqual(
            periodic_distance_bounds(first, second, [10000, 10000, 10000]).upper,
            2e-9,
        )

    def test_capsule_lower_bound_on_random_pairs(self) -> None:
        rng = np.random.default_rng(20260807)
        for _ in range(50):
            first = Cylinder(rng.uniform(-1000, 1000, 3), rng.normal(size=3), 2500, 30)
            second = Cylinder(rng.uniform(-1000, 1000, 3), rng.normal(size=3), 2500, 30)
            capsule = capsule_cylinder_distance(first, second)
            exact = distance_bounds(first, second, absolute_tolerance=1e-7)
            self.assertLessEqual(capsule, exact.upper + 1e-9)


if __name__ == "__main__":
    unittest.main()
