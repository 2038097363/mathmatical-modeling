from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


# 球与有限平底圆柱使用精确凸体支持映射。
Vec3 = NDArray[np.float64]


def _vec3(value: ArrayLike) -> Vec3:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("expected one finite three-dimensional vector")
    return result.copy()


def _unit(value: ArrayLike, min_norm: float = 1e-12) -> Vec3:
    vector = _vec3(value)
    norm = float(np.linalg.norm(vector))
    if norm <= min_norm:
        raise ValueError("axis/normal vector is degenerate")
    return vector / norm


class SupportShape(Protocol):
    center: Vec3

    def support(self, direction: ArrayLike) -> Vec3: ...

    def translated(self, shift: ArrayLike) -> "SupportShape": ...

    def characteristic_radius(self) -> float: ...


@dataclass(frozen=True)
class Sphere:
    center: Vec3
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec3(self.center))
        if not np.isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("radius must be finite and nonnegative")

    def support(self, direction: ArrayLike) -> Vec3:
        direction = _vec3(direction)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0 or self.radius == 0.0:
            return self.center.copy()
        return self.center + (self.radius / norm) * direction

    def translated(self, shift: ArrayLike) -> "Sphere":
        return Sphere(self.center + _vec3(shift), self.radius)

    def characteristic_radius(self) -> float:
        return self.radius


@dataclass(frozen=True)
class Cylinder:
    center: Vec3
    axis: Vec3
    half_length: float
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec3(self.center))
        object.__setattr__(self, "axis", _unit(self.axis))
        if not np.isfinite(self.half_length) or self.half_length < 0.0:
            raise ValueError("half_length must be finite and nonnegative")
        if not np.isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("radius must be finite and nonnegative")

    @classmethod
    def from_endpoints(
        cls,
        endpoint_a: ArrayLike,
        endpoint_b: ArrayLike,
        radius: float,
        min_axis_length: float = 1e-9,
    ) -> "Cylinder":
        endpoint_a = _vec3(endpoint_a)
        endpoint_b = _vec3(endpoint_b)
        delta = endpoint_b - endpoint_a
        length = float(np.linalg.norm(delta))
        if length <= min_axis_length:
            raise ValueError("cylinder endpoints do not define a stable axis")
        return cls(
            center=0.5 * (endpoint_a + endpoint_b),
            axis=delta / length,
            half_length=0.5 * length,
            radius=radius,
        )

    @property
    def endpoints(self) -> tuple[Vec3, Vec3]:
        offset = self.half_length * self.axis
        return self.center - offset, self.center + offset

    def support(self, direction: ArrayLike) -> Vec3:
        direction = _vec3(direction)
        axial_component = float(np.dot(direction, self.axis))
        radial_direction = direction - axial_component * self.axis
        radial_norm = float(np.linalg.norm(radial_direction))
        result = self.center + self.half_length * np.sign(axial_component) * self.axis
        if radial_norm > 0.0 and self.radius > 0.0:
            result = result + (self.radius / radial_norm) * radial_direction
        return result

    def translated(self, shift: ArrayLike) -> "Cylinder":
        return Cylinder(
            self.center + _vec3(shift), self.axis, self.half_length, self.radius
        )

    def characteristic_radius(self) -> float:
        return float(np.hypot(self.half_length, self.radius))


# 阈值判定同时保留距离上下界与闭式特例。
@dataclass(frozen=True)
class DistanceBounds:
    lower: float
    upper: float
    iterations: int
    converged: bool
    witness_a: Vec3 | None
    witness_b: Vec3 | None

    @property
    def estimate(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def classify(self, cutoff: float) -> str:
        if self.upper <= cutoff:
            return "connected"
        if self.lower > cutoff:
            return "separated"
        return "uncertain"


def _closed_form_bounds(
    distance: float,
    first: SupportShape,
    second: SupportShape,
    witness_a: Vec3 | None = None,
    witness_b: Vec3 | None = None,
) -> DistanceBounds:
    scale = max(
        1.0,
        distance,
        first.characteristic_radius(),
        second.characteristic_radius(),
        float(np.linalg.norm(first.center)),
        float(np.linalg.norm(second.center)),
    )
    guard = 64.0 * np.finfo(np.float64).eps * scale
    return DistanceBounds(
        max(0.0, distance - guard),
        distance + guard,
        0,
        True,
        witness_a,
        witness_b,
    )


def sphere_sphere_distance(first: Sphere, second: Sphere) -> float:
    return max(
        0.0,
        float(np.linalg.norm(first.center - second.center))
        - first.radius
        - second.radius,
    )


def point_cylinder_distance(point: ArrayLike, cylinder: Cylinder) -> float:
    offset = _vec3(point) - cylinder.center
    axial = float(np.dot(offset, cylinder.axis))
    radial = offset - axial * cylinder.axis
    axial_gap = max(abs(axial) - cylinder.half_length, 0.0)
    radial_gap = max(float(np.linalg.norm(radial)) - cylinder.radius, 0.0)
    return float(np.hypot(axial_gap, radial_gap))


def cylinder_sphere_distance(cylinder: Cylinder, sphere: Sphere) -> float:
    return max(
        0.0, point_cylinder_distance(sphere.center, cylinder) - sphere.radius
    )


def _projection_extent(shape: SupportShape, normal: Vec3) -> tuple[float, float]:
    if isinstance(shape, Sphere):
        center_projection = float(np.dot(normal, shape.center))
        return (
            center_projection - shape.radius,
            center_projection + shape.radius,
        )
    if isinstance(shape, Cylinder):
        cosine = float(np.clip(np.dot(normal, shape.axis), -1.0, 1.0))
        extent = (
            shape.half_length * abs(cosine)
            + shape.radius * np.sqrt(max(0.0, 1.0 - cosine * cosine))
        )
        center_projection = float(np.dot(normal, shape.center))
        return center_projection - extent, center_projection + extent
    lower = float(np.dot(normal, shape.support(-normal)))
    upper = float(np.dot(normal, shape.support(normal)))
    return lower, upper


def shape_plane_distance(
    shape: SupportShape, normal: ArrayLike, offset: float
) -> float:
    normal = _unit(normal)
    if not np.isfinite(offset):
        raise ValueError("plane offset must be finite")
    lower, upper = _projection_extent(shape, normal)
    if offset < lower:
        return lower - offset
    if offset > upper:
        return offset - upper
    return 0.0


def _parallel_cylinder_distance(first: Cylinder, second: Cylinder) -> float:
    center_offset = second.center - first.center
    axial_offset = float(np.dot(center_offset, first.axis))
    radial_offset = center_offset - axial_offset * first.axis
    axial_gap = max(
        abs(axial_offset) - first.half_length - second.half_length, 0.0
    )
    radial_gap = max(
        float(np.linalg.norm(radial_offset)) - first.radius - second.radius, 0.0
    )
    return float(np.hypot(axial_gap, radial_gap))


# GJK 在 Minkowski 差集上迭代，并保留可行上界与支持平面下界。
def _project_origin_to_convex_hull(
    points: list[Vec3], weight_tolerance: float = 1e-12
) -> tuple[Vec3, tuple[int, ...], Vec3]:
    best_norm_sq = np.inf
    best_point: Vec3 | None = None
    best_indices: tuple[int, ...] | None = None
    best_weights: Vec3 | None = None
    max_subset = min(4, len(points))

    for subset_size in range(1, max_subset + 1):
        for indices in combinations(range(len(points)), subset_size):
            vertices = np.stack([points[index] for index in indices])
            if subset_size == 1:
                weights = np.ones(1, dtype=np.float64)
            else:
                anchor = vertices[0]
                directions = (vertices[1:] - anchor).T
                coefficients = np.linalg.lstsq(
                    directions, -anchor, rcond=None
                )[0]
                weights = np.concatenate(
                    ([1.0 - float(np.sum(coefficients))], coefficients)
                )
            if float(np.min(weights)) < -weight_tolerance:
                continue
            weights = np.maximum(weights, 0.0)
            weight_sum = float(np.sum(weights))
            if weight_sum == 0.0:
                continue
            weights /= weight_sum
            projection = weights @ vertices
            norm_sq = float(np.dot(projection, projection))
            if norm_sq < best_norm_sq:
                best_norm_sq = norm_sq
                best_point = projection
                best_indices = indices
                best_weights = weights

    if best_point is None or best_indices is None or best_weights is None:
        norms = [float(np.dot(point, point)) for point in points]
        index = int(np.argmin(norms))
        return points[index].copy(), (index,), np.ones(1, dtype=np.float64)
    return best_point, best_indices, best_weights


def _minkowski_support(
    first: SupportShape, second: SupportShape, direction: Vec3
) -> tuple[Vec3, Vec3, Vec3]:
    point_a = first.support(direction)
    point_b = second.support(-direction)
    return point_a - point_b, point_a, point_b


# 关键：GJK 迭代给出凸体表面距离的上下界。
def gjk_distance_bounds(
    first: SupportShape,
    second: SupportShape,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-12,
    max_iterations: int = 256,
) -> DistanceBounds:
    if absolute_tolerance <= 0.0 or relative_tolerance < 0.0:
        raise ValueError("invalid GJK tolerance")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    initial_a = first.center.copy()
    initial_b = second.center.copy()
    simplex_z = [initial_a - initial_b]
    simplex_a = [initial_a]
    simplex_b = [initial_b]
    current = simplex_z[0].copy()
    witness_a = initial_a.copy()
    witness_b = initial_b.copy()
    lower = 0.0
    upper = float(np.linalg.norm(current))
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        upper = float(np.linalg.norm(current))
        if upper <= absolute_tolerance:
            lower = 0.0
            converged = True
            break

        support_z, support_a, support_b = _minkowski_support(
            first, second, -current
        )
        lower = max(0.0, float(np.dot(current, support_z)) / upper)
        lower = min(lower, upper)
        tolerance = absolute_tolerance + relative_tolerance * upper
        if upper - lower <= tolerance:
            converged = True
            break

        duplicate_scale = max(
            1.0,
            first.characteristic_radius(),
            second.characteristic_radius(),
            float(np.linalg.norm(first.center)),
            float(np.linalg.norm(second.center)),
        )
        duplicate_tolerance = 64.0 * np.finfo(np.float64).eps * duplicate_scale
        if any(
            float(np.linalg.norm(support_z - old_point)) <= duplicate_tolerance
            for old_point in simplex_z
        ):
            break

        candidate_z = simplex_z + [support_z]
        candidate_a = simplex_a + [support_a]
        candidate_b = simplex_b + [support_b]
        projected, active_indices, weights = _project_origin_to_convex_hull(
            candidate_z
        )
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm > upper + duplicate_tolerance:
            break

        simplex_z = [candidate_z[index] for index in active_indices]
        simplex_a = [candidate_a[index] for index in active_indices]
        simplex_b = [candidate_b[index] for index in active_indices]
        witness_a = sum(
            (weight * point for weight, point in zip(weights, simplex_a)),
            start=np.zeros(3, dtype=np.float64),
        )
        witness_b = sum(
            (weight * point for weight, point in zip(weights, simplex_b)),
            start=np.zeros(3, dtype=np.float64),
        )
        current = witness_a - witness_b

    scale = max(
        1.0,
        upper,
        first.characteristic_radius(),
        second.characteristic_radius(),
        float(np.linalg.norm(first.center)),
        float(np.linalg.norm(second.center)),
    )
    floating_guard = 256.0 * np.finfo(np.float64).eps * scale
    return DistanceBounds(
        lower=max(0.0, lower - floating_guard),
        upper=max(0.0, upper + floating_guard),
        iterations=iteration,
        converged=converged,
        witness_a=witness_a,
        witness_b=witness_b,
    )


# 关键：统一调度闭式解、平行情形和通用 GJK 距离核。
def distance_bounds(
    first: SupportShape,
    second: SupportShape,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-12,
    max_iterations: int = 256,
) -> DistanceBounds:
    if isinstance(first, Sphere) and isinstance(second, Sphere):
        distance = sphere_sphere_distance(first, second)
        return _closed_form_bounds(distance, first, second)
    if isinstance(first, Cylinder) and isinstance(second, Sphere):
        distance = cylinder_sphere_distance(first, second)
        return _closed_form_bounds(distance, first, second)
    if isinstance(first, Sphere) and isinstance(second, Cylinder):
        result = distance_bounds(
            second,
            first,
            absolute_tolerance,
            relative_tolerance,
            max_iterations,
        )
        return DistanceBounds(
            result.lower,
            result.upper,
            result.iterations,
            result.converged,
            result.witness_b,
            result.witness_a,
        )
    if isinstance(first, Cylinder) and isinstance(second, Cylinder):
        parallel_tolerance = 64.0 * np.finfo(np.float64).eps
        if (
            float(np.linalg.norm(np.cross(first.axis, second.axis)))
            <= parallel_tolerance
        ):
            distance = _parallel_cylinder_distance(first, second)
            return _closed_form_bounds(distance, first, second)
    return gjk_distance_bounds(
        first,
        second,
        absolute_tolerance,
        relative_tolerance,
        max_iterations,
    )


# 胶囊距离仅作安全下界宽相，不承担最终接触判定。
def _segment_segment_distance(
    first_a: Vec3, first_b: Vec3, second_a: Vec3, second_b: Vec3
) -> float:
    first_direction = first_b - first_a
    second_direction = second_b - second_a
    offset = first_a - second_a
    first_length_sq = float(np.dot(first_direction, first_direction))
    second_length_sq = float(np.dot(second_direction, second_direction))
    cross_term = float(np.dot(first_direction, second_direction))
    first_offset = float(np.dot(first_direction, offset))
    second_offset = float(np.dot(second_direction, offset))
    tiny = 64.0 * np.finfo(np.float64).eps * max(
        1.0, first_length_sq, second_length_sq
    )

    if first_length_sq <= tiny and second_length_sq <= tiny:
        return float(np.linalg.norm(first_a - second_a))
    if first_length_sq <= tiny:
        first_parameter = 0.0
        second_parameter = np.clip(
            second_offset / second_length_sq, 0.0, 1.0
        )
    else:
        if second_length_sq <= tiny:
            second_parameter = 0.0
            first_parameter = np.clip(-first_offset / first_length_sq, 0.0, 1.0)
        else:
            denominator = (
                first_length_sq * second_length_sq - cross_term * cross_term
            )
            if denominator > tiny:
                first_parameter = np.clip(
                    (cross_term * second_offset - first_offset * second_length_sq)
                    / denominator,
                    0.0,
                    1.0,
                )
            else:
                first_parameter = 0.0
            second_parameter = (
                cross_term * first_parameter + second_offset
            ) / second_length_sq
            if second_parameter < 0.0:
                second_parameter = 0.0
                first_parameter = np.clip(
                    -first_offset / first_length_sq, 0.0, 1.0
                )
            elif second_parameter > 1.0:
                second_parameter = 1.0
                first_parameter = np.clip(
                    (cross_term - first_offset) / first_length_sq, 0.0, 1.0
                )

    first_point = first_a + first_parameter * first_direction
    second_point = second_a + second_parameter * second_direction
    return float(np.linalg.norm(first_point - second_point))


def capsule_cylinder_distance(first: Cylinder, second: Cylinder) -> float:
    first_a, first_b = first.endpoints
    second_a, second_b = second.endpoints
    return max(
        0.0,
        _segment_segment_distance(first_a, first_b, second_a, second_b)
        - first.radius
        - second.radius,
    )


def capsule_cylinder_sphere_distance(
    cylinder: Cylinder, sphere: Sphere
) -> float:
    endpoint_a, endpoint_b = cylinder.endpoints
    direction = endpoint_b - endpoint_a
    length_sq = float(np.dot(direction, direction))
    if length_sq == 0.0:
        axis_distance = float(np.linalg.norm(sphere.center - endpoint_a))
    else:
        parameter = float(
            np.clip(
                np.dot(sphere.center - endpoint_a, direction) / length_sq,
                0.0,
                1.0,
            )
        )
        closest = endpoint_a + parameter * direction
        axis_distance = float(np.linalg.norm(sphere.center - closest))
    return max(0.0, axis_distance - cylinder.radius - sphere.radius)


# 环面最小镜像仅用于明确的周期域，Q1 附件图不调用。
def periodic_distance_bounds(
    first: SupportShape,
    second: SupportShape,
    box_lengths: ArrayLike,
    periodic_axes: tuple[bool, bool, bool] = (True, True, True),
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-12,
    max_iterations: int = 256,
) -> DistanceBounds:
    lengths = _vec3(box_lengths)
    if np.any(lengths <= 0.0):
        raise ValueError("box lengths must be positive")
    shift_choices = [(-1, 0, 1) if periodic else (0,) for periodic in periodic_axes]
    results: list[DistanceBounds] = []
    for indices in product(*shift_choices):
        shift = lengths * np.asarray(indices, dtype=np.float64)
        results.append(
            distance_bounds(
                first,
                second.translated(shift),
                absolute_tolerance,
                relative_tolerance,
                max_iterations,
            )
        )
    best_upper = min(results, key=lambda result: result.upper)
    return DistanceBounds(
        lower=min(result.lower for result in results),
        upper=best_upper.upper,
        iterations=sum(result.iterations for result in results),
        converged=all(result.converged for result in results),
        witness_a=best_upper.witness_a,
        witness_b=best_upper.witness_b,
    )


def aabb(shape: SupportShape, inflation: float = 0.0) -> tuple[Vec3, Vec3]:
    if inflation < 0.0 or not np.isfinite(inflation):
        raise ValueError("inflation must be finite and nonnegative")
    if isinstance(shape, Sphere):
        extent = np.full(3, shape.radius + inflation, dtype=np.float64)
    elif isinstance(shape, Cylinder):
        axis_sq = np.square(shape.axis)
        extent = (
            shape.half_length * np.abs(shape.axis)
            + shape.radius * np.sqrt(np.maximum(0.0, 1.0 - axis_sq))
            + inflation
        )
    else:
        unit_axes = np.eye(3, dtype=np.float64)
        lower = np.array(
            [shape.support(-axis)[i] for i, axis in enumerate(unit_axes)]
        )
        upper = np.array(
            [shape.support(axis)[i] for i, axis in enumerate(unit_axes)]
        )
        return lower - inflation, upper + inflation
    return shape.center - extent, shape.center + extent
