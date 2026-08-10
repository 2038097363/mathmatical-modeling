from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from geometry_kernel import (
    Cylinder,
    DistanceBounds,
    distance_bounds as reference_distance_bounds,
)

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        del args, kwargs

        def decorate(function):
            return function

        return decorate


_FLOAT_EPSILON = np.finfo(np.float64).eps
_FLOAT_TINY = np.finfo(np.float64).tiny
_WEIGHT_TOLERANCE = 1e-12
_FAST_FLOATING_GUARD_MULTIPLIER = 4096.0
_DEFAULT_THRESHOLD_GUARD = 1e-8


@dataclass(frozen=True)
class FastDistanceResult:
    """距离上下界及参考核回退审计记录。"""

    bounds: DistanceBounds
    used_fallback: bool
    fallback_reason: str | None
    termination_reason: str


@dataclass(frozen=True)
class FastClassificationResult:
    """严格阈值判定，区分距离收敛与决策收敛。"""

    classification: str
    bounds: DistanceBounds
    distance_converged: bool
    threshold_certified: bool
    used_fallback: bool
    fallback_reason: str | None


@njit(cache=True, nogil=True)
def _dot3(
    first_x: float,
    first_y: float,
    first_z: float,
    second_x: float,
    second_y: float,
    second_z: float,
) -> float:
    return (
        first_x * second_x
        + first_y * second_y
        + first_z * second_z
    )


@njit(cache=True, nogil=True)
def _norm3(x_value: float, y_value: float, z_value: float) -> float:
    return math.sqrt(x_value * x_value + y_value * y_value + z_value * z_value)


@njit(cache=True, nogil=True)
def _determinant3(
    a00: float,
    a01: float,
    a02: float,
    a10: float,
    a11: float,
    a12: float,
    a20: float,
    a21: float,
    a22: float,
) -> float:
    return (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )


@njit(cache=True, nogil=True)
def _support_cylinder(
    center: np.ndarray,
    axis: np.ndarray,
    half_length: float,
    radius: float,
    direction_x: float,
    direction_y: float,
    direction_z: float,
) -> tuple[float, float, float]:
    axial = _dot3(
        direction_x,
        direction_y,
        direction_z,
        axis[0],
        axis[1],
        axis[2],
    )
    support_x = center[0]
    support_y = center[1]
    support_z = center[2]
    if axial > 0.0:
        support_x += half_length * axis[0]
        support_y += half_length * axis[1]
        support_z += half_length * axis[2]
    elif axial < 0.0:
        support_x -= half_length * axis[0]
        support_y -= half_length * axis[1]
        support_z -= half_length * axis[2]

    radial_x = direction_x - axial * axis[0]
    radial_y = direction_y - axial * axis[1]
    radial_z = direction_z - axial * axis[2]
    radial_norm = _norm3(radial_x, radial_y, radial_z)
    if radial_norm > 0.0 and radius > 0.0:
        factor = radius / radial_norm
        support_x += factor * radial_x
        support_y += factor * radial_y
        support_z += factor * radial_z
    return support_x, support_y, support_z


@njit(cache=True, nogil=True)
def _minkowski_support_cylinders(
    first_center: np.ndarray,
    first_axis: np.ndarray,
    first_half_length: float,
    first_radius: float,
    second_center: np.ndarray,
    second_axis: np.ndarray,
    second_half_length: float,
    second_radius: float,
    direction_x: float,
    direction_y: float,
    direction_z: float,
) -> tuple[float, float, float]:
    first_x, first_y, first_z = _support_cylinder(
        first_center,
        first_axis,
        first_half_length,
        first_radius,
        direction_x,
        direction_y,
        direction_z,
    )
    second_x, second_y, second_z = _support_cylinder(
        second_center,
        second_axis,
        second_half_length,
        second_radius,
        -direction_x,
        -direction_y,
        -direction_z,
    )
    return first_x - second_x, first_y - second_y, first_z - second_z


@njit(cache=True, nogil=True)
def _project_origin_to_simplex(
    points: np.ndarray, point_count: int
) -> tuple[float, float, float, int, bool]:
    """投影到至多 5 点的凸包，并压缩为至多 4 点的有效单纯形。"""
    best_norm_sq = math.inf
    best_point_x = 0.0
    best_point_y = 0.0
    best_point_z = 0.0
    best_count = 0
    best_indices = np.empty(4, dtype=np.int64)
    indices = np.empty(4, dtype=np.int64)
    weights = np.empty(4, dtype=np.float64)

    for mask in range(1, 1 << point_count):
        subset_count = 0
        for point_index in range(point_count):
            if mask & (1 << point_index):
                if subset_count >= 4:
                    subset_count = 5
                    break
                indices[subset_count] = point_index
                subset_count += 1
        if subset_count > 4:
            continue

        for index in range(4):
            weights[index] = 0.0
        valid = True
        anchor_index = indices[0]
        anchor_x = points[anchor_index, 0]
        anchor_y = points[anchor_index, 1]
        anchor_z = points[anchor_index, 2]

        if subset_count == 1:
            weights[0] = 1.0
        elif subset_count == 2:
            second_index = indices[1]
            direction_x = points[second_index, 0] - anchor_x
            direction_y = points[second_index, 1] - anchor_y
            direction_z = points[second_index, 2] - anchor_z
            denominator = _dot3(
                direction_x,
                direction_y,
                direction_z,
                direction_x,
                direction_y,
                direction_z,
            )
            scale = max(
                1.0,
                _dot3(anchor_x, anchor_y, anchor_z, anchor_x, anchor_y, anchor_z),
                denominator,
            )
            if denominator <= _FLOAT_TINY * scale:
                valid = False
            else:
                coefficient = -_dot3(
                    anchor_x,
                    anchor_y,
                    anchor_z,
                    direction_x,
                    direction_y,
                    direction_z,
                ) / denominator
                weights[0] = 1.0 - coefficient
                weights[1] = coefficient
        elif subset_count == 3:
            second_index = indices[1]
            third_index = indices[2]
            first_x = points[second_index, 0] - anchor_x
            first_y = points[second_index, 1] - anchor_y
            first_z = points[second_index, 2] - anchor_z
            second_x = points[third_index, 0] - anchor_x
            second_y = points[third_index, 1] - anchor_y
            second_z = points[third_index, 2] - anchor_z
            gram_00 = _dot3(first_x, first_y, first_z, first_x, first_y, first_z)
            gram_01 = _dot3(first_x, first_y, first_z, second_x, second_y, second_z)
            gram_11 = _dot3(second_x, second_y, second_z, second_x, second_y, second_z)
            right_0 = -_dot3(anchor_x, anchor_y, anchor_z, first_x, first_y, first_z)
            right_1 = -_dot3(anchor_x, anchor_y, anchor_z, second_x, second_y, second_z)
            determinant = gram_00 * gram_11 - gram_01 * gram_01
            gram_scale = max(1.0, gram_00, gram_11)
            if abs(determinant) <= _FLOAT_TINY * gram_scale * gram_scale:
                valid = False
            else:
                first_coefficient = (
                    right_0 * gram_11 - gram_01 * right_1
                ) / determinant
                second_coefficient = (
                    gram_00 * right_1 - right_0 * gram_01
                ) / determinant
                weights[0] = 1.0 - first_coefficient - second_coefficient
                weights[1] = first_coefficient
                weights[2] = second_coefficient
        else:
            second_index = indices[1]
            third_index = indices[2]
            fourth_index = indices[3]
            first_x = points[second_index, 0] - anchor_x
            first_y = points[second_index, 1] - anchor_y
            first_z = points[second_index, 2] - anchor_z
            second_x = points[third_index, 0] - anchor_x
            second_y = points[third_index, 1] - anchor_y
            second_z = points[third_index, 2] - anchor_z
            third_x = points[fourth_index, 0] - anchor_x
            third_y = points[fourth_index, 1] - anchor_y
            third_z = points[fourth_index, 2] - anchor_z

            gram_00 = _dot3(first_x, first_y, first_z, first_x, first_y, first_z)
            gram_01 = _dot3(first_x, first_y, first_z, second_x, second_y, second_z)
            gram_02 = _dot3(first_x, first_y, first_z, third_x, third_y, third_z)
            gram_11 = _dot3(second_x, second_y, second_z, second_x, second_y, second_z)
            gram_12 = _dot3(second_x, second_y, second_z, third_x, third_y, third_z)
            gram_22 = _dot3(third_x, third_y, third_z, third_x, third_y, third_z)
            right_0 = -_dot3(anchor_x, anchor_y, anchor_z, first_x, first_y, first_z)
            right_1 = -_dot3(anchor_x, anchor_y, anchor_z, second_x, second_y, second_z)
            right_2 = -_dot3(anchor_x, anchor_y, anchor_z, third_x, third_y, third_z)
            determinant = _determinant3(
                gram_00,
                gram_01,
                gram_02,
                gram_01,
                gram_11,
                gram_12,
                gram_02,
                gram_12,
                gram_22,
            )
            gram_scale = max(1.0, gram_00, gram_11, gram_22)
            if abs(determinant) <= _FLOAT_TINY * gram_scale**3:
                valid = False
            else:
                first_coefficient = _determinant3(
                    right_0,
                    gram_01,
                    gram_02,
                    right_1,
                    gram_11,
                    gram_12,
                    right_2,
                    gram_12,
                    gram_22,
                ) / determinant
                second_coefficient = _determinant3(
                    gram_00,
                    right_0,
                    gram_02,
                    gram_01,
                    right_1,
                    gram_12,
                    gram_02,
                    right_2,
                    gram_22,
                ) / determinant
                third_coefficient = _determinant3(
                    gram_00,
                    gram_01,
                    right_0,
                    gram_01,
                    gram_11,
                    right_1,
                    gram_02,
                    gram_12,
                    right_2,
                ) / determinant
                weights[0] = (
                    1.0
                    - first_coefficient
                    - second_coefficient
                    - third_coefficient
                )
                weights[1] = first_coefficient
                weights[2] = second_coefficient
                weights[3] = third_coefficient

        if not valid:
            continue
        weight_sum = 0.0
        for index in range(subset_count):
            if weights[index] < -_WEIGHT_TOLERANCE:
                valid = False
                break
            if weights[index] < 0.0:
                weights[index] = 0.0
            weight_sum += weights[index]
        if not valid or weight_sum == 0.0 or not math.isfinite(weight_sum):
            continue

        projection_x = 0.0
        projection_y = 0.0
        projection_z = 0.0
        for index in range(subset_count):
            weights[index] /= weight_sum
            point_index = indices[index]
            projection_x += weights[index] * points[point_index, 0]
            projection_y += weights[index] * points[point_index, 1]
            projection_z += weights[index] * points[point_index, 2]
        norm_sq = _dot3(
            projection_x,
            projection_y,
            projection_z,
            projection_x,
            projection_y,
            projection_z,
        )
        if math.isfinite(norm_sq) and norm_sq < best_norm_sq:
            best_norm_sq = norm_sq
            best_point_x = projection_x
            best_point_y = projection_y
            best_point_z = projection_z
            best_count = subset_count
            for index in range(subset_count):
                best_indices[index] = indices[index]

    if best_count == 0:
        return 0.0, 0.0, 0.0, 0, False

    for index in range(best_count):
        source_index = best_indices[index]
        points[index, 0] = points[source_index, 0]
        points[index, 1] = points[source_index, 1]
        points[index, 2] = points[source_index, 2]
    return best_point_x, best_point_y, best_point_z, best_count, True


@njit(cache=True, nogil=True)
def _parallel_cylinder_bounds(
    first_center: np.ndarray,
    first_axis: np.ndarray,
    first_half_length: float,
    first_radius: float,
    second_center: np.ndarray,
    second_half_length: float,
    second_radius: float,
) -> tuple[float, float]:
    offset_x = second_center[0] - first_center[0]
    offset_y = second_center[1] - first_center[1]
    offset_z = second_center[2] - first_center[2]
    axial_offset = _dot3(
        offset_x,
        offset_y,
        offset_z,
        first_axis[0],
        first_axis[1],
        first_axis[2],
    )
    radial_x = offset_x - axial_offset * first_axis[0]
    radial_y = offset_y - axial_offset * first_axis[1]
    radial_z = offset_z - axial_offset * first_axis[2]
    axial_gap = max(
        abs(axial_offset) - first_half_length - second_half_length,
        0.0,
    )
    radial_gap = max(
        _norm3(radial_x, radial_y, radial_z) - first_radius - second_radius,
        0.0,
    )
    distance = math.sqrt(axial_gap * axial_gap + radial_gap * radial_gap)
    scale = max(
        1.0,
        distance,
        math.sqrt(first_half_length**2 + first_radius**2),
        math.sqrt(second_half_length**2 + second_radius**2),
        _norm3(first_center[0], first_center[1], first_center[2]),
        _norm3(second_center[0], second_center[1], second_center[2]),
    )
    guard = 64.0 * _FLOAT_EPSILON * scale
    return max(0.0, distance - guard), distance + guard


@njit(cache=True, nogil=True)
# 关键：使用 Numba 加速有限圆柱距离界的核心循环。
def _numba_cylinder_distance_bounds(
    first_center: np.ndarray,
    first_axis: np.ndarray,
    first_half_length: float,
    first_radius: float,
    second_center: np.ndarray,
    second_axis: np.ndarray,
    second_half_length: float,
    second_radius: float,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_iterations: int,
    cutoff: float,
    threshold_guard: float,
) -> tuple[float, float, int, bool, int]:
    cross_x = first_axis[1] * second_axis[2] - first_axis[2] * second_axis[1]
    cross_y = first_axis[2] * second_axis[0] - first_axis[0] * second_axis[2]
    cross_z = first_axis[0] * second_axis[1] - first_axis[1] * second_axis[0]
    if _norm3(cross_x, cross_y, cross_z) <= 64.0 * _FLOAT_EPSILON:
        lower, upper = _parallel_cylinder_bounds(
            first_center,
            first_axis,
            first_half_length,
            first_radius,
            second_center,
            second_half_length,
            second_radius,
        )
        return lower, upper, 0, True, 5

    simplex = np.empty((5, 3), dtype=np.float64)
    current_x = first_center[0] - second_center[0]
    current_y = first_center[1] - second_center[1]
    current_z = first_center[2] - second_center[2]
    simplex[0, 0] = current_x
    simplex[0, 1] = current_y
    simplex[0, 2] = current_z
    simplex_count = 1
    lower = 0.0
    upper = _norm3(current_x, current_y, current_z)
    converged = False
    status = 3
    iteration = 0
    characteristic_first = math.sqrt(first_half_length**2 + first_radius**2)
    characteristic_second = math.sqrt(second_half_length**2 + second_radius**2)
    center_norm_first = _norm3(first_center[0], first_center[1], first_center[2])
    center_norm_second = _norm3(second_center[0], second_center[1], second_center[2])
    duplicate_scale = max(
        1.0,
        characteristic_first,
        characteristic_second,
        center_norm_first,
        center_norm_second,
    )
    duplicate_tolerance = 64.0 * _FLOAT_EPSILON * duplicate_scale
    duplicate_tolerance_sq = duplicate_tolerance * duplicate_tolerance
    numeric_guard = (
        _FAST_FLOATING_GUARD_MULTIPLIER
        * _FLOAT_EPSILON
        * max(duplicate_scale, upper)
    )

    for iteration in range(1, max_iterations + 1):
        upper = _norm3(current_x, current_y, current_z)
        if upper <= absolute_tolerance:
            lower = 0.0
            converged = True
            status = 0
            break
        if cutoff >= 0.0 and upper + numeric_guard <= cutoff - threshold_guard:
            lower = min(lower, upper)
            status = 6
            break

        support_x, support_y, support_z = _minkowski_support_cylinders(
            first_center,
            first_axis,
            first_half_length,
            first_radius,
            second_center,
            second_axis,
            second_half_length,
            second_radius,
            -current_x,
            -current_y,
            -current_z,
        )
        lower = max(
            0.0,
            _dot3(
                current_x,
                current_y,
                current_z,
                support_x,
                support_y,
                support_z,
            )
            / upper,
        )
        lower = min(lower, upper)
        if cutoff >= 0.0 and lower - numeric_guard > cutoff + threshold_guard:
            status = 7
            break
        tolerance = absolute_tolerance + relative_tolerance * upper
        if upper - lower <= tolerance:
            converged = True
            status = 0
            break

        duplicate = False
        for point_index in range(simplex_count):
            difference_x = support_x - simplex[point_index, 0]
            difference_y = support_y - simplex[point_index, 1]
            difference_z = support_z - simplex[point_index, 2]
            if (
                difference_x * difference_x
                + difference_y * difference_y
                + difference_z * difference_z
                <= duplicate_tolerance_sq
            ):
                duplicate = True
                break
        if duplicate:
            status = 1
            break

        simplex[simplex_count, 0] = support_x
        simplex[simplex_count, 1] = support_y
        simplex[simplex_count, 2] = support_z
        projected_x, projected_y, projected_z, simplex_count, projection_ok = (
            _project_origin_to_simplex(simplex, simplex_count + 1)
        )
        if not projection_ok:
            status = 4
            break
        projected_norm = _norm3(projected_x, projected_y, projected_z)
        if projected_norm > upper + duplicate_tolerance:
            status = 2
            break
        current_x = projected_x
        current_y = projected_y
        current_z = projected_z

    scale = max(
        1.0,
        upper,
        characteristic_first,
        characteristic_second,
        center_norm_first,
        center_norm_second,
    )
    floating_guard = _FAST_FLOATING_GUARD_MULTIPLIER * _FLOAT_EPSILON * scale
    return (
        max(0.0, lower - floating_guard),
        max(0.0, upper + floating_guard),
        iteration,
        converged,
        status,
    )


def _reference_result(
    first: Cylinder,
    second: Cylinder,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_iterations: int,
    reason: str,
) -> FastDistanceResult:
    return FastDistanceResult(
        bounds=reference_distance_bounds(
            first,
            second,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
        ),
        used_fallback=True,
        fallback_reason=reason,
        termination_reason="reference_fallback",
    )


def fast_cylinder_distance_diagnostics(
    first: Cylinder,
    second: Cylinder,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-12,
    max_iterations: int = 256,
    *,
    cutoff: float | None = None,
    threshold_guard: float = _DEFAULT_THRESHOLD_GUARD,
) -> FastDistanceResult:
    """返回保守距离界；距 ``cutoff`` 不超过保护量时改用参考核复算。"""
    if not isinstance(first, Cylinder) or not isinstance(second, Cylinder):
        raise TypeError("fast geometry only accepts two Cylinder instances")
    if absolute_tolerance <= 0.0 or relative_tolerance < 0.0:
        raise ValueError("invalid GJK tolerance")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff < 0.0):
        raise ValueError("cutoff must be finite and nonnegative")
    if not np.isfinite(threshold_guard) or threshold_guard < 0.0:
        raise ValueError("threshold_guard must be finite and nonnegative")
    if not NUMBA_AVAILABLE:
        return _reference_result(
            first,
            second,
            absolute_tolerance,
            relative_tolerance,
            max_iterations,
            "numba_unavailable",
        )

    lower, upper, iterations, converged, status = _numba_cylinder_distance_bounds(
        first.center,
        first.axis,
        first.half_length,
        first.radius,
        second.center,
        second.axis,
        second.half_length,
        second.radius,
        absolute_tolerance,
        relative_tolerance,
        max_iterations,
        -1.0 if cutoff is None else cutoff,
        threshold_guard,
    )
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower < 0.0
        or upper < lower
    ):
        return _reference_result(
            first,
            second,
            absolute_tolerance,
            relative_tolerance,
            max_iterations,
            "invalid_fast_bounds",
        )
    if status not in (0, 5, 6, 7):
        status_reasons = {
            1: "duplicate_support_before_convergence",
            2: "projection_progress_guard",
            3: "iteration_limit",
            4: "simplex_projection_failure",
        }
        return _reference_result(
            first,
            second,
            absolute_tolerance,
            relative_tolerance,
            max_iterations,
            status_reasons.get(status, "unknown_fast_status"),
        )

    fast_bounds = DistanceBounds(
        lower=lower,
        upper=upper,
        iterations=iterations,
        converged=converged,
        witness_a=None,
        witness_b=None,
    )
    if cutoff is not None:
        classification = fast_bounds.classify(cutoff)
        guard = max(threshold_guard, 8.0 * absolute_tolerance)
        if classification == "uncertain" or min(
            abs(fast_bounds.lower - cutoff),
            abs(fast_bounds.upper - cutoff),
        ) <= guard:
            return _reference_result(
                first,
                second,
                absolute_tolerance,
                relative_tolerance,
                max_iterations,
                "threshold_guard",
            )
    termination_reasons = {
        0: "distance_tolerance",
        5: "parallel_closed_form",
        6: "connected_upper_bound_certificate",
        7: "separated_lower_bound_certificate",
    }
    return FastDistanceResult(
        fast_bounds,
        False,
        None,
        termination_reasons[status],
    )


def fast_cylinder_distance_bounds(
    first: Cylinder,
    second: Cylinder,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-12,
    max_iterations: int = 256,
    *,
    cutoff: float | None = None,
    threshold_guard: float = _DEFAULT_THRESHOLD_GUARD,
) -> DistanceBounds:
    """返回圆柱距离界；严格阈值判定时必须传入 ``cutoff``。"""
    return fast_cylinder_distance_diagnostics(
        first,
        second,
        absolute_tolerance,
        relative_tolerance,
        max_iterations,
        cutoff=cutoff,
        threshold_guard=threshold_guard,
    ).bounds


# 关键：依据距离界与接触阈值返回确定分类或不确定状态。
def fast_cylinder_classify(
    first: Cylinder,
    second: Cylinder,
    cutoff: float,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-12,
    max_iterations: int = 256,
    *,
    threshold_guard: float = _DEFAULT_THRESHOLD_GUARD,
) -> FastClassificationResult:
    """依据认证距离界分类，并在临界区回退参考核。"""
    result = fast_cylinder_distance_diagnostics(
        first,
        second,
        absolute_tolerance,
        relative_tolerance,
        max_iterations,
        cutoff=cutoff,
        threshold_guard=threshold_guard,
    )
    classification = result.bounds.classify(cutoff)
    return FastClassificationResult(
        classification=classification,
        bounds=result.bounds,
        distance_converged=result.bounds.converged,
        threshold_certified=classification != "uncertain",
        used_fallback=result.used_fallback,
        fallback_reason=result.fallback_reason,
    )


def warm_up_fast_geometry() -> None:
    if not NUMBA_AVAILABLE:
        return
    first = Cylinder(
        np.zeros(3, dtype=np.float64),
        np.array([1.0, 0.0, 0.0]),
        1.0,
        0.1,
    )
    second = Cylinder(
        np.array([2.5, 0.5, 0.25], dtype=np.float64),
        np.array([0.0, 1.0, 0.0]),
        1.0,
        0.1,
    )
    fast_cylinder_distance_bounds(first, second)


__all__ = [
    "FastDistanceResult",
    "FastClassificationResult",
    "NUMBA_AVAILABLE",
    "fast_cylinder_distance_bounds",
    "fast_cylinder_distance_diagnostics",
    "fast_cylinder_classify",
    "warm_up_fast_geometry",
]
