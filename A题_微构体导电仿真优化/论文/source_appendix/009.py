"""Q4 主边界 D 下固定 A/B 数量的混合微构体仿真。

A 沿用 ``microstructure_sim`` 的 D 模式：按中心线与三轴边界交点切段、映回
基础立方体，同源片段不添加内部边。B 则按完整三维集合严格处理：枚举原球
与相邻周期胞的交集，将每个非零体积球片映回基础盒，并把各片视为独立凸体。
映回后不再使用最小镜像或 ghost；A 的半径侧缘和斜切面仍保留中心线切段近似。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from geometry_kernel import (
    Cylinder,
    Sphere,
    SupportShape,
    aabb,
    capsule_cylinder_distance,
    capsule_cylinder_sphere_distance,
    cylinder_sphere_distance,
    distance_bounds,
    shape_plane_distance,
    sphere_sphere_distance,
)
from microstructure_sim import (
    ALGORITHM_VERSION as A_ALGORITHM_VERSION,
    BoundaryMode,
    CylinderFragment,
    NarrowPhaseUncertainError,
    SimulationConfig,
    clopper_pearson_interval,
    draw_isotropic_directions,
    draw_uniform_centers,
    evaluate_fragment_contact,
    fragment_trial,
    wilson_interval,
)
from pareto_connectivity import (
    ParetoSearchDiagnostics,
    axis_threshold_pareto_result,
    design_is_connected,
    pareto_prune_labels,
)


Vec3 = NDArray[np.float64]
SCHEMA_VERSION = 1
ALGORITHM_VERSION = "mixed-fixed-design-d-clipped-sphere-v3"
B_CENTER_QUANTITY_CODE = 211

BOUNDARY_CONTRACT = {
    "mode": "D",
    "periodic_axes": "X/Y/Z explicit fragment relocation; no minimum image",
    "same_source_rule": (
        "relocated fragments are independent; same-source fragment pairs are excluded"
    ),
    "A_geometry": "reuse D-mode centerline cuts and mapped flat-cylinder fragments",
    "B_geometry": "exact ball-cell intersections mapped to the base box",
    "limitation": (
        "A uses centerline cuts: radius-side overflow and oblique Boolean cut faces "
        "are not reconstructed. B spherical fragments use exact convex intersections."
    ),
}


@dataclass(frozen=True)
class MixedSimulationConfig:
    n_a: int
    n_b: int
    trial_count: int
    master_seed: int = 20_260_801
    stream_id: int = 4
    boundary_mode: str = "D"
    box_length_nm: float = 10_000.0
    a_length_nm: float = 5_000.0
    a_radius_nm: float = 30.0
    b_radius_nm: float = 200.0
    contact_cutoff_nm: float = 1.8
    cell_size_nm: float = 625.0
    broad_phase_guard_nm: float = 1e-8
    gjk_absolute_tolerance_nm: float = 1e-10
    gjk_relative_tolerance: float = 1e-13
    gjk_max_iterations: int = 512
    algorithm_version: str = ALGORITHM_VERSION
    a_algorithm_version: str = A_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        parsed_boundary = (
            self.boundary_mode.value
            if isinstance(self.boundary_mode, BoundaryMode)
            else str(self.boundary_mode)
        )
        object.__setattr__(self, "boundary_mode", parsed_boundary)
        if self.n_a < 0 or self.n_b < 0:
            raise ValueError("n_a 与 n_b 必须为非负整数")
        if self.trial_count < 1:
            raise ValueError("trial_count 必须为正整数")
        if self.master_seed < 0 or self.stream_id < 0:
            raise ValueError("SeedSequence 标识必须非负")
        if self.boundary_mode != "D":
            raise ValueError("混合固定设计内核当前只实现主边界 D")
        positive = (
            self.box_length_nm,
            self.a_length_nm,
            self.a_radius_nm,
            self.b_radius_nm,
            self.contact_cutoff_nm,
            self.cell_size_nm,
            self.gjk_absolute_tolerance_nm,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("几何尺度和绝对容差必须为有限正数")
        if not np.isfinite(self.broad_phase_guard_nm) or self.broad_phase_guard_nm < 0.0:
            raise ValueError("宽相保护量必须有限且非负")
        if not np.isfinite(self.gjk_relative_tolerance) or self.gjk_relative_tolerance < 0.0:
            raise ValueError("GJK 相对容差必须有限且非负")
        if self.gjk_max_iterations < 1:
            raise ValueError("GJK 最大迭代次数必须为正")
        if self.b_radius_nm > self.box_length_nm:
            raise ValueError("球半径超过盒长时相邻周期胞枚举不完整")
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError("混合内核算法版本不一致")
        if self.a_algorithm_version != A_ALGORITHM_VERSION:
            raise ValueError("A 内核算法版本不一致")

    @property
    def half_box_nm(self) -> float:
        return 0.5 * self.box_length_nm

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": self.algorithm_version,
            "a_algorithm_version": self.a_algorithm_version,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "trial_count": self.trial_count,
            "master_seed": self.master_seed,
            "stream_id": self.stream_id,
            "boundary_mode": self.boundary_mode,
            "box_length_nm": self.box_length_nm,
            "a_length_nm": self.a_length_nm,
            "a_radius_nm": self.a_radius_nm,
            "b_radius_nm": self.b_radius_nm,
            "contact_cutoff_nm": self.contact_cutoff_nm,
            "cell_size_nm": self.cell_size_nm,
            "broad_phase_guard_nm": self.broad_phase_guard_nm,
            "gjk_absolute_tolerance_nm": self.gjk_absolute_tolerance_nm,
            "gjk_relative_tolerance": self.gjk_relative_tolerance,
            "gjk_max_iterations": self.gjk_max_iterations,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MixedSimulationConfig":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("不支持的混合仿真配置版本")
        values = dict(payload)
        values.pop("schema_version", None)
        return cls(**values)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(canonical).hexdigest().upper()

    def a_config(self) -> SimulationConfig:
        if self.n_a < 1:
            raise ValueError("n_a=0 时不存在 A 子配置")
        return SimulationConfig(
            max_count=self.n_a,
            trial_count=self.trial_count,
            boundary_mode=BoundaryMode.D,
            master_seed=self.master_seed,
            stream_id=self.stream_id,
            box_length_nm=self.box_length_nm,
            cylinder_length_nm=self.a_length_nm,
            cylinder_radius_nm=self.a_radius_nm,
            contact_cutoff_nm=self.contact_cutoff_nm,
            cell_size_nm=self.cell_size_nm,
            broad_phase_guard_nm=self.broad_phase_guard_nm,
            gjk_absolute_tolerance_nm=self.gjk_absolute_tolerance_nm,
            gjk_relative_tolerance=self.gjk_relative_tolerance,
            gjk_max_iterations=self.gjk_max_iterations,
        )


def _finite_vec3(value: ArrayLike, name: str) -> Vec3:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须为有限三维向量")
    return result.copy()


def _ball_slack(radius: float, relative_point: ArrayLike) -> float:
    point = np.asarray(relative_point, dtype=np.float64)
    maximum = max(float(radius), float(np.max(np.abs(point))))
    if maximum == 0.0:
        return 0.0
    _, exponent = math.frexp(maximum)
    scale = math.ldexp(1.0, exponent - 1)
    scaled_radius = float(radius) / scale
    scaled_point = point / scale
    radius_sq = scaled_radius * scaled_radius
    terms = [radius_sq]
    fused_multiply_add = getattr(math, "fma", None)
    if fused_multiply_add is None:
        raise RuntimeError("截球认证距离要求提供 math.fma 的 Python 3.13 或更高版本")
    terms.append(fused_multiply_add(scaled_radius, scaled_radius, -radius_sq))
    for value in scaled_point:
        square = float(value) * float(value)
        terms.append(-square)
        terms.append(-fused_multiply_add(float(value), float(value), -square))
    return math.fsum(terms)


@dataclass(frozen=True)
class ClippedSphere:
    sphere_center: Vec3
    radius: float
    box_lower: Vec3
    box_upper: Vec3

    def __post_init__(self) -> None:
        sphere_center = _finite_vec3(self.sphere_center, "球心")
        box_lower = _finite_vec3(self.box_lower, "盒下界")
        box_upper = _finite_vec3(self.box_upper, "盒上界")
        if np.any(box_lower > box_upper):
            raise ValueError("盒下界不能超过上界")
        if not np.isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("球半径必须有限且非负")
        nearest = np.clip(sphere_center, box_lower, box_upper)
        nearest_offset = nearest - sphere_center
        if _ball_slack(self.radius, nearest_offset) < 0.0:
            raise ValueError("球与轴对齐盒没有交集")
        object.__setattr__(self, "sphere_center", sphere_center)
        object.__setattr__(self, "box_lower", box_lower)
        object.__setattr__(self, "box_upper", box_upper)

    @property
    def center(self) -> Vec3:
        return np.clip(self.sphere_center, self.box_lower, self.box_upper)

    def coordinate_interval(self, axis: int) -> tuple[float, float]:
        if axis not in (0, 1, 2):
            raise ValueError("坐标轴编号必须为 0、1 或 2")
        nearest = np.clip(self.sphere_center, self.box_lower, self.box_upper)
        other_axes = [index for index in range(3) if index != axis]
        offsets = nearest[other_axes] - self.sphere_center[other_axes]
        perpendicular_sq = math.fsum(float(value) ** 2 for value in offsets)
        radius_sq = self.radius * self.radius
        projected_radius = math.sqrt(max(0.0, radius_sq - perpendicular_sq))
        lower = max(
            float(self.box_lower[axis]),
            float(self.sphere_center[axis]) - projected_radius,
        )
        upper = min(
            float(self.box_upper[axis]),
            float(self.sphere_center[axis]) + projected_radius,
        )
        scale = max(1.0, abs(lower), abs(upper), self.radius)
        guard = 64.0 * np.finfo(np.float64).eps * scale
        if lower > upper + guard:
            raise RuntimeError("截球坐标投影为空")
        if lower > upper:
            midpoint = 0.5 * (lower + upper)
            return midpoint, midpoint
        return lower, upper

    def exact_aabb(self, inflation: float = 0.0) -> tuple[Vec3, Vec3]:
        if inflation < 0.0 or not np.isfinite(inflation):
            raise ValueError("包围盒膨胀量必须有限且非负")
        intervals = [self.coordinate_interval(axis) for axis in range(3)]
        lower = np.asarray([interval[0] for interval in intervals]) - inflation
        upper = np.asarray([interval[1] for interval in intervals]) + inflation
        return lower, upper

    def _active_set_support_relative(
        self,
        unit_direction: Vec3,
        relative_lower: Vec3,
        relative_upper: Vec3,
        nearest_relative: Vec3,
    ) -> Vec3:
        radius_sq = self.radius * self.radius
        best = nearest_relative.copy()
        epsilon = np.finfo(np.float64).eps
        box_scale = max(
            1.0,
            float(np.max(np.abs(relative_lower))),
            float(np.max(np.abs(relative_upper))),
            self.radius,
        )
        box_guard = 64.0 * epsilon * box_scale
        for statuses in product((-1, 0, 1), repeat=3):
            candidate = nearest_relative.copy()
            fixed = np.asarray([status != 0 for status in statuses])
            for axis, status in enumerate(statuses):
                if status < 0:
                    candidate[axis] = relative_lower[axis]
                elif status > 0:
                    candidate[axis] = relative_upper[axis]
            fixed_sq = float(np.dot(candidate[fixed], candidate[fixed]))
            if fixed_sq > radius_sq:
                continue
            free = ~fixed
            free_direction = unit_direction[free]
            free_norm = float(np.linalg.norm(free_direction))
            if np.any(free) and free_norm > 0.0:
                remaining_sq = max(0.0, radius_sq - fixed_sq)
                candidate[free] = (
                    math.sqrt(remaining_sq) / free_norm
                ) * free_direction
                candidate[free] *= 1.0 - 32.0 * epsilon
            if np.any(candidate < relative_lower - box_guard) or np.any(
                candidate > relative_upper + box_guard
            ):
                continue
            candidate = np.clip(candidate, relative_lower, relative_upper)
            if _ball_slack(self.radius, candidate) < 0.0:
                continue
            improvement = math.fsum(
                float(unit_direction[axis])
                * float(candidate[axis] - best[axis])
                for axis in range(3)
            )
            if improvement > 0.0:
                best = candidate
        return best

    def support(self, direction: ArrayLike) -> Vec3:
        direction_vec = _finite_vec3(direction, "支持方向")
        direction_scale = float(np.max(np.abs(direction_vec)))
        relative_lower = self.box_lower - self.sphere_center
        relative_upper = self.box_upper - self.sphere_center
        nearest_relative = np.clip(
            np.zeros(3, dtype=np.float64), relative_lower, relative_upper
        )
        if direction_scale == 0.0 or self.radius == 0.0:
            return self.sphere_center + nearest_relative
        if _ball_slack(self.radius, nearest_relative) <= 0.0:
            return self.sphere_center + nearest_relative

        scaled_direction = direction_vec / direction_scale
        unit_direction = scaled_direction / float(np.linalg.norm(scaled_direction))
        box_maximizer = np.where(
            unit_direction > 0.0,
            relative_upper,
            np.where(unit_direction < 0.0, relative_lower, nearest_relative),
        )
        if _ball_slack(self.radius, box_maximizer) >= 0.0:
            return self.sphere_center + box_maximizer
        relative = self._active_set_support_relative(
            unit_direction,
            relative_lower,
            relative_upper,
            nearest_relative,
        )
        return self.sphere_center + relative

    def translated(self, shift: ArrayLike) -> "ClippedSphere":
        shift_vec = _finite_vec3(shift, "平移量")
        return ClippedSphere(
            self.sphere_center + shift_vec,
            self.radius,
            self.box_lower + shift_vec,
            self.box_upper + shift_vec,
        )

    def characteristic_radius(self) -> float:
        return self.radius + float(np.linalg.norm(self.center - self.sphere_center))


MixedShape = Cylinder | Sphere | ClippedSphere


@dataclass(frozen=True)
class SphereFragment:
    source_index: int
    fragment_index: int
    cell_shift: tuple[int, int, int]
    shape: Sphere | ClippedSphere


@dataclass(frozen=True)
class MixedTrialGeometry:
    a_centers: Vec3
    a_directions: Vec3
    b_centers: Vec3


@dataclass(frozen=True)
class ExactContactResult:
    connected: bool
    pair_type: str
    method: str
    broad_phase_rejected: bool
    narrow_phase_calls: int
    distance_nm: float | None
    lower_bound_nm: float
    converged: bool | None


@dataclass(frozen=True)
class FixedDesignDiagnostics:
    a_fragment_count: int
    b_fragment_count: int
    clipped_b_fragment_count: int
    processed_a_particles: int
    processed_b_particles: int
    candidate_pairs: int
    component_skips: int
    same_source_skips: int
    broad_phase_rejections: int
    narrow_phase_calls: int
    narrow_nonconverged: int
    aa_contacts: int
    ab_contacts: int
    bb_contacts: int
    electrode_contacts: int
    internal_a_edges: int

    def to_dict(self) -> dict[str, int]:
        return {
            "a_fragment_count": self.a_fragment_count,
            "b_fragment_count": self.b_fragment_count,
            "clipped_b_fragment_count": self.clipped_b_fragment_count,
            "processed_a_particles": self.processed_a_particles,
            "processed_b_particles": self.processed_b_particles,
            "candidate_pairs": self.candidate_pairs,
            "component_skips": self.component_skips,
            "same_source_skips": self.same_source_skips,
            "broad_phase_rejections": self.broad_phase_rejections,
            "narrow_phase_calls": self.narrow_phase_calls,
            "narrow_nonconverged": self.narrow_nonconverged,
            "aa_contacts": self.aa_contacts,
            "ab_contacts": self.ab_contacts,
            "bb_contacts": self.bb_contacts,
            "electrode_contacts": self.electrode_contacts,
            "internal_a_edges": self.internal_a_edges,
        }


@dataclass(frozen=True)
class FixedDesignResult:
    conductive: bool
    diagnostics: FixedDesignDiagnostics


@dataclass(frozen=True)
class ParetoTrialResult:
    connectivity_frontier: tuple[tuple[int, int], ...]
    diagnostics: FixedDesignDiagnostics
    pareto_diagnostics: ParetoSearchDiagnostics


# A 沿用 101/103 随机流，B 球心使用独立 211 随机流。
def _b_rng(config: MixedSimulationConfig, trial_id: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [config.master_seed, config.stream_id, trial_id, B_CENTER_QUANTITY_CODE]
    )
    return np.random.default_rng(sequence)


# 关键：在固定圆柱样本上生成第二类球形填料几何。
def generate_mixed_trial(
    config: MixedSimulationConfig, trial_id: int
) -> MixedTrialGeometry:
    if trial_id < 0 or trial_id >= config.trial_count:
        raise ValueError("trial_id 超出固定试验数")
    if config.n_a:
        a_config = config.a_config()
        a_centers = draw_uniform_centers(a_config, trial_id)
        a_directions = draw_isotropic_directions(a_config, trial_id)
    else:
        a_centers = np.empty((0, 3), dtype=np.float64)
        a_directions = np.empty((0, 3), dtype=np.float64)
    b_centers = _b_rng(config, trial_id).uniform(
        -config.half_box_nm, config.half_box_nm, size=(config.n_b, 3)
    )
    return MixedTrialGeometry(a_centers, a_directions, b_centers)


def fragment_sphere(
    center: ArrayLike, source_index: int, config: MixedSimulationConfig
) -> list[SphereFragment]:
    center_vec = _finite_vec3(center, "B 球心")
    lower = np.full(3, -config.half_box_nm, dtype=np.float64)
    upper = np.full(3, config.half_box_nm, dtype=np.float64)
    cell_options: list[tuple[int, ...]] = []
    for axis in range(3):
        options = [0]
        if center_vec[axis] - config.b_radius_nm < lower[axis]:
            options.insert(0, -1)
        if center_vec[axis] + config.b_radius_nm > upper[axis]:
            options.append(1)
        cell_options.append(tuple(options))

    fragments: list[SphereFragment] = []
    for cell in product(*cell_options):
        cell_vec = np.asarray(cell, dtype=np.float64)
        shifted_center = center_vec - config.box_length_nm * cell_vec
        nearest = np.clip(shifted_center, lower, upper)
        if _ball_slack(config.b_radius_nm, nearest - shifted_center) <= 0.0:
            continue
        fully_inside = bool(
            np.all(shifted_center - config.b_radius_nm >= lower)
            and np.all(shifted_center + config.b_radius_nm <= upper)
        )
        shape: Sphere | ClippedSphere
        if fully_inside:
            shape = Sphere(shifted_center, config.b_radius_nm)
        else:
            shape = ClippedSphere(
                shifted_center, config.b_radius_nm, lower, upper
            )
        fragments.append(
            SphereFragment(
                source_index=source_index,
                fragment_index=len(fragments),
                cell_shift=tuple(-int(value) for value in cell),
                shape=shape,
            )
        )
    if not fragments:
        raise RuntimeError("基础胞内球未生成任何非零体积片段")
    return fragments


def _parallel_cylinder_distance(first: Cylinder, second: Cylinder) -> float:
    offset = second.center - first.center
    axial_offset = float(np.dot(offset, first.axis))
    radial_offset = offset - axial_offset * first.axis
    axial_gap = max(
        abs(axial_offset) - first.half_length - second.half_length, 0.0
    )
    radial_gap = max(
        float(np.linalg.norm(radial_offset)) - first.radius - second.radius, 0.0
    )
    return float(np.hypot(axial_gap, radial_gap))


def _closed_form_connected(
    distance_nm: float,
    cutoff_nm: float,
    first: SupportShape,
    second: SupportShape,
) -> bool:
    scale = max(
        1.0,
        distance_nm,
        cutoff_nm,
        first.characteristic_radius(),
        second.characteristic_radius(),
        float(np.linalg.norm(first.center)),
        float(np.linalg.norm(second.center)),
    )
    floating_guard = 64.0 * np.finfo(np.float64).eps * scale
    return distance_nm <= cutoff_nm + floating_guard


def _mixed_aabb(
    shape: MixedShape, inflation: float = 0.0
) -> tuple[Vec3, Vec3]:
    if isinstance(shape, ClippedSphere):
        return shape.exact_aabb(inflation)
    return aabb(shape, inflation=inflation)


def _axis_plane_distance(
    shape: MixedShape, axis: int, offset: float
) -> float:
    if isinstance(shape, ClippedSphere):
        lower, upper = shape.coordinate_interval(axis)
        if offset < lower:
            return lower - offset
        if offset > upper:
            return offset - upper
        return 0.0
    normal = np.zeros(3, dtype=np.float64)
    normal[axis] = 1.0
    return shape_plane_distance(shape, normal, offset)


def _aabb_gap(first: SupportShape, second: SupportShape) -> float:
    first_lower, first_upper = _mixed_aabb(first)
    second_lower, second_upper = _mixed_aabb(second)
    axis_gaps = np.maximum(
        np.maximum(first_lower - second_upper, second_lower - first_upper), 0.0
    )
    return float(np.linalg.norm(axis_gaps))


def _generic_gjk_contact(
    first: SupportShape,
    second: SupportShape,
    pair_type: str,
    config: MixedSimulationConfig,
) -> ExactContactResult:
    cutoff = config.contact_cutoff_nm
    guard = config.broad_phase_guard_nm
    broad_gap = _aabb_gap(first, second)
    if broad_gap > cutoff + guard:
        return ExactContactResult(
            False,
            pair_type,
            "aabb_lower_bound",
            True,
            0,
            None,
            max(0.0, broad_gap - guard),
            None,
        )
    bounds = distance_bounds(
        first,
        second,
        absolute_tolerance=config.gjk_absolute_tolerance_nm,
        relative_tolerance=config.gjk_relative_tolerance,
        max_iterations=config.gjk_max_iterations,
    )
    calls = 1
    classification = bounds.classify(cutoff)
    if classification == "uncertain":
        tighter = distance_bounds(
            first,
            second,
            absolute_tolerance=min(1e-12, config.gjk_absolute_tolerance_nm * 0.01),
            relative_tolerance=min(1e-14, config.gjk_relative_tolerance * 0.1),
            max_iterations=max(2048, 4 * config.gjk_max_iterations),
        )
        calls += 1
        if tighter.width < bounds.width:
            bounds = tighter
        classification = bounds.classify(cutoff)
    if classification == "uncertain":
        raise NarrowPhaseUncertainError(
            f"混合仿真 {pair_type} 距离界跨阈值: "
            f"[{bounds.lower:.16g},{bounds.upper:.16g}] nm"
        )
    return ExactContactResult(
        classification == "connected",
        pair_type,
        "generic_convex_gjk_bounds",
        False,
        calls,
        bounds.estimate,
        bounds.lower,
        bounds.converged,
    )


# 三类实体对先作安全排除，再以各自精确集合距离裁决。
# 关键：综合闭式判定、包围盒筛选和 GJK 精确接触判定。
def evaluate_exact_contact(
    first: MixedShape,
    second: MixedShape,
    config: MixedSimulationConfig,
) -> ExactContactResult:
    cutoff = config.contact_cutoff_nm
    guard = config.broad_phase_guard_nm

    if isinstance(first, Cylinder) and isinstance(second, Cylinder):
        capsule_gap = capsule_cylinder_distance(first, second)
        if capsule_gap > cutoff + guard:
            return ExactContactResult(
                False,
                "A-A",
                "capsule_lower_bound",
                True,
                0,
                None,
                max(0.0, capsule_gap - guard),
                None,
            )
        parallel_tolerance = 64.0 * np.finfo(np.float64).eps
        if float(np.linalg.norm(np.cross(first.axis, second.axis))) <= parallel_tolerance:
            exact = _parallel_cylinder_distance(first, second)
            return ExactContactResult(
                _closed_form_connected(exact, cutoff, first, second),
                "A-A",
                "parallel_flat_cylinder_closed_form",
                False,
                1,
                exact,
                exact,
                True,
            )
        bounds = distance_bounds(
            first,
            second,
            absolute_tolerance=config.gjk_absolute_tolerance_nm,
            relative_tolerance=config.gjk_relative_tolerance,
            max_iterations=config.gjk_max_iterations,
        )
        calls = 1
        classification = bounds.classify(cutoff)
        if classification == "uncertain":
            tighter = distance_bounds(
                first,
                second,
                absolute_tolerance=min(1e-12, config.gjk_absolute_tolerance_nm * 0.01),
                relative_tolerance=min(1e-14, config.gjk_relative_tolerance * 0.1),
                max_iterations=max(2048, 4 * config.gjk_max_iterations),
            )
            calls += 1
            if tighter.width < bounds.width:
                bounds = tighter
            classification = bounds.classify(cutoff)
        if classification == "uncertain":
            raise NarrowPhaseUncertainError(
                "混合仿真 A-A 距离界跨阈值: "
                f"[{bounds.lower:.16g},{bounds.upper:.16g}] nm"
            )
        return ExactContactResult(
            classification == "connected",
            "A-A",
            "flat_cylinder_gjk_bounds",
            False,
            calls,
            bounds.estimate,
            bounds.lower,
            bounds.converged,
        )

    if isinstance(first, Sphere) and isinstance(second, Cylinder):
        first, second = second, first
    if isinstance(first, Cylinder) and isinstance(second, Sphere):
        capsule_gap = capsule_cylinder_sphere_distance(first, second)
        if capsule_gap > cutoff + guard:
            return ExactContactResult(
                False,
                "A-B",
                "capsule_lower_bound",
                True,
                0,
                None,
                max(0.0, capsule_gap - guard),
                None,
            )
        exact = cylinder_sphere_distance(first, second)
        return ExactContactResult(
            _closed_form_connected(exact, cutoff, first, second),
            "A-B",
            "flat_cylinder_sphere_closed_form",
            False,
            1,
            exact,
            exact,
            True,
        )

    if isinstance(first, Sphere) and isinstance(second, Sphere):
        exact = sphere_sphere_distance(first, second)
        return ExactContactResult(
            _closed_form_connected(exact, cutoff, first, second),
            "B-B",
            "sphere_sphere_closed_form",
            False,
            1,
            exact,
            exact,
            True,
        )
    if isinstance(first, ClippedSphere) or isinstance(second, ClippedSphere):
        parent_first: Cylinder | Sphere = (
            Sphere(first.sphere_center, first.radius)
            if isinstance(first, ClippedSphere)
            else first
        )
        parent_second: Cylinder | Sphere = (
            Sphere(second.sphere_center, second.radius)
            if isinstance(second, ClippedSphere)
            else second
        )
        if isinstance(parent_first, Sphere) and isinstance(parent_second, Cylinder):
            parent_first, parent_second = parent_second, parent_first
        if isinstance(parent_first, Cylinder) and isinstance(parent_second, Sphere):
            parent_gap = cylinder_sphere_distance(parent_first, parent_second)
            pair_type = "A-B"
        elif isinstance(parent_first, Sphere) and isinstance(parent_second, Sphere):
            parent_gap = sphere_sphere_distance(parent_first, parent_second)
            pair_type = "B-B"
        else:
            raise RuntimeError("截球母体下界类型不一致")
        parent_scale = max(
            1.0,
            parent_gap,
            cutoff,
            parent_first.characteristic_radius(),
            parent_second.characteristic_radius(),
            float(np.linalg.norm(parent_first.center)),
            float(np.linalg.norm(parent_second.center)),
        )
        parent_guard = max(
            guard, 64.0 * np.finfo(np.float64).eps * parent_scale
        )
        if parent_gap > cutoff + parent_guard:
            return ExactContactResult(
                False,
                pair_type,
                "parent_shape_lower_bound",
                True,
                0,
                None,
                max(0.0, parent_gap - parent_guard),
                None,
            )
    if isinstance(first, (Cylinder, Sphere, ClippedSphere)) and isinstance(
        second, (Cylinder, Sphere, ClippedSphere)
    ):
        pair_type = (
            "A-B"
            if isinstance(first, Cylinder) or isinstance(second, Cylinder)
            else "B-B"
        )
        return _generic_gjk_contact(first, second, pair_type, config)
    raise TypeError("仅支持 Cylinder、Sphere 与 ClippedSphere")


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            next_node = self.parent[node]
            self.parent[node] = root
            node = next_node
        return root

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.size[root_first] < self.size[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        self.size[root_first] += self.size[root_second]


@dataclass(frozen=True)
class _GeometryInstance:
    node: int
    shape: MixedShape
    source_kind: str
    source_index: int
    a_fragment: CylinderFragment | None = None


class _SpatialHash:
    def __init__(self, config: MixedSimulationConfig) -> None:
        self.config = config
        self.buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    def _cells(self, shape: SupportShape) -> tuple[tuple[int, int, int], ...]:
        lower, upper = _mixed_aabb(
            shape, inflation=self.config.contact_cutoff_nm
        )
        ranges = []
        for axis in range(3):
            low = math.floor(float(lower[axis]) / self.config.cell_size_nm)
            high = math.floor(float(upper[axis]) / self.config.cell_size_nm)
            ranges.append(range(low, high + 1))
        return tuple(
            (first, second, third)
            for first in ranges[0]
            for second in ranges[1]
            for third in ranges[2]
        )

    def candidates(self, shape: SupportShape) -> list[int]:
        result: set[int] = set()
        for cell in self._cells(shape):
            result.update(self.buckets.get(cell, ()))
        return sorted(result)

    def add(self, instance_index: int, shape: SupportShape) -> None:
        for cell in self._cells(shape):
            self.buckets[cell].append(instance_index)


def _validate_geometry(
    geometry: MixedTrialGeometry, config: MixedSimulationConfig
) -> None:
    expected_a = (config.n_a, 3)
    expected_b = (config.n_b, 3)
    if geometry.a_centers.shape != expected_a or geometry.a_directions.shape != expected_a:
        raise ValueError(f"A 几何数组形状必须为 {expected_a}")
    if geometry.b_centers.shape != expected_b:
        raise ValueError(f"B 球心数组形状必须为 {expected_b}")
    arrays = (geometry.a_centers, geometry.a_directions, geometry.b_centers)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("混合几何数组必须有限")
    if config.n_b and np.any(np.abs(geometry.b_centers) > config.half_box_nm):
        raise ValueError("B 球心必须位于基础立方体内")


# D 边界接触图仅使用映回基础盒的独立片段。
def solve_fixed_design(
    geometry: MixedTrialGeometry,
    config: MixedSimulationConfig,
    *,
    use_spatial_index: bool = True,
) -> FixedDesignResult:
    _validate_geometry(geometry, config)
    if config.n_a:
        a_groups = fragment_trial(
            geometry.a_centers, geometry.a_directions, config.a_config()
        )
    else:
        a_groups = []
    b_groups = [
        fragment_sphere(center, source_index, config)
        for source_index, center in enumerate(geometry.b_centers)
    ]
    flat_a_fragments = [fragment for group in a_groups for fragment in group]
    flat_b_fragments = [fragment for group in b_groups for fragment in group]
    a_offsets: list[int] = []
    offset = 0
    for group in a_groups:
        a_offsets.append(offset)
        offset += len(group)
    b_offsets: list[int] = []
    for group in b_groups:
        b_offsets.append(offset)
        offset += len(group)

    a_fragment_count = len(flat_a_fragments)
    b_fragment_count = len(flat_b_fragments)
    left_node = offset
    right_node = left_node + 1
    union_find = _UnionFind(right_node + 1)
    spatial_hash = _SpatialHash(config)
    instances: list[_GeometryInstance] = []
    a_config = config.a_config() if config.n_a else None

    counters = {
        "processed_a_particles": 0,
        "processed_b_particles": 0,
        "candidate_pairs": 0,
        "component_skips": 0,
        "same_source_skips": 0,
        "broad_phase_rejections": 0,
        "narrow_phase_calls": 0,
        "narrow_nonconverged": 0,
        "aa_contacts": 0,
        "ab_contacts": 0,
        "bb_contacts": 0,
        "electrode_contacts": 0,
        "internal_a_edges": 0,
    }

    def conductive() -> bool:
        return union_find.find(left_node) == union_find.find(right_node)

    def diagnostics() -> FixedDesignDiagnostics:
        return FixedDesignDiagnostics(
            a_fragment_count=a_fragment_count,
            b_fragment_count=b_fragment_count,
            clipped_b_fragment_count=sum(
                isinstance(fragment.shape, ClippedSphere)
                for fragment in flat_b_fragments
            ),
            **{name: int(value) for name, value in counters.items()},
        )

    def candidate_indices(shape: MixedShape) -> list[int]:
        if use_spatial_index:
            return spatial_hash.candidates(shape)
        return list(range(len(instances)))

    def test_against_active(
        node: int,
        shape: MixedShape,
        source_kind: str,
        source_index: int,
        a_fragment: CylinderFragment | None = None,
    ) -> None:
        for instance_index in candidate_indices(shape):
            other = instances[instance_index]
            counters["candidate_pairs"] += 1
            if (
                other.source_kind == source_kind
                and other.source_index == source_index
            ):
                counters["same_source_skips"] += 1
                continue
            if union_find.find(node) == union_find.find(other.node):
                counters["component_skips"] += 1
                continue
            if a_fragment is not None and other.a_fragment is not None:
                if a_config is None:
                    raise RuntimeError("A-A 判定缺少 A 子配置")
                a_result = evaluate_fragment_contact(
                    a_fragment, other.a_fragment, a_config
                )
                pair_type = "A-A"
                connected = a_result.connected
                broad_rejected = a_result.broad_phase_rejected
                narrow_calls = a_result.narrow_phase_calls
                converged = a_result.converged
            else:
                result = evaluate_exact_contact(shape, other.shape, config)
                pair_type = result.pair_type
                connected = result.connected
                broad_rejected = result.broad_phase_rejected
                narrow_calls = result.narrow_phase_calls
                converged = result.converged
            counters["broad_phase_rejections"] += int(broad_rejected)
            counters["narrow_phase_calls"] += narrow_calls
            counters["narrow_nonconverged"] += int(converged is False)
            if connected:
                union_find.union(node, other.node)
                key = pair_type.lower().replace("-", "") + "_contacts"
                counters[key] += 1

    def add_instance(
        node: int,
        shape: MixedShape,
        source_kind: str,
        source_index: int,
        a_fragment: CylinderFragment | None = None,
    ) -> None:
        instance_index = len(instances)
        instances.append(
            _GeometryInstance(
                node, shape, source_kind, source_index, a_fragment
            )
        )
        if use_spatial_index:
            spatial_hash.add(instance_index, shape)

    def connect_electrodes(node: int, shape: SupportShape) -> None:
        if _axis_plane_distance(
            shape, 0, -config.half_box_nm
        ) <= config.contact_cutoff_nm:
            union_find.union(node, left_node)
            counters["electrode_contacts"] += 1
        if _axis_plane_distance(
            shape, 0, config.half_box_nm
        ) <= config.contact_cutoff_nm:
            union_find.union(node, right_node)
            counters["electrode_contacts"] += 1

    for source_index, group in enumerate(a_groups):
        nodes = list(
            range(a_offsets[source_index], a_offsets[source_index] + len(group))
        )
        for node, fragment in zip(nodes, group, strict=True):
            cylinder = fragment.cylinder
            connect_electrodes(node, cylinder)
            test_against_active(node, cylinder, "A", source_index, fragment)
            add_instance(node, cylinder, "A", source_index, fragment)
        counters["processed_a_particles"] += 1
        if conductive():
            return FixedDesignResult(True, diagnostics())

    for source_index, group in enumerate(b_groups):
        nodes = list(
            range(b_offsets[source_index], b_offsets[source_index] + len(group))
        )
        for node, fragment in zip(nodes, group, strict=True):
            connect_electrodes(node, fragment.shape)
            test_against_active(node, fragment.shape, "B", source_index)
            add_instance(node, fragment.shape, "B", source_index)
        counters["processed_b_particles"] += 1
        if conductive():
            return FixedDesignResult(True, diagnostics())

    return FixedDesignResult(False, diagnostics())


# 关键：一次求出给定混合微构体的二维 Pareto 连通前沿。
def solve_pareto_connectivity(
    geometry: MixedTrialGeometry,
    config: MixedSimulationConfig,
    *,
    use_spatial_index: bool = True,
) -> ParetoTrialResult:
    _validate_geometry(geometry, config)
    if config.n_a:
        a_groups = fragment_trial(
            geometry.a_centers, geometry.a_directions, config.a_config()
        )
    else:
        a_groups = []
    b_groups = [
        fragment_sphere(center, source_index, config)
        for source_index, center in enumerate(geometry.b_centers)
    ]
    flat_a_fragments = [fragment for group in a_groups for fragment in group]
    flat_b_fragments = [fragment for group in b_groups for fragment in group]
    total_fragments = len(flat_a_fragments) + len(flat_b_fragments)
    left_node = total_fragments
    right_node = left_node + 1
    adjacency: dict[int, set[int]] = {
        node: set() for node in range(right_node + 1)
    }
    thresholds: dict[int, tuple[int, int]] = {
        left_node: (0, 0),
        right_node: (0, 0),
    }
    spatial_hash = _SpatialHash(config)
    instances: list[_GeometryInstance] = []
    full_b_instance_indices: list[int] = []
    a_config = config.a_config() if config.n_a else None
    counters = {
        "candidate_pairs": 0,
        "component_skips": 0,
        "same_source_skips": 0,
        "broad_phase_rejections": 0,
        "narrow_phase_calls": 0,
        "narrow_nonconverged": 0,
        "aa_contacts": 0,
        "ab_contacts": 0,
        "bb_contacts": 0,
        "electrode_contacts": 0,
        "internal_a_edges": 0,
    }

    def add_edge(first: int, second: int) -> None:
        adjacency[first].add(second)
        adjacency[second].add(first)

    def candidate_indices(shape: MixedShape) -> list[int]:
        if use_spatial_index:
            return spatial_hash.candidates(shape)
        return list(range(len(instances)))

    def test_and_link(
        node: int,
        shape: MixedShape,
        source_kind: str,
        source_index: int,
        a_fragment: CylinderFragment | None = None,
    ) -> None:
        for instance_index in candidate_indices(shape):
            other = instances[instance_index]
            if (
                use_spatial_index
                and isinstance(shape, Sphere)
                and isinstance(other.shape, Sphere)
            ):
                continue
            counters["candidate_pairs"] += 1
            if (
                other.source_kind == source_kind
                and other.source_index == source_index
            ):
                counters["same_source_skips"] += 1
                continue
            if a_fragment is not None and other.a_fragment is not None:
                if a_config is None:
                    raise RuntimeError("A-A 判定缺少 A 子配置")
                a_result = evaluate_fragment_contact(
                    a_fragment, other.a_fragment, a_config
                )
                pair_type = "A-A"
                connected = a_result.connected
                broad_rejected = a_result.broad_phase_rejected
                narrow_calls = a_result.narrow_phase_calls
                converged = a_result.converged
            else:
                result = evaluate_exact_contact(shape, other.shape, config)
                pair_type = result.pair_type
                connected = result.connected
                broad_rejected = result.broad_phase_rejected
                narrow_calls = result.narrow_phase_calls
                converged = result.converged
            counters["broad_phase_rejections"] += int(broad_rejected)
            counters["narrow_phase_calls"] += narrow_calls
            counters["narrow_nonconverged"] += int(converged is False)
            if connected:
                add_edge(node, other.node)
                key = pair_type.lower().replace("-", "") + "_contacts"
                counters[key] += 1

    def connect_electrodes(node: int, shape: SupportShape) -> None:
        if _axis_plane_distance(
            shape, 0, -config.half_box_nm
        ) <= config.contact_cutoff_nm:
            add_edge(node, left_node)
            counters["electrode_contacts"] += 1
        if _axis_plane_distance(
            shape, 0, config.half_box_nm
        ) <= config.contact_cutoff_nm:
            add_edge(node, right_node)
            counters["electrode_contacts"] += 1

    def add_instance(
        node: int,
        shape: MixedShape,
        source_kind: str,
        source_index: int,
        a_fragment: CylinderFragment | None = None,
    ) -> None:
        instance_index = len(instances)
        instances.append(
            _GeometryInstance(
                node, shape, source_kind, source_index, a_fragment
            )
        )
        if isinstance(shape, Sphere):
            full_b_instance_indices.append(instance_index)
        if use_spatial_index:
            spatial_hash.add(instance_index, shape)

    node = 0
    for source_index, group in enumerate(a_groups):
        for fragment in group:
            thresholds[node] = (source_index + 1, 0)
            connect_electrodes(node, fragment.cylinder)
            test_and_link(
                node, fragment.cylinder, "A", source_index, fragment
            )
            add_instance(
                node, fragment.cylinder, "A", source_index, fragment
            )
            node += 1
    for source_index, group in enumerate(b_groups):
        for fragment in group:
            thresholds[node] = (0, source_index + 1)
            connect_electrodes(node, fragment.shape)
            test_and_link(node, fragment.shape, "B", source_index)
            add_instance(node, fragment.shape, "B", source_index)
            node += 1
    if node != total_fragments:
        raise RuntimeError("二维前沿图节点计数不一致")

    if use_spatial_index and len(full_b_instance_indices) > 1:
        centers = np.vstack(
            [instances[index].shape.center for index in full_b_instance_indices]
        )
        center_scale = float(np.max(np.linalg.norm(centers, axis=1)))
        floating_guard = 64.0 * np.finfo(np.float64).eps * max(
            1.0,
            center_scale,
            config.b_radius_nm,
            config.contact_cutoff_nm,
        )
        query_radius = (
            2.0 * config.b_radius_nm
            + config.contact_cutoff_nm
            + max(config.broad_phase_guard_nm, floating_guard)
        )
        tree_pairs = cKDTree(centers).query_pairs(
            query_radius, output_type="ndarray"
        )
        for local_first, local_second in tree_pairs:
            first = instances[full_b_instance_indices[int(local_first)]]
            second = instances[full_b_instance_indices[int(local_second)]]
            counters["candidate_pairs"] += 1
            if (
                first.source_kind == second.source_kind
                and first.source_index == second.source_index
            ):
                counters["same_source_skips"] += 1
                continue
            result = evaluate_exact_contact(first.shape, second.shape, config)
            counters["broad_phase_rejections"] += int(
                result.broad_phase_rejected
            )
            counters["narrow_phase_calls"] += result.narrow_phase_calls
            counters["narrow_nonconverged"] += int(result.converged is False)
            if result.connected:
                add_edge(first.node, second.node)
                counters["bb_contacts"] += 1

    pareto = axis_threshold_pareto_result(
        adjacency, thresholds, left_node, right_node
    )
    diagnostics = FixedDesignDiagnostics(
        a_fragment_count=len(flat_a_fragments),
        b_fragment_count=len(flat_b_fragments),
        clipped_b_fragment_count=sum(
            isinstance(fragment.shape, ClippedSphere)
            for fragment in flat_b_fragments
        ),
        processed_a_particles=config.n_a,
        processed_b_particles=config.n_b,
        **{name: int(value) for name, value in counters.items()},
    )
    return ParetoTrialResult(pareto.labels, diagnostics, pareto.diagnostics)


def run_one_trial(config: MixedSimulationConfig, trial_id: int) -> dict[str, Any]:
    result = solve_fixed_design(generate_mixed_trial(config, trial_id), config)
    return {
        "trial_id": trial_id,
        "conductive": result.conductive,
        **result.diagnostics.to_dict(),
    }


def run_one_pareto_trial(
    config: MixedSimulationConfig, trial_id: int
) -> dict[str, Any]:
    result = solve_pareto_connectivity(generate_mixed_trial(config, trial_id), config)
    return {
        "trial_id": trial_id,
        "connectivity_frontier": [
            list(label) for label in result.connectivity_frontier
        ],
        **result.diagnostics.to_dict(),
        "pareto_search": asdict(result.pareto_diagnostics),
    }


# 固定设计按确定 trial ID 分片，并作指纹核验和确定性合并。
def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_trial_ids(
    config: MixedSimulationConfig, trial_ids: Sequence[int]
) -> dict[str, Any]:
    ids = [int(value) for value in trial_ids]
    if ids != sorted(set(ids)):
        raise ValueError("分片 trial_id 必须严格递增且不重复")
    if any(value < 0 or value >= config.trial_count for value in ids):
        raise ValueError("分片 trial_id 超出固定试验数")
    started = time.perf_counter()
    records = [run_one_trial(config, trial_id) for trial_id in ids]
    return {
        "kind": "mixed_fixed_design_shard",
        "schema_version": SCHEMA_VERSION,
        "configuration": config.to_dict(),
        "configuration_fingerprint": config.fingerprint,
        "trial_ids": ids,
        "records": records,
        "runtime_seconds": time.perf_counter() - started,
    }


def _validate_shard(
    payload: dict[str, Any],
    config: MixedSimulationConfig,
    expected_ids: Sequence[int] | None = None,
) -> None:
    if payload.get("kind") != "mixed_fixed_design_shard":
        raise ValueError("文件不是混合固定设计分片")
    if payload.get("configuration_fingerprint") != config.fingerprint:
        raise ValueError("混合分片配置指纹不兼容")
    stored = MixedSimulationConfig.from_dict(payload.get("configuration", {}))
    if stored.fingerprint != config.fingerprint:
        raise ValueError("混合分片内嵌配置与指纹不一致")
    ids = [int(value) for value in payload.get("trial_ids", [])]
    record_ids = [int(row["trial_id"]) for row in payload.get("records", [])]
    if ids != sorted(set(ids)) or record_ids != ids:
        raise ValueError("混合分片 trial_id 或记录顺序无效")
    if expected_ids is not None and ids != [int(value) for value in expected_ids]:
        raise ValueError("续跑分片范围与请求不一致")


def run_shard(
    config: MixedSimulationConfig,
    trial_ids: Sequence[int],
    output_path: Path | str | None = None,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    path = Path(output_path) if output_path is not None else None
    if path is not None and resume and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        _validate_shard(payload, config, trial_ids)
        return payload
    payload = run_trial_ids(config, trial_ids)
    if path is not None:
        _atomic_write_json(path, payload)
    return payload


def merge_shard_payloads(
    payloads: Sequence[dict[str, Any]],
    config: MixedSimulationConfig,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    indexed: dict[int, dict[str, Any]] = {}
    runtime_seconds = 0.0
    for payload in payloads:
        _validate_shard(payload, config)
        runtime_seconds += float(payload.get("runtime_seconds", 0.0))
        for record in payload["records"]:
            trial_id = int(record["trial_id"])
            if trial_id in indexed:
                raise ValueError(f"合并发现重复 trial_id: {trial_id}")
            indexed[trial_id] = record
    expected = set(range(config.trial_count))
    present = set(indexed)
    if require_complete and present != expected:
        raise ValueError(
            f"混合分片不完整: missing={sorted(expected-present)}, "
            f"extra={sorted(present-expected)}"
        )
    records = [indexed[index] for index in sorted(indexed)]
    successes = int(sum(bool(record["conductive"]) for record in records))
    trials = len(records)
    if trials:
        wilson = wilson_interval(successes, trials)
        exact = clopper_pearson_interval(successes, trials)
        estimate = successes / trials
    else:
        wilson = (math.nan, math.nan)
        exact = (math.nan, math.nan)
        estimate = math.nan
    diagnostic_fields = (
        "a_fragment_count",
        "b_fragment_count",
        "clipped_b_fragment_count",
        "processed_a_particles",
        "processed_b_particles",
        "candidate_pairs",
        "component_skips",
        "same_source_skips",
        "broad_phase_rejections",
        "narrow_phase_calls",
        "narrow_nonconverged",
        "aa_contacts",
        "ab_contacts",
        "bb_contacts",
        "electrode_contacts",
        "internal_a_edges",
    )
    return {
        "kind": "mixed_fixed_design_samples",
        "schema_version": SCHEMA_VERSION,
        "configuration": config.to_dict(),
        "configuration_fingerprint": config.fingerprint,
        "boundary_contract": BOUNDARY_CONTRACT,
        "records": records,
        "successes": successes,
        "trials": trials,
        "probability_estimate": estimate,
        "wilson95_interval": list(wilson),
        "clopper_pearson95_interval": list(exact),
        "diagnostics_total": {
            field: int(sum(int(record[field]) for record in records))
            for field in diagnostic_fields
        },
        "shard_runtime_seconds": runtime_seconds,
    }


def merge_shards(
    paths: Sequence[Path | str], config: MixedSimulationConfig
) -> dict[str, Any]:
    payloads = [
        json.loads(Path(path).read_text(encoding="utf-8-sig")) for path in paths
    ]
    return merge_shard_payloads(payloads, config)


def _worker_run_shard(
    arguments: tuple[dict[str, Any], tuple[int, ...]]
) -> dict[str, Any]:
    config_payload, trial_ids = arguments
    return run_trial_ids(MixedSimulationConfig.from_dict(config_payload), trial_ids)


def run_fixed_design_simulation(
    config: MixedSimulationConfig,
    output_dir: Path | str,
    *,
    workers: int = 1,
    batch_size: int = 8,
    resume: bool = True,
    merged_name: str = "mixed_fixed_design_samples.json",
) -> Path:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers 和 batch_size 必须为正整数")
    directory = Path(output_dir)
    shard_dir = directory / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    batches = [
        tuple(range(start, min(config.trial_count, start + batch_size)))
        for start in range(0, config.trial_count, batch_size)
    ]
    paths = [
        shard_dir / f"shard_{batch[0]:06d}_{batch[-1]:06d}.json" for batch in batches
    ]
    missing: list[tuple[tuple[int, ...], Path]] = []
    for batch, path in zip(batches, paths, strict=True):
        if resume and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            _validate_shard(payload, config, batch)
        else:
            missing.append((batch, path))
    if workers == 1:
        for batch, path in missing:
            payload = run_trial_ids(config, batch)
            _validate_shard(payload, config, batch)
            _atomic_write_json(path, payload)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_target = {
                executor.submit(
                    _worker_run_shard, (config.to_dict(), batch)
                ): (batch, path)
                for batch, path in missing
            }
            for future in as_completed(future_to_target):
                batch, path = future_to_target[future]
                payload = future.result()
                _validate_shard(payload, config, batch)
                _atomic_write_json(path, payload)
    merged = merge_shards(paths, config)
    merged_path = directory / merged_name
    _atomic_write_json(merged_path, merged)
    return merged_path


def load_fixed_design_artifact(
    path: Path | str,
) -> tuple[MixedSimulationConfig, NDArray[np.bool_], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("kind") != "mixed_fixed_design_samples":
        raise ValueError("文件不是混合固定设计样本")
    config = MixedSimulationConfig.from_dict(payload.get("configuration", {}))
    if payload.get("configuration_fingerprint") != config.fingerprint:
        raise ValueError("混合固定设计样本指纹不一致")
    records = payload.get("records", [])
    ids = [int(record["trial_id"]) for record in records]
    if ids != list(range(config.trial_count)):
        raise ValueError("混合固定设计 trial_id 不完整或未排序")
    samples = np.asarray([bool(record["conductive"]) for record in records])
    return config, samples, records


def _record_frontier(record: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    labels = tuple(
        (int(label[0]), int(label[1]))
        for label in record.get("connectivity_frontier", [])
    )
    if labels != pareto_prune_labels(labels):
        raise ValueError("二维连通前沿不是确定性非支配序列")
    return labels


def run_pareto_trial_ids(
    config: MixedSimulationConfig, trial_ids: Sequence[int]
) -> dict[str, Any]:
    ids = [int(value) for value in trial_ids]
    if ids != sorted(set(ids)):
        raise ValueError("二维前沿分片 trial_id 必须严格递增且不重复")
    if any(value < 0 or value >= config.trial_count for value in ids):
        raise ValueError("二维前沿分片 trial_id 超出固定试验数")
    started = time.perf_counter()
    records = [run_one_pareto_trial(config, trial_id) for trial_id in ids]
    return {
        "kind": "mixed_pareto_frontier_shard",
        "schema_version": SCHEMA_VERSION,
        "configuration": config.to_dict(),
        "configuration_fingerprint": config.fingerprint,
        "trial_ids": ids,
        "records": records,
        "runtime_seconds": time.perf_counter() - started,
    }


def _validate_pareto_shard(
    payload: dict[str, Any],
    config: MixedSimulationConfig,
    expected_ids: Sequence[int] | None = None,
) -> None:
    if payload.get("kind") != "mixed_pareto_frontier_shard":
        raise ValueError("文件不是混合二维前沿分片")
    if payload.get("configuration_fingerprint") != config.fingerprint:
        raise ValueError("混合二维前沿分片配置指纹不兼容")
    stored = MixedSimulationConfig.from_dict(payload.get("configuration", {}))
    if stored.fingerprint != config.fingerprint:
        raise ValueError("混合二维前沿分片内嵌配置与指纹不一致")
    ids = [int(value) for value in payload.get("trial_ids", [])]
    records = payload.get("records", [])
    record_ids = [int(record["trial_id"]) for record in records]
    if ids != sorted(set(ids)) or record_ids != ids:
        raise ValueError("混合二维前沿分片 trial_id 或记录顺序无效")
    if expected_ids is not None and ids != [int(value) for value in expected_ids]:
        raise ValueError("续跑二维前沿分片范围与请求不一致")
    for record in records:
        _record_frontier(record)


def merge_pareto_shard_payloads(
    payloads: Sequence[dict[str, Any]],
    config: MixedSimulationConfig,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    indexed: dict[int, dict[str, Any]] = {}
    runtime_seconds = 0.0
    for payload in payloads:
        _validate_pareto_shard(payload, config)
        runtime_seconds += float(payload.get("runtime_seconds", 0.0))
        for record in payload["records"]:
            trial_id = int(record["trial_id"])
            if trial_id in indexed:
                raise ValueError(f"合并二维前沿发现重复 trial_id: {trial_id}")
            indexed[trial_id] = record
    expected = set(range(config.trial_count))
    present = set(indexed)
    if require_complete and present != expected:
        raise ValueError(
            f"混合二维前沿分片不完整: missing={sorted(expected-present)}, "
            f"extra={sorted(present-expected)}"
        )
    records = [indexed[index] for index in sorted(indexed)]
    diagnostic_fields = tuple(FixedDesignDiagnostics.__dataclass_fields__)
    pareto_fields = tuple(ParetoSearchDiagnostics.__dataclass_fields__)
    return {
        "kind": "mixed_pareto_frontier_samples",
        "schema_version": SCHEMA_VERSION,
        "configuration": config.to_dict(),
        "configuration_fingerprint": config.fingerprint,
        "boundary_contract": BOUNDARY_CONTRACT,
        "records": records,
        "trials": len(records),
        "diagnostics_total": {
            field: int(sum(int(record[field]) for record in records))
            for field in diagnostic_fields
        },
        "pareto_search_total": {
            field: int(
                sum(int(record["pareto_search"][field]) for record in records)
            )
            for field in pareto_fields
        },
        "shard_runtime_seconds": runtime_seconds,
    }


def merge_pareto_shards(
    paths: Sequence[Path | str], config: MixedSimulationConfig
) -> dict[str, Any]:
    payloads = [
        json.loads(Path(path).read_text(encoding="utf-8-sig")) for path in paths
    ]
    return merge_pareto_shard_payloads(payloads, config)


def _worker_run_pareto_shard(
    arguments: tuple[dict[str, Any], tuple[int, ...]]
) -> dict[str, Any]:
    config_payload, trial_ids = arguments
    return run_pareto_trial_ids(
        MixedSimulationConfig.from_dict(config_payload), trial_ids
    )


def run_pareto_frontier_simulation(
    config: MixedSimulationConfig,
    output_dir: Path | str,
    *,
    workers: int = 1,
    batch_size: int = 8,
    resume: bool = True,
    merged_name: str = "mixed_pareto_frontier_samples.json",
) -> Path:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers 和 batch_size 必须为正整数")
    directory = Path(output_dir)
    shard_dir = directory / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    batches = [
        tuple(range(start, min(config.trial_count, start + batch_size)))
        for start in range(0, config.trial_count, batch_size)
    ]
    paths = [
        shard_dir / f"shard_{batch[0]:06d}_{batch[-1]:06d}.json"
        for batch in batches
    ]
    missing: list[tuple[tuple[int, ...], Path]] = []
    for batch, path in zip(batches, paths, strict=True):
        if resume and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            _validate_pareto_shard(payload, config, batch)
        else:
            missing.append((batch, path))
    if workers == 1:
        for batch, path in missing:
            payload = run_pareto_trial_ids(config, batch)
            _validate_pareto_shard(payload, config, batch)
            _atomic_write_json(path, payload)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_target = {
                executor.submit(
                    _worker_run_pareto_shard, (config.to_dict(), batch)
                ): (batch, path)
                for batch, path in missing
            }
            for future in as_completed(future_to_target):
                batch, path = future_to_target[future]
                payload = future.result()
                _validate_pareto_shard(payload, config, batch)
                _atomic_write_json(path, payload)
    merged = merge_pareto_shards(paths, config)
    merged_path = directory / merged_name
    _atomic_write_json(merged_path, merged)
    return merged_path


def load_pareto_frontier_artifact(
    path: Path | str,
) -> tuple[
    MixedSimulationConfig,
    list[tuple[tuple[int, int], ...]],
    list[dict[str, Any]],
]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("kind") != "mixed_pareto_frontier_samples":
        raise ValueError("文件不是混合二维连通前沿样本")
    config = MixedSimulationConfig.from_dict(payload.get("configuration", {}))
    if payload.get("configuration_fingerprint") != config.fingerprint:
        raise ValueError("混合二维连通前沿样本指纹不一致")
    records = payload.get("records", [])
    ids = [int(record["trial_id"]) for record in records]
    if ids != list(range(config.trial_count)):
        raise ValueError("混合二维连通前沿 trial_id 不完整或未排序")
    frontiers = [_record_frontier(record) for record in records]
    return config, frontiers, records


def connectivity_samples_at_design(
    frontiers: Sequence[Sequence[tuple[int, int]]], n_a: int, n_b: int
) -> NDArray[np.bool_]:
    if n_a < 0 or n_b < 0:
        raise ValueError("介质数量必须非负")
    return np.asarray(
        [design_is_connected(frontier, n_a, n_b) for frontier in frontiers],
        dtype=np.bool_,
    )
