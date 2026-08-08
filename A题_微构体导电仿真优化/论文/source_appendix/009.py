# AI 工具：OpenAI Codex；模型/版本：GPT-5 系列；开发机构：OpenAI。
# 版本发布日期：2025-08-07（GPT-5 系列公开快照日期）；本程序由参赛队逐行复核并对结果负责。
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta, norm

from fast_geometry import fast_cylinder_classify, warm_up_fast_geometry
from geometry_kernel import (
    Cylinder,
    aabb,
    capsule_cylinder_distance,
    distance_bounds,
    shape_plane_distance,
)


Vec3 = NDArray[np.float64]

BOX_LENGTH_NM = 10_000.0
CYLINDER_LENGTH_NM = 5_000.0
CYLINDER_RADIUS_NM = 30.0
CONTACT_CUTOFF_NM = 1.8
PROJECT_SEED = 20_260_801
Q2_COUNTS = (354, 424, 495, 707)
ALGORITHM_VERSION = "flat-cylinder-periodic-numba-cached-v4"
SCHEMA_VERSION = 1


class BoundaryMode(str, Enum):
    D = "D"
    B = "B"
    A = "A"


@dataclass(frozen=True)
class BoundarySpec:
    mode: BoundaryMode
    periodic_axes: tuple[bool, bool, bool]
    minimum_image_axes: tuple[bool, bool, bool]
    clip_x_axis: bool
    connect_same_source: bool
    role: str
    description: str
    implementation_limitations: tuple[str, ...]


BOUNDARY_SPECS = {
    BoundaryMode.D: BoundarySpec(
        mode=BoundaryMode.D,
        periodic_axes=(True, True, True),
        minimum_image_axes=(False, False, False),
        clip_x_axis=False,
        connect_same_source=False,
        role="primary",
        description="三轴显式回绕，截断片段均作为独立介质且不添加同源内部边",
        implementation_limitations=(
            "按题设中心线交点切段；各轴半径侧缘越界及斜切面未作完整三维布尔回绕",
        ),
    ),
    BoundaryMode.B: BoundarySpec(
        mode=BoundaryMode.B,
        periodic_axes=(False, True, True),
        minimum_image_axes=(False, True, True),
        clip_x_axis=True,
        connect_same_source=True,
        role="boundary_sensitivity",
        description="X 方向裁剪中心线，Y/Z 显式回绕，同源片段保持内部连接",
        implementation_limitations=(
            "X 边界仅裁剪中心线后建立较短平底圆柱，不等于完整三维圆柱与 X slab 的布尔交",
            "Y/Z 按轴线交点分段；各轴半径侧缘越界及斜切面未作完整三维布尔回绕",
        ),
    ),
    BoundaryMode.A: BoundarySpec(
        mode=BoundaryMode.A,
        periodic_axes=(True, True, True),
        minimum_image_axes=(True, True, True),
        clip_x_axis=False,
        connect_same_source=True,
        role="sensitivity_only",
        description="三轴显式回绕，同源片段保持内部连接",
        implementation_limitations=(
            "按题设中心线交点切段；各轴半径侧缘越界及斜切面未作完整三维布尔回绕",
        ),
    ),
}


def boundary_spec(mode: BoundaryMode | str) -> BoundarySpec:
    try:
        parsed = mode if isinstance(mode, BoundaryMode) else BoundaryMode(mode)
    except ValueError as exc:
        raise ValueError(f"未知边界模式: {mode}") from exc
    return BOUNDARY_SPECS[parsed]


@dataclass(frozen=True)
class SimulationConfig:
    max_count: int
    trial_count: int
    boundary_mode: BoundaryMode | str = BoundaryMode.D
    master_seed: int = PROJECT_SEED
    stream_id: int = 2
    box_length_nm: float = BOX_LENGTH_NM
    cylinder_length_nm: float = CYLINDER_LENGTH_NM
    cylinder_radius_nm: float = CYLINDER_RADIUS_NM
    contact_cutoff_nm: float = CONTACT_CUTOFF_NM
    cell_size_nm: float = 625.0
    broad_phase_guard_nm: float = 1e-8
    gjk_absolute_tolerance_nm: float = 1e-10
    gjk_relative_tolerance: float = 1e-13
    gjk_max_iterations: int = 512
    algorithm_version: str = ALGORITHM_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_mode", boundary_spec(self.boundary_mode).mode)
        if self.max_count < 1 or self.trial_count < 1:
            raise ValueError("max_count 和 trial_count 必须为正整数")
        if self.master_seed < 0 or self.stream_id < 0:
            raise ValueError("SeedSequence 标识必须为非负整数")
        positive = (
            self.box_length_nm,
            self.cylinder_length_nm,
            self.cylinder_radius_nm,
            self.contact_cutoff_nm,
            self.cell_size_nm,
            self.gjk_absolute_tolerance_nm,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("几何尺度和数值容差必须为有限正数")
        if not np.isfinite(self.broad_phase_guard_nm) or self.broad_phase_guard_nm < 0.0:
            raise ValueError("宽相保护量必须有限且非负")
        if not np.isfinite(self.gjk_relative_tolerance) or self.gjk_relative_tolerance < 0.0:
            raise ValueError("GJK 相对容差必须有限且非负")
        if self.gjk_max_iterations < 1:
            raise ValueError("GJK 最大迭代次数必须为正")
        if self.cylinder_length_nm > self.box_length_nm:
            raise ValueError("当前切段合同要求圆柱轴长不超过立方体边长")
        if self.algorithm_version != ALGORITHM_VERSION:
            raise ValueError("配置算法版本与当前实现不一致")

    @property
    def half_box_nm(self) -> float:
        return 0.5 * self.box_length_nm

    @property
    def half_cylinder_nm(self) -> float:
        return 0.5 * self.cylinder_length_nm

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": self.algorithm_version,
            "max_count": self.max_count,
            "trial_count": self.trial_count,
            "boundary_mode": self.boundary_mode.value,
            "master_seed": self.master_seed,
            "stream_id": self.stream_id,
            "box_length_nm": self.box_length_nm,
            "cylinder_length_nm": self.cylinder_length_nm,
            "cylinder_radius_nm": self.cylinder_radius_nm,
            "contact_cutoff_nm": self.contact_cutoff_nm,
            "cell_size_nm": self.cell_size_nm,
            "broad_phase_guard_nm": self.broad_phase_guard_nm,
            "gjk_absolute_tolerance_nm": self.gjk_absolute_tolerance_nm,
            "gjk_relative_tolerance": self.gjk_relative_tolerance,
            "gjk_max_iterations": self.gjk_max_iterations,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SimulationConfig":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("不支持的仿真配置版本")
        values = dict(payload)
        values.pop("schema_version", None)
        return cls(**values)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(canonical).hexdigest().upper()


@dataclass(frozen=True)
class CylinderFragment:
    source_index: int
    fragment_index: int
    t_start: float
    t_end: float
    cell_shift: tuple[int, int, int]
    cylinder: Cylinder
    aabb_lower: Vec3 | None = None
    aabb_upper: Vec3 | None = None

    def __post_init__(self) -> None:
        if self.aabb_lower is None or self.aabb_upper is None:
            lower, upper = aabb(self.cylinder)
            object.__setattr__(self, "aabb_lower", lower)
            object.__setattr__(self, "aabb_upper", upper)


@dataclass(frozen=True)
class PairContactResult:
    connected: bool
    broad_phase_rejected: bool
    narrow_phase_calls: int
    lower_nm: float
    upper_nm: float | None
    iterations: int
    converged: bool | None
    reference_fallbacks: int = 0
    distance_converged: bool | None = None


@dataclass(frozen=True)
class TrialDiagnostics:
    fragment_count: int
    candidate_pairs: int
    component_skips: int
    broad_phase_rejections: int
    narrow_phase_calls: int
    narrow_nonconverged: int
    narrow_distance_nonconverged: int
    narrow_reference_fallbacks: int
    physical_contacts: int
    electrode_contacts: int
    internal_edges: int
    witness_nodes: tuple[str, ...] | None = None
    witness_edge_types: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_count": self.fragment_count,
            "candidate_pairs": self.candidate_pairs,
            "component_skips": self.component_skips,
            "broad_phase_rejections": self.broad_phase_rejections,
            "narrow_phase_calls": self.narrow_phase_calls,
            "narrow_nonconverged": self.narrow_nonconverged,
            "narrow_distance_nonconverged": self.narrow_distance_nonconverged,
            "narrow_reference_fallbacks": self.narrow_reference_fallbacks,
            "physical_contacts": self.physical_contacts,
            "electrode_contacts": self.electrode_contacts,
            "internal_edges": self.internal_edges,
            "witness_nodes": list(self.witness_nodes) if self.witness_nodes else None,
            "witness_edge_types": (
                list(self.witness_edge_types) if self.witness_edge_types else None
            ),
        }


@dataclass(frozen=True)
class TrialResult:
    first_connection_index: int
    diagnostics: TrialDiagnostics


class NarrowPhaseUncertainError(RuntimeError):
    pass


# 模块1：用稳定整数 SeedSequence 分流生成 IID 中心与各向同性方向。
def _stable_rng(config: SimulationConfig, trial_id: int, quantity_code: int) -> np.random.Generator:
    if trial_id < 0:
        raise ValueError("trial_id 必须非负")
    sequence = np.random.SeedSequence(
        [config.master_seed, config.stream_id, trial_id, quantity_code]
    )
    return np.random.default_rng(sequence)


def draw_uniform_centers(config: SimulationConfig, trial_id: int, count: int | None = None) -> Vec3:
    size = config.max_count if count is None else int(count)
    if size < 0 or size > config.max_count:
        raise ValueError("生成数量必须位于 [0,max_count]")
    rng = _stable_rng(config, trial_id, 101)
    return rng.uniform(-config.half_box_nm, config.half_box_nm, size=(size, 3))


def draw_isotropic_directions(
    config: SimulationConfig, trial_id: int, count: int | None = None
) -> Vec3:
    size = config.max_count if count is None else int(count)
    if size < 0 or size > config.max_count:
        raise ValueError("生成数量必须位于 [0,max_count]")
    rng = _stable_rng(config, trial_id, 103)
    uniforms = rng.random((size, 2))
    cosine = 2.0 * uniforms[:, 0] - 1.0
    azimuth = 2.0 * math.pi * uniforms[:, 1]
    radial = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    return np.column_stack(
        (radial * np.cos(azimuth), radial * np.sin(azimuth), cosine)
    )


def generate_trial(config: SimulationConfig, trial_id: int) -> tuple[Vec3, Vec3]:
    if trial_id >= config.trial_count:
        raise ValueError("trial_id 超出固定试验数")
    return (
        draw_uniform_centers(config, trial_id),
        draw_isotropic_directions(config, trial_id),
    )


# 模块2：在原始无界轴线上求交，再逐区间映射为基础立方体内的平底圆柱。
def _merge_cuts(values: Iterable[float], tolerance: float = 1e-12) -> list[float]:
    ordered = sorted(float(value) for value in values)
    merged: list[float] = []
    for value in ordered:
        if not merged or value - merged[-1] > tolerance:
            merged.append(value)
        else:
            merged[-1] = 0.5 * (merged[-1] + value)
    return merged


def _clip_parameter_to_x_slab(
    endpoint_a: Vec3, endpoint_b: Vec3, half_box_nm: float
) -> tuple[float, float] | None:
    delta_x = float(endpoint_b[0] - endpoint_a[0])
    if abs(delta_x) <= 1e-15:
        if -half_box_nm <= endpoint_a[0] <= half_box_nm:
            return 0.0, 1.0
        return None
    first = (-half_box_nm - endpoint_a[0]) / delta_x
    second = (half_box_nm - endpoint_a[0]) / delta_x
    lower = max(0.0, min(first, second))
    upper = min(1.0, max(first, second))
    if upper - lower <= 1e-14:
        return None
    return float(lower), float(upper)


def split_particle_axis(
    center: ArrayLike,
    direction: ArrayLike,
    source_index: int,
    config: SimulationConfig,
) -> list[CylinderFragment]:
    center_vec = np.asarray(center, dtype=np.float64)
    direction_vec = np.asarray(direction, dtype=np.float64)
    if center_vec.shape != (3,) or direction_vec.shape != (3,):
        raise ValueError("中心与方向必须是三维向量")
    if not np.all(np.isfinite(center_vec)) or not np.all(np.isfinite(direction_vec)):
        raise ValueError("中心与方向必须有限")
    direction_norm = float(np.linalg.norm(direction_vec))
    if direction_norm <= 1e-12:
        raise ValueError("圆柱方向退化")
    direction_vec = direction_vec / direction_norm
    endpoint_a = center_vec - config.half_cylinder_nm * direction_vec
    endpoint_b = center_vec + config.half_cylinder_nm * direction_vec
    delta = endpoint_b - endpoint_a
    spec = boundary_spec(config.boundary_mode)

    parameter_range = (0.0, 1.0)
    if spec.clip_x_axis:
        clipped = _clip_parameter_to_x_slab(
            endpoint_a, endpoint_b, config.half_box_nm
        )
        if clipped is None:
            return []
        parameter_range = clipped

    cuts: list[float] = [parameter_range[0], parameter_range[1]]
    for axis, periodic in enumerate(spec.periodic_axes):
        if not periodic or abs(float(delta[axis])) <= 1e-15:
            continue
        for boundary in (-config.half_box_nm, config.half_box_nm):
            parameter = float((boundary - endpoint_a[axis]) / delta[axis])
            if parameter_range[0] + 1e-12 < parameter < parameter_range[1] - 1e-12:
                cuts.append(parameter)
    cuts = _merge_cuts(cuts)

    fragments: list[CylinderFragment] = []
    for fragment_index, (start_t, end_t) in enumerate(zip(cuts[:-1], cuts[1:])):
        start = endpoint_a + start_t * delta
        end = endpoint_a + end_t * delta
        midpoint = 0.5 * (start + end)
        cell_shift = np.zeros(3, dtype=np.int64)
        physical_shift = np.zeros(3, dtype=np.float64)
        for axis, periodic in enumerate(spec.periodic_axes):
            if not periodic:
                continue
            cell = math.floor(
                (float(midpoint[axis]) + config.half_box_nm) / config.box_length_nm
            )
            cell_shift[axis] = -cell
            physical_shift[axis] = -cell * config.box_length_nm
        mapped_start = start + physical_shift
        mapped_end = end + physical_shift
        for axis, periodic in enumerate(spec.periodic_axes):
            if periodic:
                mapped_start[axis] = float(
                    np.clip(mapped_start[axis], -config.half_box_nm, config.half_box_nm)
                )
                mapped_end[axis] = float(
                    np.clip(mapped_end[axis], -config.half_box_nm, config.half_box_nm)
                )
        length = float(np.linalg.norm(mapped_end - mapped_start))
        if length <= 1e-9:
            continue
        fragments.append(
            CylinderFragment(
                source_index=source_index,
                fragment_index=fragment_index,
                t_start=start_t,
                t_end=end_t,
                cell_shift=tuple(int(value) for value in cell_shift),
                cylinder=Cylinder.from_endpoints(
                    mapped_start, mapped_end, config.cylinder_radius_nm
                ),
            )
        )
    return fragments


def fragment_trial(
    centers: ArrayLike, directions: ArrayLike, config: SimulationConfig
) -> list[list[CylinderFragment]]:
    centers_array = np.asarray(centers, dtype=np.float64)
    directions_array = np.asarray(directions, dtype=np.float64)
    expected = (config.max_count, 3)
    if centers_array.shape != expected or directions_array.shape != expected:
        raise ValueError(f"试验数组形状必须为 {expected}")
    return [
        split_particle_axis(center, direction, index, config)
        for index, (center, direction) in enumerate(
            zip(centers_array, directions_array, strict=True)
        )
    ]


# 模块3：胶囊仅作安全排除，候选接触全部由正式平底圆柱 GJK 距离界裁决。
def _evaluate_cylinder_contact(
    first: Cylinder,
    second: Cylinder,
    config: SimulationConfig,
) -> PairContactResult:
    capsule_gap = capsule_cylinder_distance(first, second)
    if capsule_gap > config.contact_cutoff_nm + config.broad_phase_guard_nm:
        return PairContactResult(
            connected=False,
            broad_phase_rejected=True,
            narrow_phase_calls=0,
            lower_nm=max(0.0, capsule_gap - config.broad_phase_guard_nm),
            upper_nm=None,
            iterations=0,
            converged=None,
        )

    accelerated = fast_cylinder_classify(
        first,
        second,
        config.contact_cutoff_nm,
        absolute_tolerance=config.gjk_absolute_tolerance_nm,
        relative_tolerance=config.gjk_relative_tolerance,
        max_iterations=config.gjk_max_iterations,
        threshold_guard=max(1e-8, config.broad_phase_guard_nm),
    )
    bounds = accelerated.bounds
    calls = 1
    reference_fallbacks = int(accelerated.used_fallback)
    classification = accelerated.classification
    if classification == "uncertain":
        tighter = distance_bounds(
            first,
            second,
            absolute_tolerance=min(1e-12, config.gjk_absolute_tolerance_nm * 0.01),
            relative_tolerance=min(1e-14, config.gjk_relative_tolerance * 0.1),
            max_iterations=max(2048, 4 * config.gjk_max_iterations),
        )
        calls += 1
        reference_fallbacks += 1
        if tighter.width < bounds.width:
            bounds = tighter
        classification = bounds.classify(config.contact_cutoff_nm)
    if classification == "uncertain":
        raise NarrowPhaseUncertainError(
            "平底圆柱 GJK 距离界跨越接触阈值: "
            f"[{bounds.lower:.16g},{bounds.upper:.16g}] nm"
        )
    return PairContactResult(
        connected=classification == "connected",
        broad_phase_rejected=False,
        narrow_phase_calls=calls,
        lower_nm=bounds.lower,
        upper_nm=bounds.upper,
        iterations=bounds.iterations,
        converged=True,
        reference_fallbacks=reference_fallbacks,
        distance_converged=bounds.converged,
    )


def _periodic_contact_shifts(
    first: CylinderFragment,
    second: CylinderFragment,
    config: SimulationConfig,
) -> tuple[tuple[float, float, float], ...]:
    spec = boundary_spec(config.boundary_mode)
    first_lower = first.aabb_lower
    first_upper = first.aabb_upper
    second_lower = second.aabb_lower
    second_upper = second.aabb_upper
    if (
        first_lower is None
        or first_upper is None
        or second_lower is None
        or second_upper is None
    ):
        raise RuntimeError("圆柱片段缺少 AABB 缓存")
    choices = [
        (-config.box_length_nm, 0.0, config.box_length_nm)
        if periodic
        else (0.0,)
        for periodic in spec.minimum_image_axes
    ]
    shifts: list[tuple[float, float, float]] = []
    cutoff = config.contact_cutoff_nm + config.broad_phase_guard_nm
    cutoff_sq = cutoff * cutoff
    for shift_x, shift_y, shift_z in product(*choices):
        if shift_x == 0.0 and shift_y == 0.0 and shift_z == 0.0:
            shifts.append((0.0, 0.0, 0.0))
            continue
        gap_sq = 0.0
        for axis, shift in enumerate((shift_x, shift_y, shift_z)):
            gap = max(
                float(first_lower[axis]) - (float(second_upper[axis]) + shift),
                (float(second_lower[axis]) + shift) - float(first_upper[axis]),
                0.0,
            )
            gap_sq += gap * gap
        if gap_sq <= cutoff_sq:
            shifts.append((shift_x, shift_y, shift_z))
    return tuple(shifts)


def evaluate_fragment_contact(
    first: CylinderFragment,
    second: CylinderFragment,
    config: SimulationConfig,
) -> PairContactResult:
    shifts = _periodic_contact_shifts(first, second, config)
    if not shifts:
        return PairContactResult(
            connected=False,
            broad_phase_rejected=True,
            narrow_phase_calls=0,
            lower_nm=config.contact_cutoff_nm + config.broad_phase_guard_nm,
            upper_nm=None,
            iterations=0,
            converged=None,
        )

    evaluations: list[PairContactResult] = []
    for shift in shifts:
        shifted_second = (
            second.cylinder
            if shift == (0.0, 0.0, 0.0)
            else second.cylinder.translated(np.asarray(shift, dtype=np.float64))
        )
        evaluation = _evaluate_cylinder_contact(
            first.cylinder,
            shifted_second,
            config,
        )
        evaluations.append(evaluation)
        if evaluation.connected:
            return PairContactResult(
                connected=True,
                broad_phase_rejected=False,
                narrow_phase_calls=sum(item.narrow_phase_calls for item in evaluations),
                lower_nm=min(item.lower_nm for item in evaluations),
                upper_nm=evaluation.upper_nm,
                iterations=sum(item.iterations for item in evaluations),
                converged=all(item.converged is not False for item in evaluations),
                reference_fallbacks=sum(
                    item.reference_fallbacks for item in evaluations
                ),
                distance_converged=all(
                    item.distance_converged is not False for item in evaluations
                ),
            )

    finite_uppers = [
        item.upper_nm for item in evaluations if item.upper_nm is not None
    ]
    return PairContactResult(
        connected=False,
        broad_phase_rejected=all(item.broad_phase_rejected for item in evaluations),
        narrow_phase_calls=sum(item.narrow_phase_calls for item in evaluations),
        lower_nm=min(item.lower_nm for item in evaluations),
        upper_nm=min(finite_uppers) if finite_uppers else None,
        iterations=sum(item.iterations for item in evaluations),
        converged=(
            None
            if not finite_uppers
            else all(item.converged is not False for item in evaluations)
        ),
        reference_fallbacks=sum(item.reference_fallbacks for item in evaluations),
        distance_converged=(
            None
            if not finite_uppers
            else all(
                item.distance_converged is not False for item in evaluations
            )
        ),
    )


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

    def union(self, first: int, second: int) -> bool:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return False
        if self.size[root_first] < self.size[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        self.size[root_first] += self.size[root_second]
        return True


class _SpatialHash:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.cell_count = int(math.ceil(config.box_length_nm / config.cell_size_nm))
        self.buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        spec = boundary_spec(config.boundary_mode)
        choices = [
            (-1, 0, 1) if periodic else (0,)
            for periodic in spec.minimum_image_axes
        ]
        self.query_shifts = tuple(
            config.box_length_nm * np.asarray(indices, dtype=np.float64)
            for indices in product(*choices)
        )

    def _cells(
        self, lower: Vec3, upper: Vec3
    ) -> tuple[tuple[int, int, int], ...]:
        lower = lower - self.config.contact_cutoff_nm
        upper = upper + self.config.contact_cutoff_nm
        if np.any(upper < -self.config.half_box_nm) or np.any(
            lower > self.config.half_box_nm
        ):
            return ()
        lower = np.maximum(lower, -self.config.half_box_nm)
        upper = np.minimum(upper, self.config.half_box_nm)
        index_ranges: list[range] = []
        for axis in range(3):
            low = math.floor(
                (float(lower[axis]) + self.config.half_box_nm)
                / self.config.cell_size_nm
            )
            high = math.floor(
                (float(upper[axis]) + self.config.half_box_nm)
                / self.config.cell_size_nm
            )
            low = min(self.cell_count - 1, max(0, low))
            high = min(self.cell_count - 1, max(0, high))
            index_ranges.append(range(low, high + 1))
        return tuple(
            (first, second, third)
            for first in index_ranges[0]
            for second in index_ranges[1]
            for third in index_ranges[2]
        )

    def candidates(self, fragment: CylinderFragment) -> list[int]:
        if fragment.aabb_lower is None or fragment.aabb_upper is None:
            raise RuntimeError("圆柱片段缺少 AABB 缓存")
        result: set[int] = set()
        for shift in self.query_shifts:
            for cell in self._cells(
                fragment.aabb_lower + shift,
                fragment.aabb_upper + shift,
            ):
                result.update(self.buckets.get(cell, ()))
        return sorted(result)

    def add(self, node: int, fragment: CylinderFragment) -> None:
        if fragment.aabb_lower is None or fragment.aabb_upper is None:
            raise RuntimeError("圆柱片段缺少 AABB 缓存")
        for cell in self._cells(fragment.aabb_lower, fragment.aabb_upper):
            self.buckets[cell].append(node)


def _witness_path(
    adjacency: dict[int, list[tuple[int, str]]],
    labels: Sequence[str],
    left_node: int,
    right_node: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    queue: deque[int] = deque([left_node])
    predecessor: dict[int, tuple[int, str] | None] = {left_node: None}
    while queue:
        node = queue.popleft()
        if node == right_node:
            break
        for neighbor, edge_type in adjacency.get(node, []):
            if neighbor not in predecessor:
                predecessor[neighbor] = (node, edge_type)
                queue.append(neighbor)
    if right_node not in predecessor:
        raise RuntimeError("并查集已连通但路径见证恢复失败")
    nodes: list[int] = [right_node]
    edge_types: list[str] = []
    while nodes[-1] != left_node:
        previous, edge_type = predecessor[nodes[-1]]  # type: ignore[misc]
        nodes.append(previous)
        edge_types.append(edge_type)
    nodes.reverse()
    edge_types.reverse()
    return tuple(labels[node] for node in nodes), tuple(edge_types)


# 模块4：按原粒子前缀增量加入全部片段，并以并查集判断两电极首次连通时刻。
def first_connection_prefix(
    centers: ArrayLike,
    directions: ArrayLike,
    config: SimulationConfig,
    *,
    include_witness: bool = False,
    use_spatial_index: bool = True,
) -> TrialResult:
    groups = fragment_trial(centers, directions, config)
    flat_fragments = [fragment for group in groups for fragment in group]
    offsets: list[int] = []
    position = 0
    for group in groups:
        offsets.append(position)
        position += len(group)

    fragment_count = len(flat_fragments)
    left_node = fragment_count
    right_node = fragment_count + 1
    union_find = _UnionFind(fragment_count + 2)
    spec = boundary_spec(config.boundary_mode)
    spatial_hash = _SpatialHash(config)
    active_nodes: list[int] = []
    adjacency: dict[int, list[tuple[int, str]]] = defaultdict(list)
    labels = [
        f"A{fragment.source_index + 1}:F{fragment.fragment_index + 1}"
        for fragment in flat_fragments
    ] + ["LEFT_ELECTRODE", "RIGHT_ELECTRODE"]

    candidate_pairs = 0
    component_skips = 0
    broad_rejections = 0
    narrow_calls = 0
    narrow_nonconverged = 0
    narrow_distance_nonconverged = 0
    narrow_reference_fallbacks = 0
    physical_contacts = 0
    electrode_contacts = 0
    internal_edges = 0

    def join(first: int, second: int, edge_type: str) -> None:
        union_find.union(first, second)
        if include_witness:
            adjacency[first].append((second, edge_type))
            adjacency[second].append((first, edge_type))

    for source_index, group in enumerate(groups):
        nodes = list(range(offsets[source_index], offsets[source_index] + len(group)))
        if spec.connect_same_source and len(nodes) > 1:
            for node in nodes[1:]:
                join(nodes[0], node, "same_source_internal")
                internal_edges += 1

        for node, fragment in zip(nodes, group, strict=True):
            left_gap = shape_plane_distance(
                fragment.cylinder, (1.0, 0.0, 0.0), -config.half_box_nm
            )
            right_gap = shape_plane_distance(
                fragment.cylinder, (1.0, 0.0, 0.0), config.half_box_nm
            )
            if left_gap <= config.contact_cutoff_nm:
                join(node, left_node, "left_electrode_contact")
                electrode_contacts += 1
            if right_gap <= config.contact_cutoff_nm:
                join(node, right_node, "right_electrode_contact")
                electrode_contacts += 1

            candidates = (
                spatial_hash.candidates(fragment)
                if use_spatial_index
                else list(active_nodes)
            )
            for other_node in candidates:
                candidate_pairs += 1
                if union_find.find(node) == union_find.find(other_node):
                    component_skips += 1
                    continue
                evaluation = evaluate_fragment_contact(
                    fragment, flat_fragments[other_node], config
                )
                broad_rejections += int(evaluation.broad_phase_rejected)
                narrow_calls += evaluation.narrow_phase_calls
                if evaluation.converged is False:
                    narrow_nonconverged += 1
                if evaluation.distance_converged is False:
                    narrow_distance_nonconverged += 1
                narrow_reference_fallbacks += evaluation.reference_fallbacks
                if evaluation.connected:
                    join(node, other_node, "flat_cylinder_contact")
                    physical_contacts += 1

            if use_spatial_index:
                spatial_hash.add(node, fragment)
            active_nodes.append(node)

        if union_find.find(left_node) == union_find.find(right_node):
            witness_nodes = None
            witness_edges = None
            if include_witness:
                witness_nodes, witness_edges = _witness_path(
                    adjacency, labels, left_node, right_node
                )
            diagnostics = TrialDiagnostics(
                fragment_count=fragment_count,
                candidate_pairs=candidate_pairs,
                component_skips=component_skips,
                broad_phase_rejections=broad_rejections,
                narrow_phase_calls=narrow_calls,
                narrow_nonconverged=narrow_nonconverged,
                narrow_distance_nonconverged=narrow_distance_nonconverged,
                narrow_reference_fallbacks=narrow_reference_fallbacks,
                physical_contacts=physical_contacts,
                electrode_contacts=electrode_contacts,
                internal_edges=internal_edges,
                witness_nodes=witness_nodes,
                witness_edge_types=witness_edges,
            )
            return TrialResult(source_index + 1, diagnostics)

    diagnostics = TrialDiagnostics(
        fragment_count=fragment_count,
        candidate_pairs=candidate_pairs,
        component_skips=component_skips,
        broad_phase_rejections=broad_rejections,
        narrow_phase_calls=narrow_calls,
        narrow_nonconverged=narrow_nonconverged,
        narrow_distance_nonconverged=narrow_distance_nonconverged,
        narrow_reference_fallbacks=narrow_reference_fallbacks,
        physical_contacts=physical_contacts,
        electrode_contacts=electrode_contacts,
        internal_edges=internal_edges,
    )
    return TrialResult(config.max_count + 1, diagnostics)


def run_one_trial(config: SimulationConfig, trial_id: int) -> dict[str, Any]:
    centers, directions = generate_trial(config, trial_id)
    result = first_connection_prefix(centers, directions, config)
    return {
        "trial_id": trial_id,
        "first_connection_index": result.first_connection_index,
        "censored": result.first_connection_index == config.max_count + 1,
        **result.diagnostics.to_dict(),
    }


# 模块5：固定试验数按确定批次分片，支持指纹核验、续跑与多进程确定性合并。
def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_trial_ids(config: SimulationConfig, trial_ids: Sequence[int]) -> dict[str, Any]:
    ids = [int(value) for value in trial_ids]
    if ids != sorted(set(ids)):
        raise ValueError("分片 trial_id 必须严格递增且不重复")
    if any(value < 0 or value >= config.trial_count for value in ids):
        raise ValueError("分片 trial_id 超出固定试验数")
    started = time.perf_counter()
    records = [run_one_trial(config, trial_id) for trial_id in ids]
    return {
        "kind": "microstructure_trial_shard",
        "schema_version": SCHEMA_VERSION,
        "configuration": config.to_dict(),
        "configuration_fingerprint": config.fingerprint,
        "trial_ids": ids,
        "records": records,
        "runtime_seconds": time.perf_counter() - started,
    }


def _validate_shard(
    payload: dict[str, Any], config: SimulationConfig, expected_ids: Sequence[int] | None = None
) -> None:
    if payload.get("kind") != "microstructure_trial_shard":
        raise ValueError("文件不是仿真分片")
    if payload.get("configuration_fingerprint") != config.fingerprint:
        raise ValueError("分片配置指纹不兼容")
    stored = SimulationConfig.from_dict(payload.get("configuration", {}))
    if stored.fingerprint != config.fingerprint:
        raise ValueError("分片内嵌配置与指纹不一致")
    ids = [int(value) for value in payload.get("trial_ids", [])]
    record_ids = [int(row["trial_id"]) for row in payload.get("records", [])]
    if ids != sorted(set(ids)) or record_ids != ids:
        raise ValueError("分片 trial_id 或记录顺序无效")
    if expected_ids is not None and ids != [int(value) for value in expected_ids]:
        raise ValueError("续跑分片范围与请求不一致")


def run_shard(
    config: SimulationConfig,
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
    config: SimulationConfig,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    records: dict[int, dict[str, Any]] = {}
    runtime_seconds = 0.0
    for payload in payloads:
        _validate_shard(payload, config)
        runtime_seconds += float(payload.get("runtime_seconds", 0.0))
        for record in payload["records"]:
            trial_id = int(record["trial_id"])
            if trial_id in records:
                raise ValueError(f"合并发现重复 trial_id: {trial_id}")
            records[trial_id] = record
    expected = set(range(config.trial_count))
    present = set(records)
    if require_complete and present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ValueError(f"分片不完整或越界: missing={missing}, extra={extra}")
    ordered = [records[index] for index in sorted(records)]
    diagnostic_fields = (
        "fragment_count",
        "candidate_pairs",
        "component_skips",
        "broad_phase_rejections",
        "narrow_phase_calls",
        "narrow_nonconverged",
        "narrow_distance_nonconverged",
        "narrow_reference_fallbacks",
        "physical_contacts",
        "electrode_contacts",
        "internal_edges",
    )
    totals = {
        field: int(sum(int(record[field]) for record in ordered))
        for field in diagnostic_fields
    }
    return {
        "kind": "microstructure_threshold_samples",
        "schema_version": SCHEMA_VERSION,
        "configuration": config.to_dict(),
        "configuration_fingerprint": config.fingerprint,
        "records": ordered,
        "diagnostics_total": totals,
        "censored_trials": int(sum(bool(record["censored"]) for record in ordered)),
        "shard_runtime_seconds": runtime_seconds,
    }


def merge_shards(
    paths: Sequence[Path | str],
    config: SimulationConfig,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    payloads = [
        json.loads(Path(path).read_text(encoding="utf-8-sig")) for path in paths
    ]
    return merge_shard_payloads(payloads, config, require_complete=require_complete)


def _worker_run_shard(arguments: tuple[dict[str, Any], tuple[int, ...]]) -> dict[str, Any]:
    config_payload, trial_ids = arguments
    return run_trial_ids(SimulationConfig.from_dict(config_payload), trial_ids)


def run_simulation(
    config: SimulationConfig,
    output_dir: Path | str,
    *,
    workers: int = 1,
    batch_size: int = 8,
    resume: bool = True,
    merged_name: str = "threshold_samples.json",
) -> Path:
    if workers < 1 or batch_size < 1:
        raise ValueError("workers 和 batch_size 必须为正整数")
    warm_up_fast_geometry()
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
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=warm_up_fast_geometry,
        ) as executor:
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

    merged = merge_shards(paths, config, require_complete=True)
    merged_path = directory / merged_name
    _atomic_write_json(merged_path, merged)
    return merged_path


def load_threshold_artifact(
    path: Path | str,
) -> tuple[SimulationConfig, NDArray[np.int64], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("kind") != "microstructure_threshold_samples":
        raise ValueError("文件不是阈值样本流")
    config = SimulationConfig.from_dict(payload.get("configuration", {}))
    if payload.get("configuration_fingerprint") != config.fingerprint:
        raise ValueError("阈值样本流配置指纹不一致")
    records = payload.get("records", [])
    ids = [int(record["trial_id"]) for record in records]
    if ids != list(range(config.trial_count)):
        raise ValueError("阈值样本流 trial_id 不完整或未排序")
    samples = np.asarray(
        [int(record["first_connection_index"]) for record in records],
        dtype=np.int64,
    )
    if np.any(samples < 1) or np.any(samples > config.max_count + 1):
        raise ValueError("阈值样本超出合法范围")
    return config, samples, records


# 模块6：由同一首次导通样本流计算概率、区间与整数阈值。
def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("二项计数无效")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必须位于 (0,1)")
    z_value = float(norm.ppf(0.5 + 0.5 * confidence))
    estimate = successes / trials
    denominator = 1.0 + z_value * z_value / trials
    center = (estimate + z_value * z_value / (2.0 * trials)) / denominator
    half_width = z_value * math.sqrt(
        estimate * (1.0 - estimate) / trials
        + z_value * z_value / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def clopper_pearson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("二项计数无效")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必须位于 (0,1)")
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lower, upper


def wilson_one_sided_bounds(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("二项计数无效")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必须位于 (0,1)")
    z_value = float(norm.ppf(confidence))
    estimate = successes / trials
    denominator = 1.0 + z_value * z_value / trials
    center = (estimate + z_value * z_value / (2.0 * trials)) / denominator
    half_width = z_value * math.sqrt(
        estimate * (1.0 - estimate) / trials
        + z_value * z_value / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def clopper_pearson_one_sided_bounds(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("二项计数无效")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必须位于 (0,1)")
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))
    )
    return lower, upper


def probability_at_prefix(
    first_connection_samples: ArrayLike,
    count: int,
    max_count: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    samples = np.asarray(first_connection_samples, dtype=np.int64)
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("首次导通样本必须是一维非空数组")
    if count < 0 or count > max_count:
        raise ValueError("查询粒子数必须位于 [0,max_count]")
    if np.any(samples < 1) or np.any(samples > max_count + 1):
        raise ValueError("首次导通样本包含非法值")
    successes = int(np.count_nonzero(samples <= count))
    trials = int(samples.size)
    wilson = wilson_interval(successes, trials, confidence)
    exact = clopper_pearson_interval(successes, trials, confidence)
    wilson_one_sided = wilson_one_sided_bounds(successes, trials, confidence)
    exact_one_sided = clopper_pearson_one_sided_bounds(
        successes, trials, confidence
    )
    return {
        "count": int(count),
        "successes": successes,
        "trials": trials,
        "estimate": successes / trials,
        "confidence": confidence,
        "wilson_interval": [wilson[0], wilson[1]],
        "clopper_pearson_interval": [exact[0], exact[1]],
        "wilson_one_sided_bounds": [
            wilson_one_sided[0],
            wilson_one_sided[1],
        ],
        "clopper_pearson_one_sided_bounds": [
            exact_one_sided[0],
            exact_one_sided[1],
        ],
    }


def smallest_empirical_threshold(
    first_connection_samples: ArrayLike,
    max_count: int,
    target: float = 0.90,
) -> int | None:
    samples = np.asarray(first_connection_samples, dtype=np.int64)
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("首次导通样本必须是一维非空数组")
    if not 0.0 < target < 1.0:
        raise ValueError("target 必须位于 (0,1)")
    if np.any(samples < 1) or np.any(samples > max_count + 1):
        raise ValueError("首次导通样本包含非法值")
    required = math.ceil(target * samples.size)
    candidate = int(np.partition(samples, required - 1)[required - 1])
    return candidate if candidate <= max_count else None


def smallest_confidence_threshold(
    first_connection_samples: ArrayLike,
    max_count: int,
    target: float = 0.90,
    confidence: float = 0.95,
    interval: str = "clopper_pearson",
    sided: str = "two",
) -> int | None:
    samples = np.asarray(first_connection_samples, dtype=np.int64)
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("首次导通样本必须是一维非空数组")
    if max_count < 1 or np.any(samples < 1) or np.any(samples > max_count + 1):
        raise ValueError("首次导通样本或 max_count 非法")
    if not 0.0 < target < 1.0:
        raise ValueError("target 必须位于 (0,1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 必须位于 (0,1)")
    if interval not in {"wilson", "clopper_pearson"}:
        raise ValueError("interval 必须为 wilson 或 clopper_pearson")
    if sided not in {"one", "two"}:
        raise ValueError("sided 必须为 one 或 two")
    for count in range(1, max_count + 1):
        successes = int(np.count_nonzero(samples <= count))
        if interval == "wilson":
            bounds_function = (
                wilson_one_sided_bounds if sided == "one" else wilson_interval
            )
        else:
            bounds_function = (
                clopper_pearson_one_sided_bounds
                if sided == "one"
                else clopper_pearson_interval
            )
        lower = bounds_function(successes, int(samples.size), confidence)[0]
        if lower >= target:
            return count
    return None


def nominal_volume_percent(count: int, config: SimulationConfig | None = None) -> float:
    if count < 0:
        raise ValueError("粒子数不能为负")
    box_length = BOX_LENGTH_NM if config is None else config.box_length_nm
    length = CYLINDER_LENGTH_NM if config is None else config.cylinder_length_nm
    radius = CYLINDER_RADIUS_NM if config is None else config.cylinder_radius_nm
    particle_volume = math.pi * radius * radius * length
    return 100.0 * count * particle_volume / (box_length**3)
