from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUESTION_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = PROJECT_ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from microstructure_sim import (
    clopper_pearson_one_sided_bounds,
    wilson_one_sided_bounds,
)
from mixed_microstructure_sim import (
    BOUNDARY_CONTRACT,
    MixedSimulationConfig,
    connectivity_samples_at_design,
    load_pareto_frontier_artifact,
    run_pareto_frontier_simulation,
)
from result_registry import export_latex, register_result


SCHEMA_VERSION = 1
A_COST_WEIGHT = 567
B_COST_WEIGHT = 64
COST_SCALE_YUAN = math.pi / 120_000.0
A_VOLUME_UM3 = 0.0045 * math.pi
B_VOLUME_UM3 = 0.032 * math.pi / 3.0
BOX_VOLUME_UM3 = 1000.0

DEFAULT_SCREENING_TRIALS = 2_000
DEFAULT_CONFIRMATION_TRIALS = 50_000
DEFAULT_SCREENING_STREAM_ID = 4
DEFAULT_CONFIRMATION_STREAM_ID = 5
DEFAULT_TARGET = 0.90
DEFAULT_FAMILYWISE_CONFIDENCE = 0.95
SCREENING_CANDIDATE_RULES = ("point_estimate", "cp_lower")

SCREENING_CSV_FIELDS = (
    "n_a",
    "n_b",
    "cost_weight",
    "cost_yuan",
    "a_volume_um3",
    "b_volume_um3",
    "total_volume_um3",
    "a_volume_percent",
    "b_volume_percent",
    "total_volume_percent",
    "a_cost_yuan",
    "b_cost_yuan",
    "successes",
    "trials",
    "estimate",
    "wilson_one_sided_lower",
    "wilson_one_sided_upper",
    "clopper_pearson_one_sided_lower",
    "clopper_pearson_one_sided_upper",
    "screening_empirically_feasible",
    "screening_cp_lower_feasible",
    "configuration_fingerprint",
    "artifact_path",
    "artifact_sha256",
)

CONFIRMATION_CSV_FIELDS = (
    "role",
    "proof_bound",
    "proof_status",
    *SCREENING_CSV_FIELDS[:-5],
    "configuration_fingerprint",
    "artifact_path",
    "artifact_sha256",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_or_validate_freeze(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8-sig"))
        if stored != payload:
            raise ValueError(
                "已有 Q4 冻结文件与本次协议不一致；请改用新的输出目录"
            )
        return
    _write_json(path, payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _parse_design(text: str) -> tuple[int, int]:
    pieces = text.replace(":", ",").split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("设计必须写成 N_A,N_B")
    try:
        design = int(pieces[0]), int(pieces[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("设计数量必须为整数") from exc
    if min(design) < 0:
        raise argparse.ArgumentTypeError("设计数量必须非负")
    return design


def cost_weight(n_a: int, n_b: int) -> int:
    if n_a < 0 or n_b < 0:
        raise ValueError("介质数量必须非负")
    return A_COST_WEIGHT * int(n_a) + B_COST_WEIGHT * int(n_b)


def design_metrics(n_a: int, n_b: int) -> dict[str, Any]:
    weight = cost_weight(n_a, n_b)
    a_volume = n_a * A_VOLUME_UM3
    b_volume = n_b * B_VOLUME_UM3
    a_cost = n_a * 189.0 * math.pi / 40_000.0
    b_cost = n_b * math.pi / 1_875.0
    return {
        "n_a": int(n_a),
        "n_b": int(n_b),
        "cost_weight": weight,
        "cost_yuan": COST_SCALE_YUAN * weight,
        "a_volume_um3": a_volume,
        "b_volume_um3": b_volume,
        "total_volume_um3": a_volume + b_volume,
        "a_volume_percent": 100.0 * a_volume / BOX_VOLUME_UM3,
        "b_volume_percent": 100.0 * b_volume / BOX_VOLUME_UM3,
        "total_volume_percent": 100.0 * (a_volume + b_volume) / BOX_VOLUME_UM3,
        "a_cost_yuan": a_cost,
        "b_cost_yuan": b_cost,
    }


def cheaper_maximal_frontier(candidate_weight: int) -> list[tuple[int, int]]:
    if candidate_weight < 0:
        raise ValueError("候选成本权重不能为负")
    if candidate_weight == 0:
        return []
    limit = candidate_weight - 1
    frontier = []
    for n_a in range(limit // A_COST_WEIGHT + 1):
        n_b = (limit - A_COST_WEIGHT * n_a) // B_COST_WEIGHT
        frontier.append((n_a, n_b))
    return frontier


def equal_cost_designs(weight: int) -> list[tuple[int, int]]:
    if weight < 0:
        raise ValueError("成本权重不能为负")
    designs = []
    for n_a in range(weight // A_COST_WEIGHT + 1):
        remainder = weight - A_COST_WEIGHT * n_a
        if remainder % B_COST_WEIGHT == 0:
            designs.append((n_a, remainder // B_COST_WEIGHT))
    return designs


def count_strictly_cheaper_designs(weight: int) -> int:
    if weight <= 0:
        return 0
    limit = weight - 1
    return sum(
        (limit - A_COST_WEIGHT * n_a) // B_COST_WEIGHT + 1
        for n_a in range(limit // A_COST_WEIGHT + 1)
    )


def minimum_unexcluded_cheaper_design(
    candidate_weight: int,
    excluded_frontier: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    if candidate_weight <= 0:
        return None
    limit = candidate_weight - 1
    max_a = limit // A_COST_WEIGHT
    excluded_at_a: dict[int, int] = {}
    for n_a, n_b in excluded_frontier:
        excluded_at_a[n_a] = max(excluded_at_a.get(n_a, -1), n_b)

    best: tuple[int, int] | None = None
    best_weight: int | None = None
    covered_b = -1
    for n_a in range(max_a, -1, -1):
        covered_b = max(covered_b, excluded_at_a.get(n_a, -1))
        maximum_b = (limit - A_COST_WEIGHT * n_a) // B_COST_WEIGHT
        first_unexcluded_b = covered_b + 1
        if first_unexcluded_b <= maximum_b:
            weight = cost_weight(n_a, first_unexcluded_b)
            if best_weight is None or weight < best_weight:
                best = (n_a, first_unexcluded_b)
                best_weight = weight
    return best


def _axis_values(maximum: int, step: int) -> list[int]:
    if maximum < 0 or step < 1:
        raise ValueError("探索上限必须非负且步长必须为正")
    values = list(range(0, maximum + 1, step))
    if values[-1] != maximum:
        values.append(maximum)
    return values


def screening_designs(
    *,
    max_n_a: int,
    max_n_b: int,
    step_n_a: int,
    step_n_b: int,
    explicit_designs: Sequence[tuple[int, int]] | None = None,
    maximum_designs: int = 500,
) -> list[tuple[int, int]]:
    if maximum_designs < 1:
        raise ValueError("最大探索设计数必须为正整数")
    if explicit_designs:
        designs = sorted(
            {(int(n_a), int(n_b)) for n_a, n_b in explicit_designs},
            key=lambda value: (cost_weight(*value), value[0], value[1]),
        )
        if any(min(design) < 0 for design in designs):
            raise ValueError("探索设计数量必须非负")
    else:
        designs = [
            (n_a, n_b)
            for n_a in _axis_values(max_n_a, step_n_a)
            for n_b in _axis_values(max_n_b, step_n_b)
        ]
        designs.sort(key=lambda value: (cost_weight(*value), value[0], value[1]))
    if len(designs) > maximum_designs:
        raise ValueError(
            f"探索设计数 {len(designs)} 超过上限 {maximum_designs}；请增大步长或显式提高上限"
        )
    return designs


def _design_tag(n_a: int, n_b: int) -> str:
    return f"A{n_a:06d}_B{n_b:06d}"


def _probability_fields(
    successes: int, trials: int, confidence: float
) -> dict[str, Any]:
    wilson = wilson_one_sided_bounds(successes, trials, confidence)
    exact = clopper_pearson_one_sided_bounds(successes, trials, confidence)
    return {
        "successes": int(successes),
        "trials": int(trials),
        "estimate": successes / trials,
        "confidence": confidence,
        "wilson_one_sided_lower": wilson[0],
        "wilson_one_sided_upper": wilson[1],
        "clopper_pearson_one_sided_lower": exact[0],
        "clopper_pearson_one_sided_upper": exact[1],
    }


def integer_domain_success_counts(
    frontiers: Sequence[Sequence[tuple[int, int]]],
    max_n_a: int,
    max_n_b: int,
) -> np.ndarray:
    if max_n_a < 0 or max_n_b < 0:
        raise ValueError("完整整数域上限必须非负")
    if len(frontiers) > np.iinfo(np.int32).max:
        raise ValueError("试验数超过 int32 成功次数容量")
    difference = np.zeros((max_n_a + 1, max_n_b + 2), dtype=np.int32)
    rows = np.arange(max_n_a + 1, dtype=np.intp)
    sentinel = max_n_b + 1

    for frontier in frontiers:
        requirements = np.full(max_n_a + 1, sentinel, dtype=np.intp)
        labels = tuple((int(first), int(second)) for first, second in frontier)
        for index, (first, second) in enumerate(labels):
            start = max(0, first)
            stop = (
                max_n_a + 1
                if index + 1 == len(labels)
                else min(max_n_a + 1, labels[index + 1][0])
            )
            if start < stop and 0 <= second <= max_n_b:
                requirements[start:stop] = second
        np.add.at(difference, (rows, requirements), 1)

    cumulative = np.cumsum(difference, axis=1, dtype=np.int32)
    return cumulative[:, : max_n_b + 1]


def minimum_empirically_feasible_design(
    success_counts: np.ndarray,
    *,
    trials: int,
    target: float,
) -> tuple[int, int, int] | None:
    counts = np.asarray(success_counts)
    if counts.ndim != 2 or min(counts.shape) < 1:
        raise ValueError("成功次数矩阵必须为非空二维数组")
    if trials < 1 or not 0.0 < target < 1.0:
        raise ValueError("试验数必须为正且目标概率必须位于 (0,1)")
    if np.any(counts < 0) or np.any(counts > trials):
        raise ValueError("成功次数必须位于 [0,trials]")
    if np.any(np.diff(counts, axis=1) < 0):
        raise ValueError("每个 N_A 行的成功次数必须随 N_B 单调不减")

    required = int(math.ceil(target * trials))
    while required > 0 and (required - 1) / trials >= target:
        required -= 1
    while required <= trials and required / trials < target:
        required += 1
    if required > trials:
        return None

    best: tuple[int, int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    for n_a, row in enumerate(counts):
        n_b = int(np.searchsorted(row, required, side="left"))
        if n_b >= row.size:
            continue
        key = cost_weight(n_a, n_b), n_a, n_b
        if best_key is None or key < best_key:
            best_key = key
            best = n_a, n_b, int(row[n_b])
    return best


def minimum_screening_feasible_design(
    success_counts: np.ndarray,
    *,
    trials: int,
    target: float,
    confidence: float,
    rule: str,
) -> tuple[int, int, int, int] | None:
    if rule not in SCREENING_CANDIDATE_RULES:
        raise ValueError(f"未知探索候选规则：{rule}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("探索置信水平必须位于 (0,1)")
    if rule == "point_estimate":
        selected = minimum_empirically_feasible_design(
            success_counts, trials=trials, target=target
        )
        if selected is None:
            return None
        return selected[0], selected[1], selected[2], selected[2]

    required = next(
        (
            successes
            for successes in range(trials + 1)
            if clopper_pearson_one_sided_bounds(
                successes, trials, confidence
            )[0]
            >= target
        ),
        trials + 1,
    )
    if required > trials:
        return None
    counts = np.asarray(success_counts)
    if counts.ndim != 2 or min(counts.shape) < 1:
        raise ValueError("成功次数矩阵必须为非空二维数组")
    if np.any(counts < 0) or np.any(counts > trials):
        raise ValueError("成功次数必须位于 [0,trials]")
    if np.any(np.diff(counts, axis=1) < 0):
        raise ValueError("每个 N_A 行的成功次数必须随 N_B 单调不减")

    best: tuple[int, int, int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    for n_a, row in enumerate(counts):
        n_b = int(np.searchsorted(row, required, side="left"))
        if n_b >= row.size:
            continue
        key = cost_weight(n_a, n_b), n_a, n_b
        if best_key is None or key < best_key:
            best_key = key
            best = n_a, n_b, int(row[n_b]), required
    return best


def _write_integer_domain_counts(
    path: Path,
    counts: np.ndarray,
    *,
    trials: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            success_counts=np.asarray(counts, dtype=np.int32),
            trials=np.asarray(trials, dtype=np.int64),
            max_n_a=np.asarray(counts.shape[0] - 1, dtype=np.int64),
            max_n_b=np.asarray(counts.shape[1] - 1, dtype=np.int64),
        )
    temporary.replace(path)


def _analyze_pareto_design(
    artifact_path: Path,
    n_a: int,
    n_b: int,
    *,
    confidence: float,
) -> dict[str, Any]:
    config, frontiers, _ = load_pareto_frontier_artifact(artifact_path)
    if n_a > config.n_a or n_b > config.n_b:
        raise ValueError("查询设计超过二维静态图的最大粒子前缀")
    samples = connectivity_samples_at_design(frontiers, n_a, n_b)
    successes = int(np.count_nonzero(samples))
    payload = json.loads(artifact_path.read_text(encoding="utf-8-sig"))
    return {
        **design_metrics(n_a, n_b),
        **_probability_fields(successes, config.trial_count, confidence),
        "configuration": config.to_dict(),
        "configuration_fingerprint": config.fingerprint,
        "maximum_static_graph_design": [config.n_a, config.n_b],
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": _sha256(artifact_path),
        "boundary_contract": payload["boundary_contract"],
        "diagnostics_total": payload["diagnostics_total"],
        "pareto_search_total": payload["pareto_search_total"],
        "shard_runtime_seconds": payload["shard_runtime_seconds"],
    }


def _base_config(
    n_a: int,
    n_b: int,
    *,
    trials: int,
    master_seed: int,
    stream_id: int,
) -> MixedSimulationConfig:
    return MixedSimulationConfig(
        n_a=n_a,
        n_b=n_b,
        trial_count=trials,
        master_seed=master_seed,
        stream_id=stream_id,
        boundary_mode="D",
    )


def build_confirmation_freeze(
    screening_manifest: dict[str, Any],
    *,
    screening_json_path: Path,
    screening_csv_path: Path,
    confirmation_trials: int,
    confirmation_stream_id: int,
    familywise_confidence: float,
) -> dict[str, Any]:
    if screening_manifest.get("kind") != "q4_screening_results":
        raise ValueError("文件不是 Q4 探索结果")
    candidate = screening_manifest.get("screening_candidate")
    if candidate is None:
        raise ValueError("探索阶段没有点估计达到目标的候选，不能冻结确认")
    if confirmation_trials < 1:
        raise ValueError("确认样本数必须为正整数")
    if confirmation_stream_id == int(screening_manifest["stream_id"]):
        raise ValueError("确认 stream_id 必须与探索 stream_id 不同")
    if not 0.0 < familywise_confidence < 1.0:
        raise ValueError("联合置信水平必须位于 (0,1)")
    target = float(screening_manifest["target_probability"])
    if not 0.0 < target < 1.0:
        raise ValueError("目标概率必须位于 (0,1)")

    n_a = int(candidate["n_a"])
    n_b = int(candidate["n_b"])
    candidate_weight = cost_weight(n_a, n_b)
    frontier = cheaper_maximal_frontier(candidate_weight)
    statement_count = 1 + len(frontier)
    familywise_alpha = 1.0 - familywise_confidence
    per_statement_alpha = familywise_alpha / statement_count
    per_statement_confidence = 1.0 - per_statement_alpha

    screening_config = MixedSimulationConfig.from_dict(candidate["configuration"])
    designs = [("candidate", n_a, n_b)] + [
        ("strictly_cheaper_maximal", first, second) for first, second in frontier
    ]
    maximum_n_a = max(design_a for _, design_a, _ in designs)
    maximum_n_b = max(design_b for _, _, design_b in designs)
    confirmation_config = replace(
        screening_config,
        n_a=maximum_n_a,
        n_b=maximum_n_b,
        trial_count=int(confirmation_trials),
        stream_id=int(confirmation_stream_id),
    )
    confirmation_designs = [
        {
            "role": role,
            **design_metrics(design_a, design_b),
            "proof_bound": "lower" if role == "candidate" else "upper",
        }
        for role, design_a, design_b in designs
    ]

    return {
        "kind": "q4_confirmation_freeze",
        "schema_version": SCHEMA_VERSION,
        "question": 4,
        "source_screening": {
            "json_path": str(screening_json_path.resolve()),
            "json_sha256": _sha256(screening_json_path),
            "csv_path": str(screening_csv_path.resolve()),
            "csv_sha256": _sha256(screening_csv_path),
            "pareto_artifact_path": screening_manifest[
                "pareto_frontier_artifact"
            ],
            "pareto_artifact_sha256": screening_manifest[
                "pareto_frontier_artifact_sha256"
            ],
            "fixed_trial_count": screening_manifest["fixed_trial_count"],
            "stream_id": screening_manifest["stream_id"],
            "shared_crn_across_designs": True,
            "one_static_graph_per_trial": True,
            "maximum_static_graph_design": screening_manifest[
                "maximum_static_graph_design"
            ],
        },
        "candidate_freeze": {
            **design_metrics(n_a, n_b),
            "target_probability": target,
            "selection_rule": screening_manifest["candidate_selection_rule"],
            "screening_estimate": candidate["estimate"],
            "equal_cost_designs": [list(value) for value in equal_cost_designs(candidate_weight)],
        },
        "strictly_cheaper_domain": {
            "integer_design_count": count_strictly_cheaper_designs(candidate_weight),
            "maximal_frontier": [list(value) for value in frontier],
            "maximal_frontier_count": len(frontier),
            "coverage_rule": (
                "每个严格更便宜非负整数设计均被至少一个极大前沿点分量支配"
            ),
        },
        "confirmation_protocol": {
            "boundary_mode": "D",
            "boundary_contract": BOUNDARY_CONTRACT,
            "b_geometry_status": "exact_ball_cell_intersection_fragments",
            "a_geometry_status": "centerline_cut_approximation",
            "boundary_b_sensitivity_status": "not_run_not_claimed",
            "fixed_trial_count": int(confirmation_trials),
            "target_probability": target,
            "master_seed": screening_config.master_seed,
            "screening_stream_id": screening_config.stream_id,
            "confirmation_stream_id": int(confirmation_stream_id),
            "stream_ids_distinct": True,
            "shared_crn_across_candidate_and_frontier": True,
            "one_static_graph_per_trial": True,
            "maximum_static_graph_design": [maximum_n_a, maximum_n_b],
            "configuration": confirmation_config.to_dict(),
            "configuration_fingerprint": confirmation_config.fingerprint,
            "familywise_confidence": familywise_confidence,
            "familywise_alpha": familywise_alpha,
            "bonferroni_statement_count": statement_count,
            "per_statement_alpha": per_statement_alpha,
            "per_statement_confidence": per_statement_confidence,
            "candidate_statement": "one_sided_lower_bound",
            "frontier_statements": "one_sided_upper_bounds",
            "authoritative_interval": "clopper_pearson_one_sided",
            "diagnostic_interval": "wilson_one_sided",
        },
        "confirmation_designs": confirmation_designs,
    }


def analyze_confirmation_records(
    freeze: dict[str, Any], records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected = {
        (row["role"], int(row["n_a"]), int(row["n_b"]))
        for row in freeze["confirmation_designs"]
    }
    observed = {
        (row["role"], int(row["n_a"]), int(row["n_b"])) for row in records
    }
    if expected != observed or len(records) != len(expected):
        raise ValueError("确认记录与冻结设计集不一致")
    candidate_records = [row for row in records if row["role"] == "candidate"]
    if len(candidate_records) != 1:
        raise ValueError("确认记录必须恰有一个候选")
    candidate = candidate_records[0]
    candidate_feasible = (
        candidate["proof_status"] == "candidate_statistically_feasible"
    )
    frontier = [row for row in records if row["role"] != "candidate"]
    excluded = [
        row
        for row in frontier
        if row["proof_status"] == "strictly_cheaper_design_excluded"
    ]
    not_excluded = [
        row
        for row in frontier
        if row["proof_status"] != "strictly_cheaper_design_excluded"
    ]
    all_cheaper_excluded = len(not_excluded) == 0

    if candidate_feasible and all_cheaper_excluded:
        status = "globally_certified_minimum_cost"
        confidence_percent = 100.0 * float(
            freeze["confirmation_protocol"]["familywise_confidence"]
        )
        conclusion = (
            f"所声明 D 边界模型下的联合 {confidence_percent:g}% 最低成本"
        )
    elif candidate_feasible:
        status = "lowest_statistically_feasible_cost"
        conclusion = "最低统计可行成本（尚未排除全部更便宜方案）"
    else:
        status = "screening_candidate_not_confirmed"
        conclusion = "探索候选未通过独立确认，不报告最低成本"

    candidate_weight = int(candidate["cost_weight"])
    excluded_designs = [(int(row["n_a"]), int(row["n_b"])) for row in excluded]
    minimum_unexcluded = minimum_unexcluded_cheaper_design(
        candidate_weight, excluded_designs
    )
    if candidate_feasible:
        lower_weight = (
            candidate_weight
            if minimum_unexcluded is None
            else cost_weight(*minimum_unexcluded)
        )
        uncertainty = {
            "lower_cost_weight": lower_weight,
            "upper_cost_weight": candidate_weight,
            "lower_cost_yuan": lower_weight * COST_SCALE_YUAN,
            "upper_cost_yuan": candidate_weight * COST_SCALE_YUAN,
            "minimum_not_excluded_design": (
                None if minimum_unexcluded is None else list(minimum_unexcluded)
            ),
        }
    else:
        uncertainty = None

    return {
        "result_status": status,
        "conclusion_label_zh": conclusion,
        "candidate_statistically_feasible": candidate_feasible,
        "all_strictly_cheaper_maximal_designs_excluded": all_cheaper_excluded,
        "excluded_frontier_count": len(excluded),
        "not_excluded_frontier_count": len(not_excluded),
        "not_excluded_frontier": [
            {
                "n_a": row["n_a"],
                "n_b": row["n_b"],
                "cost_weight": row["cost_weight"],
                "estimate": row["estimate"],
                "cp_upper": row["clopper_pearson_one_sided_upper"],
            }
            for row in not_excluded
        ],
        "reported_design": (
            design_metrics(int(candidate["n_a"]), int(candidate["n_b"]))
            if candidate_feasible
            else None
        ),
        "cost_uncertainty_interval": uncertainty,
        "equal_cost_designs": freeze["candidate_freeze"]["equal_cost_designs"],
        "composition_uniqueness_claimed": False,
        "proof_note": (
            "Clopper-Pearson 为最终判据；Wilson 仅作诊断。若任一严格更便宜极大点未排除，"
            "结论自动降级。"
        ),
    }


