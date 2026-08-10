# Q4：二维成本整数域 Pareto 前沿扫描与联合确认程序
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


def run_screening(
    designs: Sequence[tuple[int, int]],
    output_dir: Path,
    *,
    trials: int,
    confidence: float,
    target: float,
    master_seed: int,
    stream_id: int,
    workers: int,
    batch_size: int,
    resume: bool,
    explicit_candidate: tuple[int, int] | None = None,
    candidate_rule: str = "point_estimate",
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("探索样本数必须为正整数")
    if not 0.0 < confidence < 1.0:
        raise ValueError("探索置信水平必须位于 (0,1)")
    if not 0.0 < target < 1.0:
        raise ValueError("目标概率必须位于 (0,1)")
    if not designs:
        raise ValueError("探索设计集不能为空")
    if candidate_rule not in SCREENING_CANDIDATE_RULES:
        raise ValueError(f"未知探索候选规则：{candidate_rule}")
    maximum_n_a = max(n_a for n_a, _ in designs)
    maximum_n_b = max(n_b for _, n_b in designs)
    config = _base_config(
        maximum_n_a,
        maximum_n_b,
        trials=trials,
        master_seed=master_seed,
        stream_id=stream_id,
    )
    artifact = run_pareto_frontier_simulation(
        config,
        output_dir
        / "screening"
        / ("pareto_max_" + _design_tag(maximum_n_a, maximum_n_b)),
        workers=workers,
        batch_size=batch_size,
        resume=resume,
    )
    stored_config, frontiers, _ = load_pareto_frontier_artifact(artifact)
    if (stored_config.n_a, stored_config.n_b) != (maximum_n_a, maximum_n_b):
        raise RuntimeError("二维静态图产物与完整整数域上限不一致")
    success_counts = integer_domain_success_counts(
        frontiers, maximum_n_a, maximum_n_b
    )
    counts_path = output_dir / "q4_screening_integer_domain_counts.npz"
    _write_integer_domain_counts(counts_path, success_counts, trials=trials)

    records = []
    for n_a, n_b in designs:
        record = _analyze_pareto_design(
            artifact, n_a, n_b, confidence=confidence
        )
        record["screening_empirically_feasible"] = record["estimate"] >= target
        record["screening_cp_lower_feasible"] = (
            record["clopper_pearson_one_sided_lower"] >= target
        )
        records.append(record)

    minimum = None
    if explicit_candidate is not None:
        candidate = select_screening_candidate(
            records, target=target, explicit_candidate=explicit_candidate
        )
    else:
        minimum = minimum_screening_feasible_design(
            success_counts,
            trials=trials,
            target=target,
            confidence=confidence,
            rule=candidate_rule,
        )
        if minimum is None:
            candidate = None
        else:
            candidate_design = minimum[0], minimum[1]
            indexed_records = {
                (int(row["n_a"]), int(row["n_b"])): row for row in records
            }
            candidate = indexed_records.get(candidate_design)
            if candidate is None:
                candidate = _analyze_pareto_design(
                    artifact,
                    candidate_design[0],
                    candidate_design[1],
                    confidence=confidence,
                )
                candidate["screening_empirically_feasible"] = True
                candidate["screening_cp_lower_feasible"] = (
                    candidate["clopper_pearson_one_sided_lower"] >= target
                )
                candidate["screening_record_role"] = "integer_domain_candidate"
                records.append(candidate)
                records.sort(
                    key=lambda row: (
                        int(row["cost_weight"]),
                        int(row["n_a"]),
                        int(row["n_b"]),
                    )
                )
    return {
        "kind": "q4_screening_results",
        "schema_version": SCHEMA_VERSION,
        "question": 4,
        "result_scope": "screening_only_not_final_probability_evidence",
        "boundary_contract": BOUNDARY_CONTRACT,
        "target_probability": target,
        "pointwise_confidence": confidence,
        "fixed_trial_count": trials,
        "master_seed": master_seed,
        "stream_id": stream_id,
        "shared_crn_across_designs": True,
        "one_static_graph_per_trial": True,
        "maximum_static_graph_design": [maximum_n_a, maximum_n_b],
        "integer_domain_bounds": {
            "n_a": [0, maximum_n_a],
            "n_b": [0, maximum_n_b],
        },
        "integer_domain_design_count": (maximum_n_a + 1) * (maximum_n_b + 1),
        "integer_domain_success_counts": str(counts_path.resolve()),
        "integer_domain_success_counts_sha256": _sha256(counts_path),
        "pareto_frontier_artifact": str(artifact.resolve()),
        "pareto_frontier_artifact_sha256": _sha256(artifact),
        "design_count": len(records),
        "records": records,
        "screening_candidate": candidate,
        "candidate_selection_rule_id": candidate_rule,
        "candidate_required_successes": None if minimum is None else minimum[3],
        "candidate_selection_rule": (
            "在完整整数域点估计不低于目标的设计中，按整数成本权重、N_A、N_B依次取最小"
            if candidate_rule == "point_estimate"
            else "在完整整数域探索单侧 Clopper-Pearson 下限不低于目标的设计中，"
            "按整数成本权重、N_A、N_B依次取最小；探索区间只用于冻结候选"
        ),
    }


def select_screening_candidate(
    records: Sequence[dict[str, Any]],
    *,
    target: float,
    explicit_candidate: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    indexed = {(int(row["n_a"]), int(row["n_b"])): row for row in records}
    if explicit_candidate is not None:
        if explicit_candidate not in indexed:
            raise ValueError("显式候选不在冻结前的探索设计集中")
        candidate = indexed[explicit_candidate]
        if float(candidate["estimate"]) < target:
            raise ValueError("显式候选的探索点估计尚未达到目标概率")
        return candidate
    feasible = [row for row in records if float(row["estimate"]) >= target]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (int(row["cost_weight"]), int(row["n_a"]), int(row["n_b"])),
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


def _validate_freeze_sources(freeze: dict[str, Any]) -> None:
    source = freeze["source_screening"]
    for path_key, hash_key in (
        ("json_path", "json_sha256"),
        ("csv_path", "csv_sha256"),
        ("pareto_artifact_path", "pareto_artifact_sha256"),
    ):
        path = Path(source[path_key])
        if not path.is_file() or _sha256(path) != source[hash_key]:
            raise ValueError("冻结后探索证据文件缺失或哈希发生变化")


def run_confirmation(
    freeze: dict[str, Any],
    output_dir: Path,
    *,
    workers: int,
    batch_size: int,
    resume: bool,
) -> list[dict[str, Any]]:
    _validate_freeze_sources(freeze)
    protocol = freeze["confirmation_protocol"]
    confidence = float(protocol["per_statement_confidence"])
    target = float(freeze["candidate_freeze"]["target_probability"])
    config = MixedSimulationConfig.from_dict(protocol["configuration"])
    if config.fingerprint != protocol["configuration_fingerprint"]:
        raise ValueError("冻结的二维确认配置指纹不一致")
    artifact = run_pareto_frontier_simulation(
        config,
        output_dir
        / "confirmation"
        / ("pareto_max_" + _design_tag(config.n_a, config.n_b)),
        workers=workers,
        batch_size=batch_size,
        resume=resume,
    )
    records = []
    for frozen in freeze["confirmation_designs"]:
        record = _analyze_pareto_design(
            artifact,
            int(frozen["n_a"]),
            int(frozen["n_b"]),
            confidence=confidence,
        )
        record["role"] = frozen["role"]
        record["proof_bound"] = frozen["proof_bound"]
        if frozen["proof_bound"] == "lower":
            record["proof_status"] = (
                "candidate_statistically_feasible"
                if record["clopper_pearson_one_sided_lower"] >= target
                else "candidate_not_confirmed"
            )
        else:
            record["proof_status"] = (
                "strictly_cheaper_design_excluded"
                if record["clopper_pearson_one_sided_upper"] < target
                else "strictly_cheaper_design_not_excluded"
            )
        records.append(record)
    return records


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


def _write_stage_summary(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "q4_summary.json"
    payload["summary_path"] = str(path.resolve())
    _write_json(path, payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    screening_json = output_dir / "q4_screening.json"
    screening_csv = output_dir / "q4_screening.csv"
    freeze_path = output_dir / "q4_confirmation_freeze.json"

    if args.stage == "confirm":
        if not freeze_path.is_file():
            raise FileNotFoundError("确认阶段要求先生成 q4_confirmation_freeze.json")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
        if freeze.get("kind") != "q4_confirmation_freeze":
            raise ValueError("文件不是 Q4 确认冻结协议")
    else:
        designs = screening_designs(
            max_n_a=args.max_n_a,
            max_n_b=args.max_n_b,
            step_n_a=args.step_n_a,
            step_n_b=args.step_n_b,
            explicit_designs=args.designs,
            maximum_designs=args.max_screening_designs,
        )
        screening = run_screening(
            designs,
            output_dir,
            trials=args.screening_trials,
            confidence=args.screening_confidence,
            target=args.target,
            master_seed=args.seed,
            stream_id=args.screening_stream_id,
            workers=args.workers,
            batch_size=args.screening_batch_size,
            resume=args.resume,
            explicit_candidate=args.candidate,
            candidate_rule=args.screening_candidate_rule,
        )
        _write_json(screening_json, screening)
        _write_csv(screening_csv, screening["records"], SCREENING_CSV_FIELDS)
        if args.stage == "screen":
            return _write_stage_summary(
                output_dir,
                {
                    "kind": "q4_stage_summary",
                    "schema_version": SCHEMA_VERSION,
                    "question": 4,
                    "result_status": "screening_complete",
                    "screening_json": str(screening_json.resolve()),
                    "screening_csv": str(screening_csv.resolve()),
                    "screening_candidate": screening["screening_candidate"],
                    "candidate_selection_rule_id": screening[
                        "candidate_selection_rule_id"
                    ],
                    "final_evidence_available": False,
                },
            )
        freeze = build_confirmation_freeze(
            screening,
            screening_json_path=screening_json,
            screening_csv_path=screening_csv,
            confirmation_trials=args.confirmation_trials,
            confirmation_stream_id=args.confirmation_stream_id,
            familywise_confidence=args.familywise_confidence,
        )
        _write_or_validate_freeze(freeze_path, freeze)
        if args.stage == "freeze":
            return _write_stage_summary(
                output_dir,
                {
                    "kind": "q4_stage_summary",
                    "schema_version": SCHEMA_VERSION,
                    "question": 4,
                    "result_status": "confirmation_protocol_frozen",
                    "freeze_path": str(freeze_path.resolve()),
                    "freeze_sha256": _sha256(freeze_path),
                    "confirmation_design_count": len(freeze["confirmation_designs"]),
                    "final_evidence_available": False,
                },
            )

    records = run_confirmation(
        freeze,
        output_dir,
        workers=args.workers,
        batch_size=args.confirmation_batch_size,
        resume=args.resume,
    )
    confirmation_json = output_dir / "q4_confirmation.json"
    confirmation_csv = output_dir / "q4_confirmation.csv"
    confirmation_payload = {
        "kind": "q4_confirmation_results",
        "schema_version": SCHEMA_VERSION,
        "question": 4,
        "freeze_path": str(freeze_path.resolve()),
        "freeze_sha256": _sha256(freeze_path),
        "records": records,
    }
    _write_json(confirmation_json, confirmation_payload)
    _write_csv(confirmation_csv, records, CONFIRMATION_CSV_FIELDS)
    decision = analyze_confirmation_records(freeze, records)
    summary = _write_stage_summary(
        output_dir,
        {
            "kind": "q4_final_summary",
            "schema_version": SCHEMA_VERSION,
            "question": 4,
            "boundary_contract": BOUNDARY_CONTRACT,
            "b_geometry_status": "exact_ball_cell_intersection_fragments",
            "a_geometry_status": "centerline_cut_approximation",
            "boundary_b_sensitivity_status": "not_run_not_claimed",
            "freeze_path": str(freeze_path.resolve()),
            "freeze_sha256": _sha256(freeze_path),
            "confirmation_json": str(confirmation_json.resolve()),
            "confirmation_json_sha256": _sha256(confirmation_json),
            "confirmation_csv": str(confirmation_csv.resolve()),
            "confirmation_csv_sha256": _sha256(confirmation_csv),
            "confirmation_records": records,
            **decision,
        },
    )
    if args.register_results:
        register_formal_results(summary, Path(summary["summary_path"]))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="问题4：D 主边界下 A/B 混合填充的两阶段最低成本统计优化"
    )
    parser.add_argument(
        "--stage", choices=("screen", "freeze", "confirm", "all"), default="screen"
    )
    parser.add_argument("--output-dir", type=Path, default=QUESTION_ROOT / "results")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument(
        "--screening-trials", type=int, default=DEFAULT_SCREENING_TRIALS
    )
    parser.add_argument(
        "--confirmation-trials", type=int, default=DEFAULT_CONFIRMATION_TRIALS
    )
    parser.add_argument("--screening-confidence", type=float, default=0.95)
    parser.add_argument(
        "--screening-candidate-rule",
        choices=SCREENING_CANDIDATE_RULES,
        default="point_estimate",
        help=(
            "探索候选规则：point_estimate 复现经验边界；cp_lower 要求探索单侧"
            " Clopper-Pearson 下限达到目标"
        ),
    )
    parser.add_argument(
        "--familywise-confidence", type=float, default=DEFAULT_FAMILYWISE_CONFIDENCE
    )
    parser.add_argument("--max-na", dest="max_n_a", type=int, default=720)
    parser.add_argument("--max-nb", dest="max_n_b", type=int, default=6000)
    parser.add_argument("--step-na", dest="step_n_a", type=int, default=120)
    parser.add_argument("--step-nb", dest="step_n_b", type=int, default=1000)
    parser.add_argument(
        "--design",
        dest="designs",
        action="append",
        type=_parse_design,
        default=None,
        help="显式探索设计 N_A,N_B；重复传入时覆盖矩形网格",
    )
    parser.add_argument(
        "--candidate",
        type=_parse_design,
        default=None,
        help="从已探索且点估计可行的设计中显式冻结候选",
    )
    parser.add_argument("--max-screening-designs", type=int, default=500)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("MATH_MODELING_SEED", "20260801")),
    )
    parser.add_argument(
        "--screening-stream-id", type=int, default=DEFAULT_SCREENING_STREAM_ID
    )
    parser.add_argument(
        "--confirmation-stream-id", type=int, default=DEFAULT_CONFIRMATION_STREAM_ID
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--screening-batch-size", type=int, default=20)
    parser.add_argument("--confirmation-batch-size", type=int, default=100)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--register-results",
        action="store_true",
        help="仅在正式独立确认形成可发布结论后写入结果注册表",
    )
    args = parser.parse_args(argv)
    if args.register_results and args.stage in {"screen", "freeze"}:
        parser.error("--register-results 只能与 --stage confirm 或 all 同用")
    return args


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "summary": result["summary_path"],
                "result_status": result["result_status"],
                "reported_design": result.get("reported_design"),
                "final_evidence_available": result.get(
                    "final_evidence_available", True
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
