from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike


def _discover_project_root(script_path: Path) -> Path:
    # 优先使用环境变量，否则向上查找项目标志目录，避免写死机器绝对路径。
    configured = os.environ.get("MCM_PROJECT_ROOT")
    candidates = [Path(configured).expanduser()] if configured else script_path.resolve().parents
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "公共代码").is_dir() and (root / "问题").is_dir():
            return root
    raise RuntimeError("无法定位项目根目录；请设置 MCM_PROJECT_ROOT")


PROJECT_ROOT = _discover_project_root(Path(__file__))
QUESTION_ROOT = PROJECT_ROOT / "问题" / "问题3"
COMMON_DIR = PROJECT_ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from microstructure_sim import (
    BoundaryMode,
    SimulationConfig,
    boundary_spec,
    load_threshold_artifact,
    nominal_volume_percent,
    probability_at_prefix,
    run_simulation,
    smallest_empirical_threshold,
)
from result_registry import export_latex, register_result


FREEZE_SCHEMA_VERSION = 1
DEFAULT_CANDIDATE_RADIUS = 4
DEFAULT_CONFIRMATION_TRIALS = 50_000
DEFAULT_CONFIRMATION_STREAM_ID = 3
DEFAULT_FAMILYWISE_CONFIDENCE = 0.95
REPORTED_VOLUME_PERCENT_DECIMALS = 2


def _round_half_up(value: float, decimals: int) -> float:
    quantum = Decimal("1").scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_or_validate_freeze(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8-sig"))
        if stored != payload:
            raise ValueError(
                "已有 Q3 冻结文件与本次协议不一致；请改用新的输出目录"
            )
        return
    _write_json(path, payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _candidate_counts(
    empirical_threshold: int,
    max_count: int,
    radius: int,
    explicit_candidates: Sequence[int] | None,
) -> list[int]:
    if radius < 0:
        raise ValueError("候选半径必须为非负整数")
    if explicit_candidates is None:
        return list(
            range(
                max(0, empirical_threshold - radius),
                min(max_count, empirical_threshold + radius) + 1,
            )
        )

    candidates = [int(value) for value in explicit_candidates]
    if not candidates:
        raise ValueError("显式候选集不能为空")
    if len(set(candidates)) != len(candidates):
        raise ValueError("显式候选集不能包含重复整数")
    if min(candidates) < 0 or max(candidates) > max_count:
        raise ValueError("显式候选必须位于 [0,探索 max_count]")
    return sorted(candidates)


# 关键：冻结候选粒子数、样本量和随机流，避免事后改参。
def build_confirmation_freeze(
    exploration_artifact: Path | str,
    *,
    target: float = 0.90,
    candidate_radius: int = DEFAULT_CANDIDATE_RADIUS,
    candidates: Sequence[int] | None = None,
    confirmation_trials: int = DEFAULT_CONFIRMATION_TRIALS,
    confirmation_stream_id: int = DEFAULT_CONFIRMATION_STREAM_ID,
    familywise_confidence: float = DEFAULT_FAMILYWISE_CONFIDENCE,
) -> tuple[dict[str, Any], SimulationConfig]:
    path = Path(exploration_artifact)
    exploration_config, samples, _ = load_threshold_artifact(path)
    if exploration_config.boundary_mode is not BoundaryMode.D:
        raise ValueError("Q3 主确认只接受 D 边界的 Q2 探索阈值流")
    if confirmation_trials < 1:
        raise ValueError("确认样本数必须为正整数")
    if confirmation_stream_id == exploration_config.stream_id:
        raise ValueError("确认 stream_id 必须与 Q2 探索 stream_id 不同")
    if confirmation_stream_id < 0:
        raise ValueError("确认 stream_id 必须非负")
    if not 0.0 < familywise_confidence < 1.0:
        raise ValueError("联合置信水平必须位于 (0,1)")

    empirical = smallest_empirical_threshold(
        samples, exploration_config.max_count, target
    )
    if empirical is None:
        raise ValueError(
            "Q2 探索流在 max_count 内未定位经验 90% 阈值；须扩大探索前缀"
        )
    frozen_candidates = _candidate_counts(
        empirical,
        exploration_config.max_count,
        candidate_radius,
        candidates,
    )
    candidate_count = len(frozen_candidates)
    statement_count = 2 * candidate_count
    familywise_alpha = 1.0 - familywise_confidence
    per_bound_alpha = familywise_alpha / statement_count
    per_bound_confidence = 1.0 - per_bound_alpha

    confirmation_config = replace(
        exploration_config,
        max_count=max(1, max(frozen_candidates)),
        trial_count=int(confirmation_trials),
        stream_id=int(confirmation_stream_id),
    )
    freeze = {
        "kind": "q3_confirmation_freeze",
        "schema_version": FREEZE_SCHEMA_VERSION,
        "question": 3,
        "source_exploration": {
            "threshold_artifact": str(path.resolve()),
            "sha256": _sha256(path),
            "configuration_fingerprint": exploration_config.fingerprint,
            "configuration": exploration_config.to_dict(),
            "fixed_trial_count": exploration_config.trial_count,
            "censored_trials": int(
                np.count_nonzero(samples == exploration_config.max_count + 1)
            ),
        },
        "candidate_freeze": {
            "selection_method": (
                "explicit_cli_candidates"
                if candidates is not None
                else "empirical_threshold_plus_minus_fixed_radius"
            ),
            "target_probability": float(target),
            "exploration_empirical_threshold": int(empirical),
            "candidate_radius": None if candidates is not None else int(candidate_radius),
            "candidates": frozen_candidates,
            "candidate_count": candidate_count,
        },
        "confirmation_protocol": {
            "boundary_mode": BoundaryMode.D.value,
            "fixed_trial_count": int(confirmation_trials),
            "shared_prefix_stream_across_candidates": True,
            "master_seed": confirmation_config.master_seed,
            "exploration_stream_id": exploration_config.stream_id,
            "confirmation_stream_id": confirmation_config.stream_id,
            "stream_ids_distinct": True,
            "familywise_confidence": float(familywise_confidence),
            "familywise_alpha": familywise_alpha,
            "bonferroni_statement_count": statement_count,
            "statements_per_candidate": ["one_sided_lower", "one_sided_upper"],
            "per_bound_alpha": per_bound_alpha,
            "per_bound_confidence": per_bound_confidence,
            "authoritative_interval": "clopper_pearson_one_sided",
            "diagnostic_interval": "wilson_one_sided",
        },
        "confirmation_configuration": confirmation_config.to_dict(),
        "confirmation_configuration_fingerprint": confirmation_config.fingerprint,
    }
    return freeze, confirmation_config


# 关键：用联合精确置信界判定满足目标的最小整数设计。
def analyze_confirmation_samples(
    first_connection_samples: ArrayLike,
    *,
    candidates: Sequence[int],
    max_count: int,
    target: float = 0.90,
    familywise_confidence: float = DEFAULT_FAMILYWISE_CONFIDENCE,
) -> dict[str, Any]:
    samples = np.asarray(first_connection_samples, dtype=np.int64)
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("确认首次导通样本必须是一维非空数组")
    frozen_candidates = [int(value) for value in candidates]
    if frozen_candidates != sorted(set(frozen_candidates)):
        raise ValueError("冻结候选必须严格递增且无重复")
    if not frozen_candidates or frozen_candidates[0] < 0:
        raise ValueError("冻结候选不能为空或含负整数")
    if frozen_candidates[-1] > max_count:
        raise ValueError("确认 max_count 未覆盖冻结候选")
    if not 0.0 < target < 1.0:
        raise ValueError("target 必须位于 (0,1)")
    if not 0.0 < familywise_confidence < 1.0:
        raise ValueError("联合置信水平必须位于 (0,1)")

    statement_count = 2 * len(frozen_candidates)
    per_bound_alpha = (1.0 - familywise_confidence) / statement_count
    per_bound_confidence = 1.0 - per_bound_alpha
    records: list[dict[str, Any]] = []
    for count in frozen_candidates:
        probability = probability_at_prefix(
            samples, count, max_count, per_bound_confidence
        )
        wilson_lower, wilson_upper = probability["wilson_one_sided_bounds"]
        exact_lower, exact_upper = probability[
            "clopper_pearson_one_sided_bounds"
        ]
        if exact_lower >= target:
            classification = "statistically_feasible"
        elif exact_upper < target:
            classification = "statistically_infeasible"
        else:
            classification = "unresolved"
        records.append(
            {
                "particle_count": count,
                "successes": probability["successes"],
                "trials": probability["trials"],
                "estimate": probability["estimate"],
                "per_bound_confidence": per_bound_confidence,
                "wilson_one_sided_bounds": {
                    "lower": wilson_lower,
                    "upper": wilson_upper,
                },
                "clopper_pearson_one_sided_bounds": {
                    "lower": exact_lower,
                    "upper": exact_upper,
                },
                "classification_by_bonferroni_cp": classification,
            }
        )

    by_count = {record["particle_count"]: record for record in records}
    feasible = [
        record["particle_count"]
        for record in records
        if record["classification_by_bonferroni_cp"] == "statistically_feasible"
    ]
    infeasible = [
        record["particle_count"]
        for record in records
        if record["classification_by_bonferroni_cp"] == "statistically_infeasible"
    ]
    unresolved = [
        record["particle_count"]
        for record in records
        if record["classification_by_bonferroni_cp"] == "unresolved"
    ]
    confirmed: list[int] = []
    for count in feasible:
        previous = by_count.get(count - 1)
        if previous is None:
            continue
        previous_upper = previous["clopper_pearson_one_sided_bounds"]["upper"]
        if previous_upper < target:
            confirmed.append(count)

    confirmed_minimum = min(confirmed) if confirmed else None
    lowest_feasible = min(feasible) if feasible else None
    minimum_integer_bracket: list[int] | None = None
    volume_percent_bracket: list[float] | None = None
    rounded_volume_percent_bracket: list[float] | None = None
    reported_volume_percent: float | None = None
    if infeasible and feasible:
        lower_count = max(infeasible) + 1
        upper_count = min(feasible)
        if lower_count <= upper_count:
            minimum_integer_bracket = [lower_count, upper_count]
            volume_percent_bracket = [
                nominal_volume_percent(lower_count),
                nominal_volume_percent(upper_count),
            ]
            rounded_volume_percent_bracket = [
                _round_half_up(value, REPORTED_VOLUME_PERCENT_DECIMALS)
                for value in volume_percent_bracket
            ]
            if rounded_volume_percent_bracket[0] == rounded_volume_percent_bracket[1]:
                reported_volume_percent = rounded_volume_percent_bracket[0]

    decision = {
        "result_status": (
            "confirmed_minimum"
            if confirmed_minimum is not None
            else "minimum_not_confirmed"
        ),
        "confirmed_minimum_integer": confirmed_minimum,
        "confirmed_predecessor_integer": (
            None if confirmed_minimum is None else confirmed_minimum - 1
        ),
        "lowest_statistically_feasible_integer": lowest_feasible,
        "lowest_statistically_feasible_scope": "frozen_candidates_only",
        "statistically_infeasible_candidates": infeasible,
        "unresolved_candidates": unresolved,
        "unresolved_integer_interval": (
            None if not unresolved else [min(unresolved), max(unresolved)]
        ),
        "minimum_integer_bracket": minimum_integer_bracket,
        "minimum_volume_fraction_percent_bracket": volume_percent_bracket,
        "requested_volume_percent_decimals": REPORTED_VOLUME_PERCENT_DECIMALS,
        "rounded_volume_fraction_percent_bracket": rounded_volume_percent_bracket,
        "reported_precision_confirmed": reported_volume_percent is not None,
        "reported_volume_fraction_percent": reported_volume_percent,
        "reported_volume_fraction_formatted": (
            None
            if reported_volume_percent is None
            else f"{reported_volume_percent:.{REPORTED_VOLUME_PERCENT_DECIMALS}f}%"
        ),
        "decision_rule": (
            "仅当冻结候选 N 的 Bonferroni-CP 单侧下限不低于目标，且冻结的 "
            "N-1 单侧上限低于目标，才确认最小整数"
        ),
        "reported_precision_rule": (
            "若联合置信下的最小整数区间换算为体积分数后在题定小数位上相同，"
            "则确认题目要求精度下的填充量；这不等同于确认唯一最小整数"
        ),
    }
    return {
        "target_probability": float(target),
        "familywise_confidence": float(familywise_confidence),
        "bonferroni_statement_count": statement_count,
        "per_bound_alpha": per_bound_alpha,
        "per_bound_confidence": per_bound_confidence,
        "candidate_records": records,
        "decision": decision,
    }


def analyze_confirmation_artifact(
    confirmation_artifact: Path | str,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    path = Path(confirmation_artifact)
    config, samples, _ = load_threshold_artifact(path)
    expected_fingerprint = freeze["confirmation_configuration_fingerprint"]
    if config.fingerprint != expected_fingerprint:
        raise ValueError("确认阈值流与冻结配置指纹不一致")
    protocol = freeze["confirmation_protocol"]
    if config.boundary_mode is not BoundaryMode.D:
        raise ValueError("确认阈值流不是 D 主边界")
    if config.stream_id != protocol["confirmation_stream_id"]:
        raise ValueError("确认阈值流 stream_id 与冻结协议不一致")
    if config.trial_count != protocol["fixed_trial_count"]:
        raise ValueError("确认阈值流样本数与冻结协议不一致")

    analysis = analyze_confirmation_samples(
        samples,
        candidates=freeze["candidate_freeze"]["candidates"],
        max_count=config.max_count,
        target=freeze["candidate_freeze"]["target_probability"],
        familywise_confidence=protocol["familywise_confidence"],
    )
    for record in analysis["candidate_records"]:
        volume = nominal_volume_percent(record["particle_count"], config)
        record["nominal_volume_percent_unrounded"] = volume
        record["nominal_volume_percent_report_2dp"] = round(volume, 2)

    selected = analysis["decision"]["confirmed_minimum_integer"]
    if selected is None:
        selected = analysis["decision"]["lowest_statistically_feasible_integer"]
    selected_volume = None if selected is None else nominal_volume_percent(selected, config)
    spec = boundary_spec(config.boundary_mode)
    return {
        "result_scope": "independent_fixed_confirmation",
        "question": 3,
        "source_threshold_artifact": freeze["source_exploration"][
            "threshold_artifact"
        ],
        "source_sha256": freeze["source_exploration"]["sha256"],
        "exploration_configuration_fingerprint": freeze["source_exploration"][
            "configuration_fingerprint"
        ],
        "empirical_integer_threshold": freeze["candidate_freeze"][
            "exploration_empirical_threshold"
        ],
        "frozen_candidates": freeze["candidate_freeze"]["candidates"],
        "confirmation_threshold_artifact": str(path.resolve()),
        "confirmation_sha256": _sha256(path),
        "confirmation_configuration_fingerprint": config.fingerprint,
        "confirmation_configuration": config.to_dict(),
        "boundary_semantics": {
            "mode": spec.mode.value,
            "role": spec.role,
            "description": spec.description,
            "implementation_limitations": list(spec.implementation_limitations),
        },
        "fixed_trial_count": config.trial_count,
        "censored_trials": int(np.count_nonzero(samples == config.max_count + 1)),
        **analysis,
        "result_status": analysis["decision"]["result_status"],
        "confirmed_minimum_integer": analysis["decision"][
            "confirmed_minimum_integer"
        ],
        "lowest_statistically_feasible_integer": analysis["decision"][
            "lowest_statistically_feasible_integer"
        ],
        "unresolved_integer_interval": analysis["decision"][
            "unresolved_integer_interval"
        ],
        "reported_integer": selected,
        "nominal_volume_percent_unrounded": selected_volume,
        "nominal_volume_percent_report_2dp": (
            None if selected_volume is None else round(selected_volume, 2)
        ),
    }


def register_formal_results(result: dict[str, Any], summary_path: Path) -> None:
    configuration = result["confirmation_configuration"]
    decision = result["decision"]
    if (
        configuration["boundary_mode"] != "D"
        or int(result["fixed_trial_count"]) < DEFAULT_CONFIRMATION_TRIALS
        or int(configuration["stream_id"]) != DEFAULT_CONFIRMATION_STREAM_ID
        or not bool(decision["reported_precision_confirmed"])
    ):
        raise ValueError("问题3尚未形成可登记的正式精度结论")

    source_artifact = summary_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    bracket = [int(value) for value in decision["minimum_integer_bracket"]]
    safe_count = int(result["reported_integer"])
    safe_record = next(
        record
        for record in result["candidate_records"]
        if int(record["particle_count"]) == safe_count
    )
    validation = (
        "独立 50000 次确认；冻结 9 个候选后对 18 个单侧陈述作 Bonferroni 校正；"
        "Clopper-Pearson 联合 95% 最小整数区间换算后在 0.01% 精度上相同"
    )
    register_result(
        "q3_reported_minimum_volume_percent",
        question=3,
        value=float(decision["reported_volume_fraction_percent"]),
        formatted=decision["reported_volume_fraction_formatted"],
        source_script="问题/问题3/src/solve.py",
        source_artifact=source_artifact,
        validation=validation,
        latex_macro="QThreeReportedVolume",
    )
    register_result(
        "q3_minimum_integer_bracket",
        question=3,
        value=bracket,
        formatted=f"{bracket[0]}--{bracket[1]}",
        unit="根",
        source_script="问题/问题3/src/solve.py",
        source_artifact=source_artifact,
        validation=validation,
        latex_macro="QThreeMinimumBracket",
    )
    register_result(
        "q3_conservative_particle_count",
        question=3,
        value=safe_count,
        formatted=str(safe_count),
        unit="根",
        source_script="问题/问题3/src/solve.py",
        source_artifact=source_artifact,
        validation=validation,
        latex_macro="QThreeSafeCount",
    )
    register_result(
        "q3_conservative_probability",
        question=3,
        value=float(safe_record["estimate"]),
        formatted=f"{100.0 * float(safe_record['estimate']):.3f}%",
        source_script="问题/问题3/src/solve.py",
        source_artifact=source_artifact,
        validation=validation,
        latex_macro="QThreeSafeProbability",
    )
    register_result(
        "q3_conservative_cp_lower",
        question=3,
        value=float(safe_record["clopper_pearson_one_sided_bounds"]["lower"]),
        formatted=(
            f"{100.0 * float(safe_record['clopper_pearson_one_sided_bounds']['lower']):.4f}%"
        ),
        source_script="问题/问题3/src/solve.py",
        source_artifact=source_artifact,
        validation=validation,
        latex_macro="QThreeSafeLower",
    )
    register_result(
        "q3_unique_minimum_integer_confirmed",
        question=3,
        value=bool(decision["confirmed_minimum_integer"] is not None),
        formatted="否" if decision["confirmed_minimum_integer"] is None else "是",
        source_script="问题/问题3/src/solve.py",
        source_artifact=source_artifact,
        validation="有限样本下仅确认题设两位百分数精度，不宣称唯一最小整数",
    )
    export_latex()


def run(args: argparse.Namespace) -> dict[str, Any]:
    freeze, confirmation_config = build_confirmation_freeze(
        args.threshold_artifact,
        target=args.target,
        candidate_radius=args.candidate_radius,
        candidates=args.candidates,
        confirmation_trials=args.confirmation_trials,
        confirmation_stream_id=args.confirmation_stream_id,
        familywise_confidence=args.familywise_confidence,
    )
    output_dir = Path(args.output_dir)
    freeze_path = output_dir / "q3_confirmation_freeze.json"
    _write_or_validate_freeze(freeze_path, freeze)

    started = time.perf_counter()
    confirmation_artifact = run_simulation(
        confirmation_config,
        output_dir / "confirmation",
        workers=args.workers,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    result = analyze_confirmation_artifact(confirmation_artifact, freeze)
    result["freeze_path"] = str(freeze_path.resolve())
    result["freeze_sha256"] = _sha256(freeze_path)
    result["wall_runtime_seconds"] = time.perf_counter() - started
    summary_path = output_dir / "q3_summary.json"
    result["summary_path"] = str(summary_path.resolve())
    _write_json(summary_path, result)
    if args.register_results:
        register_formal_results(result, summary_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="问题3：由 Q2 探索流冻结候选，再用独立固定流确认 90% 最小整数"
    )
    parser.add_argument(
        "--threshold-artifact",
        type=Path,
        default=(
            PROJECT_ROOT
            / "问题"
            / "问题2"
            / "results"
            / "D_primary_n20000"
            / "threshold_samples.json"
        ),
        help="Q2 正式 D 边界探索阈值流",
    )
    parser.add_argument("--output-dir", type=Path, default=QUESTION_ROOT / "results")
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument(
        "--familywise-confidence", type=float, default=DEFAULT_FAMILYWISE_CONFIDENCE
    )
    parser.add_argument(
        "--candidate-radius", type=int, default=DEFAULT_CANDIDATE_RADIUS
    )
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=None,
        help="显式覆盖默认 t±radius 候选集",
    )
    parser.add_argument(
        "--confirmation-trials", type=int, default=DEFAULT_CONFIRMATION_TRIALS
    )
    parser.add_argument(
        "--confirmation-stream-id",
        type=int,
        default=DEFAULT_CONFIRMATION_STREAM_ID,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--register-results",
        action="store_true",
        help="仅在正式独立确认完成后写入结果注册表并导出 LaTeX 宏",
    )
    return parser.parse_args(argv)


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "freeze": result["freeze_path"],
                "confirmation_threshold_artifact": result[
                    "confirmation_threshold_artifact"
                ],
                "summary": result["summary_path"],
                "result_status": result["decision"]["result_status"],
                "confirmed_minimum_integer": result["decision"][
                    "confirmed_minimum_integer"
                ],
                "lowest_statistically_feasible_integer": result["decision"][
                    "lowest_statistically_feasible_integer"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
