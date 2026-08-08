# AI 工具：OpenAI Codex；模型/版本：GPT-5 系列；开发机构：OpenAI。
# 版本发布日期：2025-08-07（GPT-5 系列公开快照日期）；本程序由参赛队逐行复核并对结果负责。
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUESTION_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = PROJECT_ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from microstructure_sim import (  # noqa: E402
    BoundaryMode,
    PROJECT_SEED,
    Q2_COUNTS,
    SimulationConfig,
    boundary_spec,
    load_threshold_artifact,
    nominal_volume_percent,
    probability_at_prefix,
    run_simulation,
)
from result_registry import export_latex, register_result  # noqa: E402


FORMAL_TRIAL_COUNT = 20_000
FORMAL_STREAM_ID = 2
Q2_LATEX_MACROS = {
    354: "QTwoProbabilityPointFive",
    424: "QTwoProbabilityPointSix",
    495: "QTwoProbabilityPointSeven",
    707: "QTwoProbabilityOne",
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def analyze_threshold_artifact(
    artifact_path: Path | str,
    counts: Sequence[int] = Q2_COUNTS,
    confidence: float = 0.95,
) -> dict:
    config, samples, _ = load_threshold_artifact(artifact_path)
    evaluated_counts = [int(value) for value in counts]
    if not evaluated_counts or min(evaluated_counts) < 0:
        raise ValueError("Q2 粒子数必须为非负整数")
    if max(evaluated_counts) > config.max_count:
        raise ValueError("阈值样本流 max_count 小于待评估的 Q2 粒子数")
    records = []
    for count in evaluated_counts:
        probability = probability_at_prefix(
            samples, count, config.max_count, confidence
        )
        records.append(
            {
                "nominal_target_percent": (
                    {354: 0.50, 424: 0.60, 495: 0.70, 707: 1.00}.get(count)
                ),
                "particle_count": count,
                "actual_discrete_volume_percent": nominal_volume_percent(
                    count, config
                ),
                "probability": probability,
            }
        )
    spec = boundary_spec(config.boundary_mode)
    return {
        "result_scope": "configured_monte_carlo_not_result_registry",
        "question": 2,
        "threshold_artifact": str(Path(artifact_path).resolve()),
        "configuration_fingerprint": config.fingerprint,
        "configuration": config.to_dict(),
        "boundary_semantics": {
            "mode": spec.mode.value,
            "role": spec.role,
            "description": spec.description,
            "implementation_limitations": list(spec.implementation_limitations),
            "global_minimum_image_adjacency": False,
            "flat_cylinder_narrow_phase": True,
        },
        "fixed_trial_count": config.trial_count,
        "censored_trials": int((samples == config.max_count + 1).sum()),
        "probability_records": records,
    }


def register_formal_results(result: dict, summary_path: Path) -> None:
    configuration = result["configuration"]
    counts = {
        int(record["particle_count"])
        for record in result["probability_records"]
    }
    if (
        configuration["boundary_mode"] != "D"
        or int(result["fixed_trial_count"]) < FORMAL_TRIAL_COUNT
        or int(configuration["stream_id"]) != FORMAL_STREAM_ID
        or not set(Q2_LATEX_MACROS).issubset(counts)
    ):
        raise ValueError("仅允许将 D 边界正式固定样本结果写入结果注册表")

    source_artifact = summary_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    validation = (
        "D 边界固定 20000 个独立试验；共同随机前缀保证粒子数单调；"
        "双侧 95% Wilson 与 Clopper-Pearson 区间一致"
    )
    for record in result["probability_records"]:
        count = int(record["particle_count"])
        macro = Q2_LATEX_MACROS.get(count)
        if macro is None:
            continue
        probability = record["probability"]
        estimate = float(probability["estimate"])
        register_result(
            f"q2_probability_n{count}",
            question=2,
            value=estimate,
            formatted=f"{100.0 * estimate:.3f}%",
            source_script="问题/问题2/src/solve.py",
            source_artifact=source_artifact,
            validation=validation,
            latex_macro=macro,
        )
    export_latex()


# 模块1：固定一次最大前缀模拟，四个填充量共享同一批首次导通样本。
def run(args: argparse.Namespace) -> dict:
    config = SimulationConfig(
        max_count=args.max_count,
        trial_count=args.trials,
        boundary_mode=args.boundary_mode,
        master_seed=args.seed,
        stream_id=args.stream_id,
        cell_size_nm=args.cell_size_nm,
    )
    if max(args.counts) > config.max_count:
        raise ValueError("--max-count 必须覆盖所有 --counts")
    started = time.perf_counter()
    artifact_path = run_simulation(
        config,
        args.output_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    result = analyze_threshold_artifact(
        artifact_path, counts=args.counts, confidence=args.confidence
    )
    if args.register_results:
        result["result_scope"] = "formal_fixed_monte_carlo"
    result["wall_runtime_seconds"] = time.perf_counter() - started
    summary_path = Path(args.output_dir) / "q2_summary.json"
    _write_json(summary_path, result)
    result["summary_path"] = str(summary_path.resolve())
    if args.register_results:
        register_formal_results(result, summary_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="问题2：显式边界切段下仅含介质 A 的导通概率"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="显式指定输出目录；正式路径由项目级流水线提供",
    )
    parser.add_argument("--boundary-mode", choices=[mode.value for mode in BoundaryMode], default="D")
    parser.add_argument("--max-count", type=int, default=720)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("MATH_MODELING_SEED", str(PROJECT_SEED))),
    )
    parser.add_argument("--stream-id", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cell-size-nm", type=float, default=625.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--counts", type=int, nargs="+", default=list(Q2_COUNTS))
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--register-results",
        action="store_true",
        help="仅在正式固定样本运行后写入结果注册表并导出 LaTeX 宏",
    )
    return parser.parse_args(argv)


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "threshold_artifact": result["threshold_artifact"],
                "summary": result["summary_path"],
                "trials": result["fixed_trial_count"],
                "runtime_seconds": result["wall_runtime_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
