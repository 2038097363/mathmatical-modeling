from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUESTION_ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = PROJECT_ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from result_registry import export_latex, read_registry, write_registry


A_COST_WEIGHT = 567
B_COST_WEIGHT = 64
COST_SCALE_YUAN = math.pi / 120_000.0
A_VOLUME_UM3 = 0.0045 * math.pi
B_VOLUME_UM3 = 0.032 * math.pi / 3.0
BOX_VOLUME_UM3 = 1000.0
TARGET = 0.90
CONFIDENCE = 0.95
BRANCH_FAMILY_SIZE = 619

DEFAULT_RESULT_ROOT = QUESTION_ROOT / "results" / "D_screen2000_confirm50000"
DEFAULT_COUNTS = DEFAULT_RESULT_ROOT / "q4_confirmation_integer_domain_counts.npz"
DEFAULT_Q3 = (
    PROJECT_ROOT
    / "问题"
    / "问题3"
    / "results"
    / "D_confirmation_n50000"
    / "q3_summary.json"
)
DEFAULT_ARTIFACT = (
    DEFAULT_RESULT_ROOT
    / "confirmation"
    / "pareto_max_A000619_B005483"
    / "mixed_pareto_frontier_samples.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV 记录不能为空")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def cost_weight(n_a: int, n_b: int) -> int:
    if n_a < 1 or n_b < 1:
        raise ValueError("问题四要求两类介质数量均为正整数")
    return A_COST_WEIGHT * n_a + B_COST_WEIGHT * n_b


def design_record(
    counts: np.ndarray,
    trials: int,
    n_a: int,
    n_b: int,
    *,
    family_size: int = 1,
) -> dict[str, Any]:
    successes = int(counts[n_a, n_b])
    weight = cost_weight(n_a, n_b)
    alpha = (1.0 - CONFIDENCE) / family_size
    nominal_alpha = 1.0 - CONFIDENCE
    a_volume = n_a * A_VOLUME_UM3
    b_volume = n_b * B_VOLUME_UM3
    return {
        "n_a": n_a,
        "n_b": n_b,
        "successes": successes,
        "trials": trials,
        "estimate": successes / trials,
        "cost_weight": weight,
        "cost_yuan": weight * COST_SCALE_YUAN,
        "a_cost_yuan": A_COST_WEIGHT * n_a * COST_SCALE_YUAN,
        "b_cost_yuan": B_COST_WEIGHT * n_b * COST_SCALE_YUAN,
        "a_volume_um3": a_volume,
        "b_volume_um3": b_volume,
        "total_volume_um3": a_volume + b_volume,
        "a_volume_percent": 100.0 * a_volume / BOX_VOLUME_UM3,
        "b_volume_percent": 100.0 * b_volume / BOX_VOLUME_UM3,
        "total_volume_percent": 100.0 * (a_volume + b_volume) / BOX_VOLUME_UM3,
        "cp_one_sided_95_lower": float(
            beta.ppf(nominal_alpha, successes, trials - successes + 1)
        ),
        "cp_one_sided_family_lower": float(
            beta.ppf(alpha, successes, trials - successes + 1)
        ),
        "family_size": family_size,
    }


def empirical_frontier(counts: np.ndarray, trials: int) -> list[dict[str, Any]]:
    required = math.ceil(TARGET * trials)
    records: list[dict[str, Any]] = []
    for n_a in range(1, counts.shape[0]):
        offset = int(np.searchsorted(counts[n_a, 1:], required, side="left"))
        n_b = offset + 1
        if n_b >= counts.shape[1]:
            continue
        records.append(design_record(counts, trials, n_a, n_b))
    records.sort(key=lambda row: (row["cost_weight"], row["n_a"], row["n_b"]))
    return records


def q3_certificate(q3: dict[str, Any]) -> dict[str, Any]:
    records = q3.get("candidate_records")
    if not isinstance(records, list):
        raise ValueError("问题三摘要缺少候选记录")
    matches = [
        row
        for row in records
        if isinstance(row, dict) and int(row.get("particle_count", -1)) == 616
    ]
    if len(matches) != 1:
        raise ValueError("问题三摘要必须恰含一条 616 根记录")
    row = matches[0]
    if row.get("classification_by_bonferroni_cp") != "statistically_feasible":
        raise ValueError("问题三尚未确认 616 根 A 的统计可行性")
    return {
        "particle_count": 616,
        "successes": int(row["successes"]),
        "trials": int(row["trials"]),
        "estimate": float(row["estimate"]),
        "bonferroni_statement_count": int(q3["bonferroni_statement_count"]),
        "clopper_pearson_one_sided_lower": float(
            row["clopper_pearson_one_sided_bounds"]["lower"]
        ),
        "logic": "固定 616 根 A 已联合置信可行；加入 1 个独立 B 只增加节点和边，故 (616,1) 的真实导通概率不低于全 A 设计。",
    }


def strictly_cheaper_positive_count(weight: int) -> int:
    total = 0
    for n_a in range(1, (weight - 1) // A_COST_WEIGHT + 1):
        maximum_b = (weight - 1 - A_COST_WEIGHT * n_a) // B_COST_WEIGHT
        total += max(0, maximum_b)
    return total


def update_registry(summary_path: Path, summary: dict[str, Any]) -> None:
    registry = read_registry()
    for key in [key for key in registry["results"] if key.startswith("q4_")]:
        del registry["results"][key]

    recommendation = summary["conservative_recommendation"]
    empirical = summary["empirical_minimum"]
    validation = (
        "正整数域完整成本枚举；50000 次固定共同随机数矩阵；"
        "B=1 分支 619 项 Bonferroni-CP 下界；问题三独立确认的单调性证书"
    )
    source_artifact = summary_path.relative_to(PROJECT_ROOT).as_posix()
    source_script = "问题/问题4/src/finalize_positive_domain.py"
    entries = {
        "q4_reported_n_a": (recommendation["n_a"], "616", "个", "QFourNA"),
        "q4_reported_n_b": (recommendation["n_b"], "1", "个", "QFourNB"),
        "q4_cost_weight": (
            recommendation["cost_weight"],
            str(recommendation["cost_weight"]),
            "",
            "QFourCostWeight",
        ),
        "q4_reported_cost_yuan": (
            recommendation["cost_yuan"],
            f"{recommendation['cost_yuan']:.4f}",
            "元",
            "QFourCost",
        ),
        "q4_candidate_probability": (
            recommendation["estimate"],
            f"{100.0 * recommendation['estimate']:.3f}%",
            "",
            "QFourProbability",
        ),
        "q4_candidate_cp_lower": (
            recommendation["cp_one_sided_family_lower"],
            f"{100.0 * recommendation['cp_one_sided_family_lower']:.3f}%",
            "",
            "QFourLower",
        ),
        "q4_empirical_n_a": (empirical["n_a"], "612", "个", "QFourEmpiricalNA"),
        "q4_empirical_n_b": (empirical["n_b"], "12", "个", "QFourEmpiricalNB"),
        "q4_empirical_cost_yuan": (
            empirical["cost_yuan"],
            f"{empirical['cost_yuan']:.4f}",
            "元",
            "QFourEmpiricalCost",
        ),
    }
    timestamp = __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")
    for key, (value, formatted, unit, macro) in entries.items():
        registry["results"][key] = {
            "question": 4,
            "value": value,
            "formatted": formatted,
            "unit": unit,
            "source_script": source_script,
            "source_artifact": source_artifact,
            "validation": validation,
            "latex_macro": macro,
            "updated_at": timestamp,
        }
    write_registry(registry)
    export_latex()


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = args.counts.expanduser().resolve()
    q3_path = args.q3_summary.expanduser().resolve()
    artifact_path = args.confirmation_artifact.expanduser().resolve()
    for path in (counts_path, q3_path, artifact_path):
        path.relative_to(PROJECT_ROOT)
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(counts_path) as payload:
        counts = np.asarray(payload["success_counts"], dtype=np.int32)
        trials = int(payload["trials"])
    if counts.shape != (620, 5484) or trials != 50_000:
        raise ValueError("正式成功次数矩阵必须为 620×5484 且含 50000 次试验")
    if np.any(np.diff(counts, axis=0) < 0) or np.any(np.diff(counts, axis=1) < 0):
        raise ValueError("成功次数矩阵违反粒子数单调性")

    frontier = empirical_frontier(counts, trials)
    empirical = frontier[0]
    if (empirical["n_a"], empirical["n_b"]) != (612, 12):
        raise ValueError("正整数域经验最低方案与冻结审计不一致")

    branch = [
        design_record(counts, trials, n_a, 1, family_size=BRANCH_FAMILY_SIZE)
        for n_a in range(1, counts.shape[0])
    ]
    feasible_branch = [
        row for row in branch if row["cp_one_sided_family_lower"] >= TARGET
    ]
    recommendation = min(feasible_branch, key=lambda row: row["cost_weight"])
    if (recommendation["n_a"], recommendation["n_b"]) != (616, 1):
        raise ValueError("B=1 联合置信推荐与冻结审计不一致")

    q3 = read_json(q3_path)
    q3_proof = q3_certificate(q3)
    artifact = read_json(artifact_path)
    artifact_config = artifact.get("configuration")
    if not isinstance(artifact_config, dict) or artifact_config.get("boundary_mode") != "D":
        raise ValueError("正式混合介质样本未采用 D 边界合同")

    frontier_csv = args.output_dir / "q4_positive_empirical_frontier.csv"
    branch_csv = args.output_dir / "q4_positive_b1_branch.csv"
    atomic_csv(frontier_csv, frontier)
    atomic_csv(branch_csv, branch)

    nearby_designs = [
        design_record(counts, trials, n_a, n_b, family_size=BRANCH_FAMILY_SIZE if n_b == 1 else 1)
        for n_a, n_b in (
            (611, 10),
            (612, 2),
            (612, 11),
            (612, 12),
            (613, 1),
            (614, 1),
            (615, 1),
            (616, 1),
        )
    ]
    summary = {
        "schema_version": "2.0",
        "kind": "q4_positive_domain_summary",
        "question": 4,
        "result_status": "positive_domain_empirical_minimum_with_conservative_recommendation",
        "domain": {
            "n_a": "positive_integer",
            "n_b": "positive_integer",
            "constraint": "N_A >= 1 and N_B >= 1",
            "cost_bounded_audit_shape": list(counts.shape),
        },
        "boundary_contract": artifact["boundary_contract"],
        "target_probability": TARGET,
        "fixed_trial_count": trials,
        "master_seed": int(artifact_config["master_seed"]),
        "stream_id": int(artifact_config["stream_id"]),
        "empirical_minimum": {
            **empirical,
            "role": "固定 50000 次样本下、正整数成本域完整枚举得到的经验最低成本设计",
            "strictly_cheaper_positive_designs_checked": strictly_cheaper_positive_count(
                empirical["cost_weight"]
            ),
        },
        "conservative_recommendation": {
            **recommendation,
            "role": "B=1 预定义分支的 619 项 Bonferroni-CP 可行方案，也是问题三单调性证书支持的最终推荐",
            "cost_premium_yuan": recommendation["cost_yuan"] - empirical["cost_yuan"],
            "cost_premium_percent": 100.0
            * (recommendation["cost_yuan"] / empirical["cost_yuan"] - 1.0),
        },
        "q3_monotonic_certificate": q3_proof,
        "nearby_designs": nearby_designs,
        "source_files": {
            "success_counts": {
                "path": counts_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(counts_path),
            },
            "q3_summary": {
                "path": q3_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(q3_path),
            },
            "confirmation_artifact": {
                "path": artifact_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(artifact_path),
                "configuration_fingerprint": artifact["configuration_fingerprint"],
            },
            "empirical_frontier_csv": frontier_csv.relative_to(PROJECT_ROOT).as_posix(),
            "b1_branch_csv": branch_csv.relative_to(PROJECT_ROOT).as_posix(),
        },
        "paper_claim": "固定样本经验最低配比为 (612,12)；综合联合置信裕度后推荐 (616,1)。",
    }
    summary_path = args.output_dir / "q4_positive_domain_summary.json"
    summary["summary_path"] = summary_path.relative_to(PROJECT_ROOT).as_posix()
    atomic_json(summary_path, summary)
    if args.register_results:
        update_registry(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题四正整数域结果冻结与注册")
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--q3-summary", type=Path, default=DEFAULT_Q3)
    parser.add_argument("--confirmation-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--register-results", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.relative_to(PROJECT_ROOT)
    result = build_summary(args)
    print(json.dumps({
        "summary": result["summary_path"],
        "empirical_minimum": result["empirical_minimum"],
        "conservative_recommendation": result["conservative_recommendation"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
