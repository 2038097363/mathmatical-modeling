from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT
    / "问题"
    / "问题4"
    / "results"
    / "D_screen2000_confirm50000"
)
SHARD_PATTERN = re.compile(r"^shard_(\d{6})_(\d{6})\.json$")
COST_A_WEIGHT = 567
COST_B_WEIGHT = 64
COST_SCALE_YUAN = math.pi / 120000.0
EXPECTED_SOURCE_HASHES = {
    "问题/问题4/src/solve.py": (
        "A8D30E2D8292335B3658611940FE9E40FB3CE02C8BC68F24120616B69937BB18"
    ),
    "公共代码/mixed_microstructure_sim.py": (
        "E43BDF50232AC61B2C8BF8E6609DF80E412B13FB340CB4DB94E6FA4EAD545D9F"
    ),
    "公共代码/microstructure_sim.py": (
        "222EF9B1B73B429BC5E35E2E95AFF1261F17A2F695F59429BE5DED4278D40B9C"
    ),
    "公共代码/geometry_kernel.py": (
        "652CB279E662B083E190885557A942B9F819A789E77D9C538945716C91027E32"
    ),
    "公共代码/pareto_connectivity.py": (
        "E6E3C40BE06241241D5717BB42D1E0AB10292D3FCB4BB7B7745F01D1486635DE"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="独立审计问题4正式确认分片并重建完整整数域成功次数矩阵"
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--counts-output",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "q4_confirmation_integer_domain_counts.npz",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=(
            DEFAULT_RESULT_ROOT / "q4_confirmation_integer_domain_analysis.json"
        ),
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


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


def read_json_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload, sha256_bytes(raw)


def project_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"正式产物路径必须位于项目目录内：{resolved}") from error


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    project_path(resolved)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}不存在：{resolved}")
    return resolved


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_counts_atomic(path: Path, counts: np.ndarray, trials: int) -> None:
    path = path.expanduser().resolve()
    project_path(path)
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


def canonical_record_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest().upper()


def validated_frontier(
    record: dict[str, Any], max_n_a: int, max_n_b: int
) -> tuple[tuple[int, int], ...]:
    raw = record.get("connectivity_frontier")
    if not isinstance(raw, list):
        raise ValueError("前沿记录缺少 connectivity_frontier 列表")
    frontier: list[tuple[int, int]] = []
    previous_a = -1
    previous_b = max_n_b + 1
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("前沿点必须是长度为 2 的整数列表")
        n_a, n_b = int(pair[0]), int(pair[1])
        if not (0 <= n_a <= max_n_a and 0 <= n_b <= max_n_b):
            raise ValueError(f"前沿点越出冻结整数域：({n_a},{n_b})")
        if n_a <= previous_a or n_b >= previous_b:
            raise ValueError("前沿点必须按 N_A 严格递增且按 N_B 严格递减")
        frontier.append((n_a, n_b))
        previous_a, previous_b = n_a, n_b
    return tuple(frontier)


def independent_success_counts(
    frontiers: Sequence[Sequence[tuple[int, int]]],
    max_n_a: int,
    max_n_b: int,
) -> np.ndarray:
    if max_n_a < 0 or max_n_b < 0:
        raise ValueError("整数域上限必须非负")
    if len(frontiers) > np.iinfo(np.int32).max:
        raise ValueError("试验数超过 int32 容量")
    difference = np.zeros((max_n_a + 2, max_n_b + 2), dtype=np.int32)
    right = max_n_b + 1
    for frontier in frontiers:
        for index, (n_a, n_b) in enumerate(frontier):
            stop = (
                max_n_a + 1
                if index + 1 == len(frontier)
                else int(frontier[index + 1][0])
            )
            if n_a >= stop:
                raise ValueError("前沿行带宽必须为正")
            difference[n_a, n_b] += 1
            difference[stop, n_b] -= 1
            difference[n_a, right] -= 1
            difference[stop, right] += 1
    counts = np.cumsum(
        np.cumsum(difference, axis=0, dtype=np.int32),
        axis=1,
        dtype=np.int32,
    )
    return np.asarray(counts[: max_n_a + 1, : max_n_b + 1], dtype=np.int32)


def direct_success_count(
    frontiers: Sequence[Sequence[tuple[int, int]]], n_a: int, n_b: int
) -> int:
    return sum(
        any(first <= n_a and second <= n_b for first, second in frontier)
        for frontier in frontiers
    )


def cost_weight(n_a: int, n_b: int) -> int:
    return COST_A_WEIGHT * n_a + COST_B_WEIGHT * n_b


def design_metrics(n_a: int, n_b: int, successes: int, trials: int) -> dict[str, Any]:
    weight = cost_weight(n_a, n_b)
    return {
        "n_a": int(n_a),
        "n_b": int(n_b),
        "successes": int(successes),
        "trials": int(trials),
        "estimate": float(successes / trials),
        "cost_weight": int(weight),
        "cost_yuan": float(weight * COST_SCALE_YUAN),
    }


def minimum_empirical_design(
    counts: np.ndarray, trials: int, target: float
) -> dict[str, Any] | None:
    required = int(math.ceil(target * trials))
    best: tuple[tuple[int, int, int], dict[str, Any]] | None = None
    for n_a, row in enumerate(np.asarray(counts)):
        n_b = int(np.searchsorted(row, required, side="left"))
        if n_b >= row.size:
            continue
        metrics = design_metrics(n_a, n_b, int(row[n_b]), trials)
        key = int(metrics["cost_weight"]), n_a, n_b
        if best is None or key < best[0]:
            best = key, metrics
    return None if best is None else best[1]


def audit_sources() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        path = require_file(PROJECT_ROOT / relative, relative)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"正式确认运行源文件哈希已变化：{relative}")
        files[relative] = {"sha256": actual, "matches_startup_hash": True}
    return {"passed": True, "files": files}


def audit_shards(
    shard_dir: Path,
    config: dict[str, Any],
    fingerprint: str,
    trials: int,
    batch_size: int,
) -> tuple[list[tuple[tuple[int, int], ...]], dict[str, Any], str]:
    if batch_size < 1:
        raise ValueError("分片规模必须为正整数")
    expected_ranges = [
        (start, min(trials, start + batch_size) - 1)
        for start in range(0, trials, batch_size)
    ]
    expected_names = [
        f"shard_{start:06d}_{stop:06d}.json" for start, stop in expected_ranges
    ]
    paths = sorted(shard_dir.glob("shard_*.json"), key=lambda value: value.name)
    observed_names = [path.name for path in paths]
    if observed_names != expected_names:
        missing = sorted(set(expected_names) - set(observed_names))
        extra = sorted(set(observed_names) - set(expected_names))
        raise ValueError(
            f"正式确认分片集合不完整：missing={missing[:5]}, extra={extra[:5]}"
        )

    frontiers: list[tuple[tuple[int, int], ...]] = []
    record_digest = hashlib.sha256()
    manifest_digest = hashlib.sha256()
    runtime_seconds = 0.0
    nonconverged_total = 0
    first_hash = ""
    last_hash = ""
    expected_trial = 0
    for path, (start, stop) in zip(paths, expected_ranges, strict=True):
        match = SHARD_PATTERN.fullmatch(path.name)
        if match is None or (int(match.group(1)), int(match.group(2))) != (start, stop):
            raise ValueError(f"分片文件名范围无效：{path.name}")
        payload, file_hash = read_json_with_hash(path)
        if not first_hash:
            first_hash = file_hash
        last_hash = file_hash
        manifest_digest.update(f"{path.name}\t{file_hash}\n".encode("ascii"))
        if payload.get("kind") != "mixed_pareto_frontier_shard":
            raise ValueError(f"分片类型无效：{path.name}")
        if payload.get("configuration_fingerprint") != fingerprint:
            raise ValueError(f"分片配置指纹不一致：{path.name}")
        if payload.get("configuration") != config:
            raise ValueError(f"分片内嵌配置不一致：{path.name}")
        ids = [int(value) for value in payload.get("trial_ids", [])]
        expected_ids = list(range(start, stop + 1))
        if ids != expected_ids:
            raise ValueError(f"分片 trial_ids 与文件名范围不一致：{path.name}")
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != len(ids):
            raise ValueError(f"分片记录数不一致：{path.name}")
        for trial_id, record in zip(ids, records, strict=True):
            if not isinstance(record, dict) or int(record.get("trial_id", -1)) != trial_id:
                raise ValueError(f"分片记录 trial_id 无效：{path.name}")
            if trial_id != expected_trial:
                raise ValueError("正式确认 trial_id 不连续")
            if int(record.get("processed_a_particles", -1)) != int(config["n_a"]):
                raise ValueError(f"A 粒子处理数不一致：trial={trial_id}")
            if int(record.get("processed_b_particles", -1)) != int(config["n_b"]):
                raise ValueError(f"B 粒子处理数不一致：trial={trial_id}")
            if int(record.get("internal_a_edges", -1)) != 0:
                raise ValueError(f"D 边界出现同源 A 内部边：trial={trial_id}")
            nonconverged_total += int(record.get("narrow_nonconverged", 0))
            frontiers.append(
                validated_frontier(record, int(config["n_a"]), int(config["n_b"]))
            )
            record_digest.update(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            record_digest.update(b"\n")
            expected_trial += 1
        runtime_seconds += float(payload.get("runtime_seconds", 0.0))

    return (
        frontiers,
        {
            "passed": True,
            "shard_count": len(paths),
            "batch_size": batch_size,
            "trial_count": len(frontiers),
            "trial_range": [0, len(frontiers) - 1],
            "missing_trial_count": 0,
            "duplicate_trial_count": 0,
            "configuration_fingerprint": fingerprint,
            "manifest_sha256": manifest_digest.hexdigest().upper(),
            "first_shard": expected_names[0],
            "first_shard_sha256": first_hash,
            "last_shard": expected_names[-1],
            "last_shard_sha256": last_hash,
            "shard_runtime_seconds": runtime_seconds,
            "narrow_nonconverged_total": nonconverged_total,
            "internal_a_edges_total": 0,
        },
        record_digest.hexdigest().upper(),
    )


def audit_merged_artifact(
    path: Path,
    config: dict[str, Any],
    fingerprint: str,
    trials: int,
    shard_record_digest: str,
    shard_runtime_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged, merged_hash = read_json_with_hash(path)
    if merged.get("kind") != "mixed_pareto_frontier_samples":
        raise ValueError("合并文件类型不是 mixed_pareto_frontier_samples")
    if merged.get("configuration") != config:
        raise ValueError("合并文件内嵌配置与冻结配置不一致")
    if merged.get("configuration_fingerprint") != fingerprint:
        raise ValueError("合并文件配置指纹与冻结配置不一致")
    records = merged.get("records")
    if not isinstance(records, list) or len(records) != trials:
        raise ValueError("合并文件记录数不等于冻结试验数")
    ids = [int(record.get("trial_id", -1)) for record in records]
    if ids != list(range(trials)):
        raise ValueError("合并文件 trial_id 不完整或未排序")
    merged_record_digest = canonical_record_digest(records)
    if merged_record_digest != shard_record_digest:
        raise ValueError("合并文件记录与 500 个正式分片不等价")
    stored_runtime = float(merged.get("shard_runtime_seconds", 0.0))
    tolerance = max(1e-9, abs(shard_runtime_seconds) * 1e-12)
    if abs(stored_runtime - shard_runtime_seconds) > tolerance:
        raise ValueError("合并文件分片运行时间总和不一致")
    for total_key, nested_key in (
        ("diagnostics_total", None),
        ("pareto_search_total", "pareto_search"),
    ):
        stored_totals = merged.get(total_key)
        if not isinstance(stored_totals, dict):
            raise ValueError(f"合并文件缺少 {total_key}")
        for field, stored in stored_totals.items():
            calculated = sum(
                int(record[field] if nested_key is None else record[nested_key][field])
                for record in records
            )
            if calculated != int(stored):
                raise ValueError(f"合并诊断汇总不一致：{total_key}.{field}")
    return merged, {
        "passed": True,
        "path": project_path(path),
        "sha256": merged_hash,
        "record_count": len(records),
        "trial_range": [0, trials - 1],
        "record_sequence_sha256": merged_record_digest,
        "records_equal_all_shards": True,
        "diagnostic_totals_recomputed": True,
        "shard_runtime_seconds": stored_runtime,
    }


def reconcile_final_records(
    result_root: Path,
    final: dict[str, Any],
    freeze: dict[str, Any],
    counts: np.ndarray,
    trials: int,
) -> dict[str, Any]:
    confirmation_path = require_file(result_root / "q4_confirmation.json", "确认 JSON")
    confirmation_csv_path = require_file(
        result_root / "q4_confirmation.csv", "确认 CSV"
    )
    if sha256(confirmation_path) != str(final.get("confirmation_json_sha256", "")):
        raise ValueError("最终摘要记录的确认 JSON 哈希不一致")
    if sha256(confirmation_csv_path) != str(final.get("confirmation_csv_sha256", "")):
        raise ValueError("最终摘要记录的确认 CSV 哈希不一致")
    confirmation = read_json(confirmation_path)
    records = confirmation.get("records")
    final_records = final.get("confirmation_records")
    if not isinstance(records, list) or records != final_records:
        raise ValueError("确认 JSON 与最终摘要的 620 个记录不一致")
    frozen = freeze.get("confirmation_designs")
    if not isinstance(frozen, list) or len(records) != len(frozen):
        raise ValueError("确认记录数与冻结设计数不一致")

    mismatches = []
    for expected, record in zip(frozen, records, strict=True):
        role = str(expected["role"])
        n_a, n_b = int(expected["n_a"]), int(expected["n_b"])
        if (str(record.get("role")), int(record["n_a"]), int(record["n_b"])) != (
            role,
            n_a,
            n_b,
        ):
            raise ValueError("确认记录顺序或冻结设计身份不一致")
        matrix_successes = int(counts[n_a, n_b])
        if matrix_successes != int(record["successes"]):
            mismatches.append(
                {
                    "n_a": n_a,
                    "n_b": n_b,
                    "record": int(record["successes"]),
                    "matrix": matrix_successes,
                }
            )
        if int(record["trials"]) != trials:
            raise ValueError("确认记录试验数不等于冻结试验数")
    if mismatches:
        raise ValueError(f"确认记录与完整域计数不一致：{mismatches[:3]}")

    with confirmation_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(records):
        raise ValueError("确认 CSV 行数与确认记录数不一致")
    return {
        "passed": True,
        "frozen_design_count": len(frozen),
        "confirmation_record_count": len(records),
        "csv_record_count": len(csv_rows),
        "matrix_mismatch_count": 0,
        "confirmation_json": project_path(confirmation_path),
        "confirmation_json_sha256": sha256(confirmation_path),
        "confirmation_csv": project_path(confirmation_csv_path),
        "confirmation_csv_sha256": sha256(confirmation_csv_path),
    }


def direct_cell_audit(
    frontiers: Sequence[Sequence[tuple[int, int]]], counts: np.ndarray
) -> dict[str, Any]:
    max_n_a, max_n_b = counts.shape[0] - 1, counts.shape[1] - 1
    cells = {
        (0, 0),
        (0, max_n_b),
        (max_n_a, 0),
        (max_n_a, max_n_b),
        (min(613, max_n_a), 0),
        (min(616, max_n_a), 0),
        (min(619, max_n_a), 0),
    }
    generator = np.random.default_rng(20260808)
    for _ in range(57):
        cells.add(
            (
                int(generator.integers(0, max_n_a + 1)),
                int(generator.integers(0, max_n_b + 1)),
            )
        )
    for n_a, n_b in sorted(cells):
        direct = direct_success_count(frontiers, n_a, n_b)
        if direct != int(counts[n_a, n_b]):
            raise ValueError(f"独立逐试验查询与二维差分不一致：({n_a},{n_b})")
    return {
        "passed": True,
        "checked_cell_count": len(cells),
        "seed": 20260808,
        "method": "逐试验检查是否存在分量不大于查询设计的 Pareto 点",
    }


def main() -> int:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    project_path(result_root)
    freeze_path = require_file(result_root / "q4_confirmation_freeze.json", "冻结协议")
    final_path = require_file(result_root / "q4_summary.json", "最终摘要")
    freeze = read_json(freeze_path)
    final = read_json(final_path)
    if freeze.get("kind") != "q4_confirmation_freeze":
        raise ValueError("冻结文件类型无效")
    if final.get("kind") != "q4_final_summary":
        raise ValueError("Q4 主进程尚未自然完成并写出最终摘要")
    if str(final.get("freeze_sha256", "")) != sha256(freeze_path):
        raise ValueError("最终摘要引用的冻结协议哈希不一致")

    protocol = freeze["confirmation_protocol"]
    config = protocol["configuration"]
    fingerprint = str(protocol["configuration_fingerprint"])
    trials = int(protocol["fixed_trial_count"])
    max_n_a, max_n_b = (int(value) for value in protocol["maximum_static_graph_design"])
    if (int(config["n_a"]), int(config["n_b"]), int(config["trial_count"])) != (
        max_n_a,
        max_n_b,
        trials,
    ):
        raise ValueError("冻结配置与最大静态图或试验数不一致")

    source_audit = audit_sources()
    candidate_dir = (
        result_root
        / "confirmation"
        / f"pareto_max_A{max_n_a:06d}_B{max_n_b:06d}"
    )
    shard_dir = candidate_dir / "shards"
    frontiers, shard_audit, shard_record_digest = audit_shards(
        shard_dir, config, fingerprint, trials, args.batch_size
    )
    merged_path = require_file(
        candidate_dir / "mixed_pareto_frontier_samples.json", "合并 Pareto 前沿"
    )
    _, merged_audit = audit_merged_artifact(
        merged_path,
        config,
        fingerprint,
        trials,
        shard_record_digest,
        float(shard_audit["shard_runtime_seconds"]),
    )

    counts = independent_success_counts(frontiers, max_n_a, max_n_b)
    if counts.shape != (max_n_a + 1, max_n_b + 1) or counts.dtype != np.int32:
        raise ValueError("独立重建的完整整数域矩阵形状或类型无效")
    if np.any(counts < 0) or np.any(counts > trials):
        raise ValueError("完整整数域成功次数越界")
    violations_a = int(np.count_nonzero(np.diff(counts, axis=0) < 0))
    violations_b = int(np.count_nonzero(np.diff(counts, axis=1) < 0))
    if violations_a or violations_b:
        raise ValueError("完整整数域成功次数违反二维单调性")
    direct_audit = direct_cell_audit(frontiers, counts)

    counts_path = args.counts_output.expanduser().resolve()
    write_counts_atomic(counts_path, counts, trials)
    reconciliation = reconcile_final_records(
        result_root, final, freeze, counts, trials
    )
    target = float(protocol["target_probability"])
    empirical_minimum = minimum_empirical_design(counts, trials, target)
    if empirical_minimum is None:
        raise ValueError("50,000 次完整域中没有经验概率达到目标的设计")
    candidate_record = next(
        record
        for record in final["confirmation_records"]
        if record.get("role") == "candidate"
    )
    candidate = {
        **design_metrics(
            int(candidate_record["n_a"]),
            int(candidate_record["n_b"]),
            int(candidate_record["successes"]),
            trials,
        ),
        "clopper_pearson_one_sided_lower": float(
            candidate_record["clopper_pearson_one_sided_lower"]
        ),
        "clopper_pearson_one_sided_upper": float(
            candidate_record["clopper_pearson_one_sided_upper"]
        ),
        "proof_status": candidate_record["proof_status"],
        "evidence_role": "预注册候选的独立同时置信结论",
    }
    q3_reference = design_metrics(616, 0, int(counts[616, 0]), trials)

    analysis = {
        "schema_version": 1,
        "kind": "q4_confirmation_integer_domain_analysis",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "audit_status": "passed",
        "question": 4,
        "result_status": final["result_status"],
        "conclusion_label_zh": final["conclusion_label_zh"],
        "publication_guard": (
            "描述性经验最低点由同一 50000 次确认样本事后查询，只能描述完整域样本；"
            "候选可行性与全局最低成本资格仅按冻结的 620 个 Bonferroni 单侧 CP 陈述判定。"
        ),
        "boundary_contract": final["boundary_contract"],
        "configuration": {
            "fingerprint": fingerprint,
            "trial_count": trials,
            "master_seed": int(config["master_seed"]),
            "stream_id": int(config["stream_id"]),
            "maximum_static_graph_design": [max_n_a, max_n_b],
            "integer_domain_shape": [max_n_a + 1, max_n_b + 1],
            "familywise_confidence": float(protocol["familywise_confidence"]),
            "bonferroni_statement_count": int(
                protocol["bonferroni_statement_count"]
            ),
            "per_statement_confidence": float(
                protocol["per_statement_confidence"]
            ),
        },
        "source_integrity": source_audit,
        "shard_audit": shard_audit,
        "merged_artifact_audit": merged_audit,
        "integer_domain_audit": {
            "passed": True,
            "counts_path": project_path(counts_path),
            "counts_sha256": sha256(counts_path),
            "dtype": str(counts.dtype),
            "shape": list(counts.shape),
            "minimum_count": int(counts.min()),
            "maximum_count": int(counts.max()),
            "n_a_direction_violations": violations_a,
            "n_b_direction_violations": violations_b,
            "direct_cell_audit": direct_audit,
        },
        "frozen_record_reconciliation": reconciliation,
        "statistical_results": {
            "target_probability": target,
            "candidate": candidate,
            "descriptive_empirical_minimum": {
                **empirical_minimum,
                "required_successes": int(math.ceil(target * trials)),
                "evidence_role": "同一确认样本上的完整整数域描述性经验最低点，非预注册可行性证明",
            },
            "q3_reference_design_in_confirmation_stream": {
                **q3_reference,
                "evidence_role": "跨问题数值一致性参照，不替代 Q3 的独立冻结区间",
            },
            "candidate_statistically_feasible": bool(
                final["candidate_statistically_feasible"]
            ),
            "all_strictly_cheaper_maximal_designs_excluded": bool(
                final["all_strictly_cheaper_maximal_designs_excluded"]
            ),
            "excluded_frontier_count": int(final["excluded_frontier_count"]),
            "not_excluded_frontier_count": int(
                final["not_excluded_frontier_count"]
            ),
            "cost_uncertainty_interval": final["cost_uncertainty_interval"],
        },
        "input_files": {
            "freeze": {
                "path": project_path(freeze_path),
                "sha256": sha256(freeze_path),
            },
            "final_summary": {
                "path": project_path(final_path),
                "sha256": sha256(final_path),
            },
            "merged_pareto_frontier": {
                "path": project_path(merged_path),
                "sha256": sha256(merged_path),
            },
        },
    }
    analysis_path = args.analysis_output.expanduser().resolve()
    write_json_atomic(analysis_path, analysis)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
