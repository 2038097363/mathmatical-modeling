# AI 工具：OpenAI Codex；模型/版本：GPT-5 系列；开发机构：OpenAI。
# 版本发布日期：2025-08-07（GPT-5 系列公开快照日期）；本程序由参赛队逐行复核并对结果负责。
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from render_freecad_scene import (
    atomic_copy,
    atomic_write_json,
    create_ascii_temp_dir,
    discover_freecad,
    hidden_startup_info,
    is_ascii_path,
    kill_process_tree,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = PROJECT_ROOT / "论文" / "figures"
Q4_RESULTS = PROJECT_ROOT / "问题" / "问题4" / "results"
COMMON_DIR = PROJECT_ROOT / "公共代码"
FREECAD_BUILDER = Path(__file__).with_name("build_freecad_scene.py")
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from geometry_kernel import Cylinder, Sphere  # noqa: E402
from microstructure_sim import (  # noqa: E402
    CylinderFragment,
    evaluate_fragment_contact,
    fragment_trial,
)
from mixed_microstructure_sim import (  # noqa: E402
    BOUNDARY_CONTRACT,
    ClippedSphere,
    MixedSimulationConfig,
    MixedTrialGeometry,
    SphereFragment,
    _SpatialHash,
    _axis_plane_distance,
    evaluate_exact_contact,
    fragment_sphere,
    generate_mixed_trial,
    load_pareto_frontier_artifact,
    solve_fixed_design,
)
from pareto_connectivity import design_is_connected, pareto_prune_labels  # noqa: E402


BOX_LENGTH_NM = 10_000.0
A_LENGTH_NM = 5_000.0
A_RADIUS_NM = 30.0
B_RADIUS_NM = 200.0
CONTACT_CUTOFF_NM = 1.8
PREVIEW_N_A = 18
PREVIEW_N_B = 48
DISPLAY_SEED = 20_260_807
PRIMARY_BOUNDARY = "D_truncated_fragments_independent"
FINAL_PUBLICATION_STATUS = "final_random_trial_geometry"
FINAL_RESULT_STATUSES = {
    "globally_certified_minimum_cost",
    "lowest_statistically_feasible_cost",
}
FINAL_FORBIDDEN_TEXT = ("preview", "not an optimal design", "非最优")


@dataclass(frozen=True)
class DesignSource:
    n_a: int
    n_b: int
    source_status: str
    publication_status: str
    source_path: Path | None
    source_sha256: str | None
    boundary_primary: str
    artifact_path: Path | None = None
    artifact_sha256: str | None = None
    artifact_configuration: MixedSimulationConfig | None = None
    artifact_configuration_fingerprint: str | None = None
    selected_configuration_fingerprint: str | None = None
    confirmation_proof_status: str | None = None


@dataclass(frozen=True)
class SceneInstance:
    node_id: str
    source_kind: str
    source_index: int
    fragment_index: int
    shape: Cylinder | Sphere | ClippedSphere
    record: dict[str, Any]
    a_fragment: CylinderFragment | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从问题4最终确认结果生成真实随机试验三维场景和 FCStd"
    )
    parser.add_argument(
        "--design-json",
        type=Path,
        help="问题4最终 q4_summary.json；省略时在 results 中严格查找唯一最终结果",
    )
    parser.add_argument("--freeze-json", type=Path, help="已冻结的 q4_confirmation_freeze.json")
    parser.add_argument("--confirmation-shard", type=Path, help="包含所选 trial 的正式确认分片")
    parser.add_argument("--trial-id", type=int, help="指定确认样本中的导通 trial")
    parser.add_argument("--preview", action="store_true", help="显式生成既有预览场景")
    parser.add_argument("--scene-output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--build-audit", type=Path)
    parser.add_argument("--freecadcmd-exe", type=Path)
    parser.add_argument("--display-seed", type=int, default=DISPLAY_SEED)
    parser.add_argument("--scene-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def _normalized_status(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _read_nonnegative_integer(mapping: dict[str, Any], key: str) -> int:
    if key not in mapping or isinstance(mapping[key], bool):
        raise ValueError(f"{key} 必须是非负整数")
    numeric = float(mapping[key])
    integer = int(numeric)
    if not math.isfinite(numeric) or numeric != integer or integer < 0:
        raise ValueError(f"{key} 必须是非负整数")
    return integer


def _resolve_evidence_path(raw: Any, summary_path: Path) -> Path:
    if not isinstance(raw, (str, os.PathLike)) or not str(raw).strip():
        raise ValueError("候选确认记录缺少 artifact_path")
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [summary_path.parent / path, PROJECT_ROOT / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    rendered = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"确认样本 artifact 不存在：{rendered}")


def _validate_d_contract(contract: Any, context: str) -> None:
    if not isinstance(contract, dict) or _normalized_status(contract.get("mode")) != "d":
        raise ValueError(f"{context} 不是官方主边界 D")
    same_source = _normalized_status(contract.get("same_source_rule", ""))
    if "independent" not in same_source or "excluded" not in same_source:
        raise ValueError(f"{context} 未声明截断片段独立且同源片段不连边")


def load_confirmed_design(path: Path) -> DesignSource:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("kind") != "q4_final_summary":
        raise ValueError("设计文件必须是 kind=q4_final_summary 的最终问题4结果")
    status = _normalized_status(payload.get("result_status"))
    if status not in FINAL_RESULT_STATUSES:
        raise ValueError(f"问题4结果尚不可发布：result_status={payload.get('result_status')!r}")
    _validate_d_contract(payload.get("boundary_contract"), "最终 summary 的边界合同")

    reported = payload.get("reported_design")
    if not isinstance(reported, dict):
        raise ValueError("最终 summary 缺少 reported_design")
    n_a = _read_nonnegative_integer(reported, "n_a")
    n_b = _read_nonnegative_integer(reported, "n_b")

    records = payload.get("confirmation_records")
    if not isinstance(records, list):
        raise ValueError("最终 summary 缺少 confirmation_records")
    candidates = [record for record in records if isinstance(record, dict) and record.get("role") == "candidate"]
    if len(candidates) != 1:
        raise ValueError("最终 summary 必须恰有一条 candidate 确认记录")
    candidate = candidates[0]
    if candidate.get("proof_status") != "candidate_statistically_feasible":
        raise ValueError("候选设计未通过独立统计确认")
    if _read_nonnegative_integer(candidate, "n_a") != n_a or _read_nonnegative_integer(candidate, "n_b") != n_b:
        raise ValueError("reported_design 与 candidate 确认记录数量不一致")

    artifact_path = _resolve_evidence_path(candidate.get("artifact_path"), resolved)
    artifact_hash = sha256(artifact_path)
    expected_hash = str(candidate.get("artifact_sha256", "")).strip().upper()
    if not expected_hash or artifact_hash != expected_hash:
        raise ValueError("candidate 记录的 artifact_sha256 与文件不一致")
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8-sig"))
    if artifact_payload.get("kind") != "mixed_pareto_frontier_samples":
        raise ValueError("candidate artifact 不是混合二维连通前沿样本")
    artifact_config = MixedSimulationConfig.from_dict(
        artifact_payload.get("configuration", {})
    )
    if artifact_payload.get("configuration_fingerprint") != artifact_config.fingerprint:
        raise ValueError("candidate artifact 配置指纹不一致")
    record_config_payload = candidate.get("configuration")
    if not isinstance(record_config_payload, dict):
        raise ValueError("candidate 记录缺少完整 configuration")
    record_config = MixedSimulationConfig.from_dict(record_config_payload)
    record_fingerprint = str(candidate.get("configuration_fingerprint", "")).strip().upper()
    if not record_fingerprint or record_fingerprint != record_config.fingerprint:
        raise ValueError("candidate configuration_fingerprint 与内嵌配置不一致")
    if record_fingerprint != artifact_config.fingerprint:
        raise ValueError("candidate 配置与 Pareto artifact 配置不一致")
    if n_a > artifact_config.n_a or n_b > artifact_config.n_b:
        raise ValueError("reported_design 超过确认 artifact 的最大静态图")
    if artifact_config.boundary_mode != "D":
        raise ValueError("确认 artifact 不是官方主边界 D")
    _validate_d_contract(artifact_payload.get("boundary_contract"), "确认 artifact 的边界合同")
    selected_config = replace(artifact_config, n_a=n_a, n_b=n_b)
    return DesignSource(
        n_a=n_a,
        n_b=n_b,
        source_status=status,
        publication_status=FINAL_PUBLICATION_STATUS,
        source_path=resolved,
        source_sha256=sha256(resolved),
        boundary_primary=PRIMARY_BOUNDARY,
        artifact_path=artifact_path,
        artifact_sha256=artifact_hash,
        artifact_configuration=artifact_config,
        artifact_configuration_fingerprint=artifact_config.fingerprint,
        selected_configuration_fingerprint=selected_config.fingerprint,
        confirmation_proof_status=str(candidate["proof_status"]),
    )


def _validate_frozen_screening_sources(freeze: dict[str, Any]) -> None:
    source = freeze.get("source_screening")
    if not isinstance(source, dict):
        raise ValueError("冻结协议缺少 source_screening")
    for path_key, hash_key in (
        ("json_path", "json_sha256"),
        ("csv_path", "csv_sha256"),
        ("pareto_artifact_path", "pareto_artifact_sha256"),
    ):
        evidence = Path(str(source.get(path_key, ""))).expanduser().resolve()
        expected = str(source.get(hash_key, "")).strip().upper()
        if not evidence.is_file() or not expected or sha256(evidence) != expected:
            raise ValueError(f"冻结协议的探索证据缺失或哈希不一致：{path_key}")


def _validated_shard_records(
    payload: dict[str, Any], config: MixedSimulationConfig
) -> list[dict[str, Any]]:
    if payload.get("kind") != "mixed_pareto_frontier_shard":
        raise ValueError("确认分片不是 mixed_pareto_frontier_shard")
    stored = MixedSimulationConfig.from_dict(payload.get("configuration", {}))
    fingerprint = str(payload.get("configuration_fingerprint", "")).strip().upper()
    if fingerprint != stored.fingerprint or fingerprint != config.fingerprint:
        raise ValueError("确认分片配置指纹与冻结协议不一致")
    trial_ids = [int(value) for value in payload.get("trial_ids", [])]
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("确认分片缺少 records")
    record_ids = [int(record.get("trial_id", -1)) for record in records]
    if trial_ids != sorted(set(trial_ids)) or record_ids != trial_ids:
        raise ValueError("确认分片 trial_id 不连续有序或与 records 不一致")
    for record in records:
        labels = tuple(
            (int(label[0]), int(label[1]))
            for label in record.get("connectivity_frontier", [])
        )
        if labels != pareto_prune_labels(labels):
            raise ValueError("确认分片含非确定性非支配前沿")
        record["connectivity_frontier"] = labels
    return records


def load_frozen_shard_design(freeze_path: Path, shard_path: Path) -> DesignSource:
    resolved_freeze = freeze_path.expanduser().resolve()
    freeze = json.loads(resolved_freeze.read_text(encoding="utf-8-sig"))
    if freeze.get("kind") != "q4_confirmation_freeze":
        raise ValueError("冻结文件不是 q4_confirmation_freeze")
    _validate_frozen_screening_sources(freeze)
    candidate = freeze.get("candidate_freeze")
    protocol = freeze.get("confirmation_protocol")
    if not isinstance(candidate, dict) or not isinstance(protocol, dict):
        raise ValueError("冻结协议缺少候选或确认配置")
    _validate_d_contract(protocol.get("boundary_contract"), "冻结确认边界合同")
    n_a = _read_nonnegative_integer(candidate, "n_a")
    n_b = _read_nonnegative_integer(candidate, "n_b")
    config = MixedSimulationConfig.from_dict(protocol.get("configuration", {}))
    fingerprint = str(protocol.get("configuration_fingerprint", "")).strip().upper()
    if fingerprint != config.fingerprint or config.boundary_mode != "D":
        raise ValueError("冻结确认配置指纹或主边界不一致")
    if n_a > config.n_a or n_b > config.n_b:
        raise ValueError("冻结候选超过确认最大静态图")

    resolved_shard = shard_path.expanduser().resolve()
    shard = json.loads(resolved_shard.read_text(encoding="utf-8-sig"))
    records = _validated_shard_records(shard, config)
    if not any(design_is_connected(record["connectivity_frontier"], n_a, n_b) for record in records):
        raise ValueError("确认分片中没有冻结候选的导通 trial")
    selected = replace(config, n_a=n_a, n_b=n_b)
    return DesignSource(
        n_a=n_a,
        n_b=n_b,
        source_status="frozen_candidate_confirmed_trial_geometry",
        publication_status=FINAL_PUBLICATION_STATUS,
        source_path=resolved_freeze,
        source_sha256=sha256(resolved_freeze),
        boundary_primary=PRIMARY_BOUNDARY,
        artifact_path=resolved_shard,
        artifact_sha256=sha256(resolved_shard),
        artifact_configuration=config,
        artifact_configuration_fingerprint=config.fingerprint,
        selected_configuration_fingerprint=selected.fingerprint,
        confirmation_proof_status="candidate_trial_connectivity_verified",
    )


def preview_source() -> DesignSource:
    return DesignSource(
        n_a=PREVIEW_N_A,
        n_b=PREVIEW_N_B,
        source_status="preview_not_optimal",
        publication_status="preview_not_optimal",
        source_path=None,
        source_sha256=None,
        boundary_primary=PRIMARY_BOUNDARY,
    )


def discover_design(explicit: Path | None, *, allow_preview: bool = False) -> DesignSource:
    if allow_preview:
        if explicit is not None:
            raise ValueError("--preview 不能与 --design-json 同时使用")
        return preview_source()
    if explicit is not None:
        return load_confirmed_design(explicit)
    confirmed: list[DesignSource] = []
    if Q4_RESULTS.is_dir():
        for path in sorted(Q4_RESULTS.rglob("q4_summary.json")):
            try:
                confirmed.append(load_confirmed_design(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    if len(confirmed) != 1:
        if not confirmed:
            raise FileNotFoundError("results 中没有可发布的 q4_final_summary；请先完成最终确认")
        paths = ", ".join(str(item.source_path) for item in confirmed)
        raise ValueError(f"发现多个可发布设计，必须用 --design-json 明确指定：{paths}")
    return confirmed[0]


def _source_reference(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def select_conductive_trial(source: DesignSource, explicit_trial_id: int | None = None) -> int:
    if source.artifact_path is None or source.artifact_configuration is None:
        raise ValueError("正式场景缺少确认 artifact")
    if sha256(source.artifact_path) != source.artifact_sha256:
        raise ValueError("确认 artifact 在设计解析后发生变化")
    payload = json.loads(source.artifact_path.read_text(encoding="utf-8-sig"))
    if payload.get("kind") == "mixed_pareto_frontier_samples":
        config, frontiers, records = load_pareto_frontier_artifact(source.artifact_path)
        frontier_by_trial = {
            int(record["trial_id"]): frontiers[int(record["trial_id"])]
            for record in records
        }
    elif payload.get("kind") == "mixed_pareto_frontier_shard":
        config = MixedSimulationConfig.from_dict(payload.get("configuration", {}))
        records = _validated_shard_records(payload, config)
        frontier_by_trial = {
            int(record["trial_id"]): record["connectivity_frontier"]
            for record in records
        }
    else:
        raise ValueError("正式场景证据不是确认合并 artifact 或确认分片")
    if config.fingerprint != source.artifact_configuration_fingerprint:
        raise ValueError("确认 artifact 配置指纹发生变化")
    if explicit_trial_id is not None:
        if explicit_trial_id < 0 or explicit_trial_id >= config.trial_count:
            raise ValueError("trial_id 超出确认样本范围")
        if explicit_trial_id not in frontier_by_trial:
            raise ValueError("指定 trial_id 不在确认 artifact 中")
        candidates = [explicit_trial_id]
    else:
        candidates = sorted(frontier_by_trial)
    for trial_id in candidates:
        if design_is_connected(frontier_by_trial[trial_id], source.n_a, source.n_b):
            return trial_id
    if explicit_trial_id is not None:
        raise ValueError("指定 trial 在选定设计下不导通")
    raise ValueError("确认 artifact 中没有选定设计的导通 trial")


def generate_selected_geometry(
    source: DesignSource, trial_id: int
) -> tuple[MixedSimulationConfig, MixedTrialGeometry]:
    maximum_config = source.artifact_configuration
    if maximum_config is None:
        raise ValueError("正式场景缺少最大静态图配置")
    maximum_geometry = generate_mixed_trial(maximum_config, trial_id)
    selected_config = replace(maximum_config, n_a=source.n_a, n_b=source.n_b)
    geometry = MixedTrialGeometry(
        maximum_geometry.a_centers[: source.n_a].copy(),
        maximum_geometry.a_directions[: source.n_a].copy(),
        maximum_geometry.b_centers[: source.n_b].copy(),
    )
    return selected_config, geometry


def _float_vector(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64)]


def _a_record(
    fragment: CylinderFragment,
    center: np.ndarray,
    direction: np.ndarray,
    config: MixedSimulationConfig,
) -> dict[str, Any]:
    start, end = fragment.cylinder.endpoints
    return {
        "id": f"A_s{fragment.source_index + 1:06d}_f{fragment.fragment_index + 1:03d}",
        "source_index": fragment.source_index,
        "fragment_index": fragment.fragment_index,
        "cell_shift": list(fragment.cell_shift),
        "source_center_nm": _float_vector(center),
        "source_direction": _float_vector(direction),
        "t_start": float(fragment.t_start),
        "t_end": float(fragment.t_end),
        "start_nm": _float_vector(start),
        "end_nm": _float_vector(end),
        "radius_nm": float(fragment.cylinder.radius),
        "source_length_nm": float(config.a_length_nm),
        "fragment_length_nm": float(2.0 * fragment.cylinder.half_length),
        "role": "background_a",
    }


def _b_record(
    fragment: SphereFragment,
    source_center: np.ndarray,
) -> dict[str, Any]:
    shape = fragment.shape
    if isinstance(shape, ClippedSphere):
        center = shape.sphere_center
        record = {
            "clip_box_lower_nm": _float_vector(shape.box_lower),
            "clip_box_upper_nm": _float_vector(shape.box_upper),
            "clipped": True,
        }
    else:
        center = shape.center
        record = {"clipped": False}
    return {
        "id": f"B_s{fragment.source_index + 1:06d}_f{fragment.fragment_index + 1:03d}",
        "source_index": fragment.source_index,
        "fragment_index": fragment.fragment_index,
        "cell_shift": list(fragment.cell_shift),
        "source_center_nm": _float_vector(source_center),
        "center_nm": _float_vector(center),
        "radius_nm": float(shape.radius),
        "role": "background_b",
        **record,
    }


def _pair_contact(
    first: SceneInstance,
    second: SceneInstance,
    config: MixedSimulationConfig,
) -> tuple[bool, dict[str, Any]]:
    if first.a_fragment is not None and second.a_fragment is not None:
        result = evaluate_fragment_contact(first.a_fragment, second.a_fragment, config.a_config())
        audit = {
            "contact_type": "A-A",
            "method": "evaluate_fragment_contact_D",
            "lower_bound_nm": float(result.lower_nm),
            "upper_bound_nm": None if result.upper_nm is None else float(result.upper_nm),
            "distance_nm": None,
            "broad_phase_rejected": bool(result.broad_phase_rejected),
            "narrow_phase_calls": int(result.narrow_phase_calls),
            "converged": result.converged,
        }
        return bool(result.connected), audit
    result = evaluate_exact_contact(first.shape, second.shape, config)
    audit = {
        "contact_type": result.pair_type,
        "method": result.method,
        "lower_bound_nm": float(result.lower_bound_nm),
        "upper_bound_nm": None,
        "distance_nm": None if result.distance_nm is None else float(result.distance_nm),
        "broad_phase_rejected": bool(result.broad_phase_rejected),
        "narrow_phase_calls": int(result.narrow_phase_calls),
        "converged": result.converged,
    }
    return bool(result.connected), audit


def _bfs_witness(
    adjacency: dict[str, list[tuple[str, int]]],
    edges: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    left = "electrode_left"
    right = "electrode_right"
    queue: deque[str] = deque([left])
    predecessor: dict[str, tuple[str, int] | None] = {left: None}
    while queue:
        current = queue.popleft()
        if current == right:
            break
        for neighbor, edge_index in sorted(adjacency[current], key=lambda item: (item[0], item[1])):
            if neighbor not in predecessor:
                predecessor[neighbor] = current, edge_index
                queue.append(neighbor)
    if right not in predecessor:
        raise RuntimeError("完整接触图未找到左右电极导通路径")
    nodes = [right]
    edge_indices: list[int] = []
    while nodes[-1] != left:
        previous = predecessor[nodes[-1]]
        if previous is None:
            raise RuntimeError("BFS 前驱链异常")
        parent, edge_index = previous
        edge_indices.append(edge_index)
        nodes.append(parent)
    nodes.reverse()
    edge_indices.reverse()
    witness_edges = [{**edges[index], "witness_order": order} for order, index in enumerate(edge_indices)]
    return nodes, witness_edges


def build_scene_from_geometry(
    source: DesignSource,
    config: MixedSimulationConfig,
    geometry: MixedTrialGeometry,
    trial_id: int,
    *,
    artifact_frontier_connected: bool = True,
) -> dict[str, Any]:
    if source.publication_status != FINAL_PUBLICATION_STATUS:
        raise ValueError("真实随机 trial 场景要求最终发布状态")
    if config.n_a != source.n_a or config.n_b != source.n_b:
        raise ValueError("场景配置数量与最终设计不一致")
    if config.boundary_mode != "D" or source.boundary_primary != PRIMARY_BOUNDARY:
        raise ValueError("场景只能使用官方主边界 D")
    if trial_id < 0 or trial_id >= config.trial_count:
        raise ValueError("trial_id 超出配置范围")
    fixed_result = solve_fixed_design(geometry, config)
    if not artifact_frontier_connected or not fixed_result.conductive:
        raise RuntimeError("确认前沿与固定设计求解器未共同确认该 trial 导通")

    a_groups = fragment_trial(geometry.a_centers, geometry.a_directions, config.a_config()) if config.n_a else []
    b_groups = [
        fragment_sphere(center, source_index, config)
        for source_index, center in enumerate(geometry.b_centers)
    ]
    instances: list[SceneInstance] = []
    spatial_hash = _SpatialHash(config)
    adjacency: dict[str, list[tuple[str, int]]] = defaultdict(list)
    edges: list[dict[str, Any]] = []
    counters: dict[str, int] = defaultdict(int)

    def add_edge(first: str, second: str, audit: dict[str, Any]) -> None:
        edge_index = len(edges)
        edge = {
            "edge_id": f"E{edge_index + 1:07d}",
            "nodes": [first, second],
            "same_source_pair": False,
            "connected": True,
            **audit,
        }
        edges.append(edge)
        adjacency[first].append((second, edge_index))
        adjacency[second].append((first, edge_index))

    def connect_electrodes(instance: SceneInstance) -> None:
        for side, offset in (("left", -config.half_box_nm), ("right", config.half_box_nm)):
            distance = float(_axis_plane_distance(instance.shape, 0, offset))
            if distance <= config.contact_cutoff_nm:
                add_edge(
                    f"electrode_{side}",
                    instance.node_id,
                    {
                        "contact_type": f"electrode-{instance.source_kind}",
                        "method": "axis_plane_distance",
                        "distance_nm": distance,
                        "lower_bound_nm": distance,
                        "upper_bound_nm": distance,
                        "broad_phase_rejected": False,
                        "narrow_phase_calls": 0,
                        "converged": True,
                    },
                )
                counters["electrode_contacts"] += 1

    def add_instance(instance: SceneInstance) -> None:
        for other_index in spatial_hash.candidates(instance.shape):
            other = instances[other_index]
            counters["candidate_pairs"] += 1
            if other.source_kind == instance.source_kind and other.source_index == instance.source_index:
                counters["same_source_skips"] += 1
                continue
            connected, audit = _pair_contact(instance, other, config)
            counters["broad_phase_rejections"] += int(audit["broad_phase_rejected"])
            counters["narrow_phase_calls"] += int(audit["narrow_phase_calls"])
            if connected:
                add_edge(other.node_id, instance.node_id, audit)
                counters[audit["contact_type"].lower().replace("-", "") + "_contacts"] += 1
        connect_electrodes(instance)
        instance_index = len(instances)
        instances.append(instance)
        spatial_hash.add(instance_index, instance.shape)

    for source_index, group in enumerate(a_groups):
        for fragment in group:
            record = _a_record(
                fragment,
                geometry.a_centers[source_index],
                geometry.a_directions[source_index],
                config,
            )
            add_instance(
                SceneInstance(
                    record["id"], "A", source_index, fragment.fragment_index,
                    fragment.cylinder, record, fragment,
                )
            )
    for source_index, group in enumerate(b_groups):
        for fragment in group:
            record = _b_record(fragment, geometry.b_centers[source_index])
            add_instance(
                SceneInstance(
                    record["id"], "B", source_index, fragment.fragment_index,
                    fragment.shape, record,
                )
            )

    witness_nodes, witness_edges = _bfs_witness(adjacency, edges)
    witness_set = set(witness_nodes)
    cylinders: list[dict[str, Any]] = []
    spheres: list[dict[str, Any]] = []
    for instance in instances:
        record = dict(instance.record)
        if instance.node_id in witness_set:
            record["role"] = "witness"
        if instance.source_kind == "A":
            cylinders.append(record)
        else:
            spheres.append(record)

    a_sources = {int(record["source_index"]) for record in cylinders}
    b_sources = {int(record["source_index"]) for record in spheres}
    if a_sources != set(range(config.n_a)) or b_sources != set(range(config.n_b)):
        raise RuntimeError("完整场景未覆盖全部选定源粒子")
    clipped_count = sum(bool(record["clipped"]) for record in spheres)
    scene = {
        "schema_version": 2,
        "name": f"Q4_selected_design_actual_trial_{trial_id:06d}",
        "publication_status": FINAL_PUBLICATION_STATUS,
        "visible_banner": (
            f"N_A={config.n_a}, N_B={config.n_b} | "
            f"随机导通样本 {trial_id:06d}"
        ),
        "box": {"length_nm": float(config.box_length_nm), "show": True, "transparency": 96},
        "electrodes": {
            "show": True,
            "thickness_nm": float(0.012 * config.box_length_nm),
            "transparency": 70,
        },
        "cylinders": cylinders,
        "spheres": spheres,
        "traceability": {
            "source_status": source.source_status,
            "boundary_primary": source.boundary_primary,
            "source_design_json": _source_reference(source.source_path),
            "source_design_sha256": source.source_sha256,
            "source_artifact": _source_reference(source.artifact_path),
            "source_artifact_sha256": source.artifact_sha256,
            "confirmation_proof_status": source.confirmation_proof_status,
            "design_counts": {"n_a": config.n_a, "n_b": config.n_b},
            "maximum_static_graph_design": {
                "n_a": source.artifact_configuration.n_a if source.artifact_configuration else config.n_a,
                "n_b": source.artifact_configuration.n_b if source.artifact_configuration else config.n_b,
            },
            "random_stream": {
                "master_seed": config.master_seed,
                "stream_id": config.stream_id,
                "trial_count": config.trial_count,
                "trial_id": trial_id,
            },
            "artifact_configuration_fingerprint": source.artifact_configuration_fingerprint,
            "selected_configuration_fingerprint": config.fingerprint,
            "geometry_contract": {
                **config.to_dict(),
                "boundary_mode": "D",
                "periodic_axes": [True, True, True],
                "minimum_image_axes": [False, False, False],
                "connect_same_source": False,
                "fragment_connectivity": "truncated_fragments_are_independent",
                "a_fragment_geometry": "centerline_cut_flat_cylinder",
                "b_fragment_geometry": "exact_ball_cell_intersection",
            },
            "source_particles": {
                "A": [
                    {
                        "source_index": index,
                        "center_nm": _float_vector(geometry.a_centers[index]),
                        "direction": _float_vector(geometry.a_directions[index]),
                    }
                    for index in range(config.n_a)
                ],
                "B": [
                    {"source_index": index, "center_nm": _float_vector(geometry.b_centers[index])}
                    for index in range(config.n_b)
                ],
            },
            "geometry_counts": {
                "a_source_particles": config.n_a,
                "b_source_particles": config.n_b,
                "a_fragments": len(cylinders),
                "b_fragments": len(spheres),
                "clipped_b_fragments": clipped_count,
                "all_fragments": len(instances),
                "witness_fragments": sum(instance.node_id in witness_set for instance in instances),
            },
            "contact_graph": {
                "node_count_including_electrodes": len(instances) + 2,
                "connected_edge_count": len(edges),
                "all_connected_edges": edges,
                "candidate_pair_count": int(counters["candidate_pairs"]),
                "same_source_skips": int(counters["same_source_skips"]),
                "contact_counts": {
                    "A-A": int(counters["aa_contacts"]),
                    "A-B": int(counters["ab_contacts"]),
                    "B-B": int(counters["bb_contacts"]),
                    "electrode": int(counters["electrode_contacts"]),
                },
                "broad_phase_rejections": int(counters["broad_phase_rejections"]),
                "narrow_phase_calls": int(counters["narrow_phase_calls"]),
            },
            "mixed_witness": {
                "present": True,
                "status": "actual_conductive_trial",
                "nodes": witness_nodes,
                "edges": witness_edges,
                "edge_count": len(witness_edges),
                "all_edges_geometry_verified": True,
                "same_source_edges": 0,
            },
            "cross_validation": {
                "pareto_artifact_frontier_connected": artifact_frontier_connected,
                "fixed_design_solver_conductive": fixed_result.conductive,
                "fixed_design_diagnostics": asdict(fixed_result.diagnostics),
                "reconstructed_graph_bfs_conductive": True,
            },
        },
    }
    _assert_final_scene_publication_safe(scene)
    return scene


def _assert_final_scene_publication_safe(scene: dict[str, Any]) -> None:
    if scene.get("publication_status") != FINAL_PUBLICATION_STATUS:
        return
    serialized = json.dumps(scene, ensure_ascii=False).lower()
    found = [term for term in FINAL_FORBIDDEN_TEXT if term.lower() in serialized]
    if found:
        raise ValueError(f"最终场景含禁止发布文字：{found}")


def _display_cylinder(identifier: str, start: np.ndarray, end: np.ndarray, role: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "start_nm": _float_vector(start),
        "end_nm": _float_vector(end),
        "radius_nm": A_RADIUS_NM,
        "source_length_nm": A_LENGTH_NM,
        "role": role,
    }


def _contained_random_cylinder(identifier: str, rng: np.random.Generator) -> dict[str, Any]:
    half_box = BOX_LENGTH_NM / 2.0
    half_length = A_LENGTH_NM / 2.0
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    extents = half_length * np.abs(direction)
    lower = -half_box + A_RADIUS_NM + extents
    upper = half_box - A_RADIUS_NM - extents
    center = rng.uniform(lower, upper)
    return _display_cylinder(identifier, center - half_length * direction, center + half_length * direction, "background_a")


def _background_sphere(identifier: str, rng: np.random.Generator) -> dict[str, Any]:
    half_box = BOX_LENGTH_NM / 2.0
    margin = B_RADIUS_NM + 60.0
    for _ in range(1000):
        center = rng.uniform(-half_box + margin, half_box - margin, size=3)
        if not (abs(float(center[0])) < 450.0 and abs(float(center[1])) < 520.0 and abs(float(center[2])) < 420.0):
            return {
                "id": identifier,
                "center_nm": _float_vector(center),
                "radius_nm": B_RADIUS_NM,
                "role": "background_b",
            }
    raise RuntimeError("无法生成预览背景 B 球")


def build_preview_scene(source: DesignSource, display_seed: int = DISPLAY_SEED) -> dict[str, Any]:
    if display_seed < 0:
        raise ValueError("display_seed 必须非负")
    rng = np.random.default_rng(display_seed)
    cylinders = [
        _display_cylinder("A_witness_left", np.array([-5000.0, -180.0, 0.0]), np.array([0.0, -180.0, 0.0]), "witness"),
        _display_cylinder("A_witness_right", np.array([0.0, 180.0, 0.0]), np.array([5000.0, 180.0, 0.0]), "witness"),
    ]
    spheres = [{"id": "B_witness_bridge", "center_nm": [0.0, 0.0, 0.0], "radius_nm": B_RADIUS_NM, "role": "witness"}]
    for index in range(len(cylinders), source.n_a):
        cylinders.append(_contained_random_cylinder(f"A_display_{index + 1:03d}", rng))
    for index in range(len(spheres), source.n_b):
        spheres.append(_background_sphere(f"B_display_{index + 1:03d}", rng))
    return {
        "schema_version": 1,
        "name": "Q4_mixed_preview_D_fragments_independent_not_optimal",
        "publication_status": "preview_not_optimal",
        "visible_banner": "PREVIEW - ILLUSTRATIVE, NOT AN OPTIMAL DESIGN",
        "box": {"length_nm": BOX_LENGTH_NM, "show": True, "transparency": 96},
        "electrodes": {"show": True, "thickness_nm": 120.0, "transparency": 52},
        "cylinders": cylinders,
        "spheres": spheres,
        "traceability": {
            "source_status": source.source_status,
            "boundary_primary": source.boundary_primary,
            "design_counts": {"n_a": source.n_a, "n_b": source.n_b},
            "display_seed": display_seed,
            "mixed_witness": {
                "present": True,
                "status": "illustrative_only_not_a_monte_carlo_trial",
                "nodes": ["left electrode", "A_witness_left", "B_witness_bridge", "A_witness_right", "right electrode"],
            },
        },
    }


def build_scene(
    source: DesignSource,
    display_seed: int = DISPLAY_SEED,
    trial_id: int | None = None,
) -> dict[str, Any]:
    if source.publication_status == "preview_not_optimal":
        return build_preview_scene(source, display_seed)
    selected_trial = select_conductive_trial(source, trial_id)
    return build_verified_trial_scene(source, selected_trial)


def build_verified_trial_scene(
    source: DesignSource, selected_trial: int
) -> dict[str, Any]:
    config, geometry = generate_selected_geometry(source, selected_trial)
    return build_scene_from_geometry(source, config, geometry, selected_trial)


def discover_freecadcmd(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    configured = os.environ.get("FREECADCMD_EXE")
    if configured:
        candidates.append(Path(configured))
    located = shutil.which("freecadcmd.exe") or shutil.which("FreeCADCmd.exe")
    if located:
        candidates.append(Path(located))
    try:
        candidates.append(discover_freecad(None).with_name("freecadcmd.exe"))
    except FileNotFoundError:
        pass
    candidates.append(Path.home() / "AppData" / "Local" / "Programs" / "FreeCAD 1.1" / "bin" / "FreeCADCmd.exe")
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    rendered = "\n".join(f"- {candidate}" for candidate in candidates) or "- none"
    raise FileNotFoundError(f"freecadcmd.exe not found; checked:\n{rendered}")


def _parse_builder_metadata(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for position, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stdout[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "counts" in payload and "output" in payload:
            return payload
    raise RuntimeError("FreeCAD builder stdout 中没有可识别的审计 JSON")


def _expected_group_counts(scene: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in scene.get("cylinders", []):
        counts[f"cylinder_{record.get('role', 'background_a')}"] += 1
    for record in scene.get("spheres", []):
        counts[f"sphere_{record.get('role', 'background_b')}"] += 1
    return dict(counts)


def build_fcstd(
    scene_path: Path,
    model_path: Path,
    audit_path: Path,
    freecadcmd_exe: Path | None,
    timeout: float,
) -> dict[str, Any]:
    if timeout <= 0.0:
        raise ValueError("timeout 必须为正")
    executable = discover_freecadcmd(freecadcmd_exe)
    embedded_python = executable.with_name("python.exe")
    if not embedded_python.is_file():
        raise FileNotFoundError(f"FreeCAD installation does not contain embedded python.exe: {embedded_python}")
    scene_path = scene_path.resolve()
    model_path = model_path.resolve()
    audit_path = audit_path.resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    temporary = create_ascii_temp_dir()
    process: subprocess.Popen[bytes] | None = None
    try:
        temporary_scene = temporary / "scene.json"
        temporary_builder = temporary / "build_scene.py"
        temporary_model = temporary / "scene.FCStd"
        shutil.copyfile(scene_path, temporary_scene)
        shutil.copyfile(FREECAD_BUILDER, temporary_builder)
        command = [str(embedded_python), str(temporary_builder), str(temporary_scene), str(temporary_model)]
        if any(not argument.isascii() for argument in command[1:]) or not all(is_ascii_path(Path(argument)) for argument in command[1:]):
            raise ValueError("FreeCAD 构建参数必须全部为 ASCII 路径")
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=temporary,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=hidden_startup_info(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_process_tree(process)
            stdout_bytes, stderr_bytes = process.communicate()
            raise TimeoutError(f"FreeCAD Python 构建超过 {timeout:.1f} 秒")
        elapsed = time.perf_counter() - started
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise RuntimeError(f"FreeCAD Python exited with {process.returncode}: {stderr[-2000:]}")
        if not temporary_model.is_file() or temporary_model.stat().st_size == 0:
            raise RuntimeError(f"FreeCAD Python 未生成非空 FCStd：{(stdout + chr(10) + stderr)[-3000:]}")
        builder_metadata = _parse_builder_metadata(stdout)
        expected_counts = _expected_group_counts(scene)
        if builder_metadata.get("counts") != expected_counts:
            raise RuntimeError(f"FreeCAD 对象计数与场景不一致：expected={expected_counts}, actual={builder_metadata.get('counts')}")
        expected_fragments = len(scene.get("cylinders", [])) + len(scene.get("spheres", []))
        if sum(expected_counts.values()) != expected_fragments:
            raise RuntimeError("场景分组计数没有覆盖全部片段")
        geometry_audit = builder_metadata.get("geometry_audit")
        if not isinstance(geometry_audit, dict):
            raise RuntimeError("FreeCAD builder 缺少 geometry_audit")
        trace_counts = scene.get("traceability", {}).get("geometry_counts", {})
        design_counts = scene.get("traceability", {}).get("design_counts", {})
        expected_witness = sum(
            str(record.get("role", "")) == "witness"
            for record in scene.get("cylinders", []) + scene.get("spheres", [])
        )
        geometry_checks = {
            "cylinder_fragment_count": geometry_audit.get("cylinder_fragment_count")
            == len(scene.get("cylinders", [])),
            "sphere_fragment_count": geometry_audit.get("sphere_fragment_count")
            == len(scene.get("spheres", [])),
            "total_fragment_count": geometry_audit.get("total_fragment_count")
            == expected_fragments,
            "clipped_sphere_fragment_count": geometry_audit.get(
                "clipped_sphere_fragment_count"
            )
            == sum("clip_box_lower_nm" in record for record in scene.get("spheres", [])),
            "witness_fragment_count": geometry_audit.get("witness_fragment_count")
            == expected_witness,
            "unique_source_particles_a": geometry_audit.get(
                "unique_source_particles", {}
            ).get("A")
            == design_counts.get("n_a"),
            "unique_source_particles_b": geometry_audit.get(
                "unique_source_particles", {}
            ).get("B")
            == design_counts.get("n_b"),
            "unique_geometry_ids": geometry_audit.get("unique_geometry_ids")
            == expected_fragments,
            "duplicate_geometry_ids": geometry_audit.get("duplicate_geometry_ids") == 0,
            "positive_volume_shapes": geometry_audit.get("positive_volume_shapes") is True,
            "sphere_shapes_inside_base_box": geometry_audit.get(
                "sphere_shapes_inside_base_box"
            )
            is True,
            "cylinder_centerlines_inside_base_box": geometry_audit.get(
                "cylinder_centerlines_inside_base_box"
            )
            is True,
            "scene_trace_fragment_count": trace_counts.get("all_fragments")
            == expected_fragments,
        }
        if not all(geometry_checks.values()):
            failed = [name for name, passed in geometry_checks.items() if not passed]
            raise RuntimeError(f"FreeCAD builder 几何审计不一致：{failed}")
        builder_file_checks = {
            "scene_file_sha256": builder_metadata.get("scene_file_sha256")
            == sha256(scene_path),
            "output_sha256": builder_metadata.get("output_sha256")
            == sha256(temporary_model),
            "output_size_bytes": builder_metadata.get("output_size_bytes")
            == temporary_model.stat().st_size,
        }
        if not all(builder_file_checks.values()):
            failed = [name for name, passed in builder_file_checks.items() if not passed]
            raise RuntimeError(f"FreeCAD builder 文件审计不一致：{failed}")
        atomic_copy(temporary_model, model_path)
        model_hash = sha256(model_path)
        if model_hash != sha256(temporary_model):
            raise RuntimeError("FCStd 原子复制后哈希不一致")
        audit = {
            "schema_version": 2,
            "status": "passed",
            "scene": str(scene_path),
            "scene_sha256": sha256(scene_path),
            "builder": str(FREECAD_BUILDER.resolve()),
            "builder_sha256": sha256(FREECAD_BUILDER),
            "freecadcmd_executable": str(executable),
            "freecadcmd_executable_sha256": sha256(executable),
            "freecad_python_executable": str(embedded_python),
            "freecad_python_executable_sha256": sha256(embedded_python),
            "model": str(model_path),
            "model_sha256": model_hash,
            "model_size_bytes": model_path.stat().st_size,
            "expected_group_counts": expected_counts,
            "expected_fragment_count": expected_fragments,
            "builder_metadata": builder_metadata,
            "geometry_checks": geometry_checks,
            "builder_file_checks": builder_file_checks,
            "process": {
                "return_code": process.returncode,
                "elapsed_seconds": elapsed,
                "arguments_ascii": True,
                "stdout_tail": stdout[-3000:],
                "stderr_tail": stderr[-3000:],
            },
            "temporary_directory_cleanup": "pending",
        }
    finally:
        if process is not None and process.poll() is None:
            kill_process_tree(process)
        shutil.rmtree(temporary, ignore_errors=True)
    audit["temporary_directory_cleanup"] = "ok"
    atomic_write_json(audit_path, audit)
    return audit


def _default_paths(source: DesignSource, trial_id: int | None) -> tuple[Path, Path, Path]:
    if source.publication_status == "preview_not_optimal":
        stem = "q4_mixed_preview"
    else:
        if trial_id is None:
            raise ValueError("正式产物默认路径要求 trial_id")
        stem = (
            f"q4_final_na{source.n_a:06d}_nb{source.n_b:06d}_"
            f"trial{trial_id:06d}"
        )
    return (
        FIGURE_ROOT / "data" / f"{stem}_scene.json",
        FIGURE_ROOT / "models" / f"{stem}.FCStd",
        FIGURE_ROOT / "models" / f"{stem}.build.audit.json",
    )


def main() -> int:
    args = parse_args()
    if args.preview:
        if args.freeze_json is not None or args.confirmation_shard is not None:
            raise ValueError("--preview 不能与冻结确认文件同用")
        source = discover_design(args.design_json, allow_preview=True)
    elif args.freeze_json is not None or args.confirmation_shard is not None:
        if args.design_json is not None or args.freeze_json is None or args.confirmation_shard is None:
            raise ValueError("冻结分片模式要求同时给出 --freeze-json/--confirmation-shard，且不使用 --design-json")
        source = load_frozen_shard_design(args.freeze_json, args.confirmation_shard)
    else:
        source = discover_design(args.design_json)
    selected_trial = None if args.preview else select_conductive_trial(source, args.trial_id)
    default_scene, default_model, default_audit = _default_paths(source, selected_trial)
    scene_path = (args.scene_output or default_scene).expanduser().resolve()
    model_path = (args.model_output or default_model).expanduser().resolve()
    audit_path = (args.build_audit or default_audit).expanduser().resolve()
    if source.publication_status == "preview_not_optimal":
        for path in (scene_path, model_path):
            if "preview" not in path.name.lower():
                raise ValueError(f"预览产物文件名必须含 preview：{path}")
    else:
        for path in (scene_path, model_path, audit_path):
            lowered = path.name.lower()
            if any(term in lowered for term in FINAL_FORBIDDEN_TEXT):
                raise ValueError(f"最终产物文件名含禁止文字：{path}")
    scene = (
        build_preview_scene(source, args.display_seed)
        if args.preview
        else build_verified_trial_scene(source, selected_trial)
    )
    atomic_write_json(scene_path, scene)
    build_audit = None
    if not args.scene_only:
        build_audit = build_fcstd(scene_path, model_path, audit_path, args.freecadcmd_exe, args.timeout)
    print(
        json.dumps(
            {
                "publication_status": source.publication_status,
                "trial_id": selected_trial,
                "scene": str(scene_path),
                "scene_sha256": sha256(scene_path),
                "model": str(model_path) if build_audit is not None else None,
                "model_sha256": build_audit["model_sha256"] if build_audit is not None else None,
                "build_audit": str(audit_path) if build_audit is not None else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
