#!/usr/bin/env python3
"""Build the anonymous, size-controlled support-material payload and ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
TEAM_ID_PATTERN = re.compile(r"CM\d{7}\Z")
PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?:[A-Z]:[\\/]Users[\\/][^\\/\s]+|/Users/[^/\s]+|/home/[^/\s]+)"
)
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:[A-Z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+)"
)
CONFIG_KEYS = {
    "schema_version",
    "status",
    "output_dir",
    "q1_extract",
    "threshold_samples",
    "files",
}
COPY_MODES = {"binary", "sanitized_json", "sanitized_csv", "text"}
DESTINATION_SOURCE_PREFIXES = {
    "结果摘要": (
        "问题/问题2/results/",
        "问题/问题3/results/",
        "问题/问题4/results/",
        "结果注册表.json",
        "论文/generated/results.tex",
    ),
    "数据": ("问题/问题4/results/",),
    "图件": ("论文/figures/generated/",),
    "三维模型": (
        "论文/figures/data/",
        "论文/figures/models/",
        "论文/figures/rendered/",
        "论文/figures/generated/",
    ),
}
MODE_BY_SUFFIX = {
    ".json": "sanitized_json",
    ".csv": "sanitized_csv",
    ".tex": "text",
    ".npz": "binary",
    ".png": "binary",
    ".fcstd": "binary",
}
EXPECTED_THRESHOLD_DESTINATIONS = {
    (
        "数据/q2_threshold_samples.npz",
        "数据/q2_threshold_samples.metadata.json",
    ),
    (
        "数据/q3_confirmation_threshold_samples.npz",
        "数据/q3_confirmation_threshold_samples.metadata.json",
    ),
}
REQUIRED_Q4_DESTINATIONS = {
    "结果摘要/q4_screening.json",
    "结果摘要/q4_confirmation_freeze.json",
    "结果摘要/q4_confirmation.json",
    "结果摘要/q4_confirmation.csv",
    "结果摘要/q4_summary.json",
    "数据/q4_screening_integer_domain_counts.npz",
    "数据/q4_confirmation_integer_domain_counts.npz",
    "数据/q4_confirmation_integer_domain_analysis.json",
    "图件/q4_cost_frontier.png",
    "图件/q4_cost_frontier.audit.json",
    "三维模型/q4_final_scene.json",
    "三维模型/q4_final.FCStd",
    "三维模型/q4_final_assets.audit.json",
    "三维模型/q4_final_axonometric.png",
    "三维模型/q4_final_top.png",
    "三维模型/q4_final_witness_focus.png",
}
REQUIRED_EXPLANATORY_DESTINATIONS = {
    "图件/model_workflow.png",
    "图件/validation_diagnostics.png",
    "图件/explanatory_figures.audit.json",
    "图件/q4_unresolved_boundary_evidence.png",
    "图件/q4_unresolved_boundary_evidence.audit.json",
    "图件/simulation_convergence.png",
    "图件/simulation_convergence.audit.json",
}
Q4_BOUNDARY_AUDIT_KIND = "q4_unresolved_boundary_evidence_figure_audit"
Q4_BOUNDARY_AUDIT_DESTINATION = "图件/q4_unresolved_boundary_evidence.audit.json"
Q4_BOUNDARY_PNG_DESTINATION = "图件/q4_unresolved_boundary_evidence.png"
Q4_BOUNDARY_PNG_SOURCE = "论文/figures/generated/q4_unresolved_boundary_evidence.png"
Q4_BOUNDARY_EXPECTED_INPUTS = {
    "问题/问题4/results/D_screen2000_confirm50000/q4_confirmation_integer_domain_analysis.json",
    "问题/问题4/results/D_screen2000_confirm50000/q4_summary.json",
    "问题/问题4/results/D_screen2000_confirm50000/q4_confirmation_freeze.json",
}
Q4_BOUNDARY_EVIDENCE_SCOPE = (
    "candidate_feasibility_cheaper_design_exclusion_and_unresolved_focus"
)
Q4_BOUNDARY_CLASSIFICATION_GUARD = "not_excluded_is_not_confirmed_feasible"
Q4_BOUNDARY_UNRESOLVED_SEMANTICS = "not_excluded_not_confirmed_feasible"
Q4_BOUNDARY_NEXT_ROUND_FOCUS = "46_not_excluded_maximal_designs_only"
Q4_EXPECTED_EXCLUDED_FRONTIER_COUNT = 573
Q4_EXPECTED_NOT_EXCLUDED_FRONTIER_COUNT = 46
CONVERGENCE_AUDIT_KIND = "simulation_convergence_figure_audit"
CONVERGENCE_AUDIT_DESTINATION = "图件/simulation_convergence.audit.json"
CONVERGENCE_PNG_DESTINATION = "图件/simulation_convergence.png"
CONVERGENCE_PNG_SOURCE = "论文/figures/generated/simulation_convergence.png"
CONVERGENCE_EXPECTED_INPUTS = {
    "问题/问题2/results/D_primary_n20000/q2_summary.json",
    "问题/问题2/results/D_primary_n20000/threshold_samples.json",
    "问题/问题4/results/D_screen2000_confirm50000/q4_confirmation_integer_domain_analysis.json",
    "问题/问题4/results/D_screen2000_confirm50000/confirmation/pareto_max_A000619_B005483/mixed_pareto_frontier_samples.json",
}
CONVERGENCE_EVIDENCE_SCOPE = "fixed_sample_cumulative_diagnostics"
CONVERGENCE_INTERPRETATION_GUARD = (
    "intermediate_checkpoints_diagnostic_only_final_n50000_authoritative"
)
CONVERGENCE_Q2_TRIALS = 20_000
CONVERGENCE_Q4_TRIALS = 50_000
CONVERGENCE_Q4_CHECKPOINTS = [
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    30_000,
    40_000,
    50_000,
]
Q4_EXPECTED_TRIALS = 50_000
Q4_EXPECTED_MAX_N_A = 619
Q4_EXPECTED_MAX_N_B = 5_483
Q4_SCREENING_TRIALS = 2_000
Q4_SCREENING_MAX_N_A = 720
Q4_SCREENING_MAX_N_B = 6_000
Q4_FINAL_STATUSES = {
    "globally_certified_minimum_cost",
    "lowest_statistically_feasible_cost",
}
Q4_FRONTIER_EVIDENCE_SCOPE = "candidate_feasibility_and_cheaper_design_exclusion"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest().upper()


def normalized_relative_path(raw: str, role: str) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"{role} must be a non-empty POSIX-style relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or not path.parts:
        raise ValueError(f"Unsafe {role}: {raw}")
    normalized = path.as_posix()
    if normalized != raw:
        raise ValueError(f"Non-canonical {role}: {raw}")
    return normalized


def validate_support_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict) or config.get("schema_version") != "2.0":
        raise ValueError("Support config must use schema_version 2.0")
    unknown = sorted(set(config) - CONFIG_KEYS)
    missing = sorted(CONFIG_KEYS - set(config))
    if unknown or missing:
        raise ValueError(f"Support config keys mismatch: unknown={unknown}, missing={missing}")
    if config.get("status") not in {"pending_q4", "frozen"}:
        raise ValueError("Support config status must be pending_q4 or frozen")
    if config.get("output_dir") != "支撑材料内容":
        raise ValueError("Support output must be the project-root 支撑材料内容 directory")

    q1 = config.get("q1_extract")
    if not isinstance(q1, dict) or set(q1) != {"source", "destination"}:
        raise ValueError("q1_extract must contain only source and destination")
    q1_source = normalized_relative_path(str(q1["source"]), "Q1 source")
    q1_destination = normalized_relative_path(str(q1["destination"]), "Q1 destination")
    if q1_source != "问题/问题1/results/q1_results.json":
        raise ValueError("Q1 extraction must use the frozen formal result artifact")
    if q1_destination != "结果摘要/q1_main_results.json":
        raise ValueError("Q1 extraction destination is fixed")

    thresholds = config.get("threshold_samples")
    if not isinstance(thresholds, list) or len(thresholds) != 2:
        raise ValueError("Exactly two Q2/Q3 threshold artifacts are required")
    threshold_destinations: set[tuple[str, str]] = set()
    all_destinations = {q1_destination, "README.md", "SHA256SUMS.txt"}
    for index, item in enumerate(thresholds, start=1):
        if not isinstance(item, dict) or set(item) != {
            "source",
            "destination",
            "metadata_destination",
        }:
            raise ValueError(f"Threshold item {index} has unsupported keys")
        source = normalized_relative_path(str(item["source"]), f"Threshold source {index}")
        destination = normalized_relative_path(
            str(item["destination"]), f"Threshold destination {index}"
        )
        metadata_destination = normalized_relative_path(
            str(item["metadata_destination"]), f"Threshold metadata destination {index}"
        )
        if not source.startswith(("问题/问题2/results/", "问题/问题3/results/")):
            raise ValueError(f"Threshold source {index} is outside Q2/Q3 formal results")
        if Path(destination).suffix.lower() != ".npz" or not metadata_destination.endswith(
            ".metadata.json"
        ):
            raise ValueError(f"Threshold item {index} has invalid output extensions")
        threshold_destinations.add((destination, metadata_destination))
        for relative in (destination, metadata_destination):
            if relative in all_destinations:
                raise ValueError(f"Duplicate generated destination: {relative}")
            all_destinations.add(relative)
    if threshold_destinations != EXPECTED_THRESHOLD_DESTINATIONS:
        raise ValueError("Q2/Q3 compact threshold destinations do not match the delivery contract")

    files = config.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Support config requires a non-empty files allowlist")
    file_destinations: set[str] = set()
    for index, item in enumerate(files, start=1):
        if not isinstance(item, dict) or set(item) != {"source", "destination", "mode"}:
            raise ValueError(f"Support file item {index} has unsupported keys")
        source = normalized_relative_path(str(item["source"]), f"Support source {index}")
        destination = normalized_relative_path(
            str(item["destination"]), f"Support destination {index}"
        )
        mode = str(item["mode"])
        if mode not in COPY_MODES:
            raise ValueError(f"Unsupported copy mode at item {index}: {mode}")
        category = PurePosixPath(destination).parts[0]
        prefixes = DESTINATION_SOURCE_PREFIXES.get(category)
        if prefixes is None or not source.startswith(prefixes):
            raise ValueError(
                f"Support source/destination category mismatch at item {index}: {source} -> {destination}"
            )
        suffix = Path(destination).suffix.lower()
        if MODE_BY_SUFFIX.get(suffix) != mode or Path(source).suffix.lower() != suffix:
            raise ValueError(f"Support copy mode or extension mismatch at item {index}")
        lowered_parts = {part.lower() for part in PurePosixPath(source).parts}
        if lowered_parts.intersection({"tmp", "shards", "__pycache__", ".pytest_cache"}):
            raise ValueError(f"Runtime intermediate is forbidden in support materials: {source}")
        if destination in all_destinations:
            raise ValueError(f"Duplicate generated destination: {destination}")
        all_destinations.add(destination)
        file_destinations.add(destination)
    missing_q4 = sorted(REQUIRED_Q4_DESTINATIONS - file_destinations)
    if missing_q4:
        raise ValueError(f"Q4 formal support files are incomplete: {missing_q4}")
    missing_explanatory = sorted(REQUIRED_EXPLANATORY_DESTINATIONS - file_destinations)
    if missing_explanatory:
        raise ValueError(
            f"Paper explanatory figures are incomplete: {missing_explanatory}"
        )
    return config


def project_source(project_root: Path, raw: str) -> tuple[Path, str]:
    path = (project_root / raw).resolve()
    try:
        relative = path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Source escapes project root: {raw}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, relative


def configured_source(config: dict[str, Any], project_root: Path, destination: str) -> Path:
    matches = [item for item in config["files"] if item["destination"] == destination]
    if len(matches) != 1:
        raise ValueError(f"Support destination must map to exactly one source: {destination}")
    return project_source(project_root, str(matches[0]["source"]))[0]


def recorded_project_file(record: Any, project_root: Path, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"Missing artifact record: {label}")
    raw_path = str(record.get("path") or "")
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Recorded artifact escapes the project root: {label}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if (
        str(record.get("sha256") or "").upper() != sha256_file(resolved)
        or ("size_bytes" in record and int(record["size_bytes"]) != resolved.stat().st_size)
    ):
        raise ValueError(f"Recorded artifact hash or size mismatch: {label}")
    return resolved


def validate_figure_audit_source_binding(
    audit_path: Path,
    project_root: Path,
    *,
    expected_kind: str,
    expected_inputs: set[str],
    expected_png_relative: str,
    configured_png: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    audit_path = audit_path.resolve()
    try:
        audit_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Figure audit escapes the project root") from exc
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("kind") != expected_kind:
        raise ValueError(f"Figure audit kind is invalid: {expected_kind}")

    inputs = audit.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
        raise ValueError(f"Figure audit input set is invalid: {expected_kind}")
    for relative, expected_hash in inputs.items():
        recorded_project_file(
            {"path": relative, "sha256": expected_hash},
            project_root,
            f"figure input {relative}",
        )

    expected_png = (project_root / expected_png_relative).resolve()
    configured_png = configured_png.resolve()
    if configured_png != expected_png or not configured_png.is_file():
        raise ValueError(f"Configured figure source is invalid: {expected_png_relative}")
    outputs = audit.get("outputs")
    if not isinstance(outputs, dict) or expected_png_relative not in outputs:
        raise ValueError(f"Figure audit omits its PNG output: {expected_png_relative}")
    for relative, record in outputs.items():
        if not isinstance(record, dict):
            raise ValueError(f"Figure audit output record is invalid: {relative}")
        recorded_project_file(
            {"path": relative, **record},
            project_root,
            f"figure output {relative}",
        )
    if (
        str(outputs[expected_png_relative].get("sha256") or "").upper()
        != sha256_file(configured_png)
    ):
        raise ValueError(f"Configured figure differs from its audit: {expected_png_relative}")
    return audit


def validate_q4_source_bindings(config: dict[str, Any], project_root: Path) -> None:
    destinations = {
        destination: configured_source(config, project_root, destination)
        for destination in REQUIRED_Q4_DESTINATIONS
    }
    screening = json.loads(
        destinations["结果摘要/q4_screening.json"].read_text(encoding="utf-8")
    )
    freeze = json.loads(
        destinations["结果摘要/q4_confirmation_freeze.json"].read_text(encoding="utf-8")
    )
    confirmation = json.loads(
        destinations["结果摘要/q4_confirmation.json"].read_text(encoding="utf-8")
    )
    final = json.loads(
        destinations["结果摘要/q4_summary.json"].read_text(encoding="utf-8")
    )
    analysis = json.loads(
        destinations["数据/q4_confirmation_integer_domain_analysis.json"].read_text(
            encoding="utf-8"
        )
    )
    frontier = json.loads(
        destinations["图件/q4_cost_frontier.audit.json"].read_text(encoding="utf-8")
    )
    assets = json.loads(
        destinations["三维模型/q4_final_assets.audit.json"].read_text(encoding="utf-8")
    )
    screening_counts = destinations["数据/q4_screening_integer_domain_counts.npz"]
    confirmation_counts = destinations["数据/q4_confirmation_integer_domain_counts.npz"]
    screening_path = destinations["结果摘要/q4_screening.json"]
    freeze_path = destinations["结果摘要/q4_confirmation_freeze.json"]
    confirmation_path = destinations["结果摘要/q4_confirmation.json"]
    confirmation_csv = destinations["结果摘要/q4_confirmation.csv"]
    final_path = destinations["结果摘要/q4_summary.json"]
    frontier_png = destinations["图件/q4_cost_frontier.png"]

    if (
        screening.get("integer_domain_success_counts_sha256")
        != sha256_file(screening_counts)
        or freeze.get("source_screening", {}).get("json_sha256")
        != sha256_file(screening_path)
        or confirmation.get("freeze_sha256") != sha256_file(freeze_path)
        or final.get("freeze_sha256") != sha256_file(freeze_path)
        or final.get("confirmation_json_sha256") != sha256_file(confirmation_path)
        or final.get("confirmation_csv_sha256") != sha256_file(confirmation_csv)
        or analysis.get("integer_domain_audit", {}).get("counts_sha256")
        != sha256_file(confirmation_counts)
        or analysis.get("input_files", {}).get("freeze", {}).get("sha256")
        != sha256_file(freeze_path)
        or analysis.get("input_files", {}).get("final_summary", {}).get("sha256")
        != sha256_file(final_path)
        or frontier.get("screening_sha256") != sha256_file(screening_path)
        or frontier.get("final_sha256") != sha256_file(final_path)
        or frontier.get("freeze_sha256") != sha256_file(freeze_path)
        or frontier.get("integer_domain_counts_sha256") != sha256_file(screening_counts)
        or frontier.get("output_png_sha256") != sha256_file(frontier_png)
    ):
        raise ValueError("Q4 formal source hashes do not form a consistent evidence chain")

    merged = analysis.get("merged_artifact_audit", {})
    recorded_project_file(merged, project_root, "merged confirmation frontier")
    frontier_pdf = {
        "path": frontier.get("output_pdf"),
        "sha256": frontier.get("output_pdf_sha256"),
    }
    recorded_project_file(frontier_pdf, project_root, "cost-frontier PDF")
    for name, record in assets.get("artifacts", {}).items():
        recorded_project_file(record, project_root, f"3D asset {name}")
    recorded_project_file(
        assets.get("source_confirmation_artifact"),
        project_root,
        "3D source confirmation shard",
    )

    configured_asset_sources = {
        "scene": destinations["三维模型/q4_final_scene.json"],
        "model": destinations["三维模型/q4_final.FCStd"],
        "axonometric": destinations["三维模型/q4_final_axonometric.png"],
        "top": destinations["三维模型/q4_final_top.png"],
        "witness_png": destinations["三维模型/q4_final_witness_focus.png"],
    }
    for name, path in configured_asset_sources.items():
        record = assets.get("artifacts", {}).get(name, {})
        if (
            str(record.get("sha256") or "").upper() != sha256_file(path)
            or int(record.get("size_bytes", -1)) != path.stat().st_size
        ):
            raise ValueError(f"Configured Q4 3D source differs from the asset audit: {name}")


def floats_match(left: Any, right: Any, tolerance: float = 5e-13) -> bool:
    try:
        return bool(
            np.isclose(
                float(left),
                float(right),
                atol=tolerance,
                rtol=0.0,
                equal_nan=False,
            )
        )
    except (TypeError, ValueError):
        return False


def float_sequences_match(left: Any, right: Any) -> bool:
    if (
        not isinstance(left, list)
        or not isinstance(right, list)
        or len(left) != len(right)
    ):
        return False
    return all(floats_match(a, b) for a, b in zip(left, right, strict=True))


def validate_q4_boundary_figure_audit(
    audit: dict[str, Any],
    analysis: dict[str, Any],
    final: dict[str, Any],
    freeze: dict[str, Any],
    png_path: Path,
) -> None:
    error = "Q4 unresolved-boundary figure audit is incomplete or inconsistent"
    try:
        statistical = analysis["statistical_results"]
        candidate = statistical["candidate"]
        interval = statistical["cost_uncertainty_interval"]
        final_interval = final["cost_uncertainty_interval"]
        confirmation_records = final["confirmation_records"]
        unresolved_records = sorted(
            (
                record
                for record in confirmation_records
                if record.get("role") == "strictly_cheaper_maximal"
                and record.get("proof_status")
                == "strictly_cheaper_design_not_excluded"
            ),
            key=lambda record: int(record["n_a"]),
        )
        excluded_records = [
            record
            for record in confirmation_records
            if record.get("role") == "strictly_cheaper_maximal"
            and record.get("proof_status") == "strictly_cheaper_design_excluded"
        ]
        candidate_records = [
            record
            for record in confirmation_records
            if record.get("role") == "candidate"
        ]
        reported_unresolved = sorted(
            final["not_excluded_frontier"], key=lambda record: int(record["n_a"])
        )
        excluded_count = int(statistical["excluded_frontier_count"])
        unresolved_count = int(statistical["not_excluded_frontier_count"])
        expected_designs = [
            [int(record["n_a"]), int(record["n_b"]), int(record["cost_weight"])]
            for record in unresolved_records
        ]
        reported_designs = [
            [int(record["n_a"]), int(record["n_b"]), int(record["cost_weight"])]
            for record in reported_unresolved
        ]
        unresolved_costs = [float(record["cost_yuan"]) for record in unresolved_records]
        expected_interval = [
            float(interval["lower_cost_yuan"]),
            float(interval["upper_cost_yuan"]),
        ]
        expected_unresolved_range = [min(unresolved_costs), max(unresolved_costs)]
        expected_candidate_design = [int(candidate["n_a"]), int(candidate["n_b"])]
        outputs = audit["outputs"]
        png_record = outputs[Q4_BOUNDARY_PNG_SOURCE]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc

    if (
        audit.get("schema_version") != 1
        or audit.get("kind") != Q4_BOUNDARY_AUDIT_KIND
        or not isinstance(audit.get("inputs"), dict)
        or set(audit["inputs"]) != Q4_BOUNDARY_EXPECTED_INPUTS
        or png_record.get("sha256") != sha256_file(png_path)
        or analysis.get("result_status") != final.get("result_status")
        or audit.get("result_status") != final.get("result_status")
        or audit.get("evidence_scope") != Q4_BOUNDARY_EVIDENCE_SCOPE
        or audit.get("classification_guard") != Q4_BOUNDARY_CLASSIFICATION_GUARD
        or audit.get("unresolved_semantics") != Q4_BOUNDARY_UNRESOLVED_SEMANTICS
        or audit.get("next_round_focus") != Q4_BOUNDARY_NEXT_ROUND_FOCUS
        or excluded_count != Q4_EXPECTED_EXCLUDED_FRONTIER_COUNT
        or unresolved_count != Q4_EXPECTED_NOT_EXCLUDED_FRONTIER_COUNT
        or int(final.get("excluded_frontier_count", -1)) != excluded_count
        or int(final.get("not_excluded_frontier_count", -1)) != unresolved_count
        or int(audit.get("excluded_frontier_count", -1)) != excluded_count
        or int(audit.get("not_excluded_frontier_count", -1)) != unresolved_count
        or len(confirmation_records) != excluded_count + unresolved_count + 1
        or len(excluded_records) != excluded_count
        or len(unresolved_records) != unresolved_count
        or len(candidate_records) != 1
        or reported_designs != expected_designs
        or audit.get("unresolved_designs") != expected_designs
        or audit.get("candidate_design") != expected_candidate_design
        or interval.get("minimum_not_excluded_design")
        != [Q4_EXPECTED_EXCLUDED_FRONTIER_COUNT, 0]
        or final_interval != interval
        or not float_sequences_match(audit.get("cost_interval_yuan"), expected_interval)
        or not float_sequences_match(
            audit.get("unresolved_cost_range_yuan"), expected_unresolved_range
        )
        or not floats_match(candidate.get("cost_yuan"), expected_interval[1])
        or not floats_match(
            audit.get("candidate_cp_lower"),
            candidate.get("clopper_pearson_one_sided_lower"),
        )
        or bool(statistical.get("all_strictly_cheaper_maximal_designs_excluded"))
        or bool(final.get("all_strictly_cheaper_maximal_designs_excluded"))
        or not bool(statistical.get("candidate_statistically_feasible"))
        or not bool(final.get("candidate_statistically_feasible"))
        or expected_interval[0] > expected_unresolved_range[0]
        or expected_unresolved_range[1] > expected_interval[1]
        or any(
            int(candidate.get(key, -1))
            != int(freeze.get("candidate_freeze", {}).get(key, -2))
            or int(candidate.get(key, -1))
            != int(final.get("reported_design", {}).get(key, -3))
            for key in ("n_a", "n_b", "cost_weight")
        )
    ):
        raise ValueError(error)


def validate_simulation_convergence_figure_audit(
    audit: dict[str, Any],
    q2_summary: dict[str, Any],
    analysis: dict[str, Any],
    png_path: Path,
) -> None:
    error = "Simulation-convergence figure audit is incomplete or inconsistent"
    try:
        q2_records = q2_summary["probability_records"]
        q2_estimates = {
            str(int(record["particle_count"])): float(record["probability"]["estimate"])
            for record in q2_records
        }
        statistical = analysis["statistical_results"]
        candidate = statistical["candidate"]
        configuration = analysis["configuration"]
        checkpoints = [int(value) for value in audit["q4_diagnostic_checkpoints"]]
        cp_lower = [float(value) for value in audit["q4_diagnostic_cp_lower"]]
        outputs = audit["outputs"]
        png_record = outputs[CONVERGENCE_PNG_SOURCE]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc

    q2_trial_counts = {
        int(record.get("probability", {}).get("trials", -1)) for record in q2_records
    }
    if (
        audit.get("schema_version") != 1
        or audit.get("kind") != CONVERGENCE_AUDIT_KIND
        or not isinstance(audit.get("inputs"), dict)
        or set(audit["inputs"]) != CONVERGENCE_EXPECTED_INPUTS
        or png_record.get("sha256") != sha256_file(png_path)
        or audit.get("evidence_scope") != CONVERGENCE_EVIDENCE_SCOPE
        or audit.get("interpretation_guard") != CONVERGENCE_INTERPRETATION_GUARD
        or int(q2_summary.get("fixed_trial_count", -1)) != CONVERGENCE_Q2_TRIALS
        or int(q2_summary.get("configuration", {}).get("trial_count", -1))
        != CONVERGENCE_Q2_TRIALS
        or q2_trial_counts != {CONVERGENCE_Q2_TRIALS}
        or int(audit.get("q2_trial_count", -1)) != CONVERGENCE_Q2_TRIALS
        or set(audit.get("q2_final_estimates", {})) != set(q2_estimates)
        or any(
            not floats_match(audit["q2_final_estimates"].get(key), value)
            for key, value in q2_estimates.items()
        )
        or int(configuration.get("trial_count", -1)) != CONVERGENCE_Q4_TRIALS
        or int(candidate.get("trials", -1)) != CONVERGENCE_Q4_TRIALS
        or int(audit.get("q4_trial_count", -1)) != CONVERGENCE_Q4_TRIALS
        or int(audit.get("q4_successes", -1)) != int(candidate.get("successes", -2))
        or not floats_match(audit.get("q4_final_estimate"), candidate.get("estimate"))
        or int(audit.get("q4_bonferroni_statement_count", -1))
        != int(configuration.get("bonferroni_statement_count", -2))
        or not floats_match(
            audit.get("q4_per_statement_confidence"),
            configuration.get("per_statement_confidence"),
        )
        or checkpoints != CONVERGENCE_Q4_CHECKPOINTS
        or len(cp_lower) != len(checkpoints)
        or any(not 0.0 <= value <= 1.0 for value in cp_lower)
        or not floats_match(audit.get("q4_final_cp_lower"), cp_lower[-1])
        or not floats_match(
            audit.get("q4_final_cp_lower"),
            candidate.get("clopper_pearson_one_sided_lower"),
        )
    ):
        raise ValueError(error)


def validate_explanatory_source_bindings(
    config: dict[str, Any], project_root: Path
) -> None:
    audit_path = configured_source(
        config, project_root, "图件/explanatory_figures.audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("kind") != "explanatory_figures_audit":
        raise ValueError("Explanatory figure audit kind is invalid")
    for relative, expected_hash in audit.get("inputs", {}).items():
        recorded_project_file(
            {"path": relative, "sha256": expected_hash},
            project_root,
            f"explanatory input {relative}",
        )
    for relative, record in audit.get("outputs", {}).items():
        recorded_project_file(
            {"path": relative, **record},
            project_root,
            f"explanatory output {relative}",
        )
    configured_pngs = {
        "论文/figures/generated/model_workflow.png": configured_source(
            config, project_root, "图件/model_workflow.png"
        ),
        "论文/figures/generated/validation_diagnostics.png": configured_source(
            config, project_root, "图件/validation_diagnostics.png"
        ),
    }
    for relative, source in configured_pngs.items():
        record = audit.get("outputs", {}).get(relative, {})
        if source != (project_root / relative).resolve() or record.get("sha256") != sha256_file(
            source
        ):
            raise ValueError(f"Configured explanatory figure differs from its audit: {relative}")

    q4_boundary_png = configured_source(
        config, project_root, Q4_BOUNDARY_PNG_DESTINATION
    )
    q4_boundary_audit = validate_figure_audit_source_binding(
        configured_source(config, project_root, Q4_BOUNDARY_AUDIT_DESTINATION),
        project_root,
        expected_kind=Q4_BOUNDARY_AUDIT_KIND,
        expected_inputs=Q4_BOUNDARY_EXPECTED_INPUTS,
        expected_png_relative=Q4_BOUNDARY_PNG_SOURCE,
        configured_png=q4_boundary_png,
    )
    convergence_png = configured_source(
        config, project_root, CONVERGENCE_PNG_DESTINATION
    )
    convergence_audit = validate_figure_audit_source_binding(
        configured_source(config, project_root, CONVERGENCE_AUDIT_DESTINATION),
        project_root,
        expected_kind=CONVERGENCE_AUDIT_KIND,
        expected_inputs=CONVERGENCE_EXPECTED_INPUTS,
        expected_png_relative=CONVERGENCE_PNG_SOURCE,
        configured_png=convergence_png,
    )

    analysis = json.loads(
        configured_source(
            config,
            project_root,
            "数据/q4_confirmation_integer_domain_analysis.json",
        ).read_text(encoding="utf-8")
    )
    final = json.loads(
        configured_source(config, project_root, "结果摘要/q4_summary.json").read_text(
            encoding="utf-8"
        )
    )
    freeze = json.loads(
        configured_source(
            config, project_root, "结果摘要/q4_confirmation_freeze.json"
        ).read_text(encoding="utf-8")
    )
    q2_summary = json.loads(
        configured_source(config, project_root, "结果摘要/q2_summary.json").read_text(
            encoding="utf-8"
        )
    )
    validate_q4_boundary_figure_audit(
        q4_boundary_audit, analysis, final, freeze, q4_boundary_png
    )
    validate_simulation_convergence_figure_audit(
        convergence_audit, q2_summary, analysis, convergence_png
    )


def safe_destination(root: Path, raw: str) -> tuple[Path, str]:
    normalized = normalized_relative_path(raw, "destination")
    posix = PurePosixPath(normalized)
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Destination escapes output root: {raw}") from exc
    return path, normalized


def scrub_value(value: Any, project_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            str(scrub_value(str(key), project_root)): scrub_value(item, project_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_value(item, project_root) for item in value]
    if not isinstance(value, str):
        return value

    root_native = str(project_root.resolve())
    root_slash = project_root.resolve().as_posix()
    cleaned = value
    contained_project_path = root_native in cleaned or root_slash in cleaned
    cleaned = cleaned.replace(root_native + "\\", "").replace(root_native, ".")
    cleaned = cleaned.replace(root_slash + "/", "").replace(root_slash, ".")
    if PRIVATE_PATH_PATTERN.search(cleaned):
        raise ValueError(f"Private home path remains after sanitization: {value!r}")
    if WINDOWS_ABSOLUTE_PATH_PATTERN.search(cleaned):
        raise ValueError(f"Absolute Windows path remains after sanitization: {value!r}")
    return cleaned.replace("\\", "/") if contained_project_path else cleaned


def reject_private_home_text(text: str, role: str) -> None:
    match = PRIVATE_PATH_PATTERN.search(text)
    if match:
        raise ValueError(f"{role} contains a private home path: {match.group(0)}")


def reject_private_text(text: str, role: str) -> None:
    reject_private_home_text(text, role)
    match = WINDOWS_ABSOLUTE_PATH_PATTERN.search(text)
    if match:
        raise ValueError(f"{role} contains an absolute Windows path: {match.group(0)}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_q1(source: Path, destination: Path, project_root: Path) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    selected = [
        item
        for item in raw.get("scenario_results", [])
        if item.get("scenario") == "A_row_literal"
        and item.get("internal_mode") == "disconnected_fragments"
    ]
    if len(selected) != 3 or {item.get("sheet") for item in selected} != {"组1", "组2", "组3"}:
        raise ValueError("Q1 formal result must contain exactly the three independent-fragment groups")
    conclusions = {item["sheet"]: item.get("conclusion") for item in selected}
    expected = {"组1": "nonconductive", "组2": "conductive", "组3": "conductive"}
    if conclusions != expected:
        raise ValueError(f"Unexpected Q1 conclusions: {conclusions}")
    if any(
        int(item.get("internal_edges_enabled", -1)) != 0
        or int(item.get("periodic_junction_count", -1)) != 0
        for item in selected
    ):
        raise ValueError("Q1 formal groups must not retain same-source internal edges")
    selected.sort(key=lambda item: ("组1", "组2", "组3").index(str(item["sheet"])))
    metadata = raw.get("metadata", {})
    screening = raw.get("screening", [])
    if not isinstance(screening, list) or {item.get("sheet") for item in screening} != {
        "组1",
        "组2",
        "组3",
    }:
        raise ValueError("Q1 geometry audit must contain exactly the three formal groups")
    payload = {
        "kind": "q1_independent_fragment_results",
        "formal_scope": "Only row-literal, disconnected-fragment results are included.",
        "model": "Each attachment row and every truncated fragment is an independent medium entity.",
        "constants": metadata.get("constants", {}),
        "input_hashes": metadata.get("input_hashes", {}),
        "independent_fragment_geometry_audit": screening,
        "groups": selected,
        "source_sha256": sha256_file(source),
    }
    sanitized = scrub_value(payload, project_root)
    serialized = json.dumps(sanitized, ensure_ascii=False)
    if "connected_same_particle" in serialized or "B_full_cube_periodic" in serialized:
        raise ValueError("Q1 support extract leaked a non-formal diagnostic scenario")
    write_json(destination, sanitized)


def compact_threshold_samples(
    source: Path,
    destination: Path,
    metadata_destination: Path,
    project_root: Path,
) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    records = raw.get("records")
    configuration = raw.get("configuration")
    if raw.get("kind") != "microstructure_threshold_samples" or not isinstance(records, list):
        raise ValueError(f"Unsupported threshold artifact: {source}")
    if not isinstance(configuration, dict):
        raise ValueError(f"Threshold artifact lacks configuration: {source}")

    expected = int(configuration.get("trial_count", len(records)))
    max_count = int(configuration["max_count"])
    if expected < 1 or expected > np.iinfo(np.int32).max or max_count < 1:
        raise ValueError("Threshold trial_count or max_count is outside the supported range")
    if any(not isinstance(item, dict) for item in records):
        raise ValueError("Threshold records must be JSON objects")
    raw_trial_ids = [int(item["trial_id"]) for item in records]
    raw_first_indices = [int(item["first_connection_index"]) for item in records]
    raw_censored = [bool(item["censored"]) for item in records]
    if any(index < 1 or index > max_count + 1 for index in raw_first_indices):
        raise ValueError("Threshold first-connection indices are outside 1..max_count+1")
    trial_ids = np.asarray(raw_trial_ids, dtype=np.int32)
    first_indices = np.asarray(raw_first_indices, dtype=np.int32)
    censored = np.asarray(raw_censored, dtype=np.bool_)
    if len(records) != expected or not np.array_equal(trial_ids, np.arange(expected, dtype=np.int32)):
        raise ValueError(f"Threshold trial IDs are not the complete 0..{expected - 1} sequence")
    if not np.array_equal(censored, first_indices == max_count + 1):
        raise ValueError("Threshold censoring flags do not match first-connection indices")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        trial_id=trial_ids,
        first_connection_index=first_indices,
        censored=censored,
    )
    metadata = {
        "kind": "compact_microstructure_threshold_samples",
        "schema_version": "1.0",
        "source_sha256": sha256_file(source),
        "configuration": configuration,
        "configuration_fingerprint": raw.get("configuration_fingerprint"),
        "record_count": len(records),
        "censored_trials": int(censored.sum()),
        "recomputation": {
            "count_domain": [0, max_count],
            "successes_at_count_n": "count_nonzero(first_connection_index <= n)",
            "probability_at_count_n": "successes_at_count_n / record_count",
            "censored_sentinel": max_count + 1,
        },
        "arrays": {
            "trial_id": {
                "dtype": str(trial_ids.dtype),
                "shape": list(trial_ids.shape),
                "sha256": sha256_array(trial_ids),
            },
            "first_connection_index": {
                "dtype": str(first_indices.dtype),
                "shape": list(first_indices.shape),
                "sha256": sha256_array(first_indices),
            },
            "censored": {
                "dtype": str(censored.dtype),
                "shape": list(censored.shape),
                "sha256": sha256_array(censored),
            },
        },
        "npz_sha256": sha256_file(destination),
    }
    write_json(metadata_destination, scrub_value(metadata, project_root))


def copy_entry(source: Path, destination: Path, mode: str, project_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "binary":
        shutil.copyfile(source, destination)
    elif mode == "sanitized_json":
        data = json.loads(source.read_text(encoding="utf-8"))
        write_json(destination, scrub_value(data, project_root))
    elif mode == "sanitized_csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [
                [str(scrub_value(value, project_root)) for value in row]
                for row in csv.reader(stream)
            ]
        with destination.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)
    elif mode == "text":
        text = source.read_text(encoding="utf-8")
        reject_private_text(text, f"Text source {source.name}")
        destination.write_text(text, encoding="utf-8", newline="\n")
    else:
        raise ValueError(f"Unsupported support-material copy mode: {mode}")


def npz_scalar_int(arrays: Any, name: str) -> int:
    value = np.asarray(arrays[name])
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"NPZ field {name} must be an integer scalar")
    return int(value)


def validate_integer_domain_counts(
    path: Path,
    expected_trials: int,
    expected_max_n_a: int,
    expected_max_n_b: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"success_counts", "trials", "max_n_a", "max_n_b"}:
            raise ValueError("Q4 integer-domain NPZ has unexpected fields")
        counts = np.asarray(arrays["success_counts"])
        trials = npz_scalar_int(arrays, "trials")
        max_n_a = npz_scalar_int(arrays, "max_n_a")
        max_n_b = npz_scalar_int(arrays, "max_n_b")
    expected_shape = (expected_max_n_a + 1, expected_max_n_b + 1)
    if counts.dtype != np.int32 or counts.shape != expected_shape:
        raise ValueError(
            f"Q4 integer-domain matrix must be int32 with shape {expected_shape}, got "
            f"{counts.dtype} {counts.shape}"
        )
    if (trials, max_n_a, max_n_b) != (
        expected_trials,
        expected_max_n_a,
        expected_max_n_b,
    ):
        raise ValueError("Q4 integer-domain scalar metadata does not match the frozen contract")
    if int(counts.min()) < 0 or int(counts.max()) > trials:
        raise ValueError("Q4 integer-domain success counts are outside 0..trials")
    violations_a = int(np.count_nonzero(np.diff(counts, axis=0) < 0))
    violations_b = int(np.count_nonzero(np.diff(counts, axis=1) < 0))
    if violations_a or violations_b:
        raise ValueError("Q4 integer-domain success counts violate monotonicity")
    return {
        "shape": list(counts.shape),
        "dtype": str(counts.dtype),
        "trials": trials,
        "minimum": int(counts.min()),
        "maximum": int(counts.max()),
    }


def validate_fcstd_anonymity(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"FreeCAD artifact is not a readable FCStd ZIP container: {path.name}")
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None or "Document.xml" not in bundle.namelist():
            raise ValueError(f"FreeCAD artifact is corrupt or lacks Document.xml: {path.name}")
        for member in bundle.namelist():
            reject_private_home_text(member, f"FreeCAD member name {path.name}")
            member_text = bundle.read(member).decode("latin-1")
            reject_private_home_text(
                member_text, f"FreeCAD member {path.name}:{member}"
            )
        document = bundle.read("Document.xml").decode("utf-8")
    reject_private_text(document, f"FreeCAD metadata {path.name}")
    for field in ("Company", "CreatedBy", "LastModifiedBy"):
        pattern = re.compile(
            rf'<Property name="{field}"[^>]*>\s*<String value="([^"]*)"', re.DOTALL
        )
        match = pattern.search(document)
        if match is None or match.group(1):
            raise ValueError(f"FreeCAD metadata field {field} must be present and empty: {path.name}")


def validate_q4_payload(output_root: Path) -> None:
    screening = json.loads(
        (output_root / "结果摘要" / "q4_screening.json").read_text(encoding="utf-8")
    )
    freeze = json.loads(
        (output_root / "结果摘要" / "q4_confirmation_freeze.json").read_text(encoding="utf-8")
    )
    final = json.loads(
        (output_root / "结果摘要" / "q4_summary.json").read_text(encoding="utf-8")
    )
    confirmation = json.loads(
        (output_root / "结果摘要" / "q4_confirmation.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (output_root / "数据" / "q4_confirmation_integer_domain_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    screening_counts_path = output_root / "数据" / "q4_screening_integer_domain_counts.npz"
    screening_counts_audit = validate_integer_domain_counts(
        screening_counts_path,
        Q4_SCREENING_TRIALS,
        Q4_SCREENING_MAX_N_A,
        Q4_SCREENING_MAX_N_B,
    )
    if (
        screening.get("kind") != "q4_screening_results"
        or int(screening.get("fixed_trial_count", -1)) != Q4_SCREENING_TRIALS
        or screening.get("maximum_static_graph_design")
        != [Q4_SCREENING_MAX_N_A, Q4_SCREENING_MAX_N_B]
        or screening.get("integer_domain_success_counts_sha256")
        != sha256_file(screening_counts_path)
    ):
        raise ValueError("Q4 screening artifact or integer-domain matrix is inconsistent")

    if freeze.get("kind") != "q4_confirmation_freeze":
        raise ValueError("Q4 freeze artifact kind is invalid")
    protocol = freeze.get("confirmation_protocol", {})
    maximum = protocol.get("maximum_static_graph_design")
    source_screening = freeze.get("source_screening", {})
    if (
        int(protocol.get("fixed_trial_count", -1)) != Q4_EXPECTED_TRIALS
        or maximum != [Q4_EXPECTED_MAX_N_A, Q4_EXPECTED_MAX_N_B]
        or len(freeze.get("confirmation_designs", [])) != Q4_EXPECTED_MAX_N_A + 1
        or int(source_screening.get("fixed_trial_count", -1)) != Q4_SCREENING_TRIALS
        or source_screening.get("maximum_static_graph_design")
        != [Q4_SCREENING_MAX_N_A, Q4_SCREENING_MAX_N_B]
    ):
        raise ValueError("Q4 freeze artifact does not match the 50,000-trial, 620-design contract")
    if (
        final.get("kind") != "q4_final_summary"
        or final.get("result_status") not in Q4_FINAL_STATUSES
    ):
        raise ValueError("Q4 final summary was not produced by a completed confirmation run")
    reported_design = final.get("reported_design", {})
    frozen_candidate = freeze.get("candidate_freeze", {})
    if any(
        int(reported_design.get(key, -1)) != int(frozen_candidate.get(key, -2))
        for key in ("n_a", "n_b", "cost_weight")
    ):
        raise ValueError("Q4 final design does not match the frozen candidate")
    records = confirmation.get("records")
    if confirmation.get("kind") != "q4_confirmation_results" or not isinstance(records, list):
        raise ValueError("Q4 confirmation JSON kind or records are invalid")
    if len(records) != Q4_EXPECTED_MAX_N_A + 1 or any(
        int(record.get("trials", -1)) != Q4_EXPECTED_TRIALS for record in records
    ):
        raise ValueError("Q4 confirmation JSON does not contain all 620 frozen records")
    with (output_root / "结果摘要" / "q4_confirmation.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        csv_records = list(csv.DictReader(stream))
    if len(csv_records) != len(records):
        raise ValueError("Q4 confirmation CSV row count differs from the confirmation JSON")
    if (
        confirmation.get("freeze_sha256") != final.get("freeze_sha256")
    ):
        raise ValueError("Q4 confirmation files are not bound to the final summary")

    counts_path = output_root / "数据" / "q4_confirmation_integer_domain_counts.npz"
    validate_integer_domain_counts(
        counts_path,
        Q4_EXPECTED_TRIALS,
        Q4_EXPECTED_MAX_N_A,
        Q4_EXPECTED_MAX_N_B,
    )
    config = analysis.get("configuration", {})
    integer_audit = analysis.get("integer_domain_audit", {})
    analysis_inputs = analysis.get("input_files", {})
    reconciliation = analysis.get("frozen_record_reconciliation", {})
    if (
        analysis.get("kind") != "q4_confirmation_integer_domain_analysis"
        or analysis.get("audit_status") != "passed"
        or config.get("integer_domain_shape")
        != [Q4_EXPECTED_MAX_N_A + 1, Q4_EXPECTED_MAX_N_B + 1]
        or int(config.get("trial_count", -1)) != Q4_EXPECTED_TRIALS
        or integer_audit.get("counts_sha256") != sha256_file(counts_path)
        or not integer_audit.get("passed")
        or analysis_inputs.get("freeze", {}).get("sha256") != final.get("freeze_sha256")
        or analysis_inputs.get("final_summary", {}).get("sha256") is None
        or reconciliation.get("confirmation_json_sha256")
        != final.get("confirmation_json_sha256")
        or reconciliation.get("confirmation_csv_sha256")
        != final.get("confirmation_csv_sha256")
        or not reconciliation.get("passed")
    ):
        raise ValueError("Q4 independent integer-domain audit is incomplete or inconsistent")

    frontier_png = output_root / "图件" / "q4_cost_frontier.png"
    frontier = json.loads(
        (output_root / "图件" / "q4_cost_frontier.audit.json").read_text(encoding="utf-8")
    )
    result_status = str(final.get("result_status"))
    excluded_count = int(final.get("excluded_frontier_count", -1))
    unresolved_count = int(final.get("not_excluded_frontier_count", -1))
    globally_certified = result_status == "globally_certified_minimum_cost"
    all_cheaper_excluded = bool(final.get("all_strictly_cheaper_maximal_designs_excluded"))
    if (
        frontier.get("kind") != "q4_cost_frontier_figure_audit"
        or frontier.get("screening_sha256") != source_screening.get("json_sha256")
        or frontier.get("final_sha256")
        != analysis_inputs.get("final_summary", {}).get("sha256")
        or frontier.get("freeze_sha256") != final.get("freeze_sha256")
        or frontier.get("integer_domain_counts_sha256")
        != sha256_file(screening_counts_path)
        or frontier.get("integer_domain_shape")
        != screening_counts_audit["shape"]
        or int(frontier.get("fixed_trial_count", -1)) != Q4_SCREENING_TRIALS
        or not frontier.get("monotonicity", {}).get("passed")
        or frontier.get("output_png_sha256") != sha256_file(frontier_png)
        or frontier.get("result_status") != result_status
        or frontier.get("evidence_scope") != Q4_FRONTIER_EVIDENCE_SCOPE
        or bool(frontier.get("global_minimum_certified")) != globally_certified
        or int(frontier.get("unresolved_cheaper_design_count", -1)) != unresolved_count
        or int(frontier.get("excluded_frontier_count", -1)) != excluded_count
        or int(frontier.get("not_excluded_frontier_count", -1)) != unresolved_count
        or globally_certified != all_cheaper_excluded
        or (globally_certified and unresolved_count != 0)
    ):
        raise ValueError("Q4 cost-frontier figure audit is incomplete or inconsistent")

    assets = json.loads(
        (output_root / "三维模型" / "q4_final_assets.audit.json").read_text(encoding="utf-8")
    )
    if (
        assets.get("kind") != "q4_final_3d_assets_audit"
        or assets.get("status") != "passed"
        or assets.get("design")
        != {"n_a": int(reported_design["n_a"]), "n_b": int(reported_design["n_b"])}
        or int(assets.get("geometry_counts", {}).get("a_source_particles", -1))
        != int(reported_design["n_a"])
        or int(assets.get("geometry_counts", {}).get("b_source_particles", -1))
        != int(reported_design["n_b"])
        or int(assets.get("witness", {}).get("same_source_edges", -1)) != 0
    ):
        raise ValueError("Q4 formal FreeCAD asset audit is incomplete or inconsistent")
    scene = json.loads(
        (output_root / "三维模型" / "q4_final_scene.json").read_text(encoding="utf-8")
    )
    traceability = scene.get("traceability", {})
    if (
        scene.get("publication_status") != "final_random_trial_geometry"
        or traceability.get("design_counts") != assets.get("design")
        or traceability.get("geometry_counts") != assets.get("geometry_counts")
        or len(scene.get("cylinders", [])) + len(scene.get("spheres", []))
        != int(assets.get("geometry_counts", {}).get("all_fragments", -1))
    ):
        raise ValueError("Q4 sanitized 3D scene does not match the formal asset audit")
    asset_paths = {
        "model": output_root / "三维模型" / "q4_final.FCStd",
        "axonometric": output_root / "三维模型" / "q4_final_axonometric.png",
        "top": output_root / "三维模型" / "q4_final_top.png",
        "witness_png": output_root / "三维模型" / "q4_final_witness_focus.png",
    }
    asset_records = assets.get("artifacts", {})
    for name, path in asset_paths.items():
        record = asset_records.get(name, {})
        if (
            record.get("sha256") != sha256_file(path)
            or int(record.get("size_bytes", -1)) != path.stat().st_size
        ):
            raise ValueError(f"Q4 formal 3D asset hash or size mismatch: {name}")


def validate_explanatory_payload(output_root: Path) -> None:
    audit = json.loads(
        (output_root / "图件" / "explanatory_figures.audit.json").read_text(
            encoding="utf-8"
        )
    )
    outputs = audit.get("outputs", {})
    expected = {
        "论文/figures/generated/model_workflow.png": output_root
        / "图件"
        / "model_workflow.png",
        "论文/figures/generated/validation_diagnostics.png": output_root
        / "图件"
        / "validation_diagnostics.png",
    }
    if audit.get("kind") != "explanatory_figures_audit":
        raise ValueError("Explanatory figure audit kind is invalid")
    for relative, path in expected.items():
        if outputs.get(relative, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"Explanatory figure hash mismatch: {relative}")

    figures = output_root / "图件"
    summaries = output_root / "结果摘要"
    data = output_root / "数据"
    q4_boundary_audit = json.loads(
        (figures / "q4_unresolved_boundary_evidence.audit.json").read_text(
            encoding="utf-8"
        )
    )
    convergence_audit = json.loads(
        (figures / "simulation_convergence.audit.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (data / "q4_confirmation_integer_domain_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    final = json.loads((summaries / "q4_summary.json").read_text(encoding="utf-8"))
    freeze = json.loads(
        (summaries / "q4_confirmation_freeze.json").read_text(encoding="utf-8")
    )
    q2_summary = json.loads(
        (summaries / "q2_summary.json").read_text(encoding="utf-8")
    )
    validate_q4_boundary_figure_audit(
        q4_boundary_audit,
        analysis,
        final,
        freeze,
        figures / "q4_unresolved_boundary_evidence.png",
    )
    validate_simulation_convergence_figure_audit(
        convergence_audit,
        q2_summary,
        analysis,
        figures / "simulation_convergence.png",
    )


def validate_generated_tree(output_root: Path, expected_files: set[str]) -> None:
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in support materials: {path}")
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    if actual != expected_files:
        raise ValueError(
            f"Generated support tree differs from the allowlist: "
            f"extra={sorted(actual - expected_files)}, missing={sorted(expected_files - actual)}"
        )


def support_readme() -> str:
    return """# A题支撑材料

本附件仅含论文使用的源程序、匿名化结果摘要、压缩试验样本、正式图件和 FreeCAD 模型，不重复附入赛题原始附件、运行分片、缓存或内部诊断文档。

## 目录

- `提交源码/`：压缩包根目录下的全部主程序和绘图程序，每个文件含 AI 工具说明文件头。
- `支撑材料内容/结果摘要/`：Q1-Q4 机器可读结论与冻结配置。
- `支撑材料内容/数据/`：Q2/Q3 首次导通阈值压缩数组及 Q4 二维整数域计数。
- `支撑材料内容/三维模型/`：FreeCAD `FCStd` 模型、场景数据、渲染图和审计。
- `支撑材料内容/图件/`：论文定量图与生成审计。

## 复现

环境为 Python 3.13；依赖见压缩包根目录 `requirements.txt`。先检查项目级入口及压缩数组：

```powershell
python 提交源码/run_pipeline.py --help
python -c "import numpy as np; z=np.load(r'支撑材料内容/数据/q2_threshold_samples.npz',allow_pickle=False); print([(z['first_connection_index']<=n).mean() for n in (354,424,495,707)])"
python -c "import numpy as np; z=np.load(r'支撑材料内容/数据/q3_confirmation_threshold_samples.npz',allow_pickle=False); print((z['first_connection_index']<=616).mean())"
python -c "import numpy as np; z=np.load(r'支撑材料内容/数据/q4_confirmation_integer_domain_counts.npz',allow_pickle=False); a=z['success_counts']; assert a.dtype==np.int32 and a.shape==(620,5484) and int(z['trials'])==50000; assert (np.diff(a,axis=0)>=0).all() and (np.diff(a,axis=1)>=0).all(); print(a.shape,int(a.min()),int(a.max()))"
```

三个数据命令应分别复算为问题二四个概率、问题三 `N_A=616` 的概率，以及问题四完整整数域矩阵形状和计数范围。Q2/Q3 的 NPZ 文件包含 `trial_id`、`first_connection_index` 和 `censored`；同名 metadata JSON 给出冻结配置、源文件哈希、数组哈希和复算公式。Q4 的完整域计数数组按 `N_A=0..619`、`N_B=0..5483` 排列。

`图件/` 中的 Q4 未决边界图同时保留 573 个已排除点与 46 个未决点的分类审计；未决只表示“尚不能排除”，不等同于已确认可行。收敛图固定使用 Q2 的 20000 次样本和 Q4 的 50000 次确认样本，其中中间检查点仅作诊断，正式结论只取冻结终点。

`SHA256SUMS.txt` 列出 `支撑材料内容/` 内全部文件的 SHA-256；`提交源码/source-manifest.json` 列出全部提交源码及哈希。打包程序只接收与这两份清单完全一致的文件树。
"""


def write_checksums(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}")
    (output_root / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def validate_checksum_tree(output_root: Path) -> list[Path]:
    manifest_path = output_root / "SHA256SUMS.txt"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    reject_private_text(manifest_path.read_text(encoding="utf-8"), "Support checksum manifest")
    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in support materials: {path}")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9A-F]{64}", parts[0]):
            raise ValueError(f"Invalid SHA256SUMS line {line_number}")
        relative = normalized_relative_path(parts[1], f"SHA256SUMS path {line_number}")
        if relative == "SHA256SUMS.txt" or relative in expected:
            raise ValueError(f"Duplicate or recursive SHA256SUMS path: {relative}")
        expected[relative] = parts[0]
    actual = {
        path.relative_to(output_root).as_posix(): path
        for path in output_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(expected):
        raise ValueError(
            f"Support checksum whitelist mismatch: extra={sorted(set(actual) - set(expected))}, "
            f"missing={sorted(set(expected) - set(actual))}"
        )
    for relative, path in actual.items():
        if path.is_symlink() or sha256_file(path) != expected[relative]:
            raise ValueError(f"Support payload hash mismatch: {relative}")
    return [manifest_path, *(actual[relative] for relative in sorted(actual))]


def validate_submission_tree(submission_root: Path, project_root: Path) -> list[Path]:
    manifest_path = submission_root / "source-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    for path in submission_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in submission sources: {path}")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    reject_private_text(manifest_text, "Submission source manifest")
    manifest = json.loads(manifest_text)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "1.0"
        or manifest.get("status") != "frozen"
        or manifest.get("submission_root") != "提交源码"
    ):
        raise ValueError("Submission source manifest is not frozen or has an invalid schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Submission source manifest has no files")
    expected: dict[str, tuple[Path, str]] = {}
    for index, item in enumerate(files, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Submission manifest item {index} is not an object")
        archive_relative = normalized_relative_path(
            str(item.get("submission_path") or ""), f"Submission path {index}"
        )
        prefix = "提交源码/"
        if not archive_relative.startswith(prefix):
            raise ValueError(f"Submission path {index} is outside 提交源码")
        relative = archive_relative[len(prefix) :]
        if relative == "source-manifest.json" or relative in expected:
            raise ValueError(f"Duplicate submission path: {relative}")
        path = (project_root / Path(*PurePosixPath(archive_relative).parts)).resolve()
        try:
            path.relative_to(submission_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Submission source escapes 提交源码: {archive_relative}") from exc
        expected[relative] = (path, str(item.get("submission_sha256") or "").upper())
    actual = {
        path.relative_to(submission_root).as_posix(): path
        for path in submission_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(actual) != set(expected):
        raise ValueError(
            f"Submission source whitelist mismatch: extra={sorted(set(actual) - set(expected))}, "
            f"missing={sorted(set(expected) - set(actual))}"
        )
    for relative, (path, expected_hash) in expected.items():
        if path.is_symlink() or path != actual[relative] or sha256_file(path) != expected_hash:
            raise ValueError(f"Submission source hash mismatch: {relative}")
        reject_private_home_text(
            path.read_text(encoding="utf-8"), f"Submission source {relative}"
        )
    return [manifest_path, *(actual[relative] for relative in sorted(actual))]


def prepare_support_materials(config_path: Path, project_root: Path) -> Path:
    project_root = project_root.resolve()
    config = validate_support_config(json.loads(config_path.read_text(encoding="utf-8")))
    if config.get("status") != "frozen":
        raise ValueError("Support config is not frozen; complete Q4 before building the payload")
    validate_q4_source_bindings(config, project_root)
    validate_explanatory_source_bindings(config, project_root)
    output_root = (project_root / str(config.get("output_dir"))).resolve()
    if output_root.parent != project_root or output_root.name != "支撑材料内容":
        raise ValueError("Support output must be the project-root 支撑材料内容 directory")
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)

    try:
        staging.mkdir()
        expected_destinations: set[str] = set()
        q1 = config.get("q1_extract", {})
        q1_source, _ = project_source(project_root, str(q1.get("source", "")))
        q1_destination, q1_relative = safe_destination(
            staging, str(q1.get("destination", ""))
        )
        extract_q1(q1_source, q1_destination, project_root)
        expected_destinations.add(q1_relative)

        for item in config.get("threshold_samples", []):
            source, _ = project_source(project_root, str(item["source"]))
            destination, relative = safe_destination(staging, str(item["destination"]))
            metadata_destination, metadata_relative = safe_destination(
                staging, str(item["metadata_destination"])
            )
            compact_threshold_samples(source, destination, metadata_destination, project_root)
            expected_destinations.update({relative, metadata_relative})

        for item in config.get("files", []):
            source, _ = project_source(project_root, str(item["source"]))
            destination, relative = safe_destination(staging, str(item["destination"]))
            copy_entry(source, destination, str(item["mode"]), project_root)
            expected_destinations.add(relative)

        (staging / "README.md").write_text(support_readme(), encoding="utf-8", newline="\n")
        expected_destinations.add("README.md")
        text_suffixes = {".json", ".md", ".txt", ".csv", ".tex", ".py"}
        for text_path in staging.rglob("*"):
            if text_path.is_file() and text_path.suffix.lower() in text_suffixes:
                reject_private_text(text_path.read_text(encoding="utf-8"), str(text_path))
        validate_q4_payload(staging)
        validate_explanatory_payload(staging)
        for model in (staging / "三维模型").glob("*.FCStd"):
            validate_fcstd_anonymity(model)
        validate_generated_tree(staging, expected_destinations)
        write_checksums(staging)
        validate_checksum_tree(staging)
        os.replace(staging, output_root)
        return output_root
    except Exception:
        if staging.exists() and staging.parent == project_root:
            shutil.rmtree(staging)
        raise


def package_zip(project_root: Path, team_id: str, output: Path | None = None) -> Path:
    project_root = project_root.resolve()
    if not TEAM_ID_PATTERN.fullmatch(team_id):
        raise ValueError("Team ID must be CM followed by exactly seven digits")
    submission_root = project_root / "提交源码"
    support_root = project_root / "支撑材料内容"
    roots = [submission_root, support_root]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    requirements = project_root / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    archive = (output or project_root / f"A{team_id}附件.zip").resolve()
    if archive.parent != project_root or archive.suffix.lower() != ".zip":
        raise ValueError("Support ZIP must stay in the project root and use the .zip extension")
    if archive.exists():
        raise FileExistsError(archive)

    reject_private_text(requirements.read_text(encoding="utf-8"), "requirements.txt")
    source_entries = validate_submission_tree(submission_root, project_root)
    support_entries = validate_checksum_tree(support_root)
    entries = [requirements, *source_entries, *support_entries]
    expected_names = sorted(path.relative_to(project_root).as_posix() for path in entries)
    staging_archive = archive.with_name(f".{archive.name}.staging-{os.getpid()}")
    if staging_archive.exists():
        raise FileExistsError(staging_archive)
    try:
        with zipfile.ZipFile(
            staging_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as bundle:
            for path in sorted(entries, key=lambda item: item.relative_to(project_root).as_posix()):
                relative = path.relative_to(project_root).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(2026, 8, 8, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(
                    info,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        if staging_archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Support archive exceeds the 20 MB competition limit")
        with zipfile.ZipFile(staging_archive) as bundle:
            bad = bundle.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt ZIP member: {bad}")
            names = bundle.namelist()
            unsafe = [
                name
                for name in names
                if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            ]
            if unsafe:
                raise ValueError(f"ZIP contains unsafe member paths: {unsafe}")
            if names != expected_names or len(names) != len(set(names)):
                raise ValueError("ZIP member list differs from the two frozen file manifests")
        os.replace(staging_archive, archive)
        return archive
    except Exception:
        if staging_archive.exists() and staging_archive.parent == project_root:
            staging_archive.unlink()
        raise


def default_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    return project_root, project_root / "支撑材料配置.json"


def parse_args() -> argparse.Namespace:
    project_root, config = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--config", default=str(config))
    parser.add_argument("--team-id")
    parser.add_argument("--package-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root)
    try:
        if not args.package_only:
            output = prepare_support_materials(Path(args.config), project_root)
            print(f"Prepared: {output}")
        if args.team_id:
            archive = package_zip(project_root, args.team_id)
            print(f"Packaged: {archive} ({archive.stat().st_size} bytes)")
    except Exception as exc:
        print(f"[prepare_support_materials] ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
