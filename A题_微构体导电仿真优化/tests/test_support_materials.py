from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "论文" / "src" / "prepare_support_materials.py"
SPEC = importlib.util.spec_from_file_location("prepare_support_materials", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)

SUBMISSION_MODULE_PATH = PROJECT_ROOT / "论文" / "src" / "prepare_submission_sources.py"
SUBMISSION_SPEC = importlib.util.spec_from_file_location(
    "prepare_submission_sources", SUBMISSION_MODULE_PATH
)
assert SUBMISSION_SPEC is not None and SUBMISSION_SPEC.loader is not None
submission = importlib.util.module_from_spec(SUBMISSION_SPEC)
sys.path.insert(0, str(SUBMISSION_MODULE_PATH.parent))
try:
    SUBMISSION_SPEC.loader.exec_module(submission)
finally:
    sys.path.pop(0)


def create_packagable_tree(root: Path) -> None:
    submission = root / "提交源码"
    payload = root / "支撑材料内容"
    submission.mkdir()
    payload.mkdir()
    source = submission / "main.py"
    source.write_text("print('ok')\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "status": "frozen",
        "submission_root": "提交源码",
        "files": [
            {
                "submission_path": "提交源码/main.py",
                "submission_sha256": support.sha256_file(source),
            }
        ],
    }
    (submission / "source-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (payload / "README.md").write_text("ok\n", encoding="utf-8")
    support.write_checksums(payload)
    (root / "requirements.txt").write_text("numpy\n", encoding="utf-8")


def create_explanatory_payload(root: Path) -> None:
    copies = {
        "问题/问题2/results/D_primary_n20000/q2_summary.json": "结果摘要/q2_summary.json",
        "问题/问题4/results/D_screen2000_confirm50000/q4_confirmation_freeze.json": "结果摘要/q4_confirmation_freeze.json",
        "问题/问题4/results/D_screen2000_confirm50000/q4_summary.json": "结果摘要/q4_summary.json",
        "问题/问题4/results/D_screen2000_confirm50000/q4_confirmation_integer_domain_analysis.json": "数据/q4_confirmation_integer_domain_analysis.json",
        "论文/figures/generated/model_workflow.png": "图件/model_workflow.png",
        "论文/figures/generated/validation_diagnostics.png": "图件/validation_diagnostics.png",
        "论文/figures/generated/explanatory_figures.audit.json": "图件/explanatory_figures.audit.json",
        "论文/figures/generated/q4_unresolved_boundary_evidence.png": "图件/q4_unresolved_boundary_evidence.png",
        "论文/figures/generated/q4_unresolved_boundary_evidence.audit.json": "图件/q4_unresolved_boundary_evidence.audit.json",
        "论文/figures/generated/simulation_convergence.png": "图件/simulation_convergence.png",
        "论文/figures/generated/simulation_convergence.audit.json": "图件/simulation_convergence.audit.json",
    }
    for source, destination in copies.items():
        target = root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / source, target)


def test_project_support_config_is_strict_in_current_lifecycle_state() -> None:
    config = json.loads((PROJECT_ROOT / "支撑材料配置.json").read_text(encoding="utf-8"))
    validated = support.validate_support_config(config)
    assert validated["status"] == "frozen"
    destinations = {item["destination"] for item in validated["files"]}
    assert support.REQUIRED_Q4_DESTINATIONS <= destinations
    assert support.REQUIRED_EXPLANATORY_DESTINATIONS <= destinations
    assert {
        "图件/q4_unresolved_boundary_evidence.png",
        "图件/q4_unresolved_boundary_evidence.audit.json",
        "图件/simulation_convergence.png",
        "图件/simulation_convergence.audit.json",
    } <= destinations
    assert not any(path.startswith(("测试/", "审计/")) for path in destinations)
    support.validate_q4_source_bindings(validated, PROJECT_ROOT)
    support.validate_explanatory_source_bindings(validated, PROJECT_ROOT)


def test_submission_allowlist_is_ready_to_freeze(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "论文" / "src" / "source_appendix_allowlist.json"
    allowlist = json.loads(source.read_text(encoding="utf-8"))
    assert allowlist["status"] == "frozen"
    allowlist["status"] = "frozen"
    frozen_copy = tmp_path / "source_appendix_allowlist.json"
    frozen_copy.write_text(
        json.dumps(allowlist, ensure_ascii=False), encoding="utf-8"
    )

    validated = submission.load_allowlist(frozen_copy, PROJECT_ROOT)
    q1 = next(item for item in validated["files"] if item["path"] == submission.Q1_SOURCE)
    assert len(q1.get("replacements", [])) == submission.Q1_REPLACEMENT_COUNT
    appendix_paths = [
        item["path"]
        for item in validated["files"]
        if item.get("include_in_appendix", True)
    ]
    assert appendix_paths == [
        "问题/问题1/src/solve.py",
        "问题/问题2/src/solve.py",
        "问题/问题3/src/solve.py",
        "问题/问题4/src/solve.py",
    ]
    assert len(validated["files"]) == 24
    assert validated["files"][-1]["include_in_appendix"] is False


def test_submission_sources_have_only_required_appendix_comments() -> None:
    allowlist_path = PROJECT_ROOT / "论文" / "src" / "source_appendix_allowlist.json"
    validated = submission.load_allowlist(allowlist_path, PROJECT_ROOT)
    forbidden_markers = ("# 模块", "# noqa", "# type: ignore", "# pragma:")
    expected_appendix_comments = {
        "问题/问题1/src/solve.py": [
            "# 附件行按原尺寸有限平底圆柱建模，短段不延长或补齐。",
            "# 六变量凸约束优化独立复核 GJK 的窄相阈值分类。",
            "# 胶囊宽相只作排除，候选均由平底圆柱 GJK 距离界判定。",
            "# 并查集判定贯通，路径搜索仅恢复逐边见证。",
        ],
        "问题/问题2/src/solve.py": ["# 四个填充量共享同一批首次导通样本。"],
        "问题/问题3/src/solve.py": [],
        "问题/问题4/src/solve.py": [],
    }

    for item in validated["files"]:
        path = str(item["path"])
        text = (PROJECT_ROOT / path).read_text(encoding="utf-8")
        assert not any(marker in text for marker in forbidden_markers), path
        if item.get("include_in_appendix", True):
            comments = [
                line.strip()
                for line in text.splitlines()
                if line.lstrip().startswith("#")
            ]
            assert comments == expected_appendix_comments[path]


def test_support_config_rejects_unknown_key_or_incomplete_q4() -> None:
    config = json.loads((PROJECT_ROOT / "支撑材料配置.json").read_text(encoding="utf-8"))
    unknown = deepcopy(config)
    unknown["directories"] = ["tmp"]
    with pytest.raises(ValueError, match="keys mismatch"):
        support.validate_support_config(unknown)
    incomplete = deepcopy(config)
    incomplete["files"] = [
        item
        for item in incomplete["files"]
        if item["destination"] != "数据/q4_confirmation_integer_domain_counts.npz"
    ]
    with pytest.raises(ValueError, match="Q4 formal support files are incomplete"):
        support.validate_support_config(incomplete)

    incomplete_figures = deepcopy(config)
    incomplete_figures["files"] = [
        item
        for item in incomplete_figures["files"]
        if item["destination"] != "图件/simulation_convergence.audit.json"
    ]
    with pytest.raises(ValueError, match="Paper explanatory figures are incomplete"):
        support.validate_support_config(incomplete_figures)


def test_figure_source_binding_uses_current_audit_hashes(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "input.json"
    png_path = tmp_path / "paper" / "figure.png"
    audit_path = tmp_path / "paper" / "figure.audit.json"
    input_path.parent.mkdir()
    png_path.parent.mkdir()
    input_path.write_text('{"value": 1}\n', encoding="utf-8")
    png_path.write_bytes(b"first-render")

    def write_audit() -> None:
        audit_path.write_text(
            json.dumps(
                {
                    "kind": "test_figure_audit",
                    "inputs": {"data/input.json": support.sha256_file(input_path)},
                    "outputs": {
                        "paper/figure.png": {"sha256": support.sha256_file(png_path)}
                    },
                }
            ),
            encoding="utf-8",
        )

    write_audit()
    support.validate_figure_audit_source_binding(
        audit_path,
        tmp_path,
        expected_kind="test_figure_audit",
        expected_inputs={"data/input.json"},
        expected_png_relative="paper/figure.png",
        configured_png=png_path,
    )

    png_path.write_bytes(b"re-rendered-layout")
    write_audit()
    support.validate_figure_audit_source_binding(
        audit_path,
        tmp_path,
        expected_kind="test_figure_audit",
        expected_inputs={"data/input.json"},
        expected_png_relative="paper/figure.png",
        configured_png=png_path,
    )

    input_path.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        support.validate_figure_audit_source_binding(
            audit_path,
            tmp_path,
            expected_kind="test_figure_audit",
            expected_inputs={"data/input.json"},
            expected_png_relative="paper/figure.png",
            configured_png=png_path,
        )


def test_new_figure_payload_rejects_q4_semantic_drift(tmp_path: Path) -> None:
    create_explanatory_payload(tmp_path)
    support.validate_explanatory_payload(tmp_path)

    audit_path = tmp_path / "图件" / "q4_unresolved_boundary_evidence.audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["not_excluded_frontier_count"] = 45
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="Q4 unresolved-boundary"):
        support.validate_explanatory_payload(tmp_path)


def test_new_figure_payload_rejects_intermediate_checkpoint_claim(
    tmp_path: Path,
) -> None:
    create_explanatory_payload(tmp_path)
    audit_path = tmp_path / "图件" / "simulation_convergence.audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["interpretation_guard"] = "intermediate_checkpoints_are_authoritative"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="Simulation-convergence"):
        support.validate_explanatory_payload(tmp_path)


def test_scrub_value_rewrites_project_absolute_path() -> None:
    absolute = PROJECT_ROOT / "问题" / "问题2" / "results" / "q2_summary.json"
    cleaned = support.scrub_value({"path": str(absolute)}, PROJECT_ROOT)
    assert "Users" not in cleaned["path"]
    assert cleaned["path"].replace("\\", "/") == "问题/问题2/results/q2_summary.json"


def test_scrub_value_rejects_external_absolute_path() -> None:
    private_path = "D:" + "\\external\\result.json"
    with pytest.raises(ValueError, match="Absolute Windows path"):
        support.scrub_value({"path": private_path}, PROJECT_ROOT)


def test_sanitized_csv_rewrites_project_paths_without_changing_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "destination.csv"
    project_path = PROJECT_ROOT / "问题" / "问题4" / "result.json"
    source.write_text(
        f'trials,path,note\n50000,"{project_path}",ok\n', encoding="utf-8"
    )
    support.copy_entry(source, destination, "sanitized_csv", PROJECT_ROOT)
    with destination.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "trials": "50000",
            "path": "问题/问题4/result.json",
            "note": "ok",
        }
    ]
    assert "Users" not in destination.read_text(encoding="utf-8")


def test_extract_q1_keeps_only_formal_independent_fragment_results(tmp_path: Path) -> None:
    formal = [
        {
            "scenario": "A_row_literal",
            "sheet": sheet,
            "internal_mode": "disconnected_fragments",
            "conclusion": conclusion,
            "internal_edges_enabled": 0,
            "periodic_junction_count": 0,
        }
        for sheet, conclusion in (
            ("组1", "nonconductive"),
            ("组2", "conductive"),
            ("组3", "conductive"),
        )
    ]
    diagnostic = {
        **formal[0],
        "internal_mode": "connected_same_particle",
        "internal_edges_enabled": 1,
    }
    source = tmp_path / "q1.json"
    source.write_text(
        json.dumps(
            {
                "metadata": {"constants": {"cutoff_nm": 1.8}, "input_hashes": {}},
                "screening": [{"sheet": sheet} for sheet in ("组1", "组2", "组3")],
                "scenario_results": [diagnostic, *formal],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "q1-formal.json"
    support.extract_q1(source, destination, tmp_path)
    text = destination.read_text(encoding="utf-8")
    result = json.loads(text)
    assert [item["sheet"] for item in result["groups"]] == ["组1", "组2", "组3"]
    assert all(item["internal_mode"] == "disconnected_fragments" for item in result["groups"])
    assert "connected_same_particle" not in text


def test_compact_threshold_samples_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "threshold.json"
    source.write_text(
        json.dumps(
            {
                "kind": "microstructure_threshold_samples",
                "configuration": {"trial_count": 3, "max_count": 5},
                "configuration_fingerprint": "ABC",
                "records": [
                    {"trial_id": 0, "first_connection_index": 2, "censored": False},
                    {"trial_id": 1, "first_connection_index": 6, "censored": True},
                    {"trial_id": 2, "first_connection_index": 4, "censored": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "threshold.npz"
    metadata = tmp_path / "threshold.metadata.json"
    support.compact_threshold_samples(source, output, metadata, tmp_path)

    with np.load(output) as arrays:
        assert arrays["trial_id"].tolist() == [0, 1, 2]
        assert arrays["first_connection_index"].tolist() == [2, 6, 4]
        assert arrays["censored"].tolist() == [False, True, False]
        assert [(arrays["first_connection_index"] <= count).mean() for count in (2, 4, 5)] == [
            pytest.approx(1 / 3),
            pytest.approx(2 / 3),
            pytest.approx(2 / 3),
        ]
    audit = json.loads(metadata.read_text(encoding="utf-8"))
    assert audit["record_count"] == 3
    assert audit["censored_trials"] == 1
    assert audit["recomputation"]["censored_sentinel"] == 6
    assert audit["npz_sha256"] == support.sha256_file(output)


def test_validate_integer_domain_counts_checks_shape_and_monotonicity(tmp_path: Path) -> None:
    path = tmp_path / "counts.npz"
    counts = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    np.savez_compressed(path, success_counts=counts, trials=3, max_n_a=1, max_n_b=2)
    audit = support.validate_integer_domain_counts(path, 3, 1, 2)
    assert audit == {
        "shape": [2, 3],
        "dtype": "int32",
        "trials": 3,
        "minimum": 0,
        "maximum": 3,
    }
    broken = counts.copy()
    broken[1, 1] = 0
    np.savez_compressed(path, success_counts=broken, trials=3, max_n_a=1, max_n_b=2)
    with pytest.raises(ValueError, match="monotonicity"):
        support.validate_integer_domain_counts(path, 3, 1, 2)


def test_validate_fcstd_anonymity_scans_every_container_member(tmp_path: Path) -> None:
    model = tmp_path / "model.FCStd"
    document = """<Document>
<Property name="Company"><String value=""/></Property>
<Property name="CreatedBy"><String value=""/></Property>
<Property name="LastModifiedBy"><String value=""/></Property>
</Document>"""
    with zipfile.ZipFile(model, "w") as bundle:
        bundle.writestr("Document.xml", document)
        bundle.writestr("GuiDocument.xml", r"C:\Users\private-user\scene.cache")
    with pytest.raises(ValueError, match="private home path"):
        support.validate_fcstd_anonymity(model)


def test_validate_q4_payload_requires_complete_frozen_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support, "Q4_EXPECTED_TRIALS", 10)
    monkeypatch.setattr(support, "Q4_EXPECTED_MAX_N_A", 1)
    monkeypatch.setattr(support, "Q4_EXPECTED_MAX_N_B", 2)
    monkeypatch.setattr(support, "Q4_SCREENING_TRIALS", 5)
    monkeypatch.setattr(support, "Q4_SCREENING_MAX_N_A", 2)
    monkeypatch.setattr(support, "Q4_SCREENING_MAX_N_B", 4)
    summaries = tmp_path / "结果摘要"
    data = tmp_path / "数据"
    models = tmp_path / "三维模型"
    figures = tmp_path / "图件"
    summaries.mkdir()
    data.mkdir()
    models.mkdir()
    figures.mkdir()
    screening_counts_path = data / "q4_screening_integer_domain_counts.npz"
    screening_counts = np.zeros((3, 5), dtype=np.int32)
    np.savez_compressed(
        screening_counts_path,
        success_counts=screening_counts,
        trials=5,
        max_n_a=2,
        max_n_b=4,
    )
    screening_hash = "SCREENING-JSON-SHA256"
    screening = {
        "kind": "q4_screening_results",
        "fixed_trial_count": 5,
        "maximum_static_graph_design": [2, 4],
        "integer_domain_success_counts_sha256": support.sha256_file(screening_counts_path),
    }
    (summaries / "q4_screening.json").write_text(
        json.dumps(screening), encoding="utf-8"
    )

    freeze_hash = "FREEZE-JSON-SHA256"
    final_hash = "FINAL-SUMMARY-SHA256"
    confirmation_json_hash = "CONFIRMATION-JSON-SHA256"
    candidate = {"n_a": 1, "n_b": 0, "cost_weight": 567}
    freeze = {
        "kind": "q4_confirmation_freeze",
        "confirmation_protocol": {
            "fixed_trial_count": 10,
            "maximum_static_graph_design": [1, 2],
        },
        "confirmation_designs": [{"n_a": value, "n_b": 0} for value in range(2)],
        "candidate_freeze": candidate,
        "source_screening": {
            "json_sha256": screening_hash,
            "fixed_trial_count": 5,
            "maximum_static_graph_design": [2, 4],
        },
    }
    confirmation = {
        "kind": "q4_confirmation_results",
        "freeze_sha256": freeze_hash,
        "records": [{"trials": 10} for _ in range(2)],
    }
    (summaries / "q4_confirmation_freeze.json").write_text(
        json.dumps(freeze), encoding="utf-8"
    )
    (summaries / "q4_confirmation.json").write_text(
        json.dumps(confirmation), encoding="utf-8"
    )
    confirmation_csv = summaries / "q4_confirmation.csv"
    confirmation_csv.write_text("trials\n10\n10\n", encoding="utf-8")
    final = {
        "kind": "q4_final_summary",
        "result_status": "lowest_statistically_feasible_cost",
        "reported_design": candidate,
        "freeze_sha256": freeze_hash,
        "confirmation_json_sha256": confirmation_json_hash,
        "confirmation_csv_sha256": support.sha256_file(confirmation_csv),
        "excluded_frontier_count": 1,
        "not_excluded_frontier_count": 1,
        "all_strictly_cheaper_maximal_designs_excluded": False,
    }
    (summaries / "q4_summary.json").write_text(json.dumps(final), encoding="utf-8")
    counts_path = data / "q4_confirmation_integer_domain_counts.npz"
    counts = np.zeros((2, 3), dtype=np.int32)
    np.savez_compressed(
        counts_path,
        success_counts=counts,
        trials=10,
        max_n_a=1,
        max_n_b=2,
    )
    analysis = {
        "kind": "q4_confirmation_integer_domain_analysis",
        "audit_status": "passed",
        "configuration": {"integer_domain_shape": [2, 3], "trial_count": 10},
        "integer_domain_audit": {
            "passed": True,
            "counts_sha256": support.sha256_file(counts_path),
        },
        "input_files": {
            "freeze": {"sha256": freeze_hash},
            "final_summary": {"sha256": final_hash},
        },
        "frozen_record_reconciliation": {
            "passed": True,
            "confirmation_json_sha256": confirmation_json_hash,
            "confirmation_csv_sha256": support.sha256_file(confirmation_csv),
        },
    }
    (data / "q4_confirmation_integer_domain_analysis.json").write_text(
        json.dumps(analysis), encoding="utf-8"
    )
    frontier_png = figures / "q4_cost_frontier.png"
    frontier_png.write_bytes(b"synthetic-png")
    frontier_audit = {
        "kind": "q4_cost_frontier_figure_audit",
        "screening_sha256": screening_hash,
        "final_sha256": final_hash,
        "freeze_sha256": freeze_hash,
        "integer_domain_counts_sha256": support.sha256_file(screening_counts_path),
        "integer_domain_shape": [3, 5],
        "fixed_trial_count": 5,
        "monotonicity": {"passed": True},
        "result_status": "lowest_statistically_feasible_cost",
        "evidence_scope": support.Q4_FRONTIER_EVIDENCE_SCOPE,
        "global_minimum_certified": False,
        "unresolved_cheaper_design_count": 1,
        "excluded_frontier_count": 1,
        "not_excluded_frontier_count": 1,
        "output_png_sha256": support.sha256_file(frontier_png),
    }
    (figures / "q4_cost_frontier.audit.json").write_text(
        json.dumps(frontier_audit), encoding="utf-8"
    )
    asset_paths = {
        "scene": models / "q4_final_scene.json",
        "model": models / "q4_final.FCStd",
        "axonometric": models / "q4_final_axonometric.png",
        "top": models / "q4_final_top.png",
        "witness_png": models / "q4_final_witness_focus.png",
    }
    scene_payload = {
        "publication_status": "final_random_trial_geometry",
        "traceability": {
            "design_counts": {"n_a": 1, "n_b": 0},
            "geometry_counts": {
                "a_source_particles": 1,
                "b_source_particles": 0,
                "all_fragments": 1,
            },
        },
        "cylinders": [{}],
        "spheres": [],
    }
    asset_paths["scene"].write_text(json.dumps(scene_payload), encoding="utf-8")
    for name, path in asset_paths.items():
        if name != "scene":
            path.write_bytes(f"synthetic-{name}".encode())
    asset_audit = {
        "kind": "q4_final_3d_assets_audit",
        "status": "passed",
        "design": {"n_a": 1, "n_b": 0},
        "geometry_counts": {
            "a_source_particles": 1,
            "b_source_particles": 0,
            "all_fragments": 1,
        },
        "witness": {"same_source_edges": 0},
        "artifacts": {
            name: {"sha256": support.sha256_file(path), "size_bytes": path.stat().st_size}
            for name, path in asset_paths.items()
        },
    }
    (models / "q4_final_assets.audit.json").write_text(
        json.dumps(asset_audit), encoding="utf-8"
    )
    support.validate_q4_payload(tmp_path)

    frontier_audit["integer_domain_counts_sha256"] = support.sha256_file(counts_path)
    (figures / "q4_cost_frontier.audit.json").write_text(
        json.dumps(frontier_audit), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cost-frontier"):
        support.validate_q4_payload(tmp_path)


def test_package_zip_rejects_invalid_team_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seven digits"):
        support.package_zip(tmp_path, "CM123")


def test_package_zip_contains_only_submission_roots(tmp_path: Path) -> None:
    create_packagable_tree(tmp_path)

    archive = support.package_zip(tmp_path, "CM1234567")
    assert archive.stat().st_size < support.MAX_ARCHIVE_BYTES
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert bundle.namelist() == [
            "requirements.txt",
            "提交源码/main.py",
            "提交源码/source-manifest.json",
            "支撑材料内容/README.md",
            "支撑材料内容/SHA256SUMS.txt",
        ]


@pytest.mark.parametrize(("root_name", "extra_name"), [("提交源码", "debug.log"), ("支撑材料内容", "old.json")])
def test_package_zip_rejects_files_outside_frozen_manifests(
    tmp_path: Path, root_name: str, extra_name: str
) -> None:
    create_packagable_tree(tmp_path)
    (tmp_path / root_name / extra_name).write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="whitelist mismatch"):
        support.package_zip(tmp_path, "CM1234567")
    assert not (tmp_path / "ACM1234567附件.zip").exists()


def test_package_zip_removes_oversize_staging_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    create_packagable_tree(tmp_path)
    monkeypatch.setattr(support, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds the 20 MB"):
        support.package_zip(tmp_path, "CM1234567")
    assert not (tmp_path / "ACM1234567附件.zip").exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_prepare_support_materials_rejects_pending_q4_before_writing(
    tmp_path: Path,
) -> None:
    config = json.loads((PROJECT_ROOT / "支撑材料配置.json").read_text(encoding="utf-8"))
    config["status"] = "pending_q4"
    config_path = tmp_path / "support-config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="not frozen"):
        support.prepare_support_materials(config_path, tmp_path)
    assert not (tmp_path / "支撑材料内容").exists()
