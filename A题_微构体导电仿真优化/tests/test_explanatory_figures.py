from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "论文" / "figures" / "src" / "build_explanatory_figures.py"
SPEC = importlib.util.spec_from_file_location("build_explanatory_figures", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_q4_evidence_tree(root: Path) -> Path:
    candidate = {
        "role": "candidate",
        "n_a": 619,
        "n_b": 0,
        "successes": 45554,
        "trials": 50000,
        "estimate": 0.91108,
        "cost_weight": 350973,
        "cost_yuan": 9.188451653403087,
        "clopper_pearson_one_sided_lower": 0.9061863570173373,
        "proof_status": "candidate_statistically_feasible",
    }
    cheaper = []
    for n_a in range(619):
        unresolved = n_a >= 573
        cheaper.append(
            {
                "role": "strictly_cheaper_maximal",
                "n_a": n_a,
                "n_b": 618 - n_a,
                "cost_weight": 350000 + (n_a % 40),
                "cost_yuan": 9.16 + 0.00004 * n_a,
                "clopper_pearson_one_sided_upper": (
                    0.9002 + 0.00012 * (n_a - 573)
                    if unresolved
                    else 0.86 + 0.0000695 * n_a
                ),
                "proof_status": (
                    "strictly_cheaper_design_not_excluded"
                    if unresolved
                    else "strictly_cheaper_design_excluded"
                ),
            }
        )
    unresolved_records = [record for record in cheaper if record["n_a"] >= 573]
    interval = {
        "lower_cost_weight": 324891,
        "upper_cost_weight": 350973,
        "lower_cost_yuan": 8.505626490145346,
        "upper_cost_yuan": 9.188451653403087,
        "minimum_not_excluded_design": [573, 0],
    }
    freeze = {
        "kind": "q4_confirmation_freeze",
        "confirmation_protocol": {"bonferroni_statement_count": 620},
    }
    freeze_path = root / "freeze.json"
    write_json(freeze_path, freeze)
    summary = {
        "kind": "q4_final_summary",
        "result_status": "lowest_statistically_feasible_cost",
        "candidate_statistically_feasible": True,
        "all_strictly_cheaper_maximal_designs_excluded": False,
        "excluded_frontier_count": 573,
        "not_excluded_frontier_count": 46,
        "confirmation_records": [candidate, *cheaper],
        "cost_uncertainty_interval": interval,
        "not_excluded_frontier": [
            {"n_a": record["n_a"], "n_b": record["n_b"]}
            for record in unresolved_records
        ],
    }
    summary_path = root / "summary.json"
    write_json(summary_path, summary)
    merged_path = root / "merged.json"
    write_json(merged_path, {"kind": "synthetic-placeholder"})
    analysis = {
        "kind": "q4_confirmation_integer_domain_analysis",
        "audit_status": "passed",
        "result_status": "lowest_statistically_feasible_cost",
        "boundary_contract": {"mode": "D"},
        "configuration": {
            "trial_count": 50000,
            "fingerprint": "SYNTHETIC",
            "bonferroni_statement_count": 620,
            "per_statement_confidence": 1.0 - 0.05 / 620,
        },
        "statistical_results": {
            "candidate": {key: value for key, value in candidate.items() if key != "role"},
            "excluded_frontier_count": 573,
            "not_excluded_frontier_count": 46,
            "cost_uncertainty_interval": interval,
        },
        "input_files": {
            "freeze": {"path": str(freeze_path), "sha256": FIGURES.sha256(freeze_path)},
            "final_summary": {"path": str(summary_path), "sha256": FIGURES.sha256(summary_path)},
            "merged_pareto_frontier": {
                "path": str(merged_path),
                "sha256": FIGURES.sha256(merged_path),
            },
        },
    }
    analysis_path = root / "analysis.json"
    write_json(analysis_path, analysis)
    return analysis_path


def test_workflow_uses_distinct_visual_encodings(tmp_path: Path) -> None:
    figure = FIGURES.build_workflow_figure()
    texts = {text.get_text() for text in figure.axes[0].texts}
    assert {"几何输入", "统一接触图引擎", "四问证据输出", "验证回路"} <= texts
    assert {"Q1", "Q2", "Q3", "Q4"} <= texts
    output = tmp_path / "workflow.png"
    figure.savefig(output, dpi=180)
    assert output.stat().st_size > 20_000
    FIGURES.plt.close(figure)


def test_q4_boundary_loader_and_figure_preserve_unresolved_semantics(
    tmp_path: Path,
) -> None:
    analysis_path = make_q4_evidence_tree(tmp_path)
    evidence = FIGURES.load_q4_boundary_evidence(
        analysis_path,
        project_root=tmp_path,
    )
    assert len(evidence["excluded"]) == 573
    assert len(evidence["unresolved"]) == 46
    figure, audit = FIGURES.build_q4_boundary_figure(evidence)
    assert audit["unresolved_semantics"] == "not_excluded_not_confirmed_feasible"
    assert audit["next_round_focus"] == "46_not_excluded_maximal_designs_only"
    assert audit["cost_interval_yuan"] == pytest.approx(
        [8.505626490145346, 9.188451653403087]
    )
    output = tmp_path / "q4-boundary.png"
    figure.savefig(output, dpi=180)
    assert output.stat().st_size > 20_000
    FIGURES.plt.close(figure)

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["statistical_results"]["excluded_frontier_count"] = 572
    write_json(analysis_path, analysis)
    with pytest.raises(ValueError, match="573/46"):
        FIGURES.load_q4_boundary_evidence(analysis_path, project_root=tmp_path)


def test_q4_candidate_sequence_reconstructs_formal_success_indicator(
    tmp_path: Path,
) -> None:
    success_count = 45554
    records = [
        {
            "trial_id": trial_id,
            "connectivity_frontier": [[619, 0]] if trial_id < success_count else [],
        }
        for trial_id in range(50000)
    ]
    merged = {
        "kind": "mixed_pareto_frontier_samples",
        "configuration_fingerprint": "SYNTHETIC",
        "trials": 50000,
        "records": records,
    }
    merged_path = tmp_path / "merged.json"
    write_json(merged_path, merged)
    evidence = {
        "merged_path": merged_path,
        "analysis": {
            "configuration": {"fingerprint": "SYNTHETIC", "trial_count": 50000}
        },
        "candidate": {"n_a": 619, "n_b": 0, "successes": success_count},
    }
    sequence = FIGURES.read_q4_candidate_sequence(evidence)
    assert sequence.dtype == np.bool_
    assert len(sequence) == 50000
    assert int(sequence.sum()) == success_count


def test_simulation_convergence_uses_formal_endpoints(tmp_path: Path) -> None:
    q2_counts = [1, 2, 3, 4]
    q2_samples = np.arange(20000, dtype=np.int64) % 5
    q2_full = np.asarray(
        [np.count_nonzero(q2_samples <= count) / len(q2_samples) for count in q2_counts]
    )
    q4_success = np.zeros(50000, dtype=bool)
    q4_success[:45554] = True
    confidence = 1.0 - 0.05 / 620
    official_lower = FIGURES.clopper_pearson_one_sided_bounds(
        45554, 50000, confidence
    )[0]
    evidence = {
        "analysis": {"configuration": {"per_statement_confidence": confidence}},
        "candidate": {
            "estimate": 45554 / 50000,
            "clopper_pearson_one_sided_lower": official_lower,
        },
        "statement_count": 620,
    }
    figure, audit = FIGURES.build_simulation_convergence_figure(
        q2_counts,
        q2_full,
        q2_samples,
        q4_success,
        evidence,
    )
    assert audit["q4_successes"] == 45554
    assert audit["q4_final_estimate"] == pytest.approx(0.91108)
    assert audit["q4_final_cp_lower"] == pytest.approx(official_lower)
    assert audit["interpretation_guard"].endswith("final_n50000_authoritative")
    output = tmp_path / "convergence.png"
    figure.savefig(output, dpi=180)
    assert output.stat().st_size > 20_000
    FIGURES.plt.close(figure)
