from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "论文" / "figures" / "src" / "build_q4_frontier.py"
SPEC = importlib.util.spec_from_file_location("build_q4_frontier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_monotonicity_and_empirical_boundary() -> None:
    counts = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ],
        dtype=np.int64,
    )
    assert MODULE.monotonicity_audit(counts)["passed"]
    boundary = MODULE.empirical_boundary(counts / 5.0, 0.6)
    assert np.allclose(boundary, [3.0, 2.0, 1.0], equal_nan=True)

    broken = counts.copy()
    broken[2, 2] = 0
    audit = MODULE.monotonicity_audit(broken)
    assert not audit["passed"]
    assert audit["n_a_direction_violations"] > 0


def test_load_and_render_synthetic_final_result(tmp_path: Path) -> None:
    trials = 10
    max_n_a = 2
    max_n_b = 12
    weights = (
        567 * np.arange(max_n_a + 1)[:, None]
        + 64 * np.arange(max_n_b + 1)[None, :]
    )
    counts = np.where(weights >= 695, 10, np.where(weights >= 600, 8, 2)).astype(
        np.int32
    )
    counts_path = tmp_path / "counts.npz"
    np.savez_compressed(
        counts_path,
        success_counts=counts,
        trials=np.int64(trials),
        max_n_a=np.int64(max_n_a),
        max_n_b=np.int64(max_n_b),
    )
    candidate = {"n_a": 1, "n_b": 2, "cost_weight": 695, "cost_yuan": 0.0}
    freeze = {
        "kind": "q4_confirmation_freeze",
        "candidate_freeze": candidate,
        "confirmation_protocol": {"bonferroni_statement_count": 3},
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    screening = {
        "kind": "q4_screening_results",
        "boundary_contract": {"mode": "D"},
        "target_probability": 0.9,
        "fixed_trial_count": trials,
        "integer_domain_success_counts": str(counts_path),
        "integer_domain_success_counts_sha256": MODULE.sha256(counts_path),
    }
    screening_path = tmp_path / "screening.json"
    screening_path.write_text(json.dumps(screening), encoding="utf-8")
    records = [
        {
            **candidate,
            "role": "candidate",
            "successes": 10,
            "trials": trials,
            "estimate": 1.0,
            "clopper_pearson_one_sided_lower": 0.91,
            "proof_status": "candidate_statistically_feasible",
        },
        {
            "n_a": 0,
            "n_b": 10,
            "role": "strictly_cheaper_maximal",
            "clopper_pearson_one_sided_upper": 0.88,
            "proof_status": "strictly_cheaper_design_excluded",
        },
        {
            "n_a": 1,
            "n_b": 1,
            "role": "strictly_cheaper_maximal",
            "clopper_pearson_one_sided_upper": 0.89,
            "proof_status": "strictly_cheaper_design_excluded",
        },
    ]
    final = {
        "kind": "q4_final_summary",
        "result_status": "globally_certified_minimum_cost",
        "boundary_contract": {"mode": "D"},
        "freeze_path": str(freeze_path),
        "freeze_sha256": MODULE.sha256(freeze_path),
        "reported_design": candidate,
        "confirmation_records": records,
    }
    final_path = tmp_path / "final.json"
    final_path.write_text(json.dumps(final), encoding="utf-8")

    loaded = MODULE.load_inputs(screening_path, final_path)
    assert loaded[3].shape == (3, 13)
    assert loaded[4] == trials

    pdf = tmp_path / "frontier.pdf"
    png = tmp_path / "frontier.png"
    result = MODULE.build_figure(
        screening, final, counts, trials, pdf, png, dpi=220
    )
    assert pdf.stat().st_size > 1000
    assert png.stat().st_size > 1000
    assert result["frontier_record_count"] == 2
    assert result["not_excluded_frontier_count"] == 0
    claim = MODULE.claim_scope_audit(final, result)
    assert claim["global_minimum_certified"] is True
    assert claim["unresolved_cheaper_design_count"] == 0


def test_claim_scope_keeps_unresolved_cheaper_designs_explicit() -> None:
    final = {
        "result_status": "lowest_statistically_feasible_cost",
        "excluded_frontier_count": 573,
        "not_excluded_frontier_count": 46,
    }
    figure = {
        "excluded_frontier_count": 573,
        "not_excluded_frontier_count": 46,
    }
    audit = MODULE.claim_scope_audit(final, figure)
    assert audit == {
        "evidence_scope": "candidate_feasibility_and_cheaper_design_exclusion",
        "global_minimum_certified": False,
        "unresolved_cheaper_design_count": 46,
        "claim_zh": "候选设计统计可行，仍有 46 个严格更便宜极大设计未排除",
    }
