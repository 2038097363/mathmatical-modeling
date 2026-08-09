from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "问题" / "问题4" / "src" / "finalize_positive_domain.py"
SPEC = importlib.util.spec_from_file_location("q4_positive_domain", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_positive_cost_domain_rejects_zero_counts() -> None:
    assert MODULE.cost_weight(612, 12) == 347772
    with pytest.raises(ValueError, match="正整数"):
        MODULE.cost_weight(619, 0)
    with pytest.raises(ValueError, match="正整数"):
        MODULE.cost_weight(0, 5483)


def test_frozen_positive_domain_summary_matches_counts() -> None:
    result_root = ROOT / "问题" / "问题4" / "results" / "D_screen2000_confirm50000"
    summary = json.loads(
        (result_root / "q4_positive_domain_summary.json").read_text(encoding="utf-8")
    )
    with np.load(result_root / "q4_confirmation_integer_domain_counts.npz") as payload:
        counts = payload["success_counts"]
        trials = int(payload["trials"])

    empirical = summary["empirical_minimum"]
    conservative = summary["conservative_recommendation"]
    assert summary["domain"]["constraint"] == "N_A >= 1 and N_B >= 1"
    assert (empirical["n_a"], empirical["n_b"]) == (612, 12)
    assert empirical["successes"] == int(counts[612, 12]) == 45000
    assert empirical["estimate"] == 45000 / trials
    assert (conservative["n_a"], conservative["n_b"]) == (616, 1)
    assert conservative["successes"] == int(counts[616, 1]) == 45256
    assert conservative["cp_one_sided_family_lower"] >= 0.90


def test_no_strictly_cheaper_positive_empirical_design_is_feasible() -> None:
    result_root = ROOT / "问题" / "问题4" / "results" / "D_screen2000_confirm50000"
    with np.load(result_root / "q4_confirmation_integer_domain_counts.npz") as payload:
        counts = payload["success_counts"]
        trials = int(payload["trials"])
    target_successes = int(np.ceil(0.90 * trials))
    candidate_weight = MODULE.cost_weight(612, 12)

    for n_a in range(1, counts.shape[0]):
        maximum_b = (candidate_weight - 1 - MODULE.A_COST_WEIGHT * n_a) // MODULE.B_COST_WEIGHT
        if maximum_b < 1:
            continue
        maximum_b = min(maximum_b, counts.shape[1] - 1)
        assert int(counts[n_a, maximum_b]) < target_successes
