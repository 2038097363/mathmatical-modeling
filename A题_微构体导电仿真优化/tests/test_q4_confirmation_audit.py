from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT / "问题" / "问题4" / "src" / "audit_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("audit_q4_confirmation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_independent_counts_match_direct_queries() -> None:
    frontiers = [
        ((1, 3), (3, 1)),
        ((0, 4), (2, 2)),
        (),
        ((1, 0),),
    ]
    counts = MODULE.independent_success_counts(frontiers, 3, 4)
    assert counts.dtype == np.int32
    assert counts.shape == (4, 5)
    for n_a in range(4):
        for n_b in range(5):
            assert int(counts[n_a, n_b]) == MODULE.direct_success_count(
                frontiers, n_a, n_b
            )
    assert np.count_nonzero(np.diff(counts, axis=0) < 0) == 0
    assert np.count_nonzero(np.diff(counts, axis=1) < 0) == 0


def test_empirical_minimum_uses_exact_integer_cost_order() -> None:
    frontiers = [((0, 9), (1, 0))] * 9 + [()]
    counts = MODULE.independent_success_counts(frontiers, 1, 9)
    selected = MODULE.minimum_empirical_design(counts, trials=10, target=0.9)
    assert selected is not None
    assert (selected["n_a"], selected["n_b"], selected["successes"]) == (1, 0, 9)
    assert selected["cost_weight"] == 567


def test_frontier_validation_rejects_dominated_order() -> None:
    record = {"connectivity_frontier": [[1, 3], [2, 4]]}
    try:
        MODULE.validated_frontier(record, 3, 4)
    except ValueError as error:
        assert "严格递增" in str(error)
    else:
        raise AssertionError("未拒绝 N_B 反向单调性错误")
