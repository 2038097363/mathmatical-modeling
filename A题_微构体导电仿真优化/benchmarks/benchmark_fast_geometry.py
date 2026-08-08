from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numba
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from fast_geometry import (  # noqa: E402
    fast_cylinder_distance_diagnostics,
    warm_up_fast_geometry,
)
from geometry_kernel import Cylinder, distance_bounds  # noqa: E402


ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-13
MAX_ITERATIONS = 512
CUTOFF = 1.8


def make_pairs(count: int) -> list[tuple[Cylinder, Cylinder]]:
    rng = np.random.default_rng(2026080717)
    pairs = []
    for pair_index in range(count):
        first_center = rng.uniform(-5000.0, 5000.0, 3)
        second_center = (
            rng.uniform(-5000.0, 5000.0, 3)
            if pair_index % 2
            else first_center + rng.normal(0.0, 1200.0, 3)
        )
        pairs.append(
            (
                Cylinder(
                    first_center,
                    rng.normal(size=3),
                    float(rng.uniform(10.0, 2500.0)),
                    30.0,
                ),
                Cylinder(
                    second_center,
                    rng.normal(size=3),
                    float(rng.uniform(10.0, 2500.0)),
                    30.0,
                ),
            )
        )
    return pairs


def median_runtime(function: Callable[[], object], repeats: int) -> float:
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - started)
    return statistics.median(measurements)


def run_benchmark(pair_count: int, repeats: int) -> dict[str, object]:
    pairs = make_pairs(pair_count)
    warm_up_fast_geometry()
    for first, second in pairs[:10]:
        fast_cylinder_distance_diagnostics(
            first,
            second,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
            MAX_ITERATIONS,
        )

    reference_results = [
        distance_bounds(
            first,
            second,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
            MAX_ITERATIONS,
        )
        for first, second in pairs
    ]

    reference_seconds = median_runtime(
        lambda: [
            distance_bounds(
                first,
                second,
                ABSOLUTE_TOLERANCE,
                RELATIVE_TOLERANCE,
                MAX_ITERATIONS,
            )
            for first, second in pairs
        ],
        repeats,
    )
    full_results = [
        fast_cylinder_distance_diagnostics(
            first,
            second,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
            MAX_ITERATIONS,
        )
        for first, second in pairs
    ]
    full_seconds = median_runtime(
        lambda: [
            fast_cylinder_distance_diagnostics(
                first,
                second,
                ABSOLUTE_TOLERANCE,
                RELATIVE_TOLERANCE,
                MAX_ITERATIONS,
            )
            for first, second in pairs
        ],
        repeats,
    )
    threshold_results = [
        fast_cylinder_distance_diagnostics(
            first,
            second,
            ABSOLUTE_TOLERANCE,
            RELATIVE_TOLERANCE,
            MAX_ITERATIONS,
            cutoff=CUTOFF,
        )
        for first, second in pairs
    ]
    threshold_seconds = median_runtime(
        lambda: [
            fast_cylinder_distance_diagnostics(
                first,
                second,
                ABSOLUTE_TOLERANCE,
                RELATIVE_TOLERANCE,
                MAX_ITERATIONS,
                cutoff=CUTOFF,
            )
            for first, second in pairs
        ],
        repeats,
    )
    mismatches = sum(
        reference.classify(CUTOFF) != accelerated.bounds.classify(CUTOFF)
        for reference, accelerated in zip(
            reference_results,
            threshold_results,
            strict=True,
        )
    )
    return {
        "pair_count": pair_count,
        "repeats": repeats,
        "reference_seconds": reference_seconds,
        "fast_full_distance_seconds": full_seconds,
        "fast_threshold_seconds": threshold_seconds,
        "full_distance_speedup": reference_seconds / full_seconds,
        "threshold_speedup": reference_seconds / threshold_seconds,
        "threshold_classification_mismatches": mismatches,
        "full_distance_fallbacks": sum(
            result.used_fallback for result in full_results
        ),
        "threshold_fallbacks": sum(
            result.used_fallback for result in threshold_results
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-speedup", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs < 1 or args.repeats < 1 or args.minimum_speedup <= 0.0:
        raise ValueError("benchmark arguments must be positive")
    payload = {
        "kind": "fast_geometry_benchmark",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        **run_benchmark(args.pairs, args.repeats),
    }
    output = json.dumps(payload, ensure_ascii=True, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    if payload["threshold_classification_mismatches"] != 0:
        raise SystemExit("threshold classifications differ from the reference")
    if payload["full_distance_speedup"] < args.minimum_speedup:
        raise SystemExit(
            f"full-distance speedup {payload['full_distance_speedup']:.3f}x "
            f"is below {args.minimum_speedup:.3f}x"
        )


if __name__ == "__main__":
    main()
