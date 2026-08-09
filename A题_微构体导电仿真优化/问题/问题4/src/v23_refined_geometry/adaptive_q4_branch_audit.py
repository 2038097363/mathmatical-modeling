from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_q4_pooled import one_sided_cp_lower, read_candidates, read_result, wilson


COST_A = 0.0148440253
COST_B = 0.0016755161


def pava(values: list[float], weights: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators fit for a nondecreasing sequence."""
    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append({"start": index, "end": index, "weight": weight, "mean": value})
        while len(blocks) >= 2 and float(blocks[-2]["mean"]) > float(blocks[-1]["mean"]):
            right = blocks.pop()
            left = blocks.pop()
            total_weight = float(left["weight"]) + float(right["weight"])
            mean = (
                float(left["mean"]) * float(left["weight"])
                + float(right["mean"]) * float(right["weight"])
            ) / total_weight
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": total_weight,
                    "mean": mean,
                }
            )
    fitted = [0.0] * len(values)
    for block in blocks:
        for index in range(int(block["start"]), int(block["end"]) + 1):
            fitted[index] = float(block["mean"])
    return fitted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("branch_candidates", type=Path)
    parser.add_argument("branch_results", nargs="+", type=Path)
    parser.add_argument("--incumbent-a", type=int, required=True)
    parser.add_argument("--incumbent-b", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    candidates = read_candidates(args.branch_candidates)
    keys = [(int(row["N_A"]), int(row["N_B"])) for row in candidates]
    candidate_map = {key: row for key, row in zip(keys, candidates)}
    aggregate = {key: [0, 0] for key in keys}
    for path in args.branch_results:
        result = read_result(path)
        if set(result) != set(keys):
            raise ValueError(f"candidate mismatch: {path}")
        for key, (successes, trials) in result.items():
            aggregate[key][0] += successes
            aggregate[key][1] += trials

    incumbent_cost = args.incumbent_a * COST_A + args.incumbent_b * COST_B
    rows: list[dict[str, float | int | bool]] = []
    for key in sorted(keys):
        cost = float(candidate_map[key]["cost_yuan"])
        if not cost < incumbent_cost:
            raise ValueError(f"branch point is not strictly cheaper than incumbent: {key}")
        successes, trials = aggregate[key]
        probability = successes / trials
        low, high = wilson(successes, trials)
        cp_lower = one_sided_cp_lower(successes, trials, args.alpha)
        rows.append(
            {
                "N_A": key[0],
                "N_B": key[1],
                "cost_yuan": cost,
                "cost_gap_to_incumbent": incumbent_cost - cost,
                "successes": successes,
                "trials": trials,
                "probability": probability,
                "Wilson95_low": low,
                "Wilson95_high": high,
                "CP_one_sided95_lower": cp_lower,
                "point_feasible": probability >= 0.9,
                "lower_bound_feasible": cp_lower >= 0.9,
            }
        )

    raw = [float(row["probability"]) for row in rows]
    weights = [float(row["trials"]) for row in rows]
    fitted = pava(raw, weights)
    for row, fit in zip(rows, fitted):
        row["PAVA_probability"] = fit
        row["PAVA_adjustment"] = fit - float(row["probability"])

    point_candidates = [row for row in rows if bool(row["point_feasible"])]
    lower_candidates = [row for row in rows if bool(row["lower_bound_feasible"])]
    point_update = min(point_candidates, key=lambda row: float(row["cost_yuan"]), default=None)
    lower_update = min(lower_candidates, key=lambda row: float(row["cost_yuan"]), default=None)
    max_adjustment = max(abs(float(row["PAVA_adjustment"])) for row in rows)
    summary = {
        "schema_version": "1.0",
        "method": "integer branch-and-bound with common-random-number sampling and PAVA trend audit",
        "incumbent": {
            "N_A": args.incumbent_a,
            "N_B": args.incumbent_b,
            "cost_yuan": incumbent_cost,
        },
        "branch_candidate_count": len(rows),
        "pooled_trials_per_candidate": int(rows[0]["trials"]),
        "point_estimate_status": "update_incumbent" if point_update else "branch_pruned",
        "point_estimate_update": point_update,
        "lower_bound_status": "update_incumbent" if lower_update else "branch_pruned",
        "lower_bound_update": lower_update,
        "maximum_probability": max(rows, key=lambda row: float(row["probability"])),
        "PAVA_maximum_absolute_adjustment": max_adjustment,
        "PAVA_trend_consistent_at_0_002": max_adjustment <= 0.002,
        "rows": rows,
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
