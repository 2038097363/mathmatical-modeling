from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from pareto_connectivity import pareto_minimax_result  # noqa: E402


def make_tradeoff_graph(
    frontier_size: int,
) -> tuple[dict[str, list[str]], dict[str, tuple[int, int]], str, str]:
    if frontier_size < 1:
        raise ValueError("frontier size must be positive")
    left = "LEFT"
    right = "RIGHT"
    adjacency = {left: [], right: []}
    thresholds = {left: (0, 0), right: (0, 0)}
    for index in range(1, frontier_size + 1):
        a_node = f"A{index}"
        b_node = f"B{index}"
        adjacency[a_node] = [left, b_node]
        adjacency[b_node] = [a_node, right]
        adjacency[left].append(a_node)
        adjacency[right].append(b_node)
        thresholds[a_node] = (index, 0)
        thresholds[b_node] = (0, frontier_size + 1 - index)
    return adjacency, thresholds, left, right


def make_random_graph(
    node_count: int, edge_count: int, levels: int
) -> tuple[dict[int, list[int]], dict[int, tuple[int, int]], int, int]:
    if node_count < 3 or edge_count < node_count - 1:
        raise ValueError("random graph must contain a spanning chain")
    maximum_edges = node_count * (node_count - 1) // 2
    if edge_count > maximum_edges - 1 or levels < 1:
        raise ValueError("invalid edge count or threshold level count")

    left = 0
    right = node_count - 1
    generator = random.Random(2026080737)
    edges = {(node, node + 1) for node in range(node_count - 1)}
    forbidden = (left, right)
    while len(edges) < edge_count:
        first = generator.randrange(node_count)
        second = generator.randrange(node_count)
        if first == second:
            continue
        edge = tuple(sorted((first, second)))
        if edge != forbidden:
            edges.add(edge)

    adjacency = {node: [] for node in range(node_count)}
    for first, second in sorted(edges):
        adjacency[first].append(second)
        adjacency[second].append(first)
    thresholds = {left: (0, 0), right: (0, 0)}
    for node in range(1, right):
        value = generator.randrange(1, levels + 1)
        thresholds[node] = (value, 0) if node % 2 == 0 else (0, value)
    return adjacency, thresholds, left, right


def benchmark_case(adjacency, thresholds, left, right, repeats):
    warm_result = pareto_minimax_result(
        adjacency, thresholds, left, right
    )
    runtimes = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = pareto_minimax_result(
            adjacency, thresholds, left, right
        )
        runtimes.append(time.perf_counter() - started)
        if result.labels != warm_result.labels:
            raise RuntimeError("nondeterministic Pareto frontier")
    return {
        "frontier_size": len(warm_result.labels),
        "median_seconds": statistics.median(runtimes),
        "minimum_seconds": min(runtimes),
        "maximum_seconds": max(runtimes),
        "diagnostics": asdict(warm_result.diagnostics),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-size", type=int, default=300)
    parser.add_argument("--random-nodes", type=int, default=500)
    parser.add_argument("--random-edges", type=int, default=3000)
    parser.add_argument("--levels", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    tradeoff = make_tradeoff_graph(args.frontier_size)
    random_graph = make_random_graph(
        args.random_nodes, args.random_edges, args.levels
    )
    payload = {
        "kind": "pareto_connectivity_benchmark",
        "repeats": args.repeats,
        "tradeoff_case": {
            "expected_frontier_size": args.frontier_size,
            **benchmark_case(*tradeoff, args.repeats),
        },
        "random_case": {
            "requested_nodes": args.random_nodes,
            "requested_edges": args.random_edges,
            "threshold_levels": args.levels,
            **benchmark_case(*random_graph, args.repeats),
        },
    }
    if payload["tradeoff_case"]["frontier_size"] != args.frontier_size:
        raise RuntimeError("tradeoff graph did not preserve its known frontier")
    output = json.dumps(payload, ensure_ascii=True, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
