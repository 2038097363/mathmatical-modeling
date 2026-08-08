from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from pareto_connectivity import (  # noqa: E402
    axis_threshold_pareto_result,
    design_is_connected,
    minimum_second_threshold,
    pareto_minimax_labels,
    pareto_minimax_result,
    pareto_prune_labels,
)


def _undirected_edges(adjacency):
    edges = set()
    nodes = set(adjacency)
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            nodes.add(neighbor)
            if node != neighbor:
                edges.add(frozenset((node, neighbor)))
    return nodes, edges


def _brute_connected(adjacency, thresholds, left, right, design):
    nodes, edges = _undirected_edges(adjacency)
    nodes.update((left, right))
    active = {
        node
        for node in nodes
        if thresholds[node][0] <= design[0]
        and thresholds[node][1] <= design[1]
    }
    if left not in active or right not in active:
        return False
    parent = {node: node for node in active}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for edge in edges:
        first, second = tuple(edge)
        if first in active and second in active:
            union(first, second)
    return find(left) == find(right)


def _brute_frontier(adjacency, thresholds, left, right):
    maximum_first = max(value[0] for value in thresholds.values())
    maximum_second = max(value[1] for value in thresholds.values())
    feasible = []
    for first in range(maximum_first + 1):
        for second in range(maximum_second + 1):
            if _brute_connected(
                adjacency,
                thresholds,
                left,
                right,
                (first, second),
            ):
                feasible.append((first, second))
    return pareto_prune_labels(feasible)


class ParetoConnectivityExampleTests(unittest.TestCase):
    def test_q4_axis_threshold_branches_preserve_all_tradeoffs(self) -> None:
        frontier_size = 25
        adjacency = {"LEFT": [], "RIGHT": []}
        thresholds = {"LEFT": (0, 0), "RIGHT": (0, 0)}
        for index in range(1, frontier_size + 1):
            a_node = f"A{index}"
            b_node = f"B{index}"
            adjacency["LEFT"].append(a_node)
            adjacency[a_node] = [b_node]
            adjacency[b_node] = ["RIGHT"]
            thresholds[a_node] = (index, 0)
            thresholds[b_node] = (0, frontier_size + 1 - index)
        self.assertEqual(
            pareto_minimax_labels(
                adjacency, thresholds, "LEFT", "RIGHT"
            ),
            tuple(
                (index, frontier_size + 1 - index)
                for index in range(1, frontier_size + 1)
            ),
        )

    def test_three_parallel_paths_form_expected_frontier(self) -> None:
        adjacency = {
            "LEFT": ["u", "v", "w"],
            "u": ["RIGHT"],
            "v": ["RIGHT"],
            "w": ["RIGHT"],
            "RIGHT": [],
        }
        thresholds = {
            "LEFT": (0, 0),
            "RIGHT": (0, 0),
            "u": (1, 4),
            "v": (2, 2),
            "w": (4, 1),
        }
        result = pareto_minimax_result(
            adjacency, thresholds, "LEFT", "RIGHT"
        )
        self.assertEqual(result.labels, ((1, 4), (2, 2), (4, 1)))
        self.assertFalse(design_is_connected(result.labels, 1, 3))
        self.assertTrue(design_is_connected(result.labels, 1, 4))
        self.assertTrue(design_is_connected(result.labels, 3, 2))
        self.assertEqual(minimum_second_threshold(result.labels, 0), None)
        self.assertEqual(minimum_second_threshold(result.labels, 3), 2)
        self.assertGreater(result.diagnostics.dominated_prunes, 0)

    def test_componentwise_max_is_taken_over_the_whole_path(self) -> None:
        adjacency = {
            "LEFT": ["a", "c"],
            "a": ["b"],
            "b": ["RIGHT"],
            "c": ["RIGHT"],
        }
        thresholds = {
            "LEFT": (0, 0),
            "RIGHT": (0, 0),
            "a": (3, 0),
            "b": (0, 5),
            "c": (4, 6),
        }
        self.assertEqual(
            pareto_minimax_labels(adjacency, thresholds, "LEFT", "RIGHT"),
            ((3, 5),),
        )

    def test_duplicate_path_labels_are_emitted_once(self) -> None:
        adjacency = {
            0: [2, 3, 4],
            2: [1],
            3: [1],
            4: [5],
            5: [1],
            1: [],
        }
        thresholds = {
            0: (0, 0),
            1: (0, 0),
            2: (2, 3),
            3: (2, 3),
            4: (1, 4),
            5: (2, 2),
        }
        self.assertEqual(
            pareto_minimax_labels(adjacency, thresholds, 0, 1),
            ((2, 3),),
        )

    def test_disconnected_and_same_terminal_cases(self) -> None:
        thresholds = {"LEFT": (0, 0), "RIGHT": (0, 0)}
        self.assertEqual(
            pareto_minimax_labels({}, thresholds, "LEFT", "RIGHT"), ()
        )
        self.assertEqual(
            pareto_minimax_labels(
                {}, {"X": (3, 7)}, "X", "X"
            ),
            ((3, 7),),
        )

    def test_prune_labels_is_unique_and_deterministic(self) -> None:
        labels = [(4, 5), (1, 7), (2, 4), (2, 4), (3, 6), (5, 2)]
        self.assertEqual(
            pareto_prune_labels(reversed(labels)),
            ((1, 7), (2, 4), (5, 2)),
        )

    def test_input_validation(self) -> None:
        with self.assertRaisesRegex(KeyError, "missing activation"):
            pareto_minimax_labels({0: [1]}, {0: (0, 0)}, 0, 1)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            pareto_minimax_labels({}, {0: (-1, 0)}, 0, 0)
        with self.assertRaisesRegex(TypeError, "nonnegative integers"):
            pareto_minimax_labels({}, {0: (1.5, 0)}, 0, 0)


class ParetoConnectivityBruteForceTests(unittest.TestCase):
    def test_random_small_graphs_match_every_design_point(self) -> None:
        generator = random.Random(2026080731)
        for case_index in range(160):
            node_count = generator.randint(2, 10)
            nodes = list(range(node_count))
            adjacency = {node: [] for node in nodes}
            for first in nodes:
                for second in range(first + 1, node_count):
                    if generator.random() < 0.28:
                        # Mix symmetric, one-sided, and duplicate input edges.
                        adjacency[first].append(second)
                        if generator.random() < 0.55:
                            adjacency[second].append(first)
                        if generator.random() < 0.12:
                            adjacency[first].append(second)
            thresholds = {
                node: (generator.randrange(6), generator.randrange(6))
                for node in nodes
            }
            thresholds[0] = (0, 0)
            thresholds[1] = (0, 0)

            labels = pareto_minimax_labels(
                adjacency, thresholds, 0, 1
            )
            brute = _brute_frontier(adjacency, thresholds, 0, 1)
            self.assertEqual(labels, brute, msg=f"random case {case_index}")
            self.assertEqual(
                tuple(sorted(labels)),
                labels,
                msg=f"random case {case_index}",
            )
            for first in range(6):
                for second in range(6):
                    self.assertEqual(
                        design_is_connected(labels, first, second),
                        _brute_connected(
                            adjacency,
                            thresholds,
                            0,
                            1,
                            (first, second),
                        ),
                        msg=f"random case {case_index}, design {(first, second)}",
                    )

    def test_adjacency_order_does_not_change_frontier(self) -> None:
        adjacency = {
            0: [2, 3, 4, 5],
            2: [6],
            3: [6],
            4: [1],
            5: [1],
            6: [1],
            1: [],
        }
        thresholds = {
            0: (0, 0),
            1: (0, 0),
            2: (1, 6),
            3: (2, 4),
            4: (5, 1),
            5: (3, 3),
            6: (4, 2),
        }
        reversed_adjacency = {
            node: list(reversed(neighbors))
            for node, neighbors in reversed(list(adjacency.items()))
        }
        self.assertEqual(
            pareto_minimax_labels(adjacency, thresholds, 0, 1),
            pareto_minimax_labels(reversed_adjacency, thresholds, 0, 1),
        )


class AxisThresholdSweepTests(unittest.TestCase):
    def test_parallel_axis_paths_match_generic_solver(self) -> None:
        frontier_size = 40
        adjacency = {"LEFT": [], "RIGHT": []}
        thresholds = {"LEFT": (0, 0), "RIGHT": (0, 0)}
        for index in range(1, frontier_size + 1):
            a_node = f"A{index}"
            b_node = f"B{index}"
            adjacency["LEFT"].append(a_node)
            adjacency[a_node] = [b_node]
            adjacency[b_node] = ["RIGHT"]
            thresholds[a_node] = (index, 0)
            thresholds[b_node] = (0, frontier_size + 1 - index)

        specialized = axis_threshold_pareto_result(
            adjacency, thresholds, "LEFT", "RIGHT"
        )
        generic = pareto_minimax_result(
            adjacency, thresholds, "LEFT", "RIGHT"
        )
        self.assertEqual(specialized.labels, generic.labels)
        self.assertEqual(specialized.diagnostics.node_count, frontier_size * 2 + 2)

    def test_random_axis_graphs_match_generic_and_brute_force(self) -> None:
        generator = random.Random(2026080747)
        for case_index in range(120):
            node_count = generator.randint(2, 12)
            nodes = list(range(node_count))
            adjacency = {node: [] for node in nodes}
            for first in nodes:
                for second in range(first + 1, node_count):
                    if generator.random() < 0.32:
                        adjacency[first].append(second)
                        if generator.random() < 0.5:
                            adjacency[second].append(first)
            thresholds = {0: (0, 0), 1: (0, 0)}
            for node in nodes[2:]:
                level = generator.randrange(5)
                thresholds[node] = (
                    (level, 0) if generator.random() < 0.5 else (0, level)
                )

            specialized = axis_threshold_pareto_result(
                adjacency, thresholds, 0, 1
            ).labels
            generic = pareto_minimax_result(adjacency, thresholds, 0, 1).labels
            brute = _brute_frontier(adjacency, thresholds, 0, 1)
            self.assertEqual(specialized, generic, case_index)
            self.assertEqual(specialized, brute, case_index)

    def test_non_axis_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lie on one axis"):
            axis_threshold_pareto_result(
                {0: [2], 2: [1]},
                {0: (0, 0), 1: (0, 0), 2: (1, 1)},
                0,
                1,
            )


if __name__ == "__main__":
    unittest.main()
