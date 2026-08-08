from __future__ import annotations

import heapq
import math
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TypeVar


Node = TypeVar("Node", bound=Hashable)
Label = tuple[int, int]


@dataclass(frozen=True)
class ParetoSearchDiagnostics:
    node_count: int
    edge_count: int
    generated_labels: int
    accepted_labels: int
    dominated_prunes: int
    target_prunes: int
    removed_labels: int
    stale_queue_pops: int
    peak_queue_size: int


@dataclass(frozen=True)
class ParetoConnectivityResult:
    labels: tuple[Label, ...]
    diagnostics: ParetoSearchDiagnostics


def _coerce_label(value: object, *, context: str) -> Label:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a pair of nonnegative integers")
    if len(value) != 2:
        raise ValueError(f"{context} must contain exactly two components")
    first, second = value
    if (
        not isinstance(first, Integral)
        or isinstance(first, bool)
        or not isinstance(second, Integral)
        or isinstance(second, bool)
    ):
        raise TypeError(f"{context} must be a pair of nonnegative integers")
    label = int(first), int(second)
    if label[0] < 0 or label[1] < 0:
        raise ValueError(f"{context} components must be nonnegative")
    return label


def pareto_prune_labels(labels: Iterable[Sequence[int]]) -> tuple[Label, ...]:
    """按确定顺序返回互异的非支配标签。"""
    normalized = sorted(
        {
            _coerce_label(label, context="label")
            for label in labels
        }
    )
    frontier: list[Label] = []
    best_second = math.inf
    for label in normalized:
        if label[1] < best_second:
            frontier.append(label)
            best_second = label[1]
    return tuple(frontier)


def _frontier_dominates(frontier: Sequence[Label], label: Label) -> bool:
    position = bisect_right(frontier, label) - 1
    return position >= 0 and frontier[position][1] <= label[1]


def _insert_frontier_label(
    frontier: list[Label], active: set[Label], label: Label
) -> tuple[bool, int]:
    if _frontier_dominates(frontier, label):
        return False, 0

    position = bisect_left(frontier, label)
    end = position
    while end < len(frontier) and frontier[end][1] >= label[1]:
        active.remove(frontier[end])
        end += 1
    removed = end - position
    frontier[position:end] = [label]
    active.add(label)
    return True, removed


def _normalize_undirected_graph(
    adjacency: Mapping[Node, Iterable[Node]], left: Node, right: Node
) -> tuple[list[Node], list[tuple[int, ...]], int]:
    node_indices: dict[Node, int] = {}
    nodes: list[Node] = []

    def add_node(node: Node) -> int:
        try:
            existing = node_indices.get(node)
        except TypeError as exc:
            raise TypeError("graph nodes must be hashable") from exc
        if existing is not None:
            return existing
        index = len(nodes)
        node_indices[node] = index
        nodes.append(node)
        return index

    add_node(left)
    add_node(right)
    materialized: list[tuple[Node, tuple[Node, ...]]] = []
    for node, neighbors in adjacency.items():
        add_node(node)
        neighbor_tuple = tuple(neighbors)
        for neighbor in neighbor_tuple:
            add_node(neighbor)
        materialized.append((node, neighbor_tuple))

    normalized = [set() for _ in nodes]
    for node, neighbors in materialized:
        node_index = node_indices[node]
        for neighbor in neighbors:
            neighbor_index = node_indices[neighbor]
            if node_index == neighbor_index:
                continue
            normalized[node_index].add(neighbor_index)
            normalized[neighbor_index].add(node_index)

    ordered = [tuple(sorted(neighbors)) for neighbors in normalized]
    edge_count = sum(len(neighbors) for neighbors in ordered) // 2
    return nodes, ordered, edge_count


def pareto_minimax_result(
    adjacency: Mapping[Node, Iterable[Node]],
    activation_thresholds: Mapping[Node, Sequence[int]],
    left: Node,
    right: Node,
) -> ParetoConnectivityResult:
    """求左右贯通路径的非支配分量极小极大标签。

    路径标签为沿途节点激活阈值的分量最大值；输入邻接关系按无向图解释。
    """
    nodes, neighbors, edge_count = _normalize_undirected_graph(
        adjacency, left, right
    )
    thresholds: list[Label] = []
    for node in nodes:
        if node not in activation_thresholds:
            raise KeyError(f"missing activation threshold for node {node!r}")
        thresholds.append(
            _coerce_label(
                activation_thresholds[node],
                context=f"activation threshold for node {node!r}",
            )
        )

    left_index = nodes.index(left)
    right_index = nodes.index(right)
    if left_index == right_index:
        label = thresholds[left_index]
        return ParetoConnectivityResult(
            labels=(label,),
            diagnostics=ParetoSearchDiagnostics(
                node_count=len(nodes),
                edge_count=edge_count,
                generated_labels=1,
                accepted_labels=1,
                dominated_prunes=0,
                target_prunes=0,
                removed_labels=0,
                stale_queue_pops=0,
                peak_queue_size=1,
            ),
        )

    frontiers: list[list[Label]] = [[] for _ in nodes]
    active_labels: list[set[Label]] = [set() for _ in nodes]
    initial = thresholds[left_index]
    frontiers[left_index].append(initial)
    active_labels[left_index].add(initial)
    queue: list[tuple[int, int, int]] = [
        (initial[0], initial[1], left_index)
    ]

    generated_labels = 1
    accepted_labels = 1
    dominated_prunes = 0
    target_prunes = 0
    removed_labels = 0
    stale_queue_pops = 0
    peak_queue_size = 1

    while queue:
        first, second, node_index = heapq.heappop(queue)
        label = first, second
        if label not in active_labels[node_index]:
            stale_queue_pops += 1
            continue
        if node_index == right_index:
            continue
        target_frontier = frontiers[right_index]
        if target_frontier and _frontier_dominates(target_frontier, label):
            target_prunes += 1
            continue

        for neighbor_index in neighbors[node_index]:
            neighbor_threshold = thresholds[neighbor_index]
            candidate = (
                max(label[0], neighbor_threshold[0]),
                max(label[1], neighbor_threshold[1]),
            )
            generated_labels += 1

            if (
                neighbor_index != right_index
                and target_frontier
                and _frontier_dominates(target_frontier, candidate)
            ):
                target_prunes += 1
                continue

            inserted, removed = _insert_frontier_label(
                frontiers[neighbor_index],
                active_labels[neighbor_index],
                candidate,
            )
            if not inserted:
                dominated_prunes += 1
                continue
            accepted_labels += 1
            removed_labels += removed
            heapq.heappush(
                queue,
                (candidate[0], candidate[1], neighbor_index),
            )
            peak_queue_size = max(peak_queue_size, len(queue))

    labels = tuple(frontiers[right_index])
    return ParetoConnectivityResult(
        labels=labels,
        diagnostics=ParetoSearchDiagnostics(
            node_count=len(nodes),
            edge_count=edge_count,
            generated_labels=generated_labels,
            accepted_labels=accepted_labels,
            dominated_prunes=dominated_prunes,
            target_prunes=target_prunes,
            removed_labels=removed_labels,
            stale_queue_pops=stale_queue_pops,
            peak_queue_size=peak_queue_size,
        ),
    )


def axis_threshold_pareto_result(
    adjacency: Mapping[Node, Iterable[Node]],
    activation_thresholds: Mapping[Node, Sequence[int]],
    left: Node,
    right: Node,
) -> ParetoConnectivityResult:
    """以精确并查集扫描求解轴可分激活图。

    节点阈值必须为 ``(a, 0)`` 或 ``(0, b)``；固定第一维预算后按第二维阈值
    递增加入节点，首次左右贯通值即最小可行第二维预算。
    """
    nodes, neighbors, edge_count = _normalize_undirected_graph(
        adjacency, left, right
    )
    thresholds: list[Label] = []
    for node in nodes:
        if node not in activation_thresholds:
            raise KeyError(f"missing activation threshold for node {node!r}")
        threshold = _coerce_label(
            activation_thresholds[node],
            context=f"activation threshold for node {node!r}",
        )
        if threshold[0] > 0 and threshold[1] > 0:
            raise ValueError(
                "axis threshold sweep requires every node threshold to lie on one axis"
            )
        thresholds.append(threshold)

    left_index = nodes.index(left)
    right_index = nodes.index(right)
    if left_index == right_index:
        label = thresholds[left_index]
        return ParetoConnectivityResult(
            labels=(label,),
            diagnostics=ParetoSearchDiagnostics(
                node_count=len(nodes),
                edge_count=edge_count,
                generated_labels=1,
                accepted_labels=1,
                dominated_prunes=0,
                target_prunes=0,
                removed_labels=0,
                stale_queue_pops=0,
                peak_queue_size=0,
            ),
        )

    retained = bytearray(b"\x01") * len(nodes)
    degrees = [len(node_neighbors) for node_neighbors in neighbors]
    removable = deque(
        index
        for index, degree in enumerate(degrees)
        if index not in (left_index, right_index) and degree <= 1
    )
    while removable:
        node_index = removable.popleft()
        if not retained[node_index]:
            continue
        retained[node_index] = 0
        for neighbor_index in neighbors[node_index]:
            if not retained[neighbor_index]:
                continue
            degrees[neighbor_index] -= 1
            if (
                neighbor_index not in (left_index, right_index)
                and degrees[neighbor_index] == 1
            ):
                removable.append(neighbor_index)

    reachable = bytearray(len(nodes))
    if retained[left_index]:
        reachable[left_index] = 1
        stack = [left_index]
        while stack:
            node_index = stack.pop()
            for neighbor_index in neighbors[node_index]:
                if retained[neighbor_index] and not reachable[neighbor_index]:
                    reachable[neighbor_index] = 1
                    stack.append(neighbor_index)
    if not reachable[right_index]:
        return ParetoConnectivityResult(
            labels=(),
            diagnostics=ParetoSearchDiagnostics(
                node_count=len(nodes),
                edge_count=edge_count,
                generated_labels=1,
                accepted_labels=0,
                dominated_prunes=0,
                target_prunes=0,
                removed_labels=0,
                stale_queue_pops=0,
                peak_queue_size=0,
            ),
        )

    base_nodes: list[int] = []
    first_groups: dict[int, list[int]] = {}
    second_groups: dict[int, list[int]] = {}
    for node_index, threshold in enumerate(thresholds):
        if not reachable[node_index]:
            continue
        first, second = threshold
        if first == 0 and second == 0:
            base_nodes.append(node_index)
        elif first > 0:
            first_groups.setdefault(first, []).append(node_index)
        else:
            second_groups.setdefault(second, []).append(node_index)

    first_levels = [0, *sorted(first_groups)]
    second_levels = sorted(second_groups)
    first_prefix_nodes: list[int] = []
    feasible_labels: list[Label] = []
    generated_states = 0
    skipped_second_levels = 0

    for first_level in first_levels:
        if first_level:
            first_prefix_nodes.extend(first_groups[first_level])

        parent = list(range(len(nodes)))
        sizes = [1] * len(nodes)
        active = bytearray(len(nodes))

        def find(node_index: int) -> int:
            root = node_index
            while parent[root] != root:
                root = parent[root]
            while parent[node_index] != node_index:
                next_index = parent[node_index]
                parent[node_index] = root
                node_index = next_index
            return root

        def activate(node_index: int) -> None:
            if active[node_index]:
                return
            active[node_index] = 1
            for neighbor_index in neighbors[node_index]:
                if not active[neighbor_index]:
                    continue
                first_root = find(node_index)
                second_root = find(neighbor_index)
                if first_root == second_root:
                    continue
                if sizes[first_root] < sizes[second_root]:
                    first_root, second_root = second_root, first_root
                parent[second_root] = first_root
                sizes[first_root] += sizes[second_root]

        for node_index in base_nodes:
            activate(node_index)
        for node_index in first_prefix_nodes:
            activate(node_index)

        generated_states += 1
        if find(left_index) == find(right_index):
            feasible_labels.append((first_level, 0))
            skipped_second_levels += len(second_levels)
            break

        for level_index, second_level in enumerate(second_levels):
            for node_index in second_groups[second_level]:
                activate(node_index)
            generated_states += 1
            if find(left_index) == find(right_index):
                feasible_labels.append((first_level, second_level))
                skipped_second_levels += len(second_levels) - level_index - 1
                break

    labels = pareto_prune_labels(feasible_labels)
    return ParetoConnectivityResult(
        labels=labels,
        diagnostics=ParetoSearchDiagnostics(
            node_count=len(nodes),
            edge_count=edge_count,
            generated_labels=generated_states,
            accepted_labels=len(feasible_labels),
            dominated_prunes=len(feasible_labels) - len(labels),
            target_prunes=skipped_second_levels,
            removed_labels=0,
            stale_queue_pops=0,
            peak_queue_size=0,
        ),
    )


def pareto_minimax_labels(
    adjacency: Mapping[Node, Iterable[Node]],
    activation_thresholds: Mapping[Node, Sequence[int]],
    left: Node,
    right: Node,
) -> tuple[Label, ...]:
    return pareto_minimax_result(
        adjacency,
        activation_thresholds,
        left,
        right,
    ).labels


def minimum_second_threshold(
    labels: Sequence[Label], first_threshold: int
) -> int | None:
    first = _coerce_label(
        (first_threshold, 0), context="design threshold"
    )[0]
    position = bisect_right(labels, (first, math.inf)) - 1
    return None if position < 0 else labels[position][1]


def design_is_connected(
    labels: Sequence[Label], first_threshold: int, second_threshold: int
) -> bool:
    design = _coerce_label(
        (first_threshold, second_threshold), context="design threshold"
    )
    required_second = minimum_second_threshold(labels, design[0])
    return required_second is not None and required_second <= design[1]


__all__ = [
    "Label",
    "ParetoConnectivityResult",
    "ParetoSearchDiagnostics",
    "axis_threshold_pareto_result",
    "design_is_connected",
    "minimum_second_threshold",
    "pareto_minimax_labels",
    "pareto_minimax_result",
    "pareto_prune_labels",
]
