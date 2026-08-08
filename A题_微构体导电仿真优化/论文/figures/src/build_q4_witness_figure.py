from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from render_freecad_scene import atomic_write_json, sha256


FINAL_PUBLICATION_STATUS = "final_random_trial_geometry"
COLOR_A = "#A8B0B7"
COLOR_B = "#4C78A8"
COLOR_WITNESS = "#E4572E"
COLOR_ELECTRODE = "#37474F"
COLOR_BOUNDARY = "#6B7C93"
COLOR_TOPOLOGY = "#007C83"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="由问题4正式场景生成三维导通见证与有序接触图"
    )
    parser.add_argument("scene", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--audit", type=Path)
    return parser.parse_args()


def _witness_records(scene: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace = scene.get("traceability", {})
    witness = trace.get("mixed_witness", {})
    nodes = list(witness.get("nodes", []))
    edges = list(witness.get("edges", []))
    if len(nodes) < 3 or nodes[0] != "electrode_left" or nodes[-1] != "electrode_right":
        raise ValueError("导通见证必须从左电极到右电极")
    if len(edges) != len(nodes) - 1 or int(witness.get("edge_count", -1)) != len(edges):
        raise ValueError("导通见证节点与边数不一致")
    if witness.get("same_source_edges") != 0 or not witness.get("all_edges_geometry_verified"):
        raise ValueError("导通见证含同源片段边或未经几何核验的边")

    records = {
        str(record["id"]): record
        for record in list(scene.get("cylinders", [])) + list(scene.get("spheres", []))
    }
    ordered_records: list[dict[str, Any]] = []
    for node in nodes[1:-1]:
        record = records.get(node)
        if record is None or record.get("role") != "witness":
            raise ValueError(f"见证节点没有对应的高亮几何：{node}")
        ordered_records.append(record)
    for index, edge in enumerate(edges):
        if set(edge.get("nodes", [])) != {nodes[index], nodes[index + 1]}:
            raise ValueError("有序见证边与相邻节点不一致")
        if not edge.get("connected") or edge.get("same_source_pair"):
            raise ValueError("有序见证边未通过连接检查")
    return ordered_records, edges


def _record_extrema(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    radius = float(record.get("radius_nm", 0.0))
    if "start_nm" in record:
        points = np.vstack(
            [
                np.asarray(record["start_nm"], dtype=float),
                np.asarray(record["end_nm"], dtype=float),
            ]
        )
    else:
        points = np.asarray([record["center_nm"]], dtype=float)
    return points.min(axis=0) - radius, points.max(axis=0) + radius


def _witness_focus_limits(
    records: list[dict[str, Any]], half: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    extrema = [_record_extrema(record) for record in records]
    lower = np.min(np.vstack([item[0] for item in extrema]), axis=0)
    upper = np.max(np.vstack([item[1] for item in extrema]), axis=0)
    limits: list[tuple[float, float]] = [(-half, half)]
    minimum_span = 0.16 * (2.0 * half)
    for axis in (1, 2):
        span = float(upper[axis] - lower[axis])
        padding = max(0.08 * span, 0.02 * (2.0 * half))
        low = max(-half, float(lower[axis] - padding))
        high = min(half, float(upper[axis] + padding))
        if high - low < minimum_span:
            center = 0.5 * (low + high)
            low = max(-half, center - 0.5 * minimum_span)
            high = min(half, center + 0.5 * minimum_span)
            if high - low < minimum_span:
                low = high - minimum_span if low <= -half else low
                high = low + minimum_span
        limits.append((low, high))
    return limits[0], limits[1], limits[2]


def _focus_contains_records(
    records: list[dict[str, Any]],
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> bool:
    tolerance = 1e-7
    domain_low, domain_high = limits[0]
    for record in records:
        lower, upper = _record_extrema(record)
        for axis, (low, high) in enumerate(limits):
            clipped_low = max(float(lower[axis]), domain_low)
            clipped_high = min(float(upper[axis]), domain_high)
            if clipped_low < low - tolerance or clipped_high > high + tolerance:
                return False
    return True


def _draw_focus_frame(
    ax: Any,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    vertices = np.asarray(
        [
            [x, y, z]
            for x in limits[0]
            for y in limits[1]
            for z in limits[2]
        ],
        dtype=float,
    )
    edges = [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if np.count_nonzero(vertices[i] != vertices[j]) == 1
    ]
    for first, second in edges:
        points = vertices[[first, second]]
        ax.plot(
            *points.T,
            color=COLOR_BOUNDARY,
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            alpha=0.82,
        )


def _draw_electrodes(
    ax: Any,
    half: float,
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
) -> None:
    for x_value in (-half, half):
        face = [
            (x_value, y_limits[0], z_limits[0]),
            (x_value, y_limits[1], z_limits[0]),
            (x_value, y_limits[1], z_limits[1]),
            (x_value, y_limits[0], z_limits[1]),
        ]
        collection = Poly3DCollection(
            [face],
            facecolors=COLOR_ELECTRODE,
            edgecolors=COLOR_ELECTRODE,
            linewidths=1.2,
            alpha=0.14,
        )
        ax.add_collection3d(collection)


def _record_midpoint(record: dict[str, Any]) -> np.ndarray:
    if "start_nm" in record:
        return 0.5 * (
            np.asarray(record["start_nm"], dtype=float)
            + np.asarray(record["end_nm"], dtype=float)
        )
    return np.asarray(record["center_nm"], dtype=float)


def _electrode_contact_point(
    record: dict[str, Any], x_value: float
) -> np.ndarray:
    if "start_nm" in record:
        endpoints = np.vstack(
            [
                np.asarray(record["start_nm"], dtype=float),
                np.asarray(record["end_nm"], dtype=float),
            ]
        )
        point = endpoints[int(np.argmin(np.abs(endpoints[:, 0] - x_value)))].copy()
        point[0] = x_value
        return point
    center = np.asarray(record["center_nm"], dtype=float).copy()
    center[0] = x_value
    return center


def _topology_points(records: list[dict[str, Any]], half: float) -> np.ndarray:
    return np.vstack(
        [
            _electrode_contact_point(records[0], -half),
            *[_record_midpoint(record) for record in records],
            _electrode_contact_point(records[-1], half),
        ]
    )


def _draw_topology_edges(
    ax: Any, records: list[dict[str, Any]], half: float
) -> int:
    points = _topology_points(records, half)
    for first, second in zip(points[:-1], points[1:], strict=True):
        segment = np.vstack([first, second])
        ax.plot(
            *segment.T,
            color="white",
            linewidth=5.2,
            solid_capstyle="round",
            zorder=9,
        )
        ax.plot(
            *segment.T,
            color=COLOR_TOPOLOGY,
            linewidth=2.3,
            linestyle=(0, (5, 2)),
            solid_capstyle="round",
            zorder=10,
        )
    for point, label in ((points[0], "L"), (points[-1], "R")):
        ax.text(
            *point,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=7.0,
            fontweight="bold",
            bbox={
                "boxstyle": "square,pad=0.28",
                "fc": COLOR_ELECTRODE,
                "ec": "white",
                "lw": 0.9,
            },
            zorder=20,
        )
    return len(points) - 1


def _draw_witness_geometry(ax: Any, records: list[dict[str, Any]]) -> None:
    for record in records:
        if "start_nm" in record:
            start = np.asarray(record["start_nm"], dtype=float)
            end = np.asarray(record["end_nm"], dtype=float)
            ax.plot(
                *np.vstack([start, end]).T,
                color=COLOR_WITNESS,
                linewidth=4.2,
                solid_capstyle="round",
                zorder=5,
            )
            ax.scatter(*start, color=COLOR_WITNESS, s=14, depthshade=False, zorder=6)
            ax.scatter(*end, color=COLOR_WITNESS, s=14, depthshade=False, zorder=6)
        else:
            center = np.asarray(record["center_nm"], dtype=float)
            ax.scatter(
                *center,
                color=COLOR_WITNESS,
                edgecolor="white",
                linewidth=0.7,
                s=80,
                depthshade=False,
                zorder=6,
            )


def _draw_witness_labels(ax: Any, records: list[dict[str, Any]]) -> None:
    for order, record in enumerate(records, start=1):
        midpoint = _record_midpoint(record)
        ax.text(
            *midpoint,
            str(order),
            color="#1F2933",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.20", "fc": "white", "ec": COLOR_WITNESS, "lw": 0.9},
            zorder=22,
        )


def _draw_contact_chain(
    ax: Any, scene: dict[str, Any], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    nodes = scene["traceability"]["mixed_witness"]["nodes"]
    same_source_edges = int(
        scene["traceability"]["mixed_witness"]["same_source_edges"]
    )
    y_values = np.linspace(0.86, 0.14, len(nodes))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    for index in range(len(nodes) - 1):
        ax.plot(
            [0.50, 0.50],
            [y_values[index], y_values[index + 1]],
            color=COLOR_TOPOLOGY,
            linewidth=2.3,
            solid_capstyle="round",
        )
    for index, node in enumerate(nodes):
        electrode = node.startswith("electrode_")
        face = COLOR_ELECTRODE if electrode else COLOR_WITNESS
        marker = "s" if electrode else "o"
        ax.scatter(
            [0.50],
            [y_values[index]],
            s=165 if electrode else 125,
            marker=marker,
            color=face,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        marker_text = "L" if index == 0 else "R" if index == len(nodes) - 1 else str(index)
        ax.text(
            0.50,
            y_values[index],
            marker_text,
            ha="center",
            va="center",
            color="white",
            fontsize=7.2,
            fontweight="bold",
            zorder=4,
        )
    internal_count = len(nodes) - 2
    chain_text = f"L  →  1  →  …  →  {internal_count}  →  R"
    ax.text(
        0.50,
        0.955,
        chain_text,
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#263238",
    )
    ax.text(
        0.50,
        0.045,
        f"接触边 {len(edges)}  ·  同源内部边 {same_source_edges}",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#46515C",
    )
    return {
        "ordered_labels": [
            "L",
            *[str(index) for index in range(1, len(nodes) - 1)],
            "R",
        ],
        "edge_count": len(edges),
        "same_source_internal_edge_count": same_source_edges,
        "source_ids_rendered": False,
        "contact_type_labels_rendered": False,
    }


def _pixel_audit(path: Path, expected_topology_edges: int) -> dict[str, Any]:
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.int16)
    non_white = np.any(rgb < 248, axis=2)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    orange = (red > 180) & (green > 45) & (green < 150) & (blue < 100)
    teal = (red < 55) & (green > 85) & (green < 165) & (blue > 90) & (blue < 180)
    dark = np.max(rgb, axis=2) < 110
    left_width = int(0.66 * rgb.shape[1])
    left_teal = teal[:, :left_width]
    minimum_left_teal_pixels = max(200, 70 * expected_topology_edges)
    checks = {
        "nonblank": int(non_white.sum()) > 10_000,
        "witness_orange_visible": int(orange.sum()) > 1_000,
        "topology_edges_visible_in_3d_panel": int(left_teal.sum())
        > minimum_left_teal_pixels,
        "dark_graph_and_labels_visible": int(dark.sum()) > 1_000,
    }
    return {
        "dimensions": [int(rgb.shape[1]), int(rgb.shape[0])],
        "non_white_pixel_count": int(non_white.sum()),
        "witness_orange_pixel_count": int(orange.sum()),
        "topology_teal_pixel_count": int(teal.sum()),
        "left_panel_topology_teal_pixel_count": int(left_teal.sum()),
        "minimum_left_panel_topology_teal_pixel_count": minimum_left_teal_pixels,
        "dark_pixel_count": int(dark.sum()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_figure(
    scene_path: Path, png_path: Path, pdf_path: Path, audit_path: Path
) -> dict[str, Any]:
    scene_path = scene_path.expanduser().resolve()
    png_path = png_path.expanduser().resolve()
    pdf_path = pdf_path.expanduser().resolve()
    audit_path = audit_path.expanduser().resolve()
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    if scene.get("publication_status") != FINAL_PUBLICATION_STATUS:
        raise ValueError("见证聚焦图只接受正式随机 trial 场景")
    trace = scene.get("traceability", {})
    design = trace.get("design_counts", {})
    if int(design.get("n_a", -1)) < 0 or int(design.get("n_b", -1)) < 0:
        raise ValueError("正式场景缺少设计数量")
    if int(design["n_b"]) == 0 and scene.get("spheres"):
        raise ValueError("N_B=0 时不得绘制 B 几何")
    records, edges = _witness_records(scene)
    half = 0.5 * float(scene["box"]["length_nm"])
    focus_limits = _witness_focus_limits(records, half)
    focus_contains_all = _focus_contains_records(records, focus_limits)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#46515C",
            "text.color": "#263238",
            "axes.labelcolor": "#263238",
        }
    )
    figure = plt.figure(figsize=(11.2, 6.2), facecolor="white")
    panel_widths = (1.86, 1.0)
    three_d_panel_fraction = panel_widths[0] / sum(panel_widths)
    grid = figure.add_gridspec(1, 2, width_ratios=panel_widths, wspace=0.04)
    ax3d = figure.add_subplot(grid[0, 0], projection="3d")
    chain_ax = figure.add_subplot(grid[0, 1])
    _draw_focus_frame(ax3d, focus_limits)
    _draw_electrodes(ax3d, half, focus_limits[1], focus_limits[2])
    _draw_witness_geometry(ax3d, records)
    rendered_topology_edges = _draw_topology_edges(ax3d, records, half)
    _draw_witness_labels(ax3d, records)
    ax3d.set_xlim(*focus_limits[0])
    ax3d.set_ylim(*focus_limits[1])
    ax3d.set_zlim(*focus_limits[2])
    spans = [high - low for low, high in focus_limits]
    ax3d.set_box_aspect(spans)
    ax3d.set_proj_type("ortho")
    ax3d.view_init(elev=22.0, azim=-55.0)
    ax3d.set_xlabel("x (nm)", labelpad=5)
    ax3d.set_ylabel("y (nm)", labelpad=5)
    ax3d.set_zlabel("z (nm)", labelpad=5)
    ax3d.set_xticks([-half, 0.0, half])
    ax3d.set_yticks(np.linspace(*focus_limits[1], 4)[1:])
    ax3d.set_zticks(np.linspace(*focus_limits[2], 3))
    ax3d.grid(False)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_alpha(0.0)
    trial_id = int(trace["random_stream"]["trial_id"])
    ax3d.set_title(
        f"三维导通见证（样本 {trial_id:06d}）",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax3d.text2D(
        0.01,
        0.01,
        "x 方向保留全域，y-z 按见证包围盒局部放大",
        transform=ax3d.transAxes,
        fontsize=8.2,
        color="#46515C",
    )
    chain_contract = _draw_contact_chain(chain_ax, scene, edges)
    figure.subplots_adjust(left=0.035, right=0.985, top=0.94, bottom=0.07)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_png = png_path.with_name(f".{png_path.name}.{os.getpid()}.tmp.png")
    temporary_pdf = pdf_path.with_name(f".{pdf_path.name}.{os.getpid()}.tmp.pdf")
    try:
        figure.savefig(
            temporary_png,
            dpi=300,
            facecolor="white",
            metadata={"Software": "build_q4_witness_figure.py"},
        )
        figure.savefig(
            temporary_pdf,
            facecolor="white",
            metadata={
                "Creator": "build_q4_witness_figure.py",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        os.replace(temporary_png, png_path)
        os.replace(temporary_pdf, pdf_path)
    finally:
        plt.close(figure)
        for temporary in (temporary_png, temporary_pdf):
            if temporary.exists():
                temporary.unlink()

    pixels = _pixel_audit(png_path, rendered_topology_edges)
    checks = {
        "publication_status_final": True,
        "ordered_witness_valid": True,
        "same_source_edges_zero": True,
        "b_absence_matches_n_b_zero": int(design["n_b"]) != 0 or not scene.get("spheres"),
        "focus_bounds_include_all_witness_geometry": focus_contains_all,
        "rendered_topology_edge_count_matches_witness": rendered_topology_edges
        == len(edges),
        "three_d_panel_is_about_65_percent": 0.63
        <= three_d_panel_fraction
        <= 0.67,
        "right_panel_omits_source_ids": not chain_contract["source_ids_rendered"],
        "right_panel_omits_repeated_contact_labels": not chain_contract[
            "contact_type_labels_rendered"
        ],
        "right_panel_reports_same_source_zero": chain_contract[
            "same_source_internal_edge_count"
        ]
        == 0,
        "png_pixels_passed": pixels["passed"],
        "pdf_nonempty": pdf_path.stat().st_size > 0,
    }
    audit = {
        "kind": "q4_conductive_witness_figure_audit",
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "source_scene": str(scene_path),
        "source_scene_sha256": sha256(scene_path),
        "design": {"n_a": int(design["n_a"]), "n_b": int(design["n_b"])},
        "trial_id": trial_id,
        "boundary_primary": trace.get("boundary_primary"),
        "witness_ids_in_order": [record["id"] for record in records],
        "witness_edge_ids_in_order": [edge["edge_id"] for edge in edges],
        "rendered_topology_edge_count": rendered_topology_edges,
        "render_contract": {
            "geometry": "exact scene coordinates; cylinder centerlines emphasized for print visibility",
            "background_a": "omitted in focus view; retained in axonometric overview",
            "medium_b": "absent because N_B=0; no B geometry invented",
            "topology_edges": "ordered graph edges join electrode contact markers and witness-fragment midpoints; not interpreted as current trajectories",
            "layout": {
                "three_d_panel_fraction": three_d_panel_fraction,
                "right_panel_fraction": 1.0 - three_d_panel_fraction,
            },
            "right_panel": chain_contract,
            "focus_limits_nm": {
                axis: [float(value) for value in limits]
                for axis, limits in zip(("x", "y", "z"), focus_limits, strict=True)
            },
            "view": {"projection": "orthographic", "elevation_deg": 22.0, "azimuth_deg": -55.0},
            "palette": {
                "medium_a": COLOR_A,
                "medium_b": COLOR_B,
                "witness": COLOR_WITNESS,
                "electrode": COLOR_ELECTRODE,
                "boundary": COLOR_BOUNDARY,
                "topology": COLOR_TOPOLOGY,
            },
        },
        "png": {"path": str(png_path), "sha256": sha256(png_path), "size_bytes": png_path.stat().st_size},
        "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path), "size_bytes": pdf_path.stat().st_size},
        "pixels": pixels,
        "checks": checks,
    }
    atomic_write_json(audit_path, audit)
    if audit["status"] != "passed":
        raise RuntimeError("问题4导通见证聚焦图审计未通过")
    return audit


def main() -> int:
    args = parse_args()
    audit_path = args.audit or args.png.with_suffix(".audit.json")
    result = build_figure(args.scene, args.png, args.pdf, audit_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
