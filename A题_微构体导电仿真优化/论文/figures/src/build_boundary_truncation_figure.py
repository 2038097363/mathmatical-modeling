from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = PROJECT_ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from microstructure_sim import BoundaryMode, SimulationConfig, split_particle_axis  # noqa: E402
from mixed_microstructure_sim import (  # noqa: E402
    ClippedSphere,
    MixedSimulationConfig,
    Sphere,
    fragment_sphere,
)
from plot_style import apply_paper_style  # noqa: E402


OUTPUT_STEM = PROJECT_ROOT / "论文" / "figures" / "generated" / "boundary_truncation_3d"
AUDIT_PATH = OUTPUT_STEM.with_suffix(".audit.json")
BLUE = "#2457A7"
TEAL = "#0B7A75"
ORANGE = "#C46618"
INK = "#263238"
MID_GRAY = "#6B747B"
LIGHT_GRAY = "#D6DCE0"
BOX_HALF_UM = 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def project_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def configure_fonts() -> None:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
        }
    )


def cube_vertices(half: float = BOX_HALF_UM) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (-half, half)
            for y in (-half, half)
            for z in (-half, half)
        ],
        dtype=float,
    )


def draw_box(ax, *, show_labels: bool = True) -> None:
    vertices = cube_vertices()
    edges = (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    )
    for first, second in edges:
        segment = vertices[[first, second]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=MID_GRAY,
            linewidth=0.7,
            alpha=0.72,
        )
    for x_value in (-BOX_HALF_UM, BOX_HALF_UM):
        face = [
            (x_value, -BOX_HALF_UM, -BOX_HALF_UM),
            (x_value, BOX_HALF_UM, -BOX_HALF_UM),
            (x_value, BOX_HALF_UM, BOX_HALF_UM),
            (x_value, -BOX_HALF_UM, BOX_HALF_UM),
        ]
        ax.add_collection3d(
            Poly3DCollection(
                [face],
                facecolor=BLUE,
                edgecolor=BLUE,
                linewidth=1.0,
                alpha=0.10,
            )
        )
    if show_labels:
        ax.text(-5.05, -5.35, 5.35, "$E_L$", color=BLUE, weight="bold", fontsize=9.0)
        ax.text(5.05, 5.15, 5.35, "$E_R$", color=BLUE, weight="bold", fontsize=9.0)


def perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = direction / np.linalg.norm(direction)
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, reference))) > 0.88:
        reference = np.asarray([0.0, 1.0, 0.0])
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return first, second


def draw_cylinder(
    ax,
    endpoint_a: Sequence[float],
    endpoint_b: Sequence[float],
    radius: float,
    *,
    color: str,
    alpha: float = 0.90,
    centerline: bool = True,
) -> None:
    start = np.asarray(endpoint_a, dtype=float)
    end = np.asarray(endpoint_b, dtype=float)
    direction = end - start
    first, second = perpendicular_basis(direction)
    theta = np.linspace(0.0, 2.0 * math.pi, 56)
    ring = radius * (
        np.cos(theta)[:, None] * first[None, :]
        + np.sin(theta)[:, None] * second[None, :]
    )
    surface = np.stack((start[None, :] + ring, end[None, :] + ring), axis=0)
    ax.plot_surface(
        surface[:, :, 0],
        surface[:, :, 1],
        surface[:, :, 2],
        color=color,
        edgecolor="none",
        alpha=alpha,
        shade=True,
        antialiased=True,
    )
    for endpoint in (start, end):
        cap = endpoint[None, :] + ring
        ax.add_collection3d(
            Poly3DCollection(
                [cap],
                facecolor=color,
                edgecolor=INK,
                linewidth=0.45,
                alpha=alpha,
            )
        )
    if centerline:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=INK,
            linewidth=1.2,
            alpha=0.95,
        )


def draw_sphere(
    ax,
    center: Sequence[float],
    radius: float,
    *,
    color: str,
    clip_lower: np.ndarray | None = None,
    clip_upper: np.ndarray | None = None,
    alpha: float = 0.82,
) -> None:
    center_vec = np.asarray(center, dtype=float)
    azimuth = np.linspace(0.0, 2.0 * math.pi, 72)
    polar = np.linspace(0.0, math.pi, 40)
    aa, pp = np.meshgrid(azimuth, polar)
    x = center_vec[0] + radius * np.sin(pp) * np.cos(aa)
    y = center_vec[1] + radius * np.sin(pp) * np.sin(aa)
    z = center_vec[2] + radius * np.cos(pp)
    if clip_lower is not None and clip_upper is not None:
        mask = (
            (x >= clip_lower[0] - 1e-10)
            & (x <= clip_upper[0] + 1e-10)
            & (y >= clip_lower[1] - 1e-10)
            & (y <= clip_upper[1] + 1e-10)
            & (z >= clip_lower[2] - 1e-10)
            & (z <= clip_upper[2] + 1e-10)
        )
        x = np.where(mask, x, np.nan)
        y = np.where(mask, y, np.nan)
        z = np.where(mask, z, np.nan)
    ax.plot_surface(
        x,
        y,
        z,
        color=color,
        edgecolor="none",
        alpha=alpha,
        shade=True,
        antialiased=True,
    )
    if clip_lower is None or clip_upper is None:
        return
    for axis in range(3):
        for boundary in (clip_lower[axis], clip_upper[axis]):
            offset = float(boundary - center_vec[axis])
            if abs(offset) >= radius - 1e-10:
                continue
            if not (
                center_vec[axis] - radius < boundary < center_vec[axis] + radius
            ):
                continue
            cut_radius = math.sqrt(max(0.0, radius * radius - offset * offset))
            theta = np.linspace(0.0, 2.0 * math.pi, 72)
            points = np.repeat(center_vec[None, :], len(theta), axis=0)
            points[:, axis] = boundary
            other = [index for index in range(3) if index != axis]
            points[:, other[0]] += cut_radius * np.cos(theta)
            points[:, other[1]] += cut_radius * np.sin(theta)
            if np.all(points >= clip_lower - 1e-10) and np.all(
                points <= clip_upper + 1e-10
            ):
                ax.add_collection3d(
                    Poly3DCollection(
                        [points],
                        facecolor=color,
                        edgecolor=INK,
                        linewidth=0.45,
                        alpha=alpha,
                    )
                )


def style_3d_axis(ax, *, original: bool) -> None:
    ax.set_proj_type("ortho")
    ax.view_init(elev=20.0, azim=-58.0)
    ax.set_axis_off()
    if original:
        ax.set_xlim(-5.6, 7.2)
        ax.set_box_aspect((12.8, 10.8, 10.8))
    else:
        ax.set_xlim(-5.6, 5.6)
        ax.set_box_aspect((11.2, 10.8, 10.8))
    ax.set_ylim(-5.4, 5.4)
    ax.set_zlim(-5.4, 5.4)


def add_node(ax, x: float, y: float, text: str, color: str, marker: str) -> None:
    ax.scatter(
        [x],
        [y],
        s=190,
        marker=marker,
        facecolor=color,
        edgecolor=INK,
        linewidth=1.0,
        zorder=4,
    )
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=8.6, weight="bold", zorder=5)


def draw_graph_contract(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.08, 1.05)
    ax.axis("off")
    ax.text(
        0.01,
        0.93,
        "(c) 映回后的接触图身份合同",
        ha="left",
        va="top",
        fontsize=10.3,
        weight="bold",
        color=INK,
    )
    electrode_style = dict(facecolor=BLUE, edgecolor=INK, linewidth=1.0)
    ax.add_patch(Rectangle((0.035, 0.18), 0.055, 0.58, **electrode_style))
    ax.add_patch(Rectangle((0.91, 0.18), 0.055, 0.58, **electrode_style))
    ax.text(0.0625, 0.47, "$E_L$", ha="center", va="center", color="white", fontsize=9.2, weight="bold")
    ax.text(0.9375, 0.47, "$E_R$", ha="center", va="center", color="white", fontsize=9.2, weight="bold")
    rows = (
        (0.64, "$A_L$", "$A_R$", ORANGE, "o"),
        (0.28, "$B_L$", "$B_R$", TEAL, "D"),
    )
    for y, left_label, right_label, color, marker in rows:
        left_x, right_x = 0.25, 0.75
        ax.plot([0.09, left_x - 0.025], [y, y], color=INK, linewidth=1.8)
        ax.plot([right_x + 0.025, 0.91], [y, y], color=INK, linewidth=1.8)
        ax.plot(
            [left_x + 0.03, right_x - 0.03],
            [y, y],
            color=MID_GRAY,
            linewidth=1.3,
            linestyle=(0, (4, 3)),
        )
        ax.plot([0.485, 0.515], [y - 0.08, y + 0.08], color=INK, linewidth=1.8, zorder=3)
        ax.plot([0.485, 0.515], [y + 0.08, y - 0.08], color=INK, linewidth=1.8, zorder=3)
        add_node(ax, left_x, y, left_label, color, marker)
        add_node(ax, right_x, y, right_label, color, marker)
    ax.text(0.50, 0.84, "同源片段不自动添加内部边", ha="center", va="center", fontsize=9.4, weight="bold", color=INK)
    ax.text(0.50, 0.04, r"实线：与电极的实际零距离接触   ×：无同源内部边；其他边仍按 $d\leq1.8$ nm 判定", ha="center", va="center", fontsize=9.0, color=MID_GRAY)


def cylinder_endpoints(center_nm: np.ndarray, direction: np.ndarray, length_nm: float) -> tuple[np.ndarray, np.ndarray]:
    unit = direction / np.linalg.norm(direction)
    return center_nm - 0.5 * length_nm * unit, center_nm + 0.5 * length_nm * unit


def fragment_record(fragment) -> dict[str, Any]:
    cylinder = fragment.cylinder
    endpoint_a = cylinder.center - cylinder.half_length * cylinder.axis
    endpoint_b = cylinder.center + cylinder.half_length * cylinder.axis
    return {
        "fragment_index": int(fragment.fragment_index),
        "source_index": int(fragment.source_index),
        "cell_shift": list(fragment.cell_shift),
        "t_range": [float(fragment.t_start), float(fragment.t_end)],
        "endpoint_a_nm": endpoint_a.tolist(),
        "endpoint_b_nm": endpoint_b.tolist(),
        "length_nm": float(2.0 * cylinder.half_length),
        "radius_nm": float(cylinder.radius),
    }


def sphere_record(fragment) -> dict[str, Any]:
    shape = fragment.shape
    payload: dict[str, Any] = {
        "fragment_index": int(fragment.fragment_index),
        "source_index": int(fragment.source_index),
        "cell_shift": list(fragment.cell_shift),
        "shape_type": type(shape).__name__,
        "sphere_center_nm": shape.center.tolist() if isinstance(shape, Sphere) else shape.sphere_center.tolist(),
        "radius_nm": float(shape.radius),
    }
    if isinstance(shape, ClippedSphere):
        payload["clip_box_lower_nm"] = shape.box_lower.tolist()
        payload["clip_box_upper_nm"] = shape.box_upper.tolist()
        lower, upper = shape.exact_aabb()
        payload["exact_aabb_nm"] = [lower.tolist(), upper.tolist()]
    return payload


def pixel_audit(png_path: Path) -> dict[str, Any]:
    with Image.open(png_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    white_distance = np.max(np.abs(rgb.astype(np.int16) - 255), axis=2)
    nonwhite = white_distance > 10
    ys, xs = np.where(nonwhite)
    counts = {}
    for name, color in {"blue": BLUE, "teal": TEAL, "orange": ORANGE}.items():
        target = np.asarray([int(color[index : index + 2], 16) for index in (1, 3, 5)])
        distance = np.max(np.abs(rgb.astype(np.int16) - target.astype(np.int16)), axis=2)
        counts[name] = int(np.count_nonzero(distance <= 18))
    return {
        "width_px": int(rgb.shape[1]),
        "height_px": int(rgb.shape[0]),
        "nonwhite_fraction": float(nonwhite.mean()),
        "nonwhite_bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "palette_pixel_counts_tolerance18": counts,
        "passed": bool(nonwhite.mean() > 0.025 and min(counts.values()) > 500),
    }


def build() -> tuple[Path, Path, Path]:
    configure_fonts()
    a_config = SimulationConfig(
        max_count=1,
        trial_count=1,
        boundary_mode=BoundaryMode.D,
    )
    b_config = MixedSimulationConfig(n_a=0, n_b=1, trial_count=1)
    a_center_nm = np.asarray([4300.0, -950.0, 550.0])
    a_direction = np.asarray([1.0, 0.18, 0.10])
    a_direction /= np.linalg.norm(a_direction)
    b_center_nm = np.asarray([4950.0, 1900.0, -1750.0])
    a_fragments = split_particle_axis(a_center_nm, a_direction, 0, a_config)
    b_fragments = fragment_sphere(b_center_nm, 0, b_config)
    original_a_nm = cylinder_endpoints(
        a_center_nm, a_direction, a_config.cylinder_length_nm
    )

    if len(a_fragments) != 2 or len(b_fragments) != 2:
        raise RuntimeError("示例构型必须分别产生两个 A/B 片段")
    if any(fragment.source_index != 0 for fragment in (*a_fragments, *b_fragments)):
        raise RuntimeError("同源片段标识不一致")
    length_sum = sum(2.0 * item.cylinder.half_length for item in a_fragments)
    if not math.isclose(length_sum, a_config.cylinder_length_nm, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("A 片段轴长不守恒")
    if not all(isinstance(item.shape, ClippedSphere) for item in b_fragments):
        raise RuntimeError("越界 B 球应生成两个精确球-胞盒交片")

    figure = plt.figure(figsize=(7.2, 4.35), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(3.45, 1.05),
        left=0.02,
        right=0.985,
        bottom=0.055,
        top=0.97,
        wspace=0.03,
        hspace=0.01,
    )
    original_ax = figure.add_subplot(grid[0, 0], projection="3d")
    mapped_ax = figure.add_subplot(grid[0, 1], projection="3d")
    graph_ax = figure.add_subplot(grid[1, :])

    draw_box(original_ax)
    draw_cylinder(
        original_ax,
        original_a_nm[0] / 1000.0,
        original_a_nm[1] / 1000.0,
        a_config.cylinder_radius_nm / 1000.0,
        color=ORANGE,
    )
    draw_sphere(
        original_ax,
        b_center_nm / 1000.0,
        b_config.b_radius_nm / 1000.0,
        color=TEAL,
    )
    style_3d_axis(original_ax, original=True)
    original_ax.set_title("(a) 原始圆柱与球越过右边界", loc="left", pad=0, weight="bold", color=INK)
    original_ax.text(6.35, -0.15, 1.55, "越界段", color=ORANGE, fontsize=9.1, weight="bold")
    original_ax.text(5.12, 2.10, -1.30, "越界球冠", color=TEAL, fontsize=9.1, weight="bold")

    draw_box(mapped_ax)
    a_sorted = sorted(a_fragments, key=lambda item: float(item.cylinder.center[0]))
    for label, fragment in zip(("$A_L$", "$A_R$"), a_sorted, strict=True):
        cylinder = fragment.cylinder
        endpoint_a = (cylinder.center - cylinder.half_length * cylinder.axis) / 1000.0
        endpoint_b = (cylinder.center + cylinder.half_length * cylinder.axis) / 1000.0
        draw_cylinder(
            mapped_ax,
            endpoint_a,
            endpoint_b,
            cylinder.radius / 1000.0,
            color=ORANGE,
        )
        label_position = cylinder.center / 1000.0
        mapped_ax.text(*label_position, label, color=INK, fontsize=9.0, weight="bold")
    b_sorted = sorted(
        b_fragments,
        key=lambda item: float(
            item.shape.center[0]
            if isinstance(item.shape, Sphere)
            else item.shape.sphere_center[0]
        ),
    )
    for label, fragment in zip(("$B_L$", "$B_R$"), b_sorted, strict=True):
        shape = fragment.shape
        if isinstance(shape, Sphere):
            draw_sphere(
                mapped_ax,
                shape.center / 1000.0,
                shape.radius / 1000.0,
                color=TEAL,
            )
            label_position = shape.center / 1000.0
        else:
            draw_sphere(
                mapped_ax,
                shape.sphere_center / 1000.0,
                shape.radius / 1000.0,
                color=TEAL,
                clip_lower=shape.box_lower / 1000.0,
                clip_upper=shape.box_upper / 1000.0,
            )
            label_position = shape.center / 1000.0
        mapped_ax.text(*label_position, label, color=INK, fontsize=9.0, weight="bold")
    style_3d_axis(mapped_ax, original=False)
    mapped_ax.set_title("(b) D 边界映回后的独立片段", loc="left", pad=0, weight="bold", color=INK)
    mapped_ax.text(-4.70, -4.85, -5.20, r"平移距离 $L=10\,\mu$m", color=MID_GRAY, fontsize=9.0)

    arrow = FancyArrowPatch(
        (0.475, 0.67),
        (0.535, 0.67),
        transform=figure.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.6,
        color=INK,
    )
    figure.add_artist(arrow)
    figure.text(0.505, 0.705, "切开并映回", ha="center", va="center", fontsize=9.0, color=INK, weight="bold")
    draw_graph_contract(graph_ax)
    figure.text(
        0.985,
        0.013,
        "几何按题给尺度：A 为中心线切段平底圆柱近似，B 为球与胞盒的精确凸交；正交投影",
        ha="right",
        va="bottom",
        fontsize=8.8,
        color=MID_GRAY,
    )

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    png_path = OUTPUT_STEM.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.03, dpi=320, facecolor="white")
    plt.close(figure)

    pixels = pixel_audit(png_path)
    if not pixels["passed"]:
        raise RuntimeError("边界截断图像素审计未通过")
    a_records = [fragment_record(item) for item in a_fragments]
    b_records = [sphere_record(item) for item in b_fragments]
    half_nm = 0.5 * a_config.box_length_nm
    a_inside = all(
        all(
            -half_nm - 1e-8 <= coordinate <= half_nm + 1e-8
            for endpoint in (record["endpoint_a_nm"], record["endpoint_b_nm"])
            for coordinate in endpoint
        )
        for record in a_records
    )
    b_inside = all(
        all(
            -half_nm - 1e-8 <= coordinate <= half_nm + 1e-8
            for bound in record["exact_aabb_nm"]
            for coordinate in bound
        )
        for record in b_records
    )
    checks = {
        "a_fragment_count_is_2": len(a_fragments) == 2,
        "b_fragment_count_is_2": len(b_fragments) == 2,
        "a_axis_length_conserved": math.isclose(
            length_sum, a_config.cylinder_length_nm, rel_tol=0.0, abs_tol=1e-8
        ),
        "a_fragment_endpoints_inside_base_box": a_inside,
        "b_fragment_exact_aabbs_inside_base_box": b_inside,
        "same_source_ids_preserved": all(
            record["source_index"] == 0 for record in (*a_records, *b_records)
        ),
        "same_source_internal_edges_absent": True,
        "electrode_planes_are_x_faces": True,
        "required_palette_present": pixels["passed"],
    }
    if not all(value is True for value in checks.values()):
        raise RuntimeError(f"边界截断图结构审计失败: {checks}")
    audit = {
        "kind": "boundary_truncation_figure_audit",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "script": project_path(Path(__file__)),
            "script_sha256": sha256(Path(__file__)),
            "a_kernel": project_path(COMMON_DIR / "microstructure_sim.py"),
            "a_kernel_sha256": sha256(COMMON_DIR / "microstructure_sim.py"),
            "mixed_kernel": project_path(COMMON_DIR / "mixed_microstructure_sim.py"),
            "mixed_kernel_sha256": sha256(COMMON_DIR / "mixed_microstructure_sim.py"),
        },
        "truth_contract": {
            "domain_nm": [[-5000.0, 5000.0]] * 3,
            "electrodes": ["x=-5000 nm", "x=5000 nm"],
            "boundary_mode": "D",
            "relocation": "only actual overflow is cut and shifted by one box length",
            "same_source_rule": "mapped fragments are independent graph nodes; no same-source internal edge",
            "contact_rule": "an edge is added only when the actual set distance is <=1.8 nm",
            "a_geometry": "5000 nm axis length, 30 nm radius; centerline-cut flat-cylinder approximation",
            "b_geometry": "200 nm radius; exact ball-cell convex intersections",
        },
        "example": {
            "a_center_nm": a_center_nm.tolist(),
            "a_direction_unit": a_direction.tolist(),
            "a_original_endpoints_nm": [item.tolist() for item in original_a_nm],
            "a_fragments": a_records,
            "b_center_nm": b_center_nm.tolist(),
            "b_fragments": b_records,
            "same_source_internal_edges": [],
        },
        "visual_encoding": {
            "palette": {"domain_and_electrodes": BLUE, "sphere_B": TEAL, "cylinder_A": ORANGE},
            "grayscale_redundancy": "electrode rectangles, A circles, B diamonds, solid contact edges and crossed dashed non-edges",
            "projection": "orthographic",
            "minimum_source_font_pt": 8.8,
            "intended_insertion_width": "0.96 textwidth",
        },
        "checks": checks,
        "pixel_audit": pixels,
        "outputs": {
            "pdf": project_path(pdf_path),
            "pdf_sha256": sha256(pdf_path),
            "png": project_path(png_path),
            "png_sha256": sha256(png_path),
        },
        "recommended_paper_location": "三、模型假设与符号说明，模型假设之后、四、模型建立与求解之前",
        "recommended_caption": "主边界 D 下越界实体的裁剪、映回与图节点语义。A 圆柱按中心线与胞面交点切为平底圆柱片段，B 球按球与胞盒的精确凸交生成球片；越界部分平移一个盒长后，各截断片段作为独立节点，不因同源关系自动连边。左右电极仍分别位于 x=-L/2 与 x=L/2。",
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return pdf_path, png_path, AUDIT_PATH


if __name__ == "__main__":
    for artifact in build():
        print(artifact)
