from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = PROJECT_ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from microstructure_sim import (
    clopper_pearson_one_sided_bounds,
    load_threshold_artifact,
)
from plot_style import PAPER_COLORS, SEMANTIC_COLORS, apply_paper_style, save_figure


Q1_VALIDATION = PROJECT_ROOT / "问题" / "问题1" / "results" / "independent_slsqp_validation.csv"
Q2_SUMMARY = PROJECT_ROOT / "问题" / "问题2" / "results" / "D_primary_n20000" / "q2_summary.json"
Q4_ANALYSIS = (
    PROJECT_ROOT
    / "问题"
    / "问题4"
    / "results"
    / "D_screen2000_confirm50000"
    / "q4_confirmation_integer_domain_analysis.json"
)
OUTPUT_DIR = PROJECT_ROOT / "论文" / "figures" / "generated"
WORKFLOW_STEM = OUTPUT_DIR / "model_workflow"
VALIDATION_STEM = OUTPUT_DIR / "validation_diagnostics"
Q4_BOUNDARY_STEM = OUTPUT_DIR / "q4_unresolved_boundary_evidence"
CONVERGENCE_STEM = OUTPUT_DIR / "simulation_convergence"
Q1_3D_STEM = OUTPUT_DIR / "q1_groups_3d"
AUDIT_PATH = OUTPUT_DIR / "explanatory_figures.audit.json"
Q4_BOUNDARY_AUDIT_PATH = OUTPUT_DIR / "q4_unresolved_boundary_evidence.audit.json"
CONVERGENCE_AUDIT_PATH = OUTPUT_DIR / "simulation_convergence.audit.json"
IMAGEN_WORKFLOW_ASSET = PROJECT_ROOT / "论文" / "figures" / "assets" / "model_workflow_imagen_v1.png"
IMAGEN_WORKFLOW_PROMPT = (
    PROJECT_ROOT / "论文" / "figures" / "prompts" / "model_workflow_imagen_prompt.md"
)
Q1_SCENE_PATHS = [
    PROJECT_ROOT / "论文" / "figures" / "data" / f"q1_group{index}_scene.json"
    for index in (1, 2, 3)
]


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
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#4A4A4A",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.35,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
        )
    )


def add_stage_header(ax, x: float, text: str, color: str) -> None:
    ax.add_patch(Rectangle((x, 0.835), 0.032, 0.012, facecolor=color, edgecolor="none"))
    ax.text(x + 0.042, 0.841, text, ha="left", va="center", fontsize=9.2, weight="bold", color="#252525")


def draw_microstructure_icon(ax) -> None:
    cell_x, cell_y, cell_w, cell_h = 0.045, 0.325, 0.182, 0.405
    ax.add_patch(
        Rectangle(
            (cell_x, cell_y),
            cell_w,
            cell_h,
            facecolor="#F4F7F9",
            edgecolor="#6B737A",
            linewidth=1.05,
        )
    )
    electrode = "#444444"
    ax.add_patch(Rectangle((cell_x - 0.008, cell_y), 0.011, cell_h, color=electrode))
    ax.add_patch(Rectangle((cell_x + cell_w - 0.003, cell_y), 0.011, cell_h, color=electrode))

    cylinders = [
        ((0.064, 0.408), (0.119, 0.456)),
        ((0.091, 0.646), (0.151, 0.592)),
        ((0.141, 0.372), (0.199, 0.421)),
        ((0.167, 0.685), (0.223, 0.648)),
    ]
    for start, end in cylinders:
        ax.plot(*zip(start, end), color="#2F4B5C", linewidth=6.0, solid_capstyle="round", zorder=2)
        ax.plot(*zip(start, end), color="#4C9BD6", linewidth=3.8, solid_capstyle="round", zorder=3)
    for center, radius in [((0.081, 0.548), 0.014), ((0.137, 0.514), 0.018), ((0.194, 0.542), 0.012)]:
        ax.add_patch(Circle(center, radius, facecolor="#CC79A7", edgecolor="#5A3650", linewidth=0.8, zorder=3))

    ax.text(0.136, 0.275, "A 有限圆柱   B 球/球片", ha="center", va="center", fontsize=7.7, color="#333333")
    ax.text(0.136, 0.235, "截断片段独立 · 两端为电极", ha="center", va="center", fontsize=7.4, color="#555555")


def draw_contact_graph_icon(ax) -> None:
    left_x, right_x = 0.305, 0.527
    ax.add_patch(Rectangle((left_x, 0.35), 0.012, 0.35, facecolor="#444444", edgecolor="none"))
    ax.add_patch(Rectangle((right_x, 0.35), 0.012, 0.35, facecolor="#444444", edgecolor="none"))
    nodes = {
        "a": (0.344, 0.444),
        "b": (0.371, 0.624),
        "c": (0.414, 0.527),
        "d": (0.454, 0.646),
        "e": (0.491, 0.531),
        "f": (0.462, 0.397),
        "g": (0.385, 0.386),
    }
    edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
        ("c", "e"),
        ("c", "f"),
        ("c", "g"),
        ("d", "e"),
        ("e", "f"),
        ("f", "g"),
    ]
    for first, second in edges:
        x_values, y_values = zip(nodes[first], nodes[second])
        ax.plot(x_values, y_values, color="#A9B0B5", linewidth=1.0, zorder=1)
    witness = [((left_x + 0.012, 0.45), nodes["a"]), (nodes["a"], nodes["c"]), (nodes["c"], nodes["e"]), (nodes["e"], (right_x, 0.54))]
    for start, end in witness:
        ax.plot(*zip(start, end), color="#D55E00", linewidth=2.4, solid_capstyle="round", zorder=2)
    for name, center in nodes.items():
        on_witness = name in {"a", "c", "e"}
        ax.add_patch(
            Circle(
                center,
                0.012,
                facecolor="#D55E00" if on_witness else "white",
                edgecolor="#8A3C05" if on_witness else "#26765F",
                linewidth=1.1,
                zorder=3,
            )
        )
    ax.text(0.422, 0.752, "距离 ≤ 1.8 nm  →  接触边", ha="center", va="center", fontsize=7.8, color="#333333")
    ax.text(0.422, 0.296, "宽相筛选  ·  GJK 窄相  ·  并查集  ·  路径见证", ha="center", va="center", fontsize=7.3, color="#555555")


def add_question_panel(ax, x: float, y: float, label: str, title: str, color: str) -> None:
    ax.add_patch(Rectangle((x, y), 0.184, 0.235, facecolor="#FAFAFA", edgecolor="#D8DDE1", linewidth=0.8))
    ax.add_patch(Rectangle((x, y + 0.225), 0.184, 0.010, facecolor=color, edgecolor="none"))
    ax.text(x + 0.012, y + 0.198, label, ha="left", va="center", fontsize=8.2, weight="bold", color=color)
    ax.text(x + 0.050, y + 0.198, title, ha="left", va="center", fontsize=7.5, color="#333333")


def draw_question_outputs(ax) -> None:
    x_left, x_right = 0.585, 0.792
    y_top, y_bottom = 0.51, 0.245
    add_question_panel(ax, x_left, y_top, "Q1", "确定性贯通", PAPER_COLORS[0])
    add_question_panel(ax, x_right, y_top, "Q2", "概率曲线", PAPER_COLORS[3])
    add_question_panel(ax, x_left, y_bottom, "Q3", "临界量括区", PAPER_COLORS[2])
    add_question_panel(ax, x_right, y_bottom, "Q4", "成本前沿", PAPER_COLORS[1])

    for index, (label, value, conductive) in enumerate([("组1", 0, False), ("组2", 1, True), ("组3", 1, True)]):
        x = x_left + 0.038 + 0.052 * index
        ax.add_patch(
            Circle(
                (x, y_top + 0.105),
                0.017,
                facecolor="#009E73" if conductive else "white",
                edgecolor="#009E73" if conductive else "#6F767C",
                linewidth=1.1,
            )
        )
        ax.text(x, y_top + 0.105, str(value), ha="center", va="center", fontsize=7.2, color="white" if conductive else "#555555", weight="bold")
        ax.text(x, y_top + 0.060, label, ha="center", va="center", fontsize=6.8, color="#555555")

    x_values = np.linspace(x_right + 0.025, x_right + 0.160, 50)
    normalized = np.linspace(-3.0, 3.0, 50)
    y_values = y_top + 0.045 + 0.112 / (1.0 + np.exp(-normalized))
    ax.plot(x_values, y_values, color=PAPER_COLORS[3], linewidth=1.65)
    for index in (8, 19, 29, 43):
        ax.scatter([x_values[index]], [y_values[index]], s=14, color=PAPER_COLORS[3], edgecolor="white", linewidth=0.4, zorder=4)
    ax.plot([x_right + 0.023, x_right + 0.165], [y_top + 0.045, y_top + 0.045], color="#7B8186", linewidth=0.7)
    ax.plot([x_right + 0.023, x_right + 0.023], [y_top + 0.045, y_top + 0.165], color="#7B8186", linewidth=0.7)

    interval_y = y_bottom + 0.102
    ax.plot([x_left + 0.033, x_left + 0.153], [interval_y, interval_y], color="#6D7378", linewidth=1.0)
    ax.plot([x_left + 0.064, x_left + 0.132], [interval_y, interval_y], color=PAPER_COLORS[2], linewidth=5.0, solid_capstyle="butt")
    for x, text in ((x_left + 0.064, "613"), (x_left + 0.132, "616")):
        ax.plot([x, x], [interval_y - 0.018, interval_y + 0.018], color="#26765F", linewidth=1.2)
        ax.text(x, interval_y - 0.044, text, ha="center", va="center", fontsize=6.8, color="#555555")
    ax.text(x_left + 0.098, y_bottom + 0.153, "0.87%", ha="center", va="center", fontsize=8.2, color="#26765F", weight="bold")

    q4_x = np.asarray([x_right + 0.027, x_right + 0.060, x_right + 0.092, x_right + 0.123, x_right + 0.150])
    q4_y = np.asarray([y_bottom + 0.155, y_bottom + 0.138, y_bottom + 0.111, y_bottom + 0.086, y_bottom + 0.061])
    ax.plot(q4_x, q4_y, color="#6D7378", linewidth=1.0)
    ax.scatter(q4_x[:3], q4_y[:3], s=12, color="#0072B2", zorder=3)
    ax.scatter(q4_x[3:], q4_y[3:], s=23, facecolors="none", edgecolors="#E69F00", linewidth=1.1, marker="D", zorder=3)
    ax.scatter([x_right + 0.160], [y_bottom + 0.052], s=55, marker="*", color="#009E73", edgecolor="#245A49", linewidth=0.5, zorder=4)
    ax.text(x_right + 0.096, y_bottom + 0.160, "排除 / 未决 / 上界", ha="center", va="center", fontsize=6.5, color="#555555")


def build_workflow_figure():
    configure_fonts()
    if not IMAGEN_WORKFLOW_ASSET.is_file():
        raise FileNotFoundError(f"Imagen 证据链底图不存在：{IMAGEN_WORKFLOW_ASSET}")
    background = np.asarray(Image.open(IMAGEN_WORKFLOW_ASSET).convert("RGB"))
    height, width = background.shape[:2]
    figure, ax = plt.subplots(figsize=(7.6, 4.25), constrained_layout=True)
    ax.imshow(background, extent=(0, width, height, 0), interpolation="lanczos")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    panel_bg = "#F4F7F9"
    cover_specs = [
        (36, 72, 122, 190),
        (356, 70, 140, 195),
        (808, 140, 62, 58),
        (34, 479, 452, 52),
        (742, 325, 253, 88),
        (522, 345, 54, 145),
        (565, 435, 145, 67),
        (660, 487, 331, 45),
        (0, 537, width, 35),
    ]
    for x, y, box_width, box_height in cover_specs:
        ax.add_patch(
            Rectangle(
                (x, y),
                box_width,
                box_height,
                facecolor="white" if y >= 537 else panel_bg,
                edgecolor="none",
                zorder=8,
            )
        )

    headers = [
        (356, 14, 140, "几何合同", "#2457A7"),
        (860, 14, 138, "接触图引擎", "#0B7A75"),
        (356, 307, 140, "问题一至三", "#2457A7"),
        (860, 307, 138, "问题四", "#C46618"),
    ]
    for x, y, box_width, label, color in headers:
        ax.add_patch(
            Rectangle(
                (x, y),
                box_width,
                34,
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                zorder=10,
            )
        )
        ax.text(
            x + box_width / 2,
            y + 17,
            label,
            ha="center",
            va="center",
            fontsize=10.2,
            weight="bold",
            color=color,
            zorder=11,
        )

    for x, y, label in ((34, 29, "(a)"), (535, 29, "(b)"), (34, 326, "(c)"), (535, 326, "(d)")):
        ax.add_patch(Rectangle((x - 14, y - 20), 44, 28, facecolor=panel_bg, edgecolor="none", zorder=9))
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5, weight="bold", color="#263238", zorder=11)

    ax.text(101, 130, "介质 A\n有限平底圆柱", ha="center", va="center", fontsize=10.0, color="#263238", zorder=11)
    ax.text(101, 229, "两端电极", ha="center", va="center", fontsize=10.0, weight="bold", color="#2F343B", zorder=11)
    ax.text(428, 136, "介质 B\n球 / 球片", ha="center", va="center", fontsize=10.0, color="#263238", zorder=11)
    ax.text(428, 226, "截断片段\n独立介质", ha="center", va="center", fontsize=10.0, weight="bold", color="#C46618", zorder=11)
    ax.text(840, 169, r"$d\leq 1.8\,\mathrm{nm}$", ha="center", va="center", fontsize=10.2, weight="bold", color="#C46618", zorder=11)
    ax.text(925, 268, "宽相筛选\nGJK 窄相", ha="center", va="center", fontsize=9.0, weight="bold", color="#0B7A75", zorder=11)

    for x, y, value in ((96, 405, "0"), (80, 458, "1"), (130, 444, "1")):
        ax.text(x, y, value, ha="center", va="center", fontsize=10.0, weight="bold", color="white", zorder=11)
    for x, label in ((96, "Q1 贯通结论"), (263, "Q2 概率曲线"), (415, "Q3 临界区间")):
        ax.text(x, 509, label, ha="center", va="center", fontsize=9.6, weight="bold", color="#263238", zorder=11)

    legend_items = [
        (758, 350, "o", "#2457A7", "已排除"),
        (758, 379, "D", "#C46618", "未决"),
        (758, 406, "*", "#0B7A75", "确认候选"),
    ]
    for x, y, marker, color, label in legend_items:
        ax.scatter([x], [y], s=65 if marker != "*" else 95, marker=marker, facecolors="none" if marker == "D" else color, edgecolors=color, linewidth=1.2, zorder=11)
        ax.text(x + 20, y, label, ha="left", va="center", fontsize=9.7, color="#263238", zorder=11)
    ax.text(630, 457, "573 已排除\n46 尚不能排除", ha="center", va="center", fontsize=8.8, weight="bold", color="#35424A", linespacing=1.35, zorder=11)
    ax.text(545, 418, "证据状态", rotation=90, ha="center", va="center", fontsize=9.5, weight="bold", color="#263238", zorder=11)
    ax.text(825, 511, "成本前沿（结构示意）", ha="center", va="center", fontsize=9.7, weight="bold", color="#263238", zorder=11)

    validation_labels = ["构型检验", "独立求解", "单调性", "独立随机流", "置信界"]
    validation_markers = ["o", "s", "^", "D", "P"]
    validation_x = np.linspace(105, width - 105, len(validation_labels))
    ax.plot([45, width - 45], [553, 553], color="#C7D0D8", linewidth=0.8, zorder=10)
    for x, label, marker in zip(validation_x, validation_labels, validation_markers, strict=True):
        ax.scatter([x], [553], s=42, marker=marker, facecolor="white", edgecolor="#0B7A75", linewidth=1.25, zorder=11)
        ax.text(x + 16, 553, label, ha="left", va="center", fontsize=9.0, color="#35424A", zorder=11)
    return figure


def build_q1_3d_figure():
    configure_fonts()
    scenes = [json.loads(path.read_text(encoding="utf-8-sig")) for path in Q1_SCENE_PATHS]
    figure = plt.figure(figsize=(10.8, 3.65), constrained_layout=True)
    titles = ["组 1：不导通", "组 2：见证路径 4 个片段", "组 3：见证路径 4 个片段"]
    for panel_index, (scene, title) in enumerate(zip(scenes, titles, strict=True), start=1):
        ax = figure.add_subplot(1, 3, panel_index, projection="3d")
        length = float(scene["box"]["length_nm"])
        x_scale = 0.5 * length
        yz_scale = 0.05 * length

        vertices = np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        box_edges = [
            (first, second)
            for first in range(8)
            for second in range(first + 1, 8)
            if np.count_nonzero(vertices[first] != vertices[second]) == 1
        ]
        for first, second in box_edges:
            points = vertices[[first, second]]
            ax.plot(*points.T, color="#9AA5AD", linewidth=0.75, alpha=0.75)
        for x_value in (-1.0, 1.0):
            face = [
                (x_value, -1.0, -1.0),
                (x_value, 1.0, -1.0),
                (x_value, 1.0, 1.0),
                (x_value, -1.0, 1.0),
            ]
            ax.add_collection3d(
                Poly3DCollection(
                    [face],
                    facecolors="#2F343B",
                    edgecolors="#2F343B",
                    linewidths=1.0,
                    alpha=0.07,
                )
            )

        for record in scene["cylinders"]:
            points = np.asarray([record["start_nm"], record["end_nm"]], dtype=float)
            points[:, 0] /= x_scale
            points[:, 1:] /= yz_scale
            points[:, 1:] = np.clip(points[:, 1:], -1.0, 1.0)
            if record.get("role") == "witness":
                ax.plot(*points.T, color="#71310D", linewidth=5.2, solid_capstyle="round", zorder=5)
                ax.plot(*points.T, color="#E66C22", linewidth=3.1, solid_capstyle="round", zorder=6)
            else:
                ax.plot(
                    *points.T,
                    color="#7F8A92",
                    linewidth=1.15,
                    alpha=0.42 if panel_index < 3 else 0.22,
                    solid_capstyle="round",
                    zorder=2,
                )

        ax.set_xlim(-1.03, 1.03)
        ax.set_ylim(-1.03, 1.03)
        ax.set_zlim(-1.03, 1.03)
        ax.set_box_aspect((1.65, 1.0, 1.0))
        ax.set_proj_type("ortho")
        ax.view_init(elev=24.0, azim=-58.0)
        ax.set_xticks([-1.0, 0.0, 1.0], ["-5000", "0", "5000"])
        ax.set_yticks([-1.0, 0.0, 1.0], ["-500", "0", "500"])
        ax.set_zticks([-1.0, 0.0, 1.0], ["-500", "0", "500"])
        ax.tick_params(labelsize=7.7, pad=0)
        ax.set_xlabel("x (nm)", fontsize=8.5, labelpad=2)
        ax.set_ylabel("y (nm)", fontsize=8.5, labelpad=2)
        ax.set_zlabel("z (nm)", fontsize=8.5, labelpad=2)
        ax.set_title(f"({chr(96 + panel_index)}) {title}", fontsize=10.3, weight="bold", pad=4)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.0)

    figure.text(
        0.5,
        0.008,
        "三轴按各自显示范围归一化以辨识接触结构；几何判定始终使用原始 nm 坐标",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#46515C",
    )
    return figure


def read_q1_validation() -> dict[str, np.ndarray | int | float]:
    with Q1_VALIDATION.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    differences = np.asarray([float(row["slsqp_abs_difference_nm"]) for row in rows], dtype=float)
    same_class = np.asarray([row["slsqp_same_threshold_class"].lower() == "true" for row in rows])
    return {
        "differences": differences,
        "row_count": len(rows),
        "disagreements": int(np.count_nonzero(~same_class)),
        "max_difference_nm": float(np.max(differences)),
    }


def read_q2_blocks() -> tuple[list[int], np.ndarray, np.ndarray, int, Path, np.ndarray]:
    q2 = json.loads(Q2_SUMMARY.read_text(encoding="utf-8-sig"))
    threshold_path = Path(q2["threshold_artifact"])
    if not threshold_path.is_absolute():
        threshold_path = PROJECT_ROOT / threshold_path
    _, samples, _ = load_threshold_artifact(threshold_path)
    counts = [int(row["probability"]["count"]) for row in q2["probability_records"]]
    full = np.asarray([float(row["probability"]["estimate"]) for row in q2["probability_records"]])
    block_size = 5000
    blocks = np.asarray(
        [
            [float(np.count_nonzero(samples[start : start + block_size] <= count)) / block_size for count in counts]
            for start in range(0, len(samples), block_size)
        ]
    )
    return counts, full, blocks, block_size, threshold_path, samples


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}的 JSON 顶层必须为对象")
    return payload


def resolve_evidence_path(
    raw: Any,
    owner: Path,
    project_root: Path,
    label: str,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label}缺少路径")
    raw_path = Path(raw).expanduser()
    candidates = [raw_path] if raw_path.is_absolute() else [owner.parent / raw_path, project_root / raw_path]
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(project_root.resolve())
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"{label}不存在或越出项目目录：{candidates[0]}")


def require_hash(path: Path, expected: Any, label: str) -> str:
    actual = sha256(path)
    if str(expected or "").strip().upper() != actual:
        raise ValueError(f"{label}的 SHA-256 不一致")
    return actual


def require_close(left: Any, right: Any, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(float(left), float(right), rtol=0.0, atol=atol):
        raise ValueError(f"{label}不一致")


def load_q4_boundary_evidence(
    analysis_path: Path = Q4_ANALYSIS,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    analysis_path = analysis_path.expanduser().resolve()
    try:
        analysis_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Q4 完整域审计必须位于项目目录内") from exc
    analysis = read_json_object(analysis_path, "Q4 完整域审计")
    if analysis.get("kind") != "q4_confirmation_integer_domain_analysis":
        raise ValueError("Q4 证据图只接受 q4_confirmation_integer_domain_analysis")
    if analysis.get("audit_status") != "passed":
        raise ValueError("Q4 完整域审计尚未通过")
    if analysis.get("result_status") != "lowest_statistically_feasible_cost":
        raise ValueError("Q4 未决边界图要求存在未排除的更便宜设计")
    contract = analysis.get("boundary_contract")
    if not isinstance(contract, dict) or contract.get("mode") != "D":
        raise ValueError("Q4 完整域审计未使用 D 边界合同")

    inputs = analysis.get("input_files")
    if not isinstance(inputs, dict):
        raise ValueError("Q4 完整域审计缺少冻结输入清单")
    resolved_inputs: dict[str, Path] = {}
    for key, label in (
        ("freeze", "Q4 冻结协议"),
        ("final_summary", "Q4 最终摘要"),
        ("merged_pareto_frontier", "Q4 正式 Pareto 样本"),
    ):
        record = inputs.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Q4 完整域审计缺少{label}记录")
        path = resolve_evidence_path(record.get("path"), analysis_path, project_root, label)
        require_hash(path, record.get("sha256"), label)
        resolved_inputs[key] = path

    freeze = read_json_object(resolved_inputs["freeze"], "Q4 冻结协议")
    summary = read_json_object(resolved_inputs["final_summary"], "Q4 最终摘要")
    if freeze.get("kind") != "q4_confirmation_freeze":
        raise ValueError("Q4 冻结协议 kind 不一致")
    if summary.get("kind") != "q4_final_summary":
        raise ValueError("Q4 最终摘要 kind 不一致")
    if summary.get("result_status") != analysis.get("result_status"):
        raise ValueError("Q4 最终摘要与完整域审计状态不一致")
    if summary.get("candidate_statistically_feasible") is not True:
        raise ValueError("Q4 候选未形成统计可行上界")
    if summary.get("all_strictly_cheaper_maximal_designs_excluded") is not False:
        raise ValueError("Q4 未决边界图不得用于已完成全域排除的状态")

    configuration = analysis.get("configuration")
    protocol = freeze.get("confirmation_protocol")
    if not isinstance(configuration, dict) or not isinstance(protocol, dict):
        raise ValueError("Q4 正式确认配置不完整")
    statement_count = int(configuration.get("bonferroni_statement_count", -1))
    if statement_count != 620 or int(protocol.get("bonferroni_statement_count", -1)) != statement_count:
        raise ValueError("Q4 Bonferroni 陈述数必须为冻结的 620 项")
    if int(configuration.get("trial_count", -1)) != 50000:
        raise ValueError("Q4 未决边界图只接受 50000 次正式确认")

    records = summary.get("confirmation_records")
    if not isinstance(records, list) or len(records) != statement_count:
        raise ValueError("Q4 最终摘要的确认记录数与 620 项冻结族不一致")
    candidates = [record for record in records if isinstance(record, dict) and record.get("role") == "candidate"]
    cheaper = [
        record
        for record in records
        if isinstance(record, dict) and record.get("role") == "strictly_cheaper_maximal"
    ]
    if len(candidates) != 1 or len(cheaper) != statement_count - 1:
        raise ValueError("Q4 确认记录必须含 1 个候选和 619 个严格更便宜极大点")
    candidate = candidates[0]
    if candidate.get("proof_status") != "candidate_statistically_feasible":
        raise ValueError("Q4 候选不得标为未决或不可行")
    excluded = [record for record in cheaper if record.get("proof_status") == "strictly_cheaper_design_excluded"]
    unresolved = [
        record
        for record in cheaper
        if record.get("proof_status") == "strictly_cheaper_design_not_excluded"
    ]
    if len(excluded) + len(unresolved) != len(cheaper):
        raise ValueError("Q4 更便宜极大点含无法识别的证据状态")

    statistics = analysis.get("statistical_results")
    if not isinstance(statistics, dict):
        raise ValueError("Q4 完整域审计缺少统计结果")
    expected_excluded = int(statistics.get("excluded_frontier_count", -1))
    expected_unresolved = int(statistics.get("not_excluded_frontier_count", -1))
    if (len(excluded), len(unresolved)) != (expected_excluded, expected_unresolved):
        raise ValueError("Q4 573/46 分类与完整域审计不一致")
    if (int(summary.get("excluded_frontier_count", -1)), int(summary.get("not_excluded_frontier_count", -1))) != (
        expected_excluded,
        expected_unresolved,
    ):
        raise ValueError("Q4 573/46 分类与最终摘要不一致")

    analysis_candidate = statistics.get("candidate")
    if not isinstance(analysis_candidate, dict):
        raise ValueError("Q4 完整域审计缺少候选记录")
    for key in ("n_a", "n_b", "successes", "trials", "cost_weight"):
        if int(candidate.get(key, -1)) != int(analysis_candidate.get(key, -2)):
            raise ValueError(f"Q4 候选字段 {key} 不一致")
    for key in ("estimate", "cost_yuan", "clopper_pearson_one_sided_lower"):
        require_close(candidate.get(key), analysis_candidate.get(key), f"Q4 候选字段 {key}")

    interval = statistics.get("cost_uncertainty_interval")
    summary_interval = summary.get("cost_uncertainty_interval")
    if not isinstance(interval, dict) or not isinstance(summary_interval, dict):
        raise ValueError("Q4 成本证据区间缺失")
    for key in ("lower_cost_weight", "upper_cost_weight", "lower_cost_yuan", "upper_cost_yuan"):
        require_close(interval.get(key), summary_interval.get(key), f"Q4 成本区间字段 {key}")
    lower_cost = float(interval["lower_cost_yuan"])
    upper_cost = float(interval["upper_cost_yuan"])
    if not lower_cost < upper_cost or not np.isclose(upper_cost, float(candidate["cost_yuan"]), atol=1e-12):
        raise ValueError("Q4 成本区间必须由排除下界和候选可行上界构成")

    reported_unresolved = summary.get("not_excluded_frontier")
    if not isinstance(reported_unresolved, list):
        raise ValueError("Q4 最终摘要缺少 46 个未决点清单")
    unresolved_pairs = {(int(record["n_a"]), int(record["n_b"])) for record in unresolved}
    if unresolved_pairs != {
        (int(record["n_a"]), int(record["n_b"]))
        for record in reported_unresolved
        if isinstance(record, dict)
    }:
        raise ValueError("Q4 未决点清单与确认记录不一致")

    return {
        "analysis_path": analysis_path,
        "analysis": analysis,
        "freeze_path": resolved_inputs["freeze"],
        "freeze": freeze,
        "summary_path": resolved_inputs["final_summary"],
        "summary": summary,
        "merged_path": resolved_inputs["merged_pareto_frontier"],
        "candidate": candidate,
        "excluded": sorted(excluded, key=lambda record: int(record["n_a"])),
        "unresolved": sorted(unresolved, key=lambda record: int(record["n_a"])),
        "cost_interval": interval,
        "statement_count": statement_count,
    }


def build_q4_boundary_figure(evidence: dict[str, Any]):
    configure_fonts()
    excluded = evidence["excluded"]
    unresolved = evidence["unresolved"]
    candidate = evidence["candidate"]
    interval = evidence["cost_interval"]
    excluded_count, unresolved_count = len(excluded), len(unresolved)
    if (excluded_count, unresolved_count) != (573, 46):
        raise ValueError("正式 Q4 未决边界图必须绑定 573 个已排除点和 46 个未决点")

    figure = plt.figure(figsize=(7.6, 3.75), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[0.72, 2.65], width_ratios=[1.12, 1.0])
    count_ax = figure.add_subplot(grid[0, 0])
    bound_ax = figure.add_subplot(grid[1, 0])
    cost_ax = figure.add_subplot(grid[:, 1])

    count_ax.barh([0], [excluded_count], color="#0072B2", height=0.55, edgecolor="#1B4F72", linewidth=0.6)
    count_ax.barh(
        [0],
        [unresolved_count],
        left=[excluded_count],
        color="#F3C04B",
        hatch="////",
        height=0.55,
        edgecolor="#8B6508",
        linewidth=0.7,
    )
    count_ax.text(excluded_count / 2, 0, "已排除 573", ha="center", va="center", fontsize=8.2, color="white", weight="bold")
    count_ax.text(
        excluded_count + unresolved_count / 2,
        0,
        "未决\n46",
        ha="center",
        va="center",
        fontsize=6.5,
        linespacing=0.9,
        color="#4A3900",
        weight="bold",
    )
    count_ax.set_xlim(0, excluded_count + unresolved_count)
    count_ax.set_ylim(-0.58, 0.58)
    count_ax.set_title("(a) 619 个严格更便宜极大点的联合证据", loc="left", fontsize=9.5, pad=2)
    count_ax.axis("off")

    near_excluded = excluded[-40:]
    excluded_a = np.asarray([int(record["n_a"]) for record in near_excluded])
    excluded_upper = 100.0 * np.asarray(
        [float(record["clopper_pearson_one_sided_upper"]) for record in near_excluded]
    )
    unresolved_a = np.asarray([int(record["n_a"]) for record in unresolved])
    unresolved_upper = 100.0 * np.asarray(
        [float(record["clopper_pearson_one_sided_upper"]) for record in unresolved]
    )
    candidate_a = int(candidate["n_a"])
    candidate_lower = 100.0 * float(candidate["clopper_pearson_one_sided_lower"])
    lower_y = min(float(np.min(excluded_upper)) - 0.25, 89.0)
    upper_y = max(float(np.max(unresolved_upper)) + 0.25, candidate_lower + 0.25)
    bound_ax.axhspan(lower_y, 90.0, color="#D9EEF8", alpha=0.65, zorder=0)
    bound_ax.axhspan(90.0, upper_y, color="#FFF4D6", alpha=0.68, zorder=0)
    bound_ax.axhline(90.0, color="#222222", linewidth=1.2, linestyle="--", label="90% 判定阈值")
    bound_ax.plot(excluded_a, excluded_upper, color="#0072B2", linewidth=1.0, alpha=0.82)
    bound_ax.scatter(excluded_a, excluded_upper, s=18, color="#0072B2", marker="o", label="已排除：CP 上限 < 90%", zorder=3)
    bound_ax.plot(unresolved_a, unresolved_upper, color="#9A7000", linewidth=0.9, alpha=0.75)
    bound_ax.scatter(
        unresolved_a,
        unresolved_upper,
        s=28,
        facecolors="none",
        edgecolors="#D89000",
        linewidth=1.1,
        marker="D",
        label="未决：CP 上限 ≥ 90%",
        zorder=4,
    )
    bound_ax.scatter(
        [candidate_a],
        [candidate_lower],
        marker="*",
        s=105,
        color="#009E73",
        edgecolor="#245A49",
        linewidth=0.65,
        label="(619,0)：CP 下限 > 90%",
        zorder=5,
    )
    bound_ax.set_xlim(int(excluded_a[0]) - 2, candidate_a + 3)
    bound_ax.set_ylim(lower_y, upper_y)
    bound_ax.set_xlabel("极大点的 A 数量 $N_A$（显示最接近阈值的 40 个已排除点及全部未决点）")
    bound_ax.set_ylabel("单侧 CP 界（%）")
    bound_ax.grid(True, axis="y")
    bound_ax.legend(loc="upper left", fontsize=6.8, frameon=True, framealpha=0.94)
    bound_ax.text(
        0.985,
        0.04,
        "未决仅表示尚不能排除",
        transform=bound_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#7B5700",
        weight="bold",
    )

    lower_cost = float(interval["lower_cost_yuan"])
    upper_cost = float(interval["upper_cost_yuan"])
    x_min = lower_cost - 0.075
    x_max = upper_cost + 0.035
    cost_ax.axvspan(x_min, lower_cost, facecolor="#D9EEF8", edgecolor="none", alpha=0.9)
    cost_ax.axvspan(
        lower_cost,
        upper_cost,
        facecolor="#FFF4D6",
        edgecolor="#C48A00",
        linewidth=0.6,
        hatch="////",
        alpha=0.8,
    )
    cost_ax.hlines(0.47, x_min, upper_cost, color="#5E6367", linewidth=1.0)
    cost_ax.axvline(lower_cost, color="#0072B2", linewidth=1.2)
    cost_ax.axvline(upper_cost, color="#009E73", linewidth=1.2)

    unresolved_costs = np.asarray([float(record["cost_yuan"]) for record in unresolved])
    unresolved_y = 0.35 + 0.055 * (np.arange(unresolved_count) % 5)
    cost_ax.scatter(
        unresolved_costs,
        unresolved_y,
        facecolors="none",
        edgecolors="#D89000",
        linewidth=0.9,
        marker="D",
        s=24,
        zorder=4,
    )
    cost_ax.scatter(
        [upper_cost],
        [0.47],
        marker="*",
        s=130,
        color="#009E73",
        edgecolor="#245A49",
        linewidth=0.7,
        zorder=5,
    )
    cost_ax.add_patch(
        FancyArrowPatch(
            (lower_cost, 0.82),
            (upper_cost, 0.82),
            arrowstyle="<->",
            mutation_scale=10,
            color="#4B4F52",
            linewidth=1.0,
        )
    )
    cost_ax.text(
        (lower_cost + upper_cost) / 2,
        0.87,
        "Q4 家族内成本证据区间",
        ha="center",
        va="bottom",
        fontsize=8.0,
        weight="bold",
    )
    cost_ax.text(lower_cost, 0.14, f"$C_L$\n{lower_cost:.4f}", ha="center", va="top", fontsize=7.3, color="#005E8A")
    cost_ax.text(upper_cost, 0.14, f"可行上界\n{upper_cost:.4f}", ha="center", va="top", fontsize=7.3, color="#006B50", weight="bold")
    cost_ax.text((x_min + lower_cost) / 2, 0.56, "低于 $C_L$\n已排除", ha="center", va="center", fontsize=7.6, color="#005E8A")
    cost_ax.text(
        lower_cost + 0.22 * (upper_cost - lower_cost),
        0.58,
        "下一轮：只复核\n46 个未决极大点\n（未排除 ≠ 可行）",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#6E5000",
        weight="bold",
    )

    inset = cost_ax.inset_axes([0.43, 0.54, 0.53, 0.24])
    unresolved_a_all = np.asarray([int(record["n_a"]) for record in unresolved])
    inset.scatter(
        unresolved_costs,
        unresolved_a_all,
        facecolors="none",
        edgecolors="#D89000",
        linewidth=0.8,
        marker="D",
        s=15,
    )
    inset.scatter(
        [upper_cost],
        [int(candidate["n_a"])],
        marker="*",
        s=60,
        color="#009E73",
        edgecolor="#245A49",
        linewidth=0.55,
        zorder=4,
    )
    inset.set_xlim(float(np.min(unresolved_costs)) - 0.00015, upper_cost + 0.00015)
    inset.set_ylim(int(np.min(unresolved_a_all)) - 2, int(candidate["n_a"]) + 2)
    inset.set_title("46 个未决点局部放大", fontsize=6.8, pad=1.5)
    inset.set_xlabel("成本（元）", fontsize=6.2, labelpad=1)
    inset.set_ylabel("$N_A$", fontsize=6.2, labelpad=1)
    inset.tick_params(axis="both", labelsize=5.8, pad=1)
    inset.ticklabel_format(axis="x", style="plain", useOffset=False)
    inset.grid(True, linewidth=0.35, alpha=0.25)
    cost_ax.set_xlim(x_min, x_max)
    cost_ax.set_ylim(0.05, 0.98)
    cost_ax.set_yticks([])
    cost_ax.set_xlabel("成本（元）")
    cost_ax.set_title("(b) 成本区间与下一轮计算对象", loc="left", fontsize=9.5, pad=5)
    cost_ax.spines[["left", "right", "top"]].set_visible(False)
    cost_ax.grid(False)
    return figure, {
        "excluded_frontier_count": excluded_count,
        "not_excluded_frontier_count": unresolved_count,
        "displayed_near_boundary_excluded_count": len(near_excluded),
        "candidate_design": [int(candidate["n_a"]), int(candidate["n_b"])],
        "candidate_cp_lower": float(candidate["clopper_pearson_one_sided_lower"]),
        "cost_interval_yuan": [lower_cost, upper_cost],
        "unresolved_cost_range_yuan": [float(np.min(unresolved_costs)), float(np.max(unresolved_costs))],
        "unresolved_semantics": "not_excluded_not_confirmed_feasible",
        "next_round_focus": "46_not_excluded_maximal_designs_only",
    }


def read_q4_candidate_sequence(evidence: dict[str, Any]) -> np.ndarray:
    merged_path = Path(evidence["merged_path"])
    merged = read_json_object(merged_path, "Q4 正式 Pareto 样本")
    if merged.get("kind") != "mixed_pareto_frontier_samples":
        raise ValueError("Q4 收敛图只接受 mixed_pareto_frontier_samples")
    analysis = evidence["analysis"]
    configuration = analysis["configuration"]
    if merged.get("configuration_fingerprint") != configuration.get("fingerprint"):
        raise ValueError("Q4 正式 Pareto 样本配置指纹与完整域审计不一致")
    trials = int(configuration["trial_count"])
    records = merged.get("records")
    if not isinstance(records, list) or len(records) != trials or int(merged.get("trials", -1)) != trials:
        raise ValueError("Q4 正式 Pareto 样本数与 50000 次确认不一致")
    trial_ids = np.asarray([int(record.get("trial_id", -1)) for record in records], dtype=np.int64)
    if not np.array_equal(trial_ids, np.arange(trials, dtype=np.int64)):
        raise ValueError("Q4 正式 Pareto 样本 trial_id 不连续")

    candidate_a = int(evidence["candidate"]["n_a"])
    candidate_b = int(evidence["candidate"]["n_b"])
    success = np.zeros(trials, dtype=bool)
    for index, record in enumerate(records):
        frontier = record.get("connectivity_frontier")
        if not isinstance(frontier, list):
            raise ValueError(f"Q4 trial {index} 缺少 connectivity_frontier")
        for pair in frontier:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Q4 trial {index} 的 Pareto 点格式错误")
            n_a, n_b = int(pair[0]), int(pair[1])
            if n_a < 0 or n_b < 0:
                raise ValueError(f"Q4 trial {index} 的 Pareto 点出现负坐标")
            if n_a <= candidate_a and n_b <= candidate_b:
                success[index] = True
                break
    if int(np.count_nonzero(success)) != int(evidence["candidate"]["successes"]):
        raise ValueError("Q4 候选逐 trial 导通序列与正式成功数不一致")
    return success


def build_q4_boundary_figure_compact(evidence: dict[str, Any]):
    configure_fonts()
    excluded = evidence["excluded"]
    unresolved = evidence["unresolved"]
    candidate = evidence["candidate"]
    interval = evidence["cost_interval"]
    if (len(excluded), len(unresolved)) != (573, 46):
        raise ValueError("正式 Q4 未决边界图必须绑定 573 个已排除点和 46 个未决点")

    excluded_a = np.asarray([int(record["n_a"]) for record in excluded])
    excluded_upper = 100.0 * np.asarray(
        [float(record["clopper_pearson_one_sided_upper"]) for record in excluded]
    )
    unresolved_a = np.asarray([int(record["n_a"]) for record in unresolved])
    unresolved_upper = 100.0 * np.asarray(
        [float(record["clopper_pearson_one_sided_upper"]) for record in unresolved]
    )
    excluded_costs = np.asarray([float(record["cost_yuan"]) for record in excluded])
    unresolved_costs = np.asarray([float(record["cost_yuan"]) for record in unresolved])
    candidate_a = int(candidate["n_a"])
    candidate_lower = 100.0 * float(candidate["clopper_pearson_one_sided_lower"])
    lower_cost = float(interval["lower_cost_yuan"])
    upper_cost = float(interval["upper_cost_yuan"])

    figure, (bound_ax, cost_ax) = plt.subplots(
        1,
        2,
        figsize=(7.6, 3.45),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.08, 0.92]},
    )

    bound_ax.axhspan(89.0, 90.0, color="#EAF1FB", alpha=0.95, zorder=0)
    bound_ax.axhspan(90.0, max(float(np.max(unresolved_upper)), candidate_lower) + 0.25, color="#FBEEDD", alpha=0.9, zorder=0)
    bound_ax.axhline(90.0, color="#2F343B", linewidth=1.25, linestyle=(0, (5, 3)))
    bound_ax.scatter(
        excluded_a,
        excluded_upper,
        s=13,
        color="#2457A7",
        marker="o",
        alpha=0.78,
        linewidth=0,
        zorder=2,
    )
    bound_ax.scatter(
        unresolved_a,
        unresolved_upper,
        s=31,
        facecolors="white",
        edgecolors="#C46618",
        marker="D",
        linewidth=1.05,
        zorder=3,
    )
    bound_ax.scatter(
        [candidate_a],
        [candidate_lower],
        s=120,
        marker="*",
        color="#0B7A75",
        edgecolor="#154A47",
        linewidth=0.6,
        zorder=4,
    )
    bound_ax.text(
        0.03,
        0.12,
        "● 已排除 573",
        transform=bound_ax.transAxes,
        fontsize=9.3,
        weight="bold",
        color="#2457A7",
    )
    bound_ax.text(
        0.03,
        0.045,
        "◇ 未决 46",
        transform=bound_ax.transAxes,
        fontsize=9.3,
        weight="bold",
        color="#A84F10",
    )
    bound_ax.text(
        0.97,
        0.91,
        "候选 CP 下限 > 90%",
        transform=bound_ax.transAxes,
        ha="right",
        fontsize=9.1,
        weight="bold",
        color="#0B7A75",
    )
    bound_ax.text(
        0.97,
        0.75,
        "未决：CP 上限 ≥ 90%\n仅表示尚不能排除",
        transform=bound_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.9,
        color="#8A470F",
    )
    bound_ax.set_xlim(min(int(np.min(excluded_a)), int(np.min(unresolved_a))) - 3, candidate_a + 4)
    bound_ax.set_ylim(min(89.0, float(np.min(excluded_upper)) - 0.12), max(float(np.max(unresolved_upper)), candidate_lower) + 0.25)
    bound_ax.set_xlabel("极大点的介质 A 数量 $N_A$")
    bound_ax.set_ylabel("单侧 Clopper--Pearson 界（%）")
    bound_ax.set_title("(a) 完整整数域的 90% 证据阈值", loc="left", fontsize=10.4, pad=7)
    bound_ax.grid(True, axis="y", which="major", linewidth=0.55, alpha=0.34)
    bound_ax.grid(False, axis="x")

    x_min = min(float(np.min(excluded_costs)), lower_cost) - 0.015
    x_max = upper_cost + 0.02
    excluded_jitter = 2.75 + 0.18 * np.sin(np.arange(len(excluded_costs)) * 0.73)
    unresolved_jitter = 1.80 + 0.16 * np.sin(np.arange(len(unresolved_costs)) * 1.31)
    cost_ax.scatter(
        excluded_costs,
        excluded_jitter,
        s=11,
        color="#2457A7",
        marker="o",
        alpha=0.38,
        linewidth=0,
    )
    cost_ax.scatter(
        unresolved_costs,
        unresolved_jitter,
        s=27,
        facecolors="white",
        edgecolors="#C46618",
        marker="D",
        linewidth=1.0,
        zorder=3,
    )
    cost_ax.scatter(
        [upper_cost],
        [0.92],
        s=130,
        marker="*",
        color="#0B7A75",
        edgecolor="#154A47",
        linewidth=0.6,
        zorder=4,
    )
    cost_ax.hlines(0.15, lower_cost, upper_cost, color="#C46618", linewidth=5.0, zorder=2)
    cost_ax.scatter([lower_cost, upper_cost], [0.15, 0.15], s=44, color=["#2457A7", "#0B7A75"], zorder=3)
    cost_ax.axvline(lower_cost, color="#2457A7", linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.75)
    cost_ax.axvline(upper_cost, color="#0B7A75", linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.75)
    cost_ax.text(lower_cost, -0.13, f"下界 {lower_cost:.4f}", ha="center", va="top", fontsize=8.8, color="#2457A7")
    cost_ax.text(upper_cost, -0.13, f"上界 {upper_cost:.4f}", ha="center", va="top", fontsize=8.8, weight="bold", color="#0B7A75")
    cost_ax.text(
        0.98,
        0.98,
        "下一轮只需复核 46 个未决点",
        transform=cost_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        weight="bold",
        color="#8A470F",
    )
    cost_ax.set_xlim(x_min, x_max)
    cost_ax.set_ylim(-0.35, 3.25)
    cost_ax.set_yticks([2.75, 1.80, 0.92, 0.15])
    cost_ax.set_yticklabels(["已排除 573", "未决 46", "确认候选", "成本证据区间"])
    cost_ax.set_xlabel("成本（元）")
    cost_ax.set_title("(b) 状态分层与成本区间", loc="left", fontsize=10.4, pad=7)
    cost_ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    cost_ax.grid(True, axis="x", which="major", linewidth=0.55, alpha=0.30)
    cost_ax.grid(False, axis="y")
    cost_ax.spines[["top", "right", "left"]].set_visible(False)
    cost_ax.tick_params(axis="y", length=0)

    return figure, {
        "excluded_frontier_count": len(excluded),
        "not_excluded_frontier_count": len(unresolved),
        "displayed_near_boundary_excluded_count": len(excluded),
        "candidate_design": [candidate_a, int(candidate["n_b"])],
        "candidate_cp_lower": float(candidate["clopper_pearson_one_sided_lower"]),
        "cost_interval_yuan": [lower_cost, upper_cost],
        "unresolved_cost_range_yuan": [float(np.min(unresolved_costs)), float(np.max(unresolved_costs))],
        "unresolved_semantics": "not_excluded_not_confirmed_feasible",
        "next_round_focus": "46_not_excluded_maximal_designs_only",
    }


def build_simulation_convergence_figure(
    q2_counts: list[int],
    q2_full: np.ndarray,
    q2_samples: np.ndarray,
    q4_success: np.ndarray,
    q4_evidence: dict[str, Any],
):
    configure_fonts()
    q2_samples = np.asarray(q2_samples)
    q4_success = np.asarray(q4_success, dtype=bool)
    if q2_samples.ndim != 1 or len(q2_samples) != 20000:
        raise ValueError("Q2 收敛图要求 20000 个正式阈值样本")
    if q4_success.ndim != 1 or len(q4_success) != 50000:
        raise ValueError("Q4 收敛图要求 50000 个正式候选指示量")

    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.25), constrained_layout=True)
    q2_ax, q4_ax = axes
    q2_trials = np.arange(1, len(q2_samples) + 1, dtype=np.int64)
    q2_plot_index = np.unique(
        np.r_[np.arange(99, len(q2_samples), 100, dtype=np.int64), len(q2_samples) - 1]
    )
    line_styles = ["-", "--", "-.", ":"]
    q2_endpoint: dict[str, float] = {}
    for index, count in enumerate(q2_counts):
        indicators = q2_samples <= count
        cumulative = np.cumsum(indicators, dtype=np.int64) / q2_trials
        endpoint = float(cumulative[-1])
        if not np.isclose(endpoint, float(q2_full[index]), atol=0.0, rtol=0.0):
            raise ValueError(f"Q2 N_A={count} 的累计终值与正式摘要不一致")
        q2_endpoint[str(count)] = endpoint
        q2_ax.plot(
            q2_trials[q2_plot_index],
            100.0 * cumulative[q2_plot_index],
            color=PAPER_COLORS[index],
            linestyle=line_styles[index],
            linewidth=1.45,
        )
        q2_ax.scatter(
            [len(q2_samples)],
            [100.0 * endpoint],
            color=PAPER_COLORS[index],
            marker=["o", "s", "D", "^"][index],
            s=23,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
    q2_ax.set_xlim(0, len(q2_samples) * 1.015)
    q2_ax.set_ylim(-2.0, 102.0)
    q2_ax.set_xlabel("累计试验数 $n$")
    q2_ax.set_ylabel("累计导通概率（%）")
    q2_ax.set_title("(a) Q2 四个题给填充量的累计稳定过程", loc="left", fontsize=9.5)
    q2_ax.grid(True, axis="y")
    for index, count in enumerate(q2_counts):
        endpoint_percent = 100.0 * float(q2_full[index])
        offset = -2.0 if endpoint_percent > 96.0 else 1.8
        q2_ax.text(
            14600,
            endpoint_percent + offset,
            f"$N_A={count}$",
            ha="left",
            va="center",
            fontsize=7.1,
            color=PAPER_COLORS[index],
            weight="bold",
        )
    q2_ax.text(
        0.98,
        0.04,
        "终点：正式 $n=20000$ 估计",
        transform=q2_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#555555",
    )

    q4_trials = np.arange(1, len(q4_success) + 1, dtype=np.int64)
    q4_cumulative_success = np.cumsum(q4_success, dtype=np.int64)
    q4_estimate = q4_cumulative_success / q4_trials
    q4_plot_index = np.unique(
        np.r_[np.arange(99, len(q4_success), 100, dtype=np.int64), len(q4_success) - 1]
    )
    checkpoints = np.asarray([500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000], dtype=np.int64)
    confidence = float(q4_evidence["analysis"]["configuration"]["per_statement_confidence"])
    cp_lower = np.asarray(
        [
            clopper_pearson_one_sided_bounds(
                int(q4_cumulative_success[checkpoint - 1]),
                int(checkpoint),
                confidence,
            )[0]
            for checkpoint in checkpoints
        ],
        dtype=float,
    )
    official_lower = float(q4_evidence["candidate"]["clopper_pearson_one_sided_lower"])
    official_estimate = float(q4_evidence["candidate"]["estimate"])
    if not np.isclose(float(q4_estimate[-1]), official_estimate, atol=0.0, rtol=0.0):
        raise ValueError("Q4 候选累计终值与正式摘要不一致")
    if not np.isclose(float(cp_lower[-1]), official_lower, atol=5e-13, rtol=0.0):
        raise ValueError("Q4 50000 次诊断重算 CP 下限与正式结果不一致")

    q4_ax.axhline(90.0, color="#222222", linewidth=1.1, linestyle="--", label="90% 门槛")
    q4_ax.plot(
        q4_trials[q4_plot_index],
        100.0 * q4_estimate[q4_plot_index],
        color="#0072B2",
        linewidth=1.55,
        label="累计经验概率",
    )
    q4_ax.plot(
        checkpoints,
        100.0 * cp_lower,
        color="#D55E00",
        linewidth=1.3,
        linestyle="--",
        marker="D",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="诊断性 620 项 CP 下限",
    )
    q4_ax.fill_between(
        checkpoints,
        100.0 * cp_lower,
        100.0 * q4_estimate[checkpoints - 1],
        color="#F4D7C8",
        alpha=0.45,
        linewidth=0.0,
    )
    q4_ax.scatter(
        [50000],
        [100.0 * official_lower],
        marker="*",
        s=100,
        color="#009E73",
        edgecolor="#245A49",
        linewidth=0.7,
        zorder=5,
        label="正式 $n=50000$ 判定",
    )
    q4_ax.set_xlim(0, 51500)
    q4_ax.set_ylim(min(88.0, 100.0 * float(np.min(cp_lower)) - 0.3), max(92.0, 100.0 * float(np.max(q4_estimate[q4_plot_index])) + 0.2))
    q4_ax.set_xlabel("累计试验数 $n$")
    q4_ax.set_ylabel("概率 / 单侧下限（%）")
    q4_ax.set_title("(b) Q4 候选 (619,0) 的累计诊断", loc="left", fontsize=9.5)
    q4_ax.grid(True, axis="y")
    q4_ax.legend(loc="lower right", fontsize=6.8, frameon=True, framealpha=0.95)
    q4_ax.annotate(
        "仅终点进入正式结论",
        xy=(50000, 100.0 * official_lower),
        xytext=(31500, 91.55),
        ha="center",
        va="center",
        fontsize=7.2,
        color="#006B50",
        arrowprops={"arrowstyle": "->", "color": "#006B50", "linewidth": 0.9},
    )
    q4_ax.text(
        0.035,
        0.135,
        "中间检查点仅作诊断\n不提供序贯覆盖保证",
        transform=q4_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#7A3E18",
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.3},
    )
    return figure, {
        "q2_trial_count": int(len(q2_samples)),
        "q2_final_estimates": q2_endpoint,
        "q4_trial_count": int(len(q4_success)),
        "q4_successes": int(q4_cumulative_success[-1]),
        "q4_final_estimate": float(q4_estimate[-1]),
        "q4_bonferroni_statement_count": int(q4_evidence["statement_count"]),
        "q4_per_statement_confidence": confidence,
        "q4_diagnostic_checkpoints": checkpoints.tolist(),
        "q4_diagnostic_cp_lower": cp_lower.tolist(),
        "q4_final_cp_lower": float(cp_lower[-1]),
        "interpretation_guard": "intermediate_checkpoints_diagnostic_only_final_n50000_authoritative",
    }


def build_validation_figure(q1: dict[str, np.ndarray | int | float], counts, full, blocks):
    configure_fonts()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)

    differences = np.asarray(q1["differences"], dtype=float)
    ordered = np.sort(np.maximum(differences, 1e-15))
    ax = axes[0]
    ax.plot(np.arange(1, len(ordered) + 1), ordered, color=SEMANTIC_COLORS["primary"])
    ax.scatter(
        np.arange(1, len(ordered) + 1),
        ordered,
        s=13,
        color=SEMANTIC_COLORS["primary"],
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("按绝对差排序的窄相候选对")
    ax.set_ylabel("GJK 与 SLSQP 距离绝对差（nm）")
    ax.set_title("(a) 195 对独立数值复核")
    ax.grid(True, which="major", axis="y")
    ax.grid(False, which="minor", axis="y")
    ax.annotate(
        f"最大差 {float(q1['max_difference_nm']):.3e} nm\n阈值分类分歧 {int(q1['disagreements'])}",
        xy=(len(ordered), ordered[-1]),
        xytext=(-8, -42),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#777777"},
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.0},
    )

    ax = axes[1]
    x_base = np.arange(len(counts), dtype=float)
    markers = ["o", "s", "D", "^"]
    block_colors = ["#2457A7", "#0B7A75", "#C46618", "#69757E"]
    for block_index in range(blocks.shape[0]):
        deviation = 100.0 * (blocks[block_index, :] - full)
        x = x_base + (block_index - 1.5) * 0.075
        ax.scatter(
            x,
            deviation,
            s=34,
            marker=markers[block_index],
            facecolors="white" if block_index in {1, 2} else block_colors[block_index],
            edgecolors=block_colors[block_index],
            linewidth=1.0,
            label=f"样本块 {block_index + 1}",
            zorder=3,
        )
    minima = 100.0 * np.min(blocks - full[None, :], axis=0)
    maxima = 100.0 * np.max(blocks - full[None, :], axis=0)
    ax.vlines(x_base, minima, maxima, color="#9AA5AD", linewidth=1.1, zorder=1)
    ax.axhline(0.0, color=SEMANTIC_COLORS["reference"], linewidth=1.1)
    ax.set_xticks(x_base, [str(count) for count in counts])
    ax.set_xlabel("介质 A 数量 $N_A$（每组 4 个独立样本块）")
    ax.set_ylabel("相对 20000 次总估计的偏差（百分点）")
    ax.set_title("(b) 问题二样本分块稳定性")
    ax.grid(True, axis="y")
    ax.legend(ncol=2, loc="upper right", fontsize=8.2, frameon=False)
    return figure


def image_size(path: Path) -> list[int]:
    with Image.open(path) as image:
        return [int(image.width), int(image.height)]


def write_audit(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    q1 = read_q1_validation()
    counts, full, blocks, block_size, threshold_path, q2_samples = read_q2_blocks()
    q4_evidence = load_q4_boundary_evidence()

    workflow = build_workflow_figure()
    workflow_pdf, workflow_png = save_figure(workflow, WORKFLOW_STEM)
    plt.close(workflow)

    q1_3d = build_q1_3d_figure()
    q1_3d_pdf, q1_3d_png = save_figure(q1_3d, Q1_3D_STEM)
    plt.close(q1_3d)

    validation = build_validation_figure(q1, counts, full, blocks)
    validation_pdf, validation_png = save_figure(validation, VALIDATION_STEM)
    plt.close(validation)

    q4_boundary, q4_boundary_stats = build_q4_boundary_figure_compact(q4_evidence)
    q4_boundary_pdf, q4_boundary_png = save_figure(q4_boundary, Q4_BOUNDARY_STEM)
    plt.close(q4_boundary)

    q4_success = read_q4_candidate_sequence(q4_evidence)
    convergence, convergence_stats = build_simulation_convergence_figure(
        counts,
        full,
        q2_samples,
        q4_success,
        q4_evidence,
    )
    convergence_pdf, convergence_png = save_figure(convergence, CONVERGENCE_STEM)
    plt.close(convergence)

    block_ranges = {
        str(count): [float(np.min(blocks[:, index])), float(np.max(blocks[:, index]))]
        for index, count in enumerate(counts)
    }
    audit = {
        "schema_version": 1,
        "kind": "explanatory_figures_audit",
        "inputs": {
            project_path(Q1_VALIDATION): sha256(Q1_VALIDATION),
            project_path(Q2_SUMMARY): sha256(Q2_SUMMARY),
            project_path(threshold_path): sha256(threshold_path),
            project_path(IMAGEN_WORKFLOW_ASSET): sha256(IMAGEN_WORKFLOW_ASSET),
            project_path(IMAGEN_WORKFLOW_PROMPT): sha256(IMAGEN_WORKFLOW_PROMPT),
            **{project_path(path): sha256(path) for path in Q1_SCENE_PATHS},
        },
        "workflow_truth_contract": [
            "Imagen 仅生成不承担定量读数的二维结构底图，生成文字全部由本地矢量层覆盖",
            "几何输入只示意有限圆柱、球或球片、独立截断片段和两端电极，不编码尺寸",
            "本地矢量层明确叠加 1.8 nm 阈值、问题一 0/1/1 结论和四问证据类型",
            "定量概率、临界括区、成本前沿和三维场景仍由正式计算与 CAD 图件单独证明",
            "底部验证带只保留构型、独立求解、单调性、随机流和置信界五类证据",
        ],
        "workflow_visual_encoding": {
            "geometry": "finite_cylinders_and_spherical_fragments",
            "contact_graph": "nodes_edges_electrodes_and_orange_witness_path",
            "question_outputs": ["q1_binary", "q2_probability_curve", "q3_interval", "q4_cost_frontier"],
            "scale_status": "schematic_not_to_scale",
        },
        "validation": {
            "q1_pair_count": int(q1["row_count"]),
            "q1_threshold_class_disagreements": int(q1["disagreements"]),
            "q1_max_distance_difference_nm": float(q1["max_difference_nm"]),
            "q2_block_size": block_size,
            "q2_block_probability_ranges": block_ranges,
        },
        "outputs": {
            project_path(workflow_pdf): {"sha256": sha256(workflow_pdf)},
            project_path(workflow_png): {"sha256": sha256(workflow_png), "pixels": image_size(workflow_png)},
            project_path(q1_3d_pdf): {"sha256": sha256(q1_3d_pdf)},
            project_path(q1_3d_png): {"sha256": sha256(q1_3d_png), "pixels": image_size(q1_3d_png)},
            project_path(validation_pdf): {"sha256": sha256(validation_pdf)},
            project_path(validation_png): {"sha256": sha256(validation_png), "pixels": image_size(validation_png)},
        },
    }
    write_audit(AUDIT_PATH, audit)

    q4_inputs = {
        project_path(q4_evidence["analysis_path"]): sha256(q4_evidence["analysis_path"]),
        project_path(q4_evidence["summary_path"]): sha256(q4_evidence["summary_path"]),
        project_path(q4_evidence["freeze_path"]): sha256(q4_evidence["freeze_path"]),
    }
    q4_boundary_audit = {
        "schema_version": 1,
        "kind": "q4_unresolved_boundary_evidence_figure_audit",
        "inputs": q4_inputs,
        "result_status": q4_evidence["analysis"]["result_status"],
        "evidence_scope": "candidate_feasibility_cheaper_design_exclusion_and_unresolved_focus",
        "classification_guard": "not_excluded_is_not_confirmed_feasible",
        "unresolved_designs": [
            [int(record["n_a"]), int(record["n_b"]), int(record["cost_weight"])]
            for record in q4_evidence["unresolved"]
        ],
        **q4_boundary_stats,
        "outputs": {
            project_path(q4_boundary_pdf): {"sha256": sha256(q4_boundary_pdf)},
            project_path(q4_boundary_png): {
                "sha256": sha256(q4_boundary_png),
                "pixels": image_size(q4_boundary_png),
            },
        },
    }
    write_audit(Q4_BOUNDARY_AUDIT_PATH, q4_boundary_audit)

    convergence_audit = {
        "schema_version": 1,
        "kind": "simulation_convergence_figure_audit",
        "inputs": {
            project_path(Q2_SUMMARY): sha256(Q2_SUMMARY),
            project_path(threshold_path): sha256(threshold_path),
            project_path(q4_evidence["analysis_path"]): sha256(q4_evidence["analysis_path"]),
            project_path(q4_evidence["merged_path"]): sha256(q4_evidence["merged_path"]),
        },
        "evidence_scope": "fixed_sample_cumulative_diagnostics",
        **convergence_stats,
        "outputs": {
            project_path(convergence_pdf): {"sha256": sha256(convergence_pdf)},
            project_path(convergence_png): {
                "sha256": sha256(convergence_png),
                "pixels": image_size(convergence_png),
            },
        },
    }
    write_audit(CONVERGENCE_AUDIT_PATH, convergence_audit)

    print(
        json.dumps(
            {
                "audits": [
                    project_path(AUDIT_PATH),
                    project_path(Q4_BOUNDARY_AUDIT_PATH),
                    project_path(CONVERGENCE_AUDIT_PATH),
                ],
                "outputs": [
                    *audit["outputs"],
                    *q4_boundary_audit["outputs"],
                    *convergence_audit["outputs"],
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
