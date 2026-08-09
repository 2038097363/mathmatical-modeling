#!/usr/bin/env python3
"""Generate the v2.3 clipped-fragment manuscript figures and QA artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACK = PROJECT_ROOT / "20_mainline" / "windows_review_pack" / "华数杯A题_Windows可运行复核版"
EVIDENCE = PROJECT_ROOT / "40_candidate" / "gpt" / "strict_fragment_recompute_20260808"
Q1_EVIDENCE = PROJECT_ROOT / "40_candidate" / "gpt" / "unified_model_recompute" / "q1"
FIGURE_ROOT = EVIDENCE / "figures_v230"
QA_ROOT = EVIDENCE / "figure_qa_v230"

COLORS = {
    "paper": "#FFFFFF",
    "ink": "#20272B",
    "muted": "#68747A",
    "soft": "#DDDDDD",
    "grid": "#E7E9E9",
    "orange": "#EDBE91",
    "orange_dark": "#A96F43",
    "orange_light": "#FAF0E7",
    "sage": "#8AA9A0",
    "sage_dark": "#4F746B",
    "sage_light": "#EDF3F1",
    "cyan": "#ACD4D6",
    "cyan_dark": "#5D9295",
    "cyan_light": "#EFF8F8",
    "blue": "#A8BCCC",
    "blue_dark": "#58748C",
    "blue_light": "#EEF3F7",
    "grey_dark": "#7C858A",
    "grey_light": "#F5F6F6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure() -> str:
    available = {font.name for font in fm.fontManager.ttflist}
    font = next(
        (
            name
            for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC")
            if name in available
        ),
        None,
    )
    if font is None:
        raise RuntimeError("未找到中文字体，停止生成图件。")
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 12.0,
            "axes.labelsize": 13.0,
            "axes.titlesize": 14.0,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.0,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.dpi": 320,
        }
    )
    return font


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    from PIL import Image

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    QA_ROOT.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("svg", "pdf", "png"):
        path = FIGURE_ROOT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=320, facecolor="white")
        outputs.append(path)
    gray = QA_ROOT / f"{stem}_gray.png"
    with Image.open(outputs[-1]) as image:
        image.convert("L").save(gray)
    outputs.append(gray)
    plt.close(fig)
    return outputs


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    face: str,
    edge: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.010",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + width / 2,
        y + height * 0.29,
        body,
        ha="center",
        va="center",
        fontsize=10.3,
        linespacing=1.18,
        color=COLORS["muted"],
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=COLORS["grey_dark"],
        )
    )


def make_model_figure() -> tuple[list[Path], dict]:
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.955,
        "真实裁剪片段驱动的周期随机几何图模型",
        ha="center",
        fontsize=17.5,
        fontweight="bold",
        color=COLORS["ink"],
    )

    xs = (0.045, 0.285, 0.525, 0.765)
    width = 0.19
    stages = (
        ("原始介质", "生成 A 圆柱、B 球体\n登记 source_id"),
        ("真实非空片段", r"$P_{i,k}=[K_i\cap(\Omega+Lk)]-Lk$" "\n仅保留与基元盒相交者"),
        ("裁剪几何接触", "片段—片段最短距离\n片段—电极薄层相交"),
        ("接触图", "距离不超过 1.8 nm 连边\n并查集判定左右连通"),
    )
    faces = (COLORS["blue_light"], COLORS["cyan_light"], COLORS["sage_light"], COLORS["orange_light"])
    edges = (COLORS["blue_dark"], COLORS["cyan_dark"], COLORS["sage_dark"], COLORS["orange_dark"])
    for x, (title, body), face, edge in zip(xs, stages, faces, edges):
        add_box(ax, x, 0.665, width, 0.19, title, body, face, edge)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x0 + width + 0.008, 0.76), (x1 - 0.008, 0.76))

    ax.text(0.045, 0.565, "边界身份开关", fontsize=11.0, fontweight="bold", color=COLORS["muted"])
    add_box(
        ax,
        0.18,
        0.46,
        0.29,
        0.15,
        "D 模式：主模型",
        "每个裁剪片段独立参与导电\n同源片段不自动并边",
        COLORS["blue_light"],
        COLORS["blue_dark"],
    )
    add_box(
        ax,
        0.53,
        0.46,
        0.29,
        0.15,
        "S 模式：敏感性模型",
        "仅实际越界产生的同源片段\n共享带电状态",
        COLORS["sage_light"],
        COLORS["sage_dark"],
    )
    arrow(ax, (0.86, 0.665), (0.50, 0.635))
    ax.plot([0.325, 0.675], [0.635, 0.635], color=COLORS["grey_dark"], linewidth=1.0)
    arrow(ax, (0.325, 0.635), (0.325, 0.61))
    arrow(ax, (0.675, 0.635), (0.675, 0.61))

    outputs_spec = (
        ("Q1 构型判定", "组 1 断开\n组 2、3 导通"),
        ("Q2 概率查询", "给定体积分数\n点估计与区间"),
        ("Q3 阈值定位", "D 模式 613 根\n体积分数 0.866608%"),
        ("Q4 成本优化", "成本前沿批量筛选\n临界候选定向加样"),
    )
    for x, (title, body), face, edge in zip(xs, outputs_spec, faces, edges):
        add_box(ax, x, 0.13, width, 0.18, title, body, face, edge)
    ax.plot([0.325, 0.675], [0.415, 0.415], color=COLORS["grey_dark"], linewidth=1.0)
    arrow(ax, (0.325, 0.46), (0.325, 0.415))
    arrow(ax, (0.675, 0.46), (0.675, 0.415))
    ax.plot([0.14, 0.86], [0.37, 0.37], color=COLORS["grey_dark"], linewidth=1.0)
    arrow(ax, (0.5, 0.415), (0.5, 0.37))
    for x in (0.14, 0.38, 0.62, 0.86):
        arrow(ax, (x, 0.37), (x, 0.31))
    ax.text(
        0.5,
        0.045,
        "两种模式复用同一中心、方向、加入顺序和随机种子；差值仅由片段身份规则产生。",
        ha="center",
        fontsize=10.8,
        color=COLORS["muted"],
    )
    outputs = save_figure(fig, "fig01_clipped_fragment_graph_pipeline")
    return outputs, {
        "figure_id": "fig01_clipped_fragment_graph_pipeline",
        "claim": "只有实际非空裁剪片段参与周期接触，D为主模型，S为同随机流敏感性模型",
        "inputs": [
            PROJECT_ROOT / "20_mainline" / "windows_review_pack" / "华数杯A题_Windows可运行复核版" / "src" / "microstructure_sim.cpp"
        ],
        "type": "generic_schematic",
    }


def add_box_3d(ax) -> None:
    lo, hi = -5000.0, 5000.0
    vertices = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)])
    edges = ((0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7))
    for first, second in edges:
        points = vertices[[first, second]]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=COLORS["soft"], linewidth=0.7)
    yy, zz = np.meshgrid(np.linspace(lo, hi, 2), np.linspace(lo, hi, 2))
    ax.plot_surface(np.full_like(yy, lo), yy, zz, color=COLORS["blue"], alpha=0.06, shade=False)
    ax.plot_surface(np.full_like(yy, hi), yy, zz, color=COLORS["orange"], alpha=0.06, shade=False)


def style_3d(ax, title: str) -> None:
    add_box_3d(ax)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlim=(-5000, 5000), ylim=(-5000, 5000), zlim=(-5000, 5000))
    ax.view_init(elev=22, azim=-55)
    ax.set_xticks([-5000, 0, 5000])
    ax.set_yticks([-5000, 0, 5000])
    ax.set_zticks([-5000, 0, 5000])
    ax.set_xticklabels(["-5k", "0", "5k"])
    ax.set_yticklabels(["-5k", "0", "5k"])
    ax.set_zticklabels(["-5k", "0", "5k"])
    ax.set_xlabel("X / nm", labelpad=1)
    ax.set_ylabel("Y / nm", labelpad=1)
    ax.set_zlabel("Z / nm", labelpad=1)
    ax.set_title(title, pad=7, fontsize=14.0, fontweight="bold")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(COLORS["soft"])


def make_q1_figure() -> tuple[list[Path], dict]:
    fig = plt.figure(figsize=(10.0, 4.7))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.09, top=0.84, wspace=0.02)
    inputs: list[Path] = []
    for panel, group in enumerate((1, 2, 3), start=1):
        data_path = PACK / "data" / f"group{group}.csv"
        report_path = Q1_EVIDENCE / f"group{group}.json"
        segments = np.loadtxt(data_path, delimiter=",")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected = {int(value) for value in report["path"] if isinstance(value, int)}
        inputs.extend((data_path, report_path))
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        for index, row in enumerate(segments, start=1):
            if index in selected:
                continue
            if group == 1 and index in set(report["left_component"]):
                color, alpha, width = COLORS["blue_dark"], 0.62, 1.0
            elif group == 1 and index in set(report["right_component"]):
                color, alpha, width = COLORS["orange_dark"], 0.62, 1.0
            else:
                color = COLORS["muted"]
                alpha = 0.22 if group == 2 else 0.045
                width = 0.55 if group == 2 else 0.25
            ax.plot(row[[0, 3]], row[[1, 4]], row[[2, 5]], color=color, alpha=alpha, linewidth=width)
        for index in selected:
            row = segments[index - 1]
            ax.plot(row[[0, 3]], row[[1, 4]], row[[2, 5]], color=COLORS["orange_dark"], linewidth=4.4, solid_capstyle="round")
            ax.plot(row[[0, 3]], row[[1, 4]], row[[2, 5]], color=COLORS["orange"], linewidth=2.4, solid_capstyle="round")
        status = "导通" if report["conductive"] else "未导通"
        style_3d(ax, f"({chr(96 + panel)}) 组 {group}：{status}")
        if selected:
            ax.text2D(0.03, 0.94, "路径 " + "→".join(map(str, report["path"])), transform=ax.transAxes, fontsize=10.5, color=COLORS["orange_dark"])
        elif group == 1:
            ax.text2D(0.03, 0.94, "蓝/橙分别为左右电极连通分量", transform=ax.transAxes, fontsize=10.2, color=COLORS["muted"])
    fig.suptitle("附件微构体的等比例三维构型与导通路径", fontsize=17.0, fontweight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.012, "三轴范围均为 -5000 至 5000 nm；浅灰线保留总体结构，暖橙粗线标出路径。", ha="center", fontsize=10.8, color=COLORS["muted"])
    outputs = save_figure(fig, "fig02_q1_scaled_3d_paths_v230")
    return outputs, {
        "figure_id": "fig02_q1_scaled_3d_paths_v230",
        "claim": "附件组1不导通，组2和组3存在左右电极之间的路径",
        "inputs": inputs,
        "type": "scaled_orthographic_3d",
    }


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=COLORS["grid"], linewidth=0.7)
    ax.tick_params(colors=COLORS["muted"])
    ax.spines["left"].set_color(COLORS["grey_dark"])
    ax.spines["bottom"].set_color(COLORS["grey_dark"])


def make_probability_figure() -> tuple[list[Path], dict]:
    source = EVIDENCE / "pooled_D_v230_150000_prefix.csv"
    frame = pd.read_csv(source)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.45), constrained_layout=True)
    q2 = frame[frame["N_A"].isin([354, 424, 495, 707])]
    axes[0].errorbar(
        q2["N_A"], q2["probability"],
        yerr=np.vstack((q2["probability"] - q2["Wilson95_low"], q2["Wilson95_high"] - q2["probability"])),
        fmt="o-", markersize=6.5, capsize=4, linewidth=1.4,
        color=COLORS["blue_dark"], markerfacecolor=COLORS["blue"], markeredgewidth=1.0,
    )
    for row in q2.itertuples():
        axes[0].annotate(f"{row.probability:.4f}", (row.N_A, row.probability), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10.2, color=COLORS["ink"])
    axes[0].set(xlabel="介质 A 数量", ylabel="导通概率", title="(a) 问题二：四个指定数量")
    axes[0].set_ylim(0.0, 1.04)
    axes[0].set_xticks([354, 424, 495, 707])
    style_axes(axes[0])

    near = frame[(frame["N_A"] >= 580) & (frame["N_A"] <= 625)]
    axes[1].fill_between(near["N_A"], near["Wilson95_low"], near["Wilson95_high"], color=COLORS["blue"], alpha=0.45, label="95% Wilson 区间")
    axes[1].plot(near["N_A"], near["probability"], color=COLORS["blue_dark"], linewidth=1.7, label="D 模式点估计")
    axes[1].axhline(0.90, color=COLORS["grey_dark"], linestyle="--", linewidth=1.0)
    row = frame[frame["N_A"] == 613].iloc[0]
    axes[1].scatter([613], [row["probability"]], marker="*", s=130, color=COLORS["orange"], edgecolor=COLORS["orange_dark"], zorder=5)
    axes[1].annotate(
        "613 根\n$\\hat p$=0.901613\nWilson 下限=0.900096",
        (613, row["probability"]), xytext=(-84, 45), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["orange_dark"]},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": COLORS["orange_light"], "edgecolor": "none"},
        fontsize=10.3, color=COLORS["ink"],
    )
    axes[1].text(0.98, 0.08, "S 模式阈值：8 根\n仅作边界身份敏感性", transform=axes[1].transAxes, ha="right", fontsize=10.2, color=COLORS["sage_dark"])
    axes[1].set(xlabel="介质 A 数量", ylabel="导通概率", title="(b) 问题三：D 模式临界区间")
    axes[1].set_ylim(0.82, 0.96)
    style_axes(axes[1])
    axes[1].legend(loc="upper left")
    outputs = save_figure(fig, "fig03_q2_q3_v230")
    return outputs, {
        "figure_id": "fig03_q2_q3_v230",
        "claim": "D模式问题二概率依次为0.077133、0.216500、0.473640、0.994087，90%阈值为613根",
        "inputs": [source, EVIDENCE / "pooled_D_v230_150000_summary.json"],
        "type": "uncertainty_and_threshold",
    }


def make_q4_figure() -> tuple[list[Path], dict]:
    low_path = EVIDENCE / "q4_v230_below_612A8B_low_t1000.csv"
    mid_path = EVIDENCE / "q4_v230_below_612A8B_mid_t5000.csv"
    high_path = EVIDENCE / "q4_v230_strict_cheaper_A590_612_pooled150000.csv"
    branch_path = EVIDENCE / "q4_v230_branch_below_A612B2_A605_611_pooled150000.csv"
    local_path = EVIDENCE / "q4_v230_local_grid_A612_B1_8_pooled150000.csv"
    a_path = EVIDENCE / "pooled_D_v230_150000_prefix.csv"
    low = pd.read_csv(low_path)
    mid = pd.read_csv(mid_path)
    high = pd.read_csv(high_path)
    branch = pd.read_csv(branch_path)
    local = pd.read_csv(local_path)
    a_frame = pd.read_csv(a_path)
    row613 = a_frame[a_frame["N_A"] == 613].iloc[0]
    cost_a = 0.0148440253
    cost_b = 0.0016755161
    conservative_cost = 613 * cost_a + cost_b

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.55), constrained_layout=True)
    axes[0].scatter(low["N_A"], low["probability"], s=19, facecolors="white", edgecolors=COLORS["grey_dark"], linewidths=0.7, label="分层筛选：1000 次")
    axes[0].plot(mid["N_A"], mid["probability"], color=COLORS["sage_dark"], linewidth=1.25, alpha=0.9, label="密集筛选：5000 次")
    axes[0].plot(high["N_A"], high["probability"], color=COLORS["blue_dark"], linewidth=1.6, label="临界复算：150000 次")
    axes[0].axhline(0.90, color=COLORS["grey_dark"], linestyle="--", linewidth=1.0)
    axes[0].set(xlabel="介质 A 数量", ylabel="导通概率", title="(a) 成本前沿的分层筛选与局部加密")
    axes[0].set_ylim(0.0, 0.93)
    style_axes(axes[0])
    axes[0].legend(loc="lower right")

    axes[1].scatter(
        branch["cost_yuan"], branch["probability"], s=43,
        facecolors="white", edgecolors=COLORS["blue_dark"], linewidths=0.9,
        label="严格更低成本分支",
    )
    axes[1].plot(
        local["cost_yuan"], local["probability"], color=COLORS["blue"],
        marker="o", markersize=4.8, linewidth=1.2, label="612 A，B=1--8",
    )
    optimum = local[(local["N_A"] == 612) & (local["N_B"] == 2)].iloc[0]
    axes[1].scatter([optimum["cost_yuan"]], [optimum["probability"]], marker="*", s=155, color=COLORS["orange"], edgecolor=COLORS["orange_dark"], zorder=5)
    axes[1].errorbar(
        [conservative_cost], [row613["probability"]],
        yerr=[[row613["probability"] - row613["Wilson95_low"]], [row613["Wilson95_high"] - row613["probability"]]],
        fmt="D", markersize=6.5, capsize=4, color=COLORS["sage_dark"], markerfacecolor=COLORS["sage"], label="613 A + 1 B 保守方案",
    )
    axes[1].axhline(0.90, color=COLORS["grey_dark"], linestyle="--", linewidth=1.0)
    axes[1].annotate(
        "经验最优\n612 A + 2 B\n$\\hat p$=0.900047",
        (optimum["cost_yuan"], optimum["probability"]), xytext=(-10, 55), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["orange_dark"]},
        bbox={"boxstyle": "round,pad=0.22", "facecolor": COLORS["orange_light"], "edgecolor": "none"},
        fontsize=9.8, color=COLORS["ink"],
    )
    axes[1].annotate(
        "置信保守\n613 A + 1 B\nCP 下限≥0.900339",
        (conservative_cost, row613["probability"]), xytext=(-28, -72), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["sage_dark"]},
        bbox={"boxstyle": "round,pad=0.22", "facecolor": COLORS["sage_light"], "edgecolor": "none"},
        fontsize=9.8, color=COLORS["ink"],
    )
    axes[1].set(xlabel="总成本 / 元", ylabel="导通概率", title="(b) 临界成本带与两种判据")
    axes[1].set_xlim(9.0858, 9.1015)
    axes[1].set_ylim(0.896, 0.904)
    style_axes(axes[1])
    axes[1].legend(loc="lower left", fontsize=9.7)
    outputs = save_figure(fig, "fig04_q4_cost_frontier_v230")
    return outputs, {
        "figure_id": "fig04_q4_cost_frontier_v230",
        "claim": "固定15万次点估计的经验最优为612A加2B，95%下界可行的保守方案为613A加1B",
        "inputs": [low_path, mid_path, high_path, branch_path, local_path, a_path],
        "type": "staged_integer_cost_frontier",
    }


def write_registry(records: list[dict], font: str) -> None:
    rows = []
    detailed = []
    for record in records:
        outputs = [{"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)} for path in record.pop("outputs")]
        inputs = [{"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)} for path in record["inputs"]]
        detailed.append({**record, "inputs": inputs, "outputs": outputs})
        rows.append(
            {
                "figure_id": record["figure_id"],
                "status": "candidate",
                "type": record["type"],
                "claim": record["claim"],
                "source_data": " | ".join(item["path"] for item in inputs),
                "svg": next(item["path"] for item in outputs if item["path"].endswith(".svg")),
                "pdf": next(item["path"] for item in outputs if item["path"].endswith(".pdf")),
                "png": next(item["path"] for item in outputs if item["path"].endswith(".png") and "_gray" not in item["path"]),
                "human_review": "pending",
            }
        )
    with (EVIDENCE / "figure_registry_v230.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (EVIDENCE / "figure_manifest_v230.json").write_text(
        json.dumps({"status": "candidate", "font": font, "generator": str(Path(__file__).relative_to(PROJECT_ROOT)), "figures": detailed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    if args.project_root.resolve() != PROJECT_ROOT.resolve():
        raise SystemExit("project root mismatch")
    font = configure()
    records = []
    for builder in (make_model_figure, make_q1_figure, make_probability_figure, make_q4_figure):
        outputs, record = builder()
        record["outputs"] = outputs
        records.append(record)
    write_registry(records, font)
    print(json.dumps({"status": "candidate_ready", "figures": len(records), "font": font}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
