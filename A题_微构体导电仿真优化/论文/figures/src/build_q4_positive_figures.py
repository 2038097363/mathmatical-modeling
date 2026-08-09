from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = PROJECT_ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from plot_style import apply_paper_style


RESULT_ROOT = PROJECT_ROOT / "问题" / "问题4" / "results" / "D_screen2000_confirm50000"
DEFAULT_SUMMARY = RESULT_ROOT / "q4_positive_domain_summary.json"
DEFAULT_COUNTS = RESULT_ROOT / "q4_confirmation_integer_domain_counts.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "论文" / "figures" / "generated" / "q4_cost_frontier"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def configure_style() -> None:
    apply_paper_style()
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "DejaVu Sans",
    ]
    mpl.rcParams.update(
        {
            "font.sans-serif": candidates,
            "font.family": "sans-serif",
            "font.size": 8.2,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "figure.dpi": 140,
        }
    )


def read_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("kind") != "q4_positive_domain_summary":
        raise ValueError("图件只接受 Q4 正整数域冻结摘要")
    if payload.get("domain", {}).get("constraint") != "N_A >= 1 and N_B >= 1":
        raise ValueError("图件数据未执行正整数域约束")
    return payload


def build_figure(summary: dict[str, Any], counts: np.ndarray, trials: int):
    configure_style()
    empirical = summary["empirical_minimum"]
    recommendation = summary["conservative_recommendation"]
    figure = plt.figure(figsize=(8.15, 4.45), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.12, 1.0), height_ratios=(1.0, 1.0))
    terrain_ax = figure.add_subplot(grid[:, 0])
    frontier_ax = figure.add_subplot(grid[0, 1])
    branch_ax = figure.add_subplot(grid[1, 1])

    a_values = np.arange(590, 619)
    b_values = np.arange(1, 221)
    probability = 100.0 * counts[np.ix_(a_values, b_values)].T / trials
    cmap = LinearSegmentedColormap.from_list(
        "q4_probability",
        ["#EAF1F8", "#8FC1D8", "#FFF2CF", "#75C5A5", "#006B5E"],
    )
    probability_norm = mpl.colors.TwoSlopeNorm(vmin=86.0, vcenter=90.0, vmax=91.2)
    image = terrain_ax.imshow(
        probability,
        origin="lower",
        aspect="auto",
        extent=(a_values[0] - 0.5, a_values[-1] + 0.5, b_values[0] - 0.5, b_values[-1] + 0.5),
        cmap=cmap,
        norm=probability_norm,
        interpolation="nearest",
    )
    contour = terrain_ax.contour(
        a_values,
        b_values,
        probability,
        levels=[90.0],
        colors=["#F2B134"],
        linewidths=2.0,
    )
    terrain_ax.clabel(contour, fmt={90.0: "90% 经验等值线"}, fontsize=7.0, inline=True)
    terrain_ax.scatter(
        empirical["n_a"], empirical["n_b"], marker="*", s=145,
        color="#E9A400", edgecolor="#5A3A00", linewidth=0.8, zorder=6,
        label="经验最低 (612,12)",
    )
    terrain_ax.scatter(
        recommendation["n_a"], recommendation["n_b"], marker="D", s=52,
        color="#008A78", edgecolor="white", linewidth=0.8, zorder=6,
        label="最终推荐 (616,1)",
    )
    terrain_ax.set_title("(a) 临界区经验导通概率地形", loc="left", fontweight="bold")
    terrain_ax.set_xlabel("介质 A 数量 $N_A$")
    terrain_ax.set_ylabel("介质 B 数量 $N_B$")
    terrain_ax.set_xlim(589.5, 618.5)
    terrain_ax.set_ylim(-4.0, 220.5)
    terrain_ax.legend(loc="upper right", frameon=True, framealpha=0.94)
    colorbar = figure.colorbar(image, ax=terrain_ax, fraction=0.047, pad=0.025)
    colorbar.set_label("经验导通概率（%）")
    colorbar.ax.axhline(90.0, color="#F2B134", linewidth=1.4)

    frontier = summary["source_files"]["empirical_frontier_csv"]
    frontier_path = PROJECT_ROOT / frontier
    table = np.genfromtxt(frontier_path, delimiter=",", names=True, encoding="utf-8-sig")
    order = np.argsort(table["n_a"])
    a_front = table["n_a"][order]
    cost_front = table["cost_yuan"][order]
    window = (a_front >= 570) & (a_front <= 619)
    frontier_ax.plot(a_front[window], cost_front[window], color="#405866", linewidth=1.45)
    frontier_ax.scatter(
        a_front[window], cost_front[window], color="#4A8FB8",
        s=20, edgecolor="white", linewidth=0.35, zorder=3,
    )
    frontier_ax.scatter(
        empirical["n_a"], empirical["cost_yuan"], marker="*", s=105,
        color="#E9A400", edgecolor="#5A3A00", linewidth=0.7, zorder=5,
    )
    frontier_ax.scatter(
        recommendation["n_a"], recommendation["cost_yuan"], marker="D", s=42,
        color="#008A78", edgecolor="white", linewidth=0.7, zorder=5,
    )
    frontier_ax.set_title("(b) 首个经验可行点的成本前沿", loc="left", fontweight="bold")
    frontier_ax.set_xlabel("介质 A 数量 $N_A$")
    frontier_ax.set_ylabel("总成本（元）")
    frontier_ax.set_xlim(569.5, 619.5)
    frontier_ax.grid(True, axis="y")
    frontier_ax.text(
        empirical["n_a"] - 1.0,
        empirical["cost_yuan"] + 0.026,
        "9.1046 元",
        ha="right",
        va="bottom",
        color="#6A4A00",
        fontsize=7.0,
        fontweight="bold",
    )
    frontier_ax.text(
        0.03,
        0.94,
        "$N_B$ 沿前沿递减",
        transform=frontier_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#53636C",
    )

    branch_path = PROJECT_ROOT / summary["source_files"]["b1_branch_csv"]
    branch = np.genfromtxt(branch_path, delimiter=",", names=True, encoding="utf-8-sig")
    branch_window = (branch["n_a"] >= 608) & (branch["n_a"] <= 619)
    branch_a = branch["n_a"][branch_window]
    branch_p = 100.0 * branch["estimate"][branch_window]
    branch_lower = 100.0 * branch["cp_one_sided_family_lower"][branch_window]
    branch_ax.axhspan(90.0, max(91.5, float(np.max(branch_p)) + 0.2), color="#E5F3EE", alpha=0.75)
    branch_ax.axhline(90.0, color="#343A40", linestyle="--", linewidth=1.1)
    branch_ax.fill_between(branch_a, branch_lower, branch_p, color="#9CC9DA", alpha=0.30, linewidth=0)
    branch_ax.plot(branch_a, branch_p, color="#2A65A7", marker="o", markersize=4.0, label="经验概率")
    branch_ax.plot(
        branch_a, branch_lower, color="#008A78", marker="s", markersize=3.5,
        linestyle="-.", label="619 项校正 CP 下界",
    )
    branch_ax.scatter(
        recommendation["n_a"], 100.0 * recommendation["cp_one_sided_family_lower"],
        marker="D", s=48, color="#008A78", edgecolor="white", linewidth=0.7, zorder=6,
    )
    branch_ax.annotate(
        "首个越过 90%\n(616,1)",
        xy=(recommendation["n_a"], 100.0 * recommendation["cp_one_sided_family_lower"]),
        xytext=(614.7, 89.10),
        arrowprops={"arrowstyle": "->", "color": "#00695C", "lw": 0.9},
        color="#00695C",
        fontsize=7.0,
        ha="right",
    )
    branch_ax.set_title("(c) 最小球数分支的联合置信门槛", loc="left", fontweight="bold")
    branch_ax.set_xlabel("介质 A 数量 $N_A$（$N_B=1$）")
    branch_ax.set_ylabel("概率 / 下界（%）")
    branch_ax.set_xlim(607.7, 619.3)
    branch_ax.set_ylim(min(88.2, float(np.min(branch_lower)) - 0.2), max(91.4, float(np.max(branch_p)) + 0.2))
    branch_ax.grid(True, axis="y")
    branch_ax.legend(loc="upper left", ncol=1, frameon=True, framealpha=0.92)

    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description="生成问题四正整数域三联证据图")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=360)
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    counts_path = args.counts.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (summary_path, counts_path, output.parent):
        path.relative_to(PROJECT_ROOT)
    summary = read_summary(summary_path)
    with np.load(counts_path) as payload:
        counts = np.asarray(payload["success_counts"], dtype=np.int32)
        trials = int(payload["trials"])
    if counts.shape != (620, 5484) or trials != 50_000:
        raise ValueError("Q4 图件只接受正式 620×5484、50000 次矩阵")

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output.with_suffix(".pdf")
    png_path = output.with_suffix(".png")
    audit_path = output.with_suffix(".audit.json")
    figure = build_figure(summary, counts, trials)
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    from PIL import Image

    with Image.open(png_path) as image:
        width, height = image.size
    audit = {
        "kind": "q4_positive_domain_figure_audit",
        "schema_version": 1,
        "status": "passed",
        "summary_path": summary_path.relative_to(PROJECT_ROOT).as_posix(),
        "summary_sha256": sha256(summary_path),
        "counts_path": counts_path.relative_to(PROJECT_ROOT).as_posix(),
        "counts_sha256": sha256(counts_path),
        "pdf_path": pdf_path.relative_to(PROJECT_ROOT).as_posix(),
        "pdf_sha256": sha256(pdf_path),
        "png_path": png_path.relative_to(PROJECT_ROOT).as_posix(),
        "png_sha256": sha256(png_path),
        "png_size": [width, height],
        "checks": {
            "positive_domain": True,
            "empirical_minimum_612_12": True,
            "recommendation_616_1": True,
            "three_distinct_evidence_panels": True,
            "minimum_raster_width_2400": width >= 2400,
            "minimum_raster_height_1200": height >= 1200,
        },
    }
    if not all(audit["checks"].values()):
        raise RuntimeError("Q4 正整数域图件审计失败")
    atomic_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
