from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = PROJECT_ROOT / "公共代码"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from microstructure_sim import (
    load_threshold_artifact,
    nominal_volume_percent,
    probability_at_prefix,
)
from plot_style import SEMANTIC_COLORS, apply_paper_style, save_figure


Q2_SUMMARY = (
    PROJECT_ROOT / "问题" / "问题2" / "results" / "D_primary_n20000" / "q2_summary.json"
)
Q3_SUMMARY = (
    PROJECT_ROOT / "问题" / "问题3" / "results" / "D_confirmation_n50000" / "q3_summary.json"
)
OUTPUT_STEM = PROJECT_ROOT / "论文" / "figures" / "generated" / "q2_q3_probability"
DATA_PATH = PROJECT_ROOT / "论文" / "figures" / "data" / "q2_q3_probability_curve.csv"
AUDIT_PATH = OUTPUT_STEM.with_suffix(".audit.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def project_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def build_curve(q2: dict) -> list[dict[str, float | int]]:
    threshold_path = Path(q2["threshold_artifact"])
    config, samples, _ = load_threshold_artifact(threshold_path)
    records: list[dict[str, float | int]] = []
    for count in range(300, 721, 4):
        probability = probability_at_prefix(samples, count, config.max_count, 0.95)
        records.append(
            {
                "particle_count": count,
                "volume_percent": nominal_volume_percent(count, config),
                "estimate": float(probability["estimate"]),
                "wilson_lower": float(probability["wilson_interval"][0]),
                "wilson_upper": float(probability["wilson_interval"][1]),
            }
        )
    return records


def write_curve(records: list[dict[str, float | int]]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def make_figure(q2: dict, q3: dict, curve: list[dict[str, float | int]]):
    apply_paper_style()
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.15), constrained_layout=True)

    x = np.asarray([row["volume_percent"] for row in curve], dtype=float)
    estimate = np.asarray([row["estimate"] for row in curve], dtype=float)
    lower = np.asarray([row["wilson_lower"] for row in curve], dtype=float)
    upper = np.asarray([row["wilson_upper"] for row in curve], dtype=float)

    ax = axes[0]
    ax.fill_between(
        x,
        lower,
        upper,
        color=SEMANTIC_COLORS["primary"],
        alpha=0.16,
        linewidth=0,
        label="95% Wilson 区间",
    )
    ax.plot(x, estimate, color=SEMANTIC_COLORS["primary"], label="20000 次概率估计")

    q2_x = []
    q2_y = []
    q2_low = []
    q2_high = []
    for record in q2["probability_records"]:
        probability = record["probability"]
        q2_x.append(float(record["actual_discrete_volume_percent"]))
        q2_y.append(float(probability["estimate"]))
        q2_low.append(float(probability["wilson_interval"][0]))
        q2_high.append(float(probability["wilson_interval"][1]))
    q2_x_arr = np.asarray(q2_x)
    q2_y_arr = np.asarray(q2_y)
    ax.errorbar(
        q2_x_arr,
        q2_y_arr,
        yerr=np.vstack((q2_y_arr - q2_low, np.asarray(q2_high) - q2_y_arr)),
        fmt="o",
        color=SEMANTIC_COLORS["comparison"],
        markeredgecolor="white",
        markeredgewidth=0.7,
        capsize=3,
        label="题给四个填充量",
        zorder=4,
    )
    for x_value, y_value in zip(q2_x_arr, q2_y_arr):
        ax.annotate(
            f"{100.0 * y_value:.1f}%",
            (x_value, y_value),
            xytext=(0, 8 if y_value < 0.95 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            color=SEMANTIC_COLORS["comparison"],
        )

    volume_bracket = q3["decision"]["minimum_volume_fraction_percent_bracket"]
    ax.axvspan(
        float(volume_bracket[0]),
        float(volume_bracket[1]),
        color=SEMANTIC_COLORS["accepted"],
        alpha=0.20,
        label="问题三最小量括区",
    )
    ax.axhline(0.9, color=SEMANTIC_COLORS["reference"], linestyle="--", linewidth=1.2)
    ax.set(xlabel="介质 A 名义体积分数（%）", ylabel="微构体导通概率", xlim=(0.42, 1.03), ylim=(0, 1.025))
    ax.set_title("(a) 全范围导通概率与题给填充量")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")

    ax = axes[1]
    status_style = {
        "statistically_infeasible": (SEMANTIC_COLORS["comparison"], "统计不可行", "X"),
        "unresolved": (SEMANTIC_COLORS["warning"], "未决", "D"),
        "statistically_feasible": (SEMANTIC_COLORS["accepted"], "统计可行", "o"),
    }
    plotted_labels: set[str] = set()
    for record in q3["candidate_records"]:
        count = int(record["particle_count"])
        value = float(record["estimate"])
        bounds = record["clopper_pearson_one_sided_bounds"]
        status = str(record["classification_by_bonferroni_cp"])
        color, label, marker = status_style[status]
        shown_label = label if label not in plotted_labels else None
        plotted_labels.add(label)
        ax.errorbar(
            count,
            value,
            yerr=[[value - float(bounds["lower"])], [float(bounds["upper"]) - value]],
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=3,
            label=shown_label,
            zorder=3,
        )
    ax.axhline(0.9, color=SEMANTIC_COLORS["reference"], linestyle="--", linewidth=1.2, label="目标概率 90%")
    ax.axvspan(612.5, 616.5, color=SEMANTIC_COLORS["uncertainty"], alpha=0.10)
    ax.annotate("上限 < 90%", (612, 0.89528), xytext=(-5, -27), textcoords="offset points", ha="center", fontsize=8.3)
    ax.annotate("下限 > 90%", (616, 0.90452), xytext=(8, 18), textcoords="offset points", ha="center", fontsize=8.3)
    ax.set(xlabel="介质 A 数量（根）", ylabel="导通概率", xlim=(608.4, 617.6), ylim=(0.882, 0.913))
    ax.set_xticks(range(609, 618))
    ax.set_title("(b) 临界整数的联合置信判定")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", ncol=2)
    return figure


def main() -> None:
    q2 = read_json(Q2_SUMMARY)
    q3 = read_json(Q3_SUMMARY)
    curve = build_curve(q2)
    write_curve(curve)
    figure = make_figure(q2, q3, curve)
    pdf_path, png_path = save_figure(figure, OUTPUT_STEM)
    plt.close(figure)

    with Image.open(png_path) as image:
        dimensions = list(image.size)
    threshold_path = Path(q2["threshold_artifact"])
    audit = {
        "schema_version": 1,
        "status": "passed",
        "purpose": "支持问题二四点概率与问题三题定精度最低填充量",
        "sources": {
            project_path(Q2_SUMMARY): sha256(Q2_SUMMARY),
            project_path(Q3_SUMMARY): sha256(Q3_SUMMARY),
            project_path(threshold_path): sha256(threshold_path),
            project_path(DATA_PATH): sha256(DATA_PATH),
        },
        "sample_counts": {"q2": 20_000, "q3": 50_000},
        "q3_integer_bracket": q3["decision"]["minimum_integer_bracket"],
        "q3_reported_volume_percent": q3["decision"]["reported_volume_fraction_percent"],
        "outputs": {
            project_path(pdf_path): sha256(pdf_path),
            project_path(png_path): sha256(png_path),
            "png_dimensions": dimensions,
        },
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
