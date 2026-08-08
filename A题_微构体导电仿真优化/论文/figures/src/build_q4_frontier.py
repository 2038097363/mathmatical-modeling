from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = PROJECT_ROOT / "论文" / "figures"
FINAL_STATUSES = {
    "globally_certified_minimum_cost",
    "lowest_statistically_feasible_cost",
}
COST_A_WEIGHT = 567
COST_B_WEIGHT = 64
COST_SCALE_YUAN = math.pi / 120000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="由问题4冻结结果生成完整整数域探索、可行成本与更便宜设计排除证据图"
    )
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=FIGURE_ROOT / "generated" / "q4_cost_frontier.pdf",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=FIGURE_ROOT / "generated" / "q4_cost_frontier.png",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=FIGURE_ROOT / "generated" / "q4_cost_frontier.audit.json",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象：{path}")
    return payload


def resolve_evidence_path(raw: Any, owner: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("结果文件缺少证据路径")
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [owner.parent / path, PROJECT_ROOT / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(str(candidates[0]))


def validate_hash(path: Path, expected: Any, label: str) -> str:
    actual = sha256(path)
    normalized = str(expected or "").strip().upper()
    if normalized and normalized != actual:
        raise ValueError(f"{label} 的 SHA-256 不一致")
    return actual


def load_inputs(
    screening_path: Path, final_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, np.ndarray, int, dict[str, Any]]:
    screening_path = screening_path.expanduser().resolve()
    final_path = final_path.expanduser().resolve()
    screening = read_json(screening_path)
    final = read_json(final_path)
    if screening.get("kind") != "q4_screening_results":
        raise ValueError("--screening 必须是 q4_screening_results")
    if final.get("kind") != "q4_final_summary":
        raise ValueError("--final 必须是 q4_final_summary")
    if final.get("result_status") not in FINAL_STATUSES:
        raise ValueError("问题4尚无可写入论文的最终统计可行设计")
    for payload, label in ((screening, "screening"), (final, "final")):
        contract = payload.get("boundary_contract")
        if not isinstance(contract, dict) or contract.get("mode") != "D":
            raise ValueError(f"{label} 未使用题设截断片段独立的 D 几何合同")

    counts_path = resolve_evidence_path(
        screening.get("integer_domain_success_counts"), screening_path
    )
    validate_hash(
        counts_path,
        screening.get("integer_domain_success_counts_sha256"),
        "完整整数域计数矩阵",
    )
    with np.load(counts_path, allow_pickle=False) as archive:
        required = {"success_counts", "trials", "max_n_a", "max_n_b"}
        if set(archive.files) != required:
            raise ValueError("整数域 NPZ 字段不完整或含未冻结字段")
        counts = np.asarray(archive["success_counts"], dtype=np.int64)
        trials = int(np.asarray(archive["trials"]).item())
        max_n_a = int(np.asarray(archive["max_n_a"]).item())
        max_n_b = int(np.asarray(archive["max_n_b"]).item())
    if counts.shape != (max_n_a + 1, max_n_b + 1):
        raise ValueError("整数域计数矩阵形状与边界不一致")
    if trials != int(screening["fixed_trial_count"]):
        raise ValueError("整数域计数矩阵样本数与 screening 不一致")
    if np.any(counts < 0) or np.any(counts > trials):
        raise ValueError("整数域成功次数越界")

    freeze_path = resolve_evidence_path(final.get("freeze_path"), final_path)
    validate_hash(freeze_path, final.get("freeze_sha256"), "冻结协议")
    freeze = read_json(freeze_path)
    if freeze.get("kind") != "q4_confirmation_freeze":
        raise ValueError("最终结果引用的文件不是 Q4 冻结协议")
    candidate = final.get("reported_design")
    if not isinstance(candidate, dict):
        raise ValueError("最终结果缺少 reported_design")
    candidate_records = [
        record
        for record in final.get("confirmation_records", [])
        if isinstance(record, dict) and record.get("role") == "candidate"
    ]
    if len(candidate_records) != 1:
        raise ValueError("最终确认记录必须恰有一个 candidate")
    record = candidate_records[0]
    if record.get("proof_status") != "candidate_statistically_feasible":
        raise ValueError("最终候选未通过单侧下限检验")
    for key in ("n_a", "n_b", "cost_weight"):
        if int(candidate[key]) != int(record[key]):
            raise ValueError(f"reported_design 与 candidate 的 {key} 不一致")
    frozen_candidate = freeze.get("candidate_freeze", {})
    for key in ("n_a", "n_b", "cost_weight"):
        if int(candidate[key]) != int(frozen_candidate[key]):
            raise ValueError(f"最终设计与冻结候选的 {key} 不一致")
    return screening, final, counts_path, counts, trials, freeze


def monotonicity_audit(counts: np.ndarray) -> dict[str, Any]:
    violations_a = int(np.count_nonzero(np.diff(counts, axis=0) < 0))
    violations_b = int(np.count_nonzero(np.diff(counts, axis=1) < 0))
    return {
        "n_a_direction_violations": violations_a,
        "n_b_direction_violations": violations_b,
        "passed": violations_a == 0 and violations_b == 0,
    }


def empirical_boundary(probabilities: np.ndarray, level: float) -> np.ndarray:
    mask = probabilities >= level
    first = np.argmax(mask, axis=1).astype(np.float64)
    first[~np.any(mask, axis=1)] = np.nan
    return first


def configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10.2,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def build_figure(
    screening: dict[str, Any],
    final: dict[str, Any],
    counts: np.ndarray,
    trials: int,
    output_pdf: Path,
    output_png: Path,
    dpi: int,
) -> dict[str, Any]:
    if dpi < 200:
        raise ValueError("论文 PNG 的 dpi 不得低于 200")
    configure_fonts()
    probabilities = counts.astype(np.float64) / trials
    max_n_a, max_n_b = counts.shape[0] - 1, counts.shape[1] - 1
    candidate = final["reported_design"]
    n_a = int(candidate["n_a"])
    n_b = int(candidate["n_b"])
    candidate_weight = int(candidate["cost_weight"])
    if not (0 <= n_a <= max_n_a and 0 <= n_b <= max_n_b):
        raise ValueError("最终设计超出探索整数域")

    stride_a = max(1, math.ceil((max_n_a + 1) / 260))
    stride_b = max(1, math.ceil((max_n_b + 1) / 420))
    sampled = probabilities[::stride_a, ::stride_b].T
    extent = [0, max_n_a, 0, max_n_b]
    boundary_90 = empirical_boundary(probabilities, float(screening["target_probability"]))

    records = [record for record in final["confirmation_records"] if record.get("role") != "candidate"]
    if not records:
        raise ValueError("最终结果没有严格更便宜极大点确认记录")
    frontier_a = np.asarray([int(record["n_a"]) for record in records])
    frontier_b = np.asarray([int(record["n_b"]) for record in records])
    frontier_upper = np.asarray(
        [float(record["clopper_pearson_one_sided_upper"]) for record in records]
    )
    excluded = np.asarray(
        [record.get("proof_status") == "strictly_cheaper_design_excluded" for record in records]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.25), constrained_layout=True)
    left, right = axes
    probability_cmap = LinearSegmentedColormap.from_list(
        "paper_probability",
        ["#F7F9FC", "#C9D9EB", "#2457A7", "#0B7A75", "#F0C36B"],
    )
    image = left.imshow(
        sampled,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=probability_cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    left.plot(
        np.arange(max_n_a + 1),
        boundary_90,
        color="#C46618",
        linewidth=2.4,
        label="经验导通概率 90% 边界",
    )
    line_a = np.arange(0, min(max_n_a, candidate_weight // COST_A_WEIGHT) + 1)
    line_b = (candidate_weight - COST_A_WEIGHT * line_a) / COST_B_WEIGHT
    visible = (line_b >= 0) & (line_b <= max_n_b)
    left.plot(
        line_a[visible],
        line_b[visible],
        color="#2F343B",
        linewidth=2.0,
        linestyle="--",
        label="候选等成本线",
    )
    left.scatter(
        [n_a],
        [n_b],
        marker="*",
        s=190,
        color="#0B7A75",
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        label="冻结候选",
    )
    left.set(
        xlabel="介质 A 数量 $N_A$",
        ylabel="介质 B 数量 $N_B$",
        title=f"(a) 完整整数域探索（{trials:,} 次）",
        xlim=(0, max_n_a),
        ylim=(0, max_n_b),
    )
    left.grid(color="white", alpha=0.22, linewidth=0.6)
    left.legend(loc="upper right", frameon=True, framealpha=0.94)
    colorbar = fig.colorbar(image, ax=left, fraction=0.046, pad=0.02)
    colorbar.set_label("经验导通概率")

    frontier_cost = (COST_A_WEIGHT * frontier_a + COST_B_WEIGHT * frontier_b) * COST_SCALE_YUAN
    frontier_probability = 100.0 * probabilities[frontier_a, frontier_b]
    screening_pass = frontier_probability >= 100.0 * float(screening["target_probability"])
    right.scatter(
        frontier_cost[~screening_pass],
        frontier_probability[~screening_pass],
        s=22,
        marker="o",
        facecolor="#2457A7",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.78,
        label="探索经验率低于 90%",
        zorder=3,
    )
    right.scatter(
        frontier_cost[screening_pass],
        frontier_probability[screening_pass],
        s=28,
        marker="D",
        facecolor="#C46618",
        edgecolor="#7A3C08",
        linewidth=0.5,
        alpha=0.82,
        label="探索经验率不低于 90%",
        zorder=4,
    )
    candidate_cost = candidate_weight * COST_SCALE_YUAN
    candidate_probability = 100.0 * float(probabilities[n_a, n_b])
    right.scatter(
        [candidate_cost],
        [candidate_probability],
        marker="*",
        s=210,
        color="#0B7A75",
        edgecolor="#24483F",
        linewidth=0.7,
        zorder=5,
        label="冻结候选（探索定位）",
    )
    unresolved_count = int((~excluded).sum())
    right.axhline(
        100.0 * float(screening["target_probability"]),
        color="#2F343B",
        linewidth=1.35,
        linestyle="--",
        label="90% 探索门槛",
    )
    right.set(
        xlabel="成本（元）",
        ylabel="经验导通概率（%）",
        title=f"(b) 极大前沿的成本—概率探索（{trials:,} 次）",
    )
    right.grid(color="#d8d8d8", linewidth=0.55, alpha=0.65)
    right.legend(loc="lower right", frameon=True, framealpha=0.94)
    right.text(
        0.03,
        0.95,
        "探索样本仅用于候选定位\n正式结论由独立确认流给出",
        transform=right.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "#D8DDE1", "alpha": 0.9, "pad": 4.0},
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    with Image.open(output_png) as opened:
        dimensions = list(opened.size)
    return {
        "png_dimensions": dimensions,
        "candidate_screening_estimate": float(probabilities[n_a, n_b]),
        "frontier_record_count": len(records),
        "excluded_frontier_count": int(excluded.sum()),
        "not_excluded_frontier_count": unresolved_count,
        "screening_downsample_stride": [stride_a, stride_b],
    }


def claim_scope_audit(final: dict[str, Any], figure: dict[str, Any]) -> dict[str, Any]:
    status = str(final["result_status"])
    unresolved = int(figure["not_excluded_frontier_count"])
    reported_unresolved = final.get("not_excluded_frontier_count")
    if reported_unresolved is not None and int(reported_unresolved) != unresolved:
        raise ValueError("图中尚未排除点数与最终结果不一致")
    reported_excluded = final.get("excluded_frontier_count")
    if reported_excluded is not None and int(reported_excluded) != int(
        figure["excluded_frontier_count"]
    ):
        raise ValueError("图中已排除点数与最终结果不一致")
    globally_certified = status == "globally_certified_minimum_cost"
    if globally_certified and unresolved:
        raise ValueError("仍有更便宜设计未排除时不得标记全局最低成本")
    return {
        "evidence_scope": "candidate_feasibility_and_cheaper_design_exclusion",
        "global_minimum_certified": globally_certified,
        "unresolved_cheaper_design_count": unresolved,
        "claim_zh": (
            "全局最低成本已获统计认证"
            if globally_certified
            else f"候选设计统计可行，仍有 {unresolved} 个严格更便宜极大设计未排除"
        ),
    }


def main() -> int:
    args = parse_args()
    screening_path = args.screening.expanduser().resolve()
    final_path = args.final.expanduser().resolve()
    output_pdf = args.output_pdf.expanduser().resolve()
    output_png = args.output_png.expanduser().resolve()
    audit_path = args.audit.expanduser().resolve()
    screening, final, counts_path, counts, trials, freeze = load_inputs(
        screening_path, final_path
    )
    monotonicity = monotonicity_audit(counts)
    if not monotonicity["passed"]:
        raise ValueError("完整整数域成功次数不满足二维单调性")
    figure = build_figure(
        screening,
        final,
        counts,
        trials,
        output_pdf,
        output_png,
        args.dpi,
    )
    claim_scope = claim_scope_audit(final, figure)
    candidate_record = next(
        record for record in final["confirmation_records"] if record.get("role") == "candidate"
    )
    audit = {
        "schema_version": 1,
        "kind": "q4_cost_frontier_figure_audit",
        "screening_path": str(screening_path),
        "screening_sha256": sha256(screening_path),
        "final_path": str(final_path),
        "final_sha256": sha256(final_path),
        "freeze_path": str(resolve_evidence_path(final["freeze_path"], final_path)),
        "freeze_sha256": final["freeze_sha256"],
        "integer_domain_counts_path": str(counts_path),
        "integer_domain_counts_sha256": sha256(counts_path),
        "integer_domain_shape": list(counts.shape),
        "fixed_trial_count": trials,
        "monotonicity": monotonicity,
        "result_status": final["result_status"],
        "reported_design": final["reported_design"],
        "candidate_confirmation": {
            "successes": candidate_record["successes"],
            "trials": candidate_record["trials"],
            "estimate": candidate_record["estimate"],
            "cp_lower": candidate_record["clopper_pearson_one_sided_lower"],
            "proof_status": candidate_record["proof_status"],
        },
        "bonferroni_statement_count": freeze["confirmation_protocol"][
            "bonferroni_statement_count"
        ],
        **claim_scope,
        "output_pdf": str(output_pdf),
        "output_pdf_sha256": sha256(output_pdf),
        "output_png": str(output_png),
        "output_png_sha256": sha256(output_png),
        **figure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
