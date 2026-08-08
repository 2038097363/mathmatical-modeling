# AI 工具：OpenAI Codex；模型/版本：GPT-5 系列；开发机构：OpenAI。
# 版本发布日期：2025-08-07（GPT-5 系列公开快照日期）；本程序由参赛队逐行复核并对结果负责。
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from cycler import cycler


PAPER_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#000000",
]

# Use semantic roles consistently across figures; do not assign colors by loop order
# when the same baseline, proposed method, warning, or accepted design recurs.
SEMANTIC_COLORS = {
    "primary": "#0072B2",
    "comparison": "#D55E00",
    "accepted": "#009E73",
    "warning": "#E69F00",
    "uncertainty": "#CC79A7",
    "reference": "#333333",
}


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (6.6, 4.1),
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.9,
            "axes.prop_cycle": cycler(color=PAPER_COLORS),
            "lines.linewidth": 1.7,
            "lines.markersize": 5.5,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(figure, output_stem: str | Path) -> tuple[Path, Path]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight", transparent=False)
    figure.savefig(png_path, bbox_inches="tight", dpi=320, transparent=False)
    return pdf_path, png_path
