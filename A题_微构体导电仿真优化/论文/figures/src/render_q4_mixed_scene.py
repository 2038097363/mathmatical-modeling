from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_freecad_scene import (
    DEFAULT_CAPTURE_DELAY_MS,
    DEFAULT_LINE_WIDTH,
    DEFAULT_MAX_NON_WHITE_RATIO,
    DEFAULT_MIN_BORDER_WHITE_RATIO,
    DEFAULT_NEAR_WHITE,
    atomic_copy,
    atomic_write_json,
    create_ascii_temp_dir,
    inspect_pixels,
    render as render_base,
    sha256,
)


DEFAULT_WIDTH = 2400
DEFAULT_HEIGHT = 1800
DEFAULT_ZOOM = 0.82
DEFAULT_TIMEOUT_SECONDS = 180.0
FINAL_PUBLICATION_STATUS = "final_random_trial_geometry"
VIEW_CHOICES = ("axonometric", "top", "bottom", "front", "rear", "left", "right")
FINAL_FORBIDDEN_TEXT = ("preview", "not an optimal design", "非最优")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="渲染问题4混合介质 FCStd，叠加状态标识并审计最终 PNG"
    )
    parser.add_argument("source", type=Path, help="输入 FCStd")
    parser.add_argument("scene", type=Path, help="对应的场景 JSON")
    parser.add_argument("output", type=Path, help="最终 PNG")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--freecad-exe", type=Path)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--view", choices=VIEW_CHOICES, default="axonometric")
    parser.add_argument(
        "--focus-witness",
        action="store_true",
        help="从同一 FCStd 隐藏普通背景，只显示实际导通见证、电极和边界",
    )
    return parser.parse_args()


def _load_font(size: int, *, bold: bool) -> tuple[ImageFont.ImageFont, str]:
    windows = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = (
        ("msyhbd.ttc", "simhei.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
        if bold
        else ("msyh.ttc", "simhei.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
    )
    for name in candidates:
        path = windows / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size), str(path)
        try:
            return ImageFont.truetype(name, size=size), name
        except OSError:
            continue
    return ImageFont.load_default(), "PIL-default"


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0])


def overlay_status_banner(image: Image.Image, scene: dict[str, Any]) -> dict[str, Any]:
    width, height = image.size
    scale = min(width / DEFAULT_WIDTH, height / DEFAULT_HEIGHT)
    title_font, title_font_path = _load_font(max(22, round(40 * scale)), bold=True)
    label_font, label_font_path = _load_font(max(16, round(24 * scale)), bold=False)
    draw = ImageDraw.Draw(image)

    title = str(scene.get("visible_banner", "")).strip()
    if not title:
        raise ValueError("场景缺少 visible_banner，不能生成状态不明的论文图片")
    publication_status = str(scene.get("publication_status", ""))
    witness_present = _witness_fragment_count(scene) > 0
    legend_items = []
    if scene.get("cylinders", []):
        legend_items.append(("line", (184, 193, 201), "介质 A 圆柱片段"))
    if scene.get("spheres", []):
        legend_items.append(("circle", (66, 151, 213), "介质 B 球形片段"))
    if bool(scene.get("electrodes", {}).get("show", True)):
        legend_items.append(("block", (87, 94, 101), "左右电极"))
    if witness_present:
        witness_label = (
            "橙色：实际导通见证"
            if publication_status == FINAL_PUBLICATION_STATUS
            else "橙色：示意见证链"
        )
        legend_items.append(("line", (237, 92, 31), witness_label))
    if not legend_items:
        raise ValueError("场景没有可用于图例的三维对象")

    margin_x = max(32, round(width * 0.02))
    margin_y = max(28, round(height * 0.02))
    padding_x = max(24, round(28 * scale))
    padding_y = max(16, round(18 * scale))
    row_gap = max(10, round(14 * scale))
    sample_width = max(30, round(44 * scale))
    item_gap = max(18, round(28 * scale))
    label_gap = max(8, round(10 * scale))
    title_height = draw.textbbox((0, 0), title, font=title_font)[3]
    label_height = draw.textbbox((0, 0), "Ag", font=label_font)[3]

    legend_width = 0
    for _, _, label in legend_items:
        legend_width += sample_width + label_gap + _text_width(draw, label, label_font)
    legend_width += item_gap * (len(legend_items) - 1)
    content_width = max(_text_width(draw, title, title_font), legend_width)
    banner_width = min(width - 2 * margin_x, content_width + 2 * padding_x)
    banner_height = 2 * padding_y + title_height + row_gap + label_height
    rectangle = (
        margin_x,
        margin_y,
        margin_x + banner_width,
        margin_y + banner_height,
    )
    draw.rectangle(rectangle, fill=(31, 39, 46))
    accent_width = max(8, round(10 * scale))
    draw.rectangle(
        (rectangle[0], rectangle[1], rectangle[0] + accent_width, rectangle[3]),
        fill=(224, 75, 45),
    )
    text_x = rectangle[0] + padding_x
    title_y = rectangle[1] + padding_y - 2
    draw.text((text_x, title_y), title, font=title_font, fill=(255, 255, 255))

    cursor_x = text_x
    legend_y = title_y + title_height + row_gap
    for kind, color, label in legend_items:
        center_y = legend_y + label_height // 2
        if kind == "line":
            draw.line(
                (cursor_x, center_y, cursor_x + sample_width, center_y),
                fill=color,
                width=max(4, round(6 * scale)),
            )
        elif kind == "circle":
            radius = max(7, round(9 * scale))
            center_x = cursor_x + sample_width // 2
            draw.ellipse(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                fill=color,
                outline=(235, 239, 242),
                width=max(1, round(2 * scale)),
            )
        else:
            inset = max(4, round(6 * scale))
            draw.rectangle(
                (
                    cursor_x + inset,
                    center_y - max(7, round(9 * scale)),
                    cursor_x + sample_width - inset,
                    center_y + max(7, round(9 * scale)),
                ),
                fill=color,
                outline=(235, 239, 242),
            )
        cursor_x += sample_width + label_gap
        draw.text((cursor_x, legend_y), label, font=label_font, fill=(232, 236, 239))
        cursor_x += _text_width(draw, label, label_font) + item_gap

    return {
        "text": title,
        "rectangle_xyxy": list(rectangle),
        "title_font": title_font_path,
        "label_font": label_font_path,
        "legend_labels": [item[2] for item in legend_items],
    }


def _witness_fragment_count(scene: dict[str, Any]) -> int:
    return sum(
        str(record.get("role", "")) == "witness"
        for record in list(scene.get("cylinders", [])) + list(scene.get("spheres", []))
    )


def inspect_scene_colors(
    path: Path,
    witness_expected: bool,
    *,
    a_expected: bool = True,
    b_expected: bool = True,
    electrodes_expected: bool = True,
    b_min_pixels: int = 100,
) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    orange = (
        (red >= 180)
        & (green >= 45)
        & (green <= 165)
        & (blue <= 115)
        & (red >= green + 55)
    )
    sphere_blue = (
        (blue >= 145)
        & (blue >= red + 25)
        & (blue >= green + 12)
        & (green >= 90)
    )
    neutral_a = (
        (red >= 125)
        & (red <= 235)
        & (green >= 125)
        & (green <= 238)
        & (blue >= 125)
        & (blue <= 242)
        & (np.max(rgb, axis=2) - np.min(rgb, axis=2) <= 34)
    )
    dark = np.max(rgb, axis=2) <= 125
    counts = {
        "a_neutral_pixel_count": int(neutral_a.sum()),
        "b_blue_pixel_count": int(sphere_blue.sum()),
        "witness_orange_pixel_count": int(orange.sum()),
        "electrode_or_outline_dark_pixel_count": int(dark.sum()),
    }
    checks = {
        "a_cylinders_visible": (
            not a_expected or counts["a_neutral_pixel_count"] >= 100
        ),
        "b_spheres_visible": (
            not b_expected or counts["b_blue_pixel_count"] >= b_min_pixels
        ),
        "electrodes_or_outline_visible": (
            not electrodes_expected
            or counts["electrode_or_outline_dark_pixel_count"] >= 100
        ),
        "witness_visible": (
            not witness_expected or counts["witness_orange_pixel_count"] >= 100
        ),
    }
    return {**counts, "checks": checks, "passed": all(checks.values())}


def render_q4(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve()
    scene_path = args.scene.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audit_path = (
        args.audit.expanduser().resolve()
        if args.audit is not None
        else output.with_suffix(".audit.json")
    )
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    focus_witness = bool(getattr(args, "focus_witness", False))
    publication_status = str(scene.get("publication_status", ""))
    if publication_status not in {
        "preview_not_optimal",
        "confirmed_design_representative",
        FINAL_PUBLICATION_STATUS,
    }:
        raise ValueError(f"不支持或缺失 publication_status：{publication_status!r}")
    if publication_status == "preview_not_optimal":
        for path in (source, scene_path, output):
            if "preview" not in path.name.lower():
                raise ValueError(f"预览源与输出文件名必须含 preview：{path}")
        if "NOT AN OPTIMAL DESIGN" not in str(scene.get("visible_banner", "")):
            raise ValueError("预览场景必须在 visible_banner 中明确标注非最优")
    if publication_status == FINAL_PUBLICATION_STATUS:
        serialized = json.dumps(scene, ensure_ascii=False, sort_keys=True).casefold()
        path_text = "\n".join(str(path) for path in (source, scene_path, output, audit_path)).casefold()
        forbidden = [
            token
            for token in FINAL_FORBIDDEN_TEXT
            if token.casefold() in serialized or token.casefold() in path_text
        ]
        if forbidden:
            raise ValueError(f"正式三维产物含禁用的预览措辞：{forbidden}")
        if _witness_fragment_count(scene) < 1:
            raise ValueError("正式随机试验场景缺少实际导通见证片段")
        if focus_witness and "witness" not in output.stem.casefold():
            raise ValueError("见证聚焦视图的输出文件名必须含 witness")
    if args.width < 800 or args.height < 600:
        raise ValueError("论文三维图分辨率不得低于 800x600")

    witness_expected = _witness_fragment_count(scene) > 0
    a_expected = any(
        str(record.get("role", "background_a")) != "witness"
        for record in scene.get("cylinders", [])
    ) and not focus_witness
    background_b_count = sum(
        str(record.get("role", "background_b")) != "witness"
        for record in scene.get("spheres", [])
    )
    b_expected = background_b_count > 1 and not focus_witness
    b_min_pixels = 12 if background_b_count == 1 else 100
    electrodes_expected = bool(scene.get("electrodes", {}).get("show", True))
    scene_electrode_transparency = int(
        scene.get("electrodes", {}).get("transparency", 52)
    )
    render_electrode_transparency = max(88, scene_electrode_transparency)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "publication_status": publication_status,
        "source": str(source),
        "source_sha256": sha256(source),
        "scene": str(scene_path),
        "scene_sha256": sha256(scene_path),
        "output": str(output),
        "audit": str(audit_path),
        "renderer": str(Path(__file__).resolve()),
        "renderer_sha256": sha256(Path(__file__).resolve()),
    }
    temporary = create_ascii_temp_dir()
    try:
        raw_min_non_white_ratio = 0.0005 if focus_witness else 0.003
        raw_png = temporary / "raw.png"
        raw_audit = temporary / "raw.audit.json"
        final_png = temporary / "final.png"
        base_args = argparse.Namespace(
            source=source,
            output=raw_png,
            audit=raw_audit,
            freecad_exe=args.freecad_exe,
            width=args.width,
            height=args.height,
            timeout=args.timeout,
            capture_delay_ms=DEFAULT_CAPTURE_DELAY_MS,
            line_width=DEFAULT_LINE_WIDTH,
            view=args.view,
            zoom=args.zoom,
            near_white=DEFAULT_NEAR_WHITE,
            min_non_white_ratio=raw_min_non_white_ratio,
            max_non_white_ratio=DEFAULT_MAX_NON_WHITE_RATIO,
            min_border_white_ratio=DEFAULT_MIN_BORDER_WHITE_RATIO,
            min_margin_pixels=max(24, round(min(args.width, args.height) * 0.015)),
            hidden_styles=("background_a", "background_b") if focus_witness else (),
            electrode_transparency=render_electrode_transparency,
            electrode_wireframe=True,
        )
        base_audit = render_base(base_args)
        scene_colors = inspect_scene_colors(
            raw_png,
            witness_expected,
            a_expected=a_expected,
            b_expected=b_expected,
            electrodes_expected=electrodes_expected,
            b_min_pixels=b_min_pixels,
        )
        if not scene_colors["passed"]:
            failed = [
                name for name, passed in scene_colors["checks"].items() if not passed
            ]
            raise RuntimeError(f"三维对象颜色可见性检查失败：{failed}")

        with Image.open(raw_png) as opened:
            image = opened.convert("RGB")
        overlay_scene = dict(scene)
        if focus_witness:
            overlay_scene["visible_banner"] = (
                f"{scene['visible_banner']} | 见证聚焦"
            )
        overlay = overlay_status_banner(image, overlay_scene)
        image.save(final_png, format="PNG", optimize=True)
        final_pixels = inspect_pixels(
            final_png,
            args.width,
            args.height,
            DEFAULT_NEAR_WHITE,
            0.003,
            DEFAULT_MAX_NON_WHITE_RATIO,
            DEFAULT_MIN_BORDER_WHITE_RATIO,
            max(24, round(min(args.width, args.height) * 0.015)),
            witness_expected,
        )
        if not final_pixels["passed"]:
            failed = [
                name for name, passed in final_pixels["checks"].items() if not passed
            ]
            raise RuntimeError(f"最终叠标 PNG 像素审计失败：{failed}")
        if final_pixels["sha256"] == base_audit["pixels"]["sha256"]:
            raise RuntimeError("状态标识没有改变 PNG")

        atomic_copy(final_png, output)
        if sha256(output) != final_pixels["sha256"]:
            raise RuntimeError("最终 PNG 原子复制后哈希不一致")
        audit.update(
            {
                "status": "passed",
                "parameters": {
                    "width": args.width,
                    "height": args.height,
                    "view": args.view,
                    "zoom": args.zoom,
                    "focus_witness": focus_witness,
                    "hidden_styles": list(base_args.hidden_styles),
                    "electrode_transparency": render_electrode_transparency,
                    "electrode_wireframe": focus_witness,
                    "raw_min_non_white_ratio": raw_min_non_white_ratio,
                    "final_pixel_audit_after_overlay": True,
                },
                "freecad": {
                    "executable": base_audit["freecad_executable"],
                    "executable_sha256": base_audit["freecad_executable_sha256"],
                    "process": base_audit["process"],
                    "log": base_audit["freecad_log"],
                    "macro": base_audit["macro"],
                },
                "raw_render": {
                    "sha256": base_audit["pixels"]["sha256"],
                    "pixels": base_audit["pixels"],
                },
                "scene_color_audit": scene_colors,
                "overlay": overlay,
                "pixels": final_pixels,
                "temporary_directory_cleanup": "pending",
            }
        )
    except BaseException as error:
        audit.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        audit["temporary_directory_cleanup"] = "ok"
        atomic_write_json(audit_path, audit)
    return audit


def main() -> int:
    args = parse_args()
    result = render_q4(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "publication_status": result["publication_status"],
                "output": result["output"],
                "audit": result["audit"],
                "png_sha256": result["pixels"]["sha256"],
                "dimensions": result["pixels"]["dimensions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
