# AI 工具：OpenAI Codex；模型/版本：GPT-5 系列；开发机构：OpenAI。
# 版本发布日期：2025-08-07（GPT-5 系列公开快照日期）；本程序由参赛队逐行复核并对结果负责。
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_WIDTH = 2000
DEFAULT_HEIGHT = 1500
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_CAPTURE_DELAY_MS = 1200
DEFAULT_LINE_WIDTH = 2.0
DEFAULT_NEAR_WHITE = 250
DEFAULT_MIN_NON_WHITE_RATIO = 0.001
DEFAULT_MAX_NON_WHITE_RATIO = 0.95
DEFAULT_MIN_BORDER_WHITE_RATIO = 0.98


class RenderFailure(RuntimeError):
    def __init__(self, message: str, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an FCStd scene through a hidden full FreeCAD GUI process."
    )
    parser.add_argument("source", type=Path, help="Input FCStd file")
    parser.add_argument("output", type=Path, help="Output PNG file")
    parser.add_argument(
        "--audit",
        type=Path,
        help="Pixel-audit JSON; defaults to OUTPUT with suffix .audit.json",
    )
    parser.add_argument("--freecad-exe", type=Path, help="Path to full freecad.exe")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--capture-delay-ms", type=int, default=DEFAULT_CAPTURE_DELAY_MS)
    parser.add_argument("--line-width", type=float, default=DEFAULT_LINE_WIDTH)
    parser.add_argument("--electrode-transparency", type=int, default=8)
    parser.add_argument("--electrode-wireframe", action="store_true")
    parser.add_argument(
        "--view",
        choices=("axonometric", "top", "bottom", "front", "rear", "left", "right"),
        default="axonometric",
    )
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument(
        "--hide-style",
        dest="hidden_styles",
        action="append",
        choices=("background_a", "background_b", "witness", "candidate"),
        default=[],
    )
    parser.add_argument("--near-white", type=int, default=DEFAULT_NEAR_WHITE)
    parser.add_argument(
        "--min-non-white-ratio",
        type=float,
        default=DEFAULT_MIN_NON_WHITE_RATIO,
    )
    parser.add_argument(
        "--max-non-white-ratio",
        type=float,
        default=DEFAULT_MAX_NON_WHITE_RATIO,
    )
    parser.add_argument(
        "--min-border-white-ratio",
        type=float,
        default=DEFAULT_MIN_BORDER_WHITE_RATIO,
    )
    parser.add_argument(
        "--min-margin-pixels",
        type=int,
        help="Minimum white margin around the non-white bounding box",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def discover_freecad(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    configured = os.environ.get("FREECAD_EXE")
    if configured:
        candidates.append(Path(configured))
    located = shutil.which("freecad.exe") or shutil.which("FreeCAD.exe")
    if located:
        candidates.append(Path(located))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.extend(
            [
                Path(local_app_data) / "Programs" / "FreeCAD 1.1" / "bin" / "freecad.exe",
                Path(local_app_data) / "Programs" / "FreeCAD 1.0" / "bin" / "freecad.exe",
            ]
        )
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.extend(
            [
                Path(program_files) / "FreeCAD 1.1" / "bin" / "freecad.exe",
                Path(program_files) / "FreeCAD 1.0" / "bin" / "freecad.exe",
            ]
        )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    rendered = "\n".join(f"- {candidate}" for candidate in candidates) or "- none"
    raise FileNotFoundError(f"full freecad.exe not found; checked:\n{rendered}")


def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def create_ascii_temp_dir() -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("FREECAD_RENDER_TEMP")
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path(tempfile.gettempdir()))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Temp")
    public = os.environ.get("PUBLIC")
    if public:
        candidates.append(Path(public) / "Documents" / "FreeCADRenderTmp")
    system_drive = os.environ.get("SystemDrive", "C:")
    candidates.append(Path(system_drive + os.sep) / "FreeCADRenderTmp")

    errors: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        key = str(root).casefold()
        if key in seen or not is_ascii_path(root):
            continue
        seen.add(key)
        try:
            root.mkdir(parents=True, exist_ok=True)
            created = Path(tempfile.mkdtemp(prefix="freecad_render_", dir=root))
            if not is_ascii_path(created):
                shutil.rmtree(created)
                continue
            return created
        except OSError as error:
            errors.append(f"{root}: {error}")
    details = "\n".join(errors) or "no writable ASCII candidate"
    raise RuntimeError(f"could not create an ASCII-only temporary directory:\n{details}")


def make_macro(
    macro_path: Path,
    source: Path,
    output: Path,
    status: Path,
    width: int,
    height: int,
    delay_ms: int,
    line_width: float,
    view_name: str,
    zoom: float,
    hidden_styles: tuple[str, ...],
    electrode_transparency: int,
    electrode_wireframe: bool,
) -> None:
    config = json.dumps(
        {
            "source": str(source),
            "output": str(output),
            "status": str(status),
            "width": width,
            "height": height,
            "delay_ms": delay_ms,
            "line_width": line_width,
            "view_name": view_name,
            "zoom": zoom,
            "hidden_styles": list(hidden_styles),
            "electrode_transparency": electrode_transparency,
            "electrode_wireframe": electrode_wireframe,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    macro = textwrap.dedent(
        f"""
        from __future__ import annotations

        import hashlib
        import json
        import traceback
        from pathlib import Path

        import FreeCAD as App
        import FreeCADGui as Gui
        from PySide import QtCore
        from pivy import coin


        CONFIG = json.loads({config!r})
        SOURCE = Path(CONFIG["source"])
        OUTPUT = Path(CONFIG["output"])
        STATUS = Path(CONFIG["status"])
        WIDTH = int(CONFIG["width"])
        HEIGHT = int(CONFIG["height"])
        DELAY_MS = int(CONFIG["delay_ms"])
        LINE_WIDTH = float(CONFIG["line_width"])
        VIEW_NAME = str(CONFIG["view_name"])
        ZOOM = float(CONFIG["zoom"])
        HIDDEN_STYLES = set(str(value) for value in CONFIG["hidden_styles"])
        ELECTRODE_TRANSPARENCY = int(CONFIG["electrode_transparency"])
        ELECTRODE_WIREFRAME = bool(CONFIG["electrode_wireframe"])

        ROLE_STYLES = {{
            "background_a": ((0.72, 0.75, 0.78), 62),
            "background_b": ((0.20, 0.55, 0.82), 25),
            "witness": ((0.93, 0.36, 0.12), 0),
            "boundary_fragment": ((0.84, 0.20, 0.26), 5),
            "candidate": ((0.18, 0.64, 0.45), 10),
        }}

        state = {{
            "status": "scheduled",
            "source": str(SOURCE),
            "output": str(OUTPUT),
            "requested_dimensions": [WIDTH, HEIGHT],
        }}
        document = None
        view = None
        main_window = None


        def file_hash(path):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest().upper()


        def write_status():
            STATUS.write_text(
                json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True),
                encoding="ascii",
            )


        def shutdown():
            global document
            try:
                if document is not None:
                    App.closeDocument(document.Name)
                    document = None
            except Exception:
                state["close_document_error"] = traceback.format_exc()
                write_status()
            try:
                if main_window is not None:
                    main_window.hide()
                    main_window.close()
            finally:
                application = QtCore.QCoreApplication.instance()
                if application is not None:
                    application.quit()


        def fail(phase):
            state["status"] = "error"
            state["phase"] = phase
            state["traceback"] = traceback.format_exc()
            write_status()
            QtCore.QTimer.singleShot(0, shutdown)


        def apply_view_style(obj):
            name = obj.Name.casefold()
            style_name = "preserved"
            color = None
            transparency = None
            if name == "domain":
                style_name = "domain"
                color = (0.78, 0.82, 0.86)
                transparency = 94
                if (
                    hasattr(obj.ViewObject, "listDisplayModes")
                    and "Wireframe" in obj.ViewObject.listDisplayModes()
                ):
                    obj.ViewObject.DisplayMode = "Wireframe"
            elif "electrode" in name:
                style_name = "electrode"
                color = (0.16, 0.18, 0.20)
                transparency = ELECTRODE_TRANSPARENCY
                if (
                    ELECTRODE_WIREFRAME
                    and hasattr(obj.ViewObject, "listDisplayModes")
                    and "Wireframe" in obj.ViewObject.listDisplayModes()
                ):
                    obj.ViewObject.DisplayMode = "Wireframe"
            else:
                for role, style in ROLE_STYLES.items():
                    if role in name:
                        style_name = role
                        color, transparency = style
                        break
            if color is not None:
                obj.ViewObject.ShapeColor = color
                obj.ViewObject.LineColor = tuple(
                    max(0.0, component * 0.72) for component in color
                )
                obj.ViewObject.Transparency = transparency
            if hasattr(obj.ViewObject, "LineWidth"):
                obj.ViewObject.LineWidth = max(
                    float(obj.ViewObject.LineWidth), LINE_WIDTH
                )
            return style_name


        def capture():
            try:
                state["phase"] = "capture"
                view.redraw()
                application = QtCore.QCoreApplication.instance()
                if application is not None:
                    application.processEvents()
                saved = view.saveImage(str(OUTPUT), WIDTH, HEIGHT, "White")
                if saved is False:
                    raise RuntimeError("activeView.saveImage returned False")
                if not OUTPUT.is_file() or OUTPUT.stat().st_size == 0:
                    raise RuntimeError("saveImage did not create a non-empty PNG")
                state.update(
                    {{
                        "status": "ok",
                        "phase": "complete",
                        "freecad_version": ".".join(App.Version()[:3]),
                        "source_sha256": file_hash(SOURCE),
                        "output_size_bytes": OUTPUT.stat().st_size,
                        "output_sha256": file_hash(OUTPUT),
                    }}
                )
                write_status()
                QtCore.QTimer.singleShot(0, shutdown)
            except Exception:
                fail("capture")


        def prepare():
            global document, view
            try:
                state["phase"] = "open_document"
                if not SOURCE.is_file():
                    raise FileNotFoundError(SOURCE)
                document = App.openDocument(str(SOURCE))
                App.setActiveDocument(document.Name)
                objects = []
                for obj in document.Objects:
                    shape = getattr(obj, "Shape", None)
                    style_name = "not_shape"
                    if shape is not None and not shape.isNull():
                        obj.ViewObject.Visibility = True
                        style_name = apply_view_style(obj)
                        if style_name in HIDDEN_STYLES:
                            obj.ViewObject.Visibility = False
                    objects.append(
                        {{
                            "name": obj.Name,
                            "label": obj.Label,
                            "type": obj.TypeId,
                            "visible": bool(obj.ViewObject.Visibility),
                            "style": style_name,
                            "shape_color": list(obj.ViewObject.ShapeColor),
                            "line_color": list(obj.ViewObject.LineColor),
                            "transparency": int(obj.ViewObject.Transparency),
                            "display_mode": str(obj.ViewObject.DisplayMode),
                        }}
                    )
                document.recompute()
                gui_document = Gui.activeDocument()
                if gui_document is None:
                    raise RuntimeError("Gui.activeDocument returned None")
                view = gui_document.activeView()
                if view is None:
                    raise RuntimeError("active GUI document has no active view")
                view_methods = {{
                    "axonometric": view.viewAxonometric,
                    "top": view.viewTop,
                    "bottom": view.viewBottom,
                    "front": view.viewFront,
                    "rear": view.viewRear,
                    "left": view.viewLeft,
                    "right": view.viewRight,
                }}
                view_methods[VIEW_NAME]()
                view.fitAll()
                camera = view.getCameraNode()
                camera_zoom = {{"factor": ZOOM, "field": None}}
                for field_name in ("height", "heightAngle"):
                    if hasattr(camera, field_name):
                        field = getattr(camera, field_name)
                        before = float(field.getValue())
                        field.setValue(before / ZOOM)
                        camera_zoom.update(
                            {{"field": field_name, "before": before, "after": before / ZOOM}}
                        )
                        break
                view.redraw()
                state["objects"] = objects
                state["camera"] = {{"view": VIEW_NAME, "zoom": camera_zoom}}
                state["visible_object_count"] = sum(
                    1 for item in objects if item["visible"]
                )
                write_status()
                QtCore.QTimer.singleShot(DELAY_MS, capture)
            except Exception:
                fail("prepare")


        try:
            write_status()
            main_window = Gui.getMainWindow()
            main_window.hide()
            QtCore.QTimer.singleShot(0, prepare)
        except Exception:
            fail("startup")
        """
    ).lstrip()
    macro_path.write_text(macro, encoding="ascii")


def hidden_startup_info() -> subprocess.STARTUPINFO:
    if os.name != "nt":
        raise OSError("hidden full-GUI FreeCAD rendering is implemented for Windows")
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return startup


def kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        startupinfo=hidden_startup_info(),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_freecad(
    executable: Path,
    macro: Path,
    log: Path,
    user_config: Path,
    working_directory: Path,
    timeout: float,
) -> dict[str, Any]:
    command = [
        str(executable),
        "--log-file",
        str(log),
        "--user-cfg",
        str(user_config),
        str(macro),
    ]
    non_ascii = [argument for argument in command[1:] if not argument.isascii()]
    if non_ascii:
        raise ValueError(f"non-ASCII argument would be passed to FreeCAD: {non_ascii}")

    environment = os.environ.copy()
    environment.update(
        {
            "QT_OPENGL": "software",
            "QT_QUICK_BACKEND": "software",
            "LIBGL_ALWAYS_SOFTWARE": "1",
        }
    )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        startupinfo=hidden_startup_info(),
        creationflags=creation_flags,
    )
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process_tree(process)
        return_code = process.returncode
    finally:
        if process.poll() is None:
            kill_process_tree(process)
    elapsed = time.perf_counter() - started
    if timed_out:
        raise TimeoutError(
            f"FreeCAD PID {process.pid} exceeded the {timeout:.1f}-second timeout"
        )
    return {
        "pid": process.pid,
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "process_exited": process.poll() is not None,
        "command_arguments_ascii": True,
        "window_mode": "Windows SW_HIDE + CREATE_NO_WINDOW; macro hides Qt main window",
    }


def read_log_evidence(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"log_created": False}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    selectors = (
        "OpenGL version is:",
        "Processing file:",
        "Hide main window",
        "Finish: Event loop left",
        "FreeCAD terminating",
    )
    evidence = [line.strip() for line in lines if any(item in line for item in selectors)]
    errors = [
        line.strip()
        for line in lines
        if "Exception while processing file" in line or line.startswith("Err:")
    ]
    return {
        "log_created": True,
        "evidence": evidence,
        "errors": errors,
    }


def inspect_pixels(
    path: Path,
    width: int,
    height: int,
    near_white_threshold: int,
    min_non_white_ratio: float,
    max_non_white_ratio: float,
    min_border_white_ratio: float,
    min_margin_pixels: int,
    witness_expected: bool,
) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        image_format = image.format
        rgb = np.asarray(image.convert("RGB"))
    actual_height, actual_width = rgb.shape[:2]
    near_white = np.all(rgb >= near_white_threshold, axis=2)
    non_white = ~near_white
    ys, xs = np.nonzero(non_white)
    bbox = None
    margins = None
    if xs.size:
        bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
        margins = [
            bbox[0],
            bbox[1],
            actual_width - bbox[2],
            actual_height - bbox[3],
        ]
    border = np.concatenate(
        [
            near_white[0, :],
            near_white[-1, :],
            near_white[1:-1, 0],
            near_white[1:-1, -1],
        ]
    )
    pixel_count = int(actual_width * actual_height)
    non_white_count = int(non_white.sum())
    near_white_count = int(near_white.sum())
    non_white_ratio = non_white_count / pixel_count
    near_white_ratio = near_white_count / pixel_count
    border_white_ratio = float(border.mean())
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    witness_orange = (
        (red >= 180)
        & (green >= 45)
        & (green <= 160)
        & (blue <= 110)
        & (red >= green + 60)
        & (green >= blue + 20)
    )
    witness_orange_count = int(witness_orange.sum())
    checks = {
        "format_png": image_format == "PNG",
        "dimensions": [actual_width, actual_height] == [width, height],
        "file_nonempty": path.stat().st_size > 0,
        "non_white_ratio_min": non_white_ratio >= min_non_white_ratio,
        "non_white_ratio_max": non_white_ratio <= max_non_white_ratio,
        "non_white_bbox_present": bbox is not None,
        "border_near_white_ratio": border_white_ratio >= min_border_white_ratio,
        "bbox_margin": margins is not None and min(margins) >= min_margin_pixels,
        "witness_orange_visible": not witness_expected or witness_orange_count >= 50,
    }
    return {
        "format": image_format,
        "dimensions": [actual_width, actual_height],
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "pixel_count": pixel_count,
        "near_white_threshold": near_white_threshold,
        "near_white_pixel_count": near_white_count,
        "near_white_ratio": near_white_ratio,
        "non_white_pixel_count": non_white_count,
        "non_white_ratio": non_white_ratio,
        "non_white_bbox_xyxy_exclusive": bbox,
        "bbox_margins_ltrb": margins,
        "border_near_white_ratio": border_white_ratio,
        "witness_expected": witness_expected,
        "witness_orange_pixel_count": witness_orange_count,
        "witness_orange_ratio": witness_orange_count / pixel_count,
        "rgb_channel_min": [int(value) for value in rgb.min(axis=(0, 1))],
        "rgb_channel_max": [int(value) for value in rgb.max(axis=(0, 1))],
        "checks": checks,
        "passed": all(checks.values()),
    }


def render(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audit_path = (
        args.audit.expanduser().resolve()
        if args.audit is not None
        else output.with_suffix(".audit.json")
    )
    freecad = discover_freecad(args.freecad_exe)
    hidden_styles = tuple(getattr(args, "hidden_styles", ()) or ())
    allowed_hidden_styles = {"background_a", "background_b", "witness", "candidate"}
    if len(set(hidden_styles)) != len(hidden_styles) or not set(hidden_styles).issubset(
        allowed_hidden_styles
    ):
        raise ValueError(f"invalid or duplicate hidden styles: {hidden_styles}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.casefold() != ".fcstd":
        raise ValueError(f"input must be an FCStd file: {source}")
    if output.suffix.casefold() != ".png":
        raise ValueError(f"output must be a PNG file: {output}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    electrode_transparency = int(getattr(args, "electrode_transparency", 8))
    electrode_wireframe = bool(getattr(args, "electrode_wireframe", False))
    if not 0 <= electrode_transparency <= 100:
        raise ValueError("electrode transparency must be between 0 and 100")
    if (
        args.timeout <= 0
        or args.capture_delay_ms < 0
        or args.line_width <= 0
        or args.zoom <= 0
    ):
        raise ValueError(
            "timeout, line width, and zoom must be positive; delay must be nonnegative"
        )
    if not 0 <= args.near_white <= 255:
        raise ValueError("near-white must be between 0 and 255")
    if not (
        0 <= args.min_non_white_ratio < args.max_non_white_ratio <= 1
        and 0 <= args.min_border_white_ratio <= 1
    ):
        raise ValueError("pixel-ratio thresholds are inconsistent")
    min_margin = (
        args.min_margin_pixels
        if args.min_margin_pixels is not None
        else max(8, round(min(args.width, args.height) * 0.01))
    )
    if min_margin < 0:
        raise ValueError("minimum margin must be nonnegative")

    audit: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "audit": str(audit_path),
        "renderer": str(Path(__file__).resolve()),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "freecad_executable": str(freecad),
        "freecad_executable_sha256": sha256(freecad),
        "parameters": {
            "width": args.width,
            "height": args.height,
            "timeout_seconds": args.timeout,
            "capture_delay_ms": args.capture_delay_ms,
            "line_width": args.line_width,
            "background": "White",
            "camera": f"{args.view} + fitAll",
            "zoom": args.zoom,
            "hidden_styles": list(hidden_styles),
            "electrode_transparency": electrode_transparency,
            "electrode_wireframe": electrode_wireframe,
            "style_profile": "build_freecad_scene ROLE_STYLE with wireframe domain",
            "near_white_threshold": args.near_white,
            "min_non_white_ratio": args.min_non_white_ratio,
            "max_non_white_ratio": args.max_non_white_ratio,
            "min_border_white_ratio": args.min_border_white_ratio,
            "min_margin_pixels": min_margin,
        },
    }

    temp_directory = create_ascii_temp_dir()
    failure: BaseException | None = None
    try:
        temporary_source = temp_directory / "scene.FCStd"
        temporary_output = temp_directory / "rendered.png"
        temporary_status = temp_directory / "render_status.json"
        macro = temp_directory / "render_macro.py"
        log = temp_directory / "freecad.log"
        user_config = temp_directory / "user.cfg"
        shutil.copyfile(source, temporary_source)
        if sha256(temporary_source) != audit["source_sha256"]:
            raise RuntimeError("temporary FCStd copy hash mismatch")
        make_macro(
            macro,
            temporary_source,
            temporary_output,
            temporary_status,
            args.width,
            args.height,
            args.capture_delay_ms,
            args.line_width,
            args.view,
            args.zoom,
            hidden_styles,
            electrode_transparency,
            electrode_wireframe,
        )
        audit["process"] = run_freecad(
            freecad,
            macro,
            log,
            user_config,
            temp_directory,
            args.timeout,
        )
        audit["freecad_log"] = read_log_evidence(log)
        if audit["process"]["return_code"] != 0:
            raise RuntimeError(
                f"FreeCAD exited with code {audit['process']['return_code']}"
            )
        if not temporary_status.is_file():
            raise RuntimeError("FreeCAD macro did not write render_status.json")
        macro_status = json.loads(temporary_status.read_text(encoding="ascii"))
        audit["macro"] = macro_status
        if macro_status.get("status") != "ok":
            raise RuntimeError(
                f"FreeCAD macro failed in phase {macro_status.get('phase')}: "
                f"{macro_status.get('traceback', 'no traceback')}"
            )
        if not temporary_output.is_file():
            raise RuntimeError("FreeCAD macro reported success without a PNG")
        pixels = inspect_pixels(
            temporary_output,
            args.width,
            args.height,
            args.near_white,
            args.min_non_white_ratio,
            args.max_non_white_ratio,
            args.min_border_white_ratio,
            min_margin,
            any(
                item.get("visible") and item.get("style") == "witness"
                for item in macro_status.get("objects", [])
            ),
        )
        audit["pixels"] = pixels
        atomic_copy(temporary_output, output)
        if sha256(output) != pixels["sha256"]:
            raise RuntimeError("project PNG hash differs from audited temporary PNG")
        if not pixels["passed"]:
            failed_checks = [name for name, passed in pixels["checks"].items() if not passed]
            raise RuntimeError(f"pixel audit failed: {failed_checks}")
    except BaseException as error:
        failure = error
        audit["status"] = "failed"
        audit["error_type"] = type(error).__name__
        audit["error"] = str(error)
        audit["traceback"] = traceback.format_exc()
    finally:
        try:
            shutil.rmtree(temp_directory)
            audit["temporary_directory_cleanup"] = "ok"
        except OSError as cleanup_error:
            audit["temporary_directory_cleanup"] = "failed"
            audit["temporary_directory_cleanup_error"] = str(cleanup_error)
            if failure is None:
                failure = cleanup_error
                audit["status"] = "failed"
                audit["error_type"] = type(cleanup_error).__name__
                audit["error"] = str(cleanup_error)

    if failure is not None:
        atomic_write_json(audit_path, audit)
        raise RenderFailure(str(failure), audit)
    audit["status"] = "passed"
    atomic_write_json(audit_path, audit)
    return audit


def main() -> int:
    args = parse_args()
    audit_path = (
        args.audit.expanduser().resolve()
        if args.audit is not None
        else args.output.expanduser().resolve().with_suffix(".audit.json")
    )
    try:
        result = render(args)
    except RenderFailure as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "audit": str(audit_path),
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": result["output"],
                "audit": result["audit"],
                "png_sha256": result["pixels"]["sha256"],
                "non_white_ratio": result["pixels"]["non_white_ratio"],
                "bbox": result["pixels"]["non_white_bbox_xyxy_exclusive"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
