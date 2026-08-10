#!/usr/bin/env python3
"""Build a hash-verified LaTeX source appendix from the selected MCM code."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
SELECTION_PATH = PROJECT_DIR / "appendix-selection.json"
SOURCE_ROOT = PROJECT_DIR / "source_code"
OUTPUT_DIR = PROJECT_DIR / "source_appendix"
TEX_PATH = PROJECT_DIR / "source_appendix.tex"
REPORT_PATH = PROJECT_DIR / "source-appendix-build.json"
GENERATED_NAME_PATTERN = re.compile(r"^[0-9]{3}\.[A-Za-z0-9]+$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def project_source(raw_path: Any) -> tuple[Path, str]:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError(f"源码路径必须采用 POSIX 相对路径：{raw_path!r}")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"源码路径越出 source_code：{raw_path!r}")
    resolved_root = SOURCE_ROOT.resolve()
    source = (SOURCE_ROOT / Path(*relative.parts)).resolve()
    if source.parent != resolved_root and resolved_root not in source.parents:
        raise ValueError(f"源码路径越出 source_code：{raw_path!r}")
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"源码文件不存在或属于符号链接：{source}")
    return source, relative.as_posix()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def load_selection() -> dict[str, Any]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "1.0":
        raise ValueError("appendix-selection.json schema_version 必须为 1.0")
    files = selection.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("附录源码清单不能为空")
    return selection


def annotate_source(raw: bytes, item: dict[str, Any], line_count: int) -> tuple[bytes, list[dict[str, Any]]]:
    """Insert sparse Chinese comments before selected original source lines."""
    annotations = item.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError(f"annotations 必须为数组：{item.get('path')}")

    by_line: dict[int, str] = {}
    normalized: list[dict[str, Any]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"单行注释必须为对象：{item.get('path')}")
        before_line = annotation.get("before_line")
        comment = annotation.get("comment")
        if not isinstance(before_line, int) or not 1 <= before_line <= line_count:
            raise ValueError(f"注释行号越界：{item.get('path')}:{before_line}")
        if before_line in by_line:
            raise ValueError(f"同一源码行只能配置一条注释：{item.get('path')}:{before_line}")
        if (
            not isinstance(comment, str)
            or not comment.strip()
            or comment != comment.strip()
            or "\n" in comment
            or "\r" in comment
        ):
            raise ValueError(f"注释必须为非空单行文本：{item.get('path')}:{before_line}")
        by_line[before_line] = comment
        normalized.append({"before_line": before_line, "comment": comment})

    source_lines = raw.splitlines(keepends=True)
    chunks: list[bytes] = []
    for line_number, source_line in enumerate(source_lines, start=1):
        comment = by_line.get(line_number)
        if comment is not None:
            chunks.append(f"# 关键：{comment}\n".encode("utf-8"))
        chunks.append(source_line)
    return b"".join(chunks), normalized


def build() -> dict[str, Any]:
    selection = load_selection()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_outputs: set[str] = set()
    tex_lines = [
        r"\subsection{附录 C：核心程序源码}",
        r"\noindent 本附录排印用户确认的四个正式求解器和七个核心依赖模块。",
        r"源码基于 GitHub 提交 \texttt{" + latex_escape(selection["commit"][:7]) + r"}，",
        r"并对项目根目录发现方式作可移植适配；完整提交号与哈希见 \path{appendix-selection.json}。",
        r"编号副本由源码附录构建脚本按清单自动生成，",
        r"原始文件逐项核验 SHA-256；排印副本仅在关键逻辑前加入单行中文注释。\par",
        r"\vspace{0.5em}",
    ]
    report_files: list[dict[str, Any]] = []
    total_lines = 0

    for index, item in enumerate(selection["files"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个附录条目必须为对象")
        source, relative_path = project_source(item.get("path"))
        raw = source.read_bytes()
        digest = sha256_bytes(raw)
        expected_digest = str(item.get("sha256", "")).lower()
        if digest != expected_digest:
            raise ValueError(
                f"源码 SHA-256 不一致：{relative_path}，"
                f"expected={expected_digest} actual={digest}"
            )
        line_count = raw.count(b"\n")
        if raw and not raw.endswith(b"\n"):
            line_count += 1
        if line_count != int(item.get("lines", -1)):
            raise ValueError(
                f"源码行数不一致：{relative_path}，"
                f"expected={item.get('lines')} actual={line_count}"
            )

        output_name = f"{index:03d}{source.suffix.lower()}"
        expected_outputs.add(output_name)
        output_path = OUTPUT_DIR / output_name
        annotated_raw, annotations = annotate_source(raw, item, line_count)
        atomic_write(output_path, annotated_raw)
        copied_digest = sha256_bytes(output_path.read_bytes())
        if copied_digest != sha256_bytes(annotated_raw):
            raise RuntimeError(f"带注释副本写入后哈希不一致：{output_path}")
        annotated_line_count = annotated_raw.count(b"\n")
        if annotated_raw and not annotated_raw.endswith(b"\n"):
            annotated_line_count += 1

        appendix_id = latex_escape(str(item["appendix_id"]))
        title = latex_escape(str(item["title"]))
        tex_lines.extend(
            [
                r"\Needspace{8\baselineskip}",
                rf"\subsubsection{{附录 {appendix_id}：{title}}}",
                r"\noindent 原项目相对路径：\texttt{\detokenize{"
                + relative_path
                + r"}}\par",
                r"\vspace{0.25em}",
                r"{\fontsize{6.4pt}{7.2pt}\selectfont",
                r"\VerbatimInput[breaklines=true,breakanywhere=true,numbers=left,"
                r"numbersep=5pt,frame=single,framesep=2mm,rulecolor=\color{black!35}]{"
                + f"source_appendix/{output_name}"
                + r"}",
                r"}",
            ]
        )
        report_files.append(
            {
                "appendix_id": item["appendix_id"],
                "source_path": relative_path,
                "output_path": f"source_appendix/{output_name}",
                "upstream_sha256": item.get("upstream_sha256", digest),
                "source_sha256": digest,
                "source_lines": line_count,
                "source_verified": True,
                "path_adaptation": item.get("path_adaptation"),
                "appendix_sha256": copied_digest,
                "appendix_lines": annotated_line_count,
                "annotation_count": len(annotations),
                "annotations": annotations,
                "byte_identical": annotated_raw == raw,
            }
        )
        total_lines += line_count

    for stale in OUTPUT_DIR.iterdir():
        if (
            stale.is_file()
            and GENERATED_NAME_PATTERN.fullmatch(stale.name)
            and stale.name not in expected_outputs
        ):
            stale.unlink()

    atomic_write(TEX_PATH, ("\n".join(tex_lines) + "\n").encode("utf-8"))
    report = {
        "schema_version": "1.0",
        "status": "ready",
        "repository": selection["repository"],
        "commit": selection["commit"],
        "path_policy": "MCM_PROJECT_ROOT override, then marker-directory discovery",
        "selection_sha256": sha256_bytes(SELECTION_PATH.read_bytes()),
        "file_count": len(report_files),
        "source_total_lines": total_lines,
        "annotation_count": sum(item["annotation_count"] for item in report_files),
        "appendix_total_lines": sum(item["appendix_lines"] for item in report_files),
        "tex_path": TEX_PATH.name,
        "tex_sha256": sha256_bytes(TEX_PATH.read_bytes()),
        "files": report_files,
    }
    atomic_write(
        REPORT_PATH,
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return report


def main() -> int:
    report = build()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
