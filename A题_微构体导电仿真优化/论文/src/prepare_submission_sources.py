#!/usr/bin/env python3
"""Create audited submission-only source copies without changing executable sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from build_docx import (
    MAX_SOURCE_FILE_BYTES,
    SOURCE_LANGUAGE_BY_SUFFIX,
    read_utf8_text,
    reject_private_home_paths,
    sha256_file,
    validate_ai_header_lines,
)


Q1_SOURCE = "问题/问题1/src/solve.py"
Q1_REPLACEMENT_COUNT = 0


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def project_file(raw_path: str, project_root: Path, role: str) -> tuple[Path, str]:
    path = (project_root / raw_path).resolve()
    try:
        relative = path.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{role} must stay inside the project root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, relative


def load_allowlist(path: Path, project_root: Path) -> dict[str, Any]:
    data = json.loads(read_utf8_text(path, "Submission source allowlist"))
    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise ValueError("Submission source allowlist must use schema_version 1.0")
    if data.get("status") != "frozen":
        raise ValueError("Submission source allowlist status must be frozen")
    validate_ai_header_lines(data.get("ai_header_lines"))
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Submission source allowlist requires a non-empty files list")

    seen: set[str] = set()
    q1_seen = False
    for index, item in enumerate(files, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Allowlist item {index} must be an object")
        unknown = sorted(
            set(item)
            - {
                "path",
                "title",
                "language",
                "replacements",
                "forbidden_terms",
                "include_in_appendix",
            }
        )
        if unknown:
            raise ValueError(f"Unsupported keys in allowlist item {index}: {', '.join(unknown)}")
        source, relative = project_file(str(item.get("path") or ""), project_root, f"Allowlist item {index}")
        if relative in seen:
            raise ValueError(f"Duplicate allowlist source: {relative}")
        seen.add(relative)
        if source.stat().st_size > MAX_SOURCE_FILE_BYTES:
            raise ValueError(f"Allowlist source exceeds {MAX_SOURCE_FILE_BYTES} bytes: {relative}")
        if source.suffix.lower() not in SOURCE_LANGUAGE_BY_SUFFIX:
            raise ValueError(f"Unsupported allowlist source extension: {relative}")
        include_in_appendix = item.get("include_in_appendix", True)
        if not isinstance(include_in_appendix, bool):
            raise ValueError(f"include_in_appendix must be boolean: {relative}")
        replacements = item.get("replacements", [])
        if not isinstance(replacements, list):
            raise ValueError(f"Allowlist replacements must be a list: {relative}")
        if relative == Q1_SOURCE:
            q1_seen = True
            if len(replacements) != Q1_REPLACEMENT_COUNT:
                raise ValueError(
                    f"Q1 submission copy requires exactly {Q1_REPLACEMENT_COUNT} replacements"
                )
        elif replacements:
            raise ValueError(f"Only {Q1_SOURCE} may define content replacements")
        for replacement in replacements:
            if not isinstance(replacement, dict) or set(replacement) != {"old", "new"}:
                raise ValueError(f"Each replacement must contain only old/new: {relative}")
            if not str(replacement["old"]) or replacement["old"] == replacement["new"]:
                raise ValueError(f"Invalid replacement in {relative}")
    if Q1_SOURCE in seen and not q1_seen:
        raise ValueError("Q1 allowlist entry is missing")
    return data


def prepare_submission_sources(allowlist_path: Path, output_dir: Path, project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    allowlist_path = allowlist_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.parent != project_root or output_dir.name != "提交源码":
        raise ValueError("Submission source output must be the project-root 提交源码 directory")
    if output_dir.exists():
        raise FileExistsError(f"Submission source output already exists: {output_dir}")
    allowlist = load_allowlist(allowlist_path, project_root)
    header_lines = validate_ai_header_lines(allowlist["ai_header_lines"])
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")

    manifest_files: list[dict[str, Any]] = []
    try:
        staging.mkdir(parents=False)
        for item in allowlist["files"]:
            source, relative = project_file(str(item["path"]), project_root, "Allowlist source")
            raw = source.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                raise ValueError(f"Submission sources must not contain a UTF-8 BOM: {relative}")
            source_text = read_utf8_text(source, "Allowlist source")
            reject_private_home_paths(source_text, "Allowlist source")
            newline = "\r\n" if b"\r\n" in raw else "\n"
            transformed = source_text
            replacement_audit: list[dict[str, Any]] = []
            for replacement in item.get("replacements", []):
                old = str(replacement["old"])
                new = str(replacement["new"])
                count = transformed.count(old)
                if count != 1:
                    raise ValueError(f"Controlled replacement must match exactly once in {relative}: {old!r}")
                transformed = transformed.replace(old, new, 1)
                replacement_audit.append(
                    {
                        "old_sha256": text_sha256(old),
                        "new_sha256": text_sha256(new),
                        "count": count,
                    }
                )
            forbidden = [str(term) for term in item.get("forbidden_terms", []) if str(term)]
            leaked = [term for term in forbidden if term in transformed]
            if leaked:
                raise ValueError(f"Submission source still contains forbidden Q1 report terms: {leaked}")
            submission_text = newline.join(header_lines) + newline + transformed
            reject_private_home_paths(submission_text, "Submission source copy")

            staged_path = staging / Path(relative)
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_bytes(submission_text.encode("utf-8"))
            final_relative = (Path("提交源码") / Path(relative)).as_posix()
            language = str(item.get("language") or SOURCE_LANGUAGE_BY_SUFFIX[source.suffix.lower()])
            manifest_files.append(
                {
                    "title": str(item.get("title") or relative),
                    "source_path": relative,
                    "source_sha256": sha256_file(source),
                    "source_bytes": source.stat().st_size,
                    "submission_path": final_relative,
                    "submission_sha256": sha256_file(staged_path),
                    "submission_bytes": staged_path.stat().st_size,
                    "language": language,
                    "replacement_audit": replacement_audit,
                }
            )

        manifest = {
            "schema_version": "1.0",
            "status": "frozen",
            "submission_root": "提交源码",
            "allowlist": allowlist_path.relative_to(project_root).as_posix(),
            "allowlist_sha256": sha256_file(allowlist_path),
            "ai_header_lines": header_lines,
            "files": manifest_files,
        }
        manifest_path = staging / "source-manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        if staging.exists() and staging.parent == project_root:
            shutil.rmtree(staging)
        raise


def default_paths() -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    return project_root, Path(__file__).resolve().parent / "source_appendix_allowlist.json", project_root / "提交源码"


def parse_args() -> argparse.Namespace:
    project_root, allowlist, output = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(project_root))
    parser.add_argument("--allowlist", default=str(allowlist))
    parser.add_argument("--output-dir", default=str(output))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = prepare_submission_sources(
            Path(args.allowlist), Path(args.output_dir), Path(args.project_root)
        )
    except Exception as exc:
        print(f"[prepare_submission_sources] ERROR: {exc}")
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
