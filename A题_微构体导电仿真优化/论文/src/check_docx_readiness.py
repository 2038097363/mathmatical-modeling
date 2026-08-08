#!/usr/bin/env python3
"""Report whether paper content and assets are ready for a formal DOCX build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_docx import (
    load_figure_map,
    load_content_map,
    load_metadata,
    materialize_content_slot,
    parse_front_matter,
    parse_keywords,
    parse_markdown,
    resolve_figure_path,
    sha256_file,
)


def check_readiness(
    markdown: Path,
    metadata_file: Path,
    figure_map_file: Path,
    content_map_file: Path | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    front_matter, body = parse_front_matter(markdown.read_text(encoding="utf-8"))
    title, blocks = parse_markdown(body)
    metadata = load_metadata(metadata_file)
    figure_map, figure_base = load_figure_map(figure_map_file)
    content_map, content_base = load_content_map(content_map_file)

    abstract = str(metadata.get("abstract") or front_matter.get("abstract") or "").strip()
    keywords = parse_keywords(metadata.get("keywords", front_matter.get("keywords")))
    placeholders = [block.text for block in blocks if block.kind == "placeholder"]
    resolved_slots: list[str] = []
    unresolved_slots: list[str] = []
    resolved_figure_slots: list[str] = []
    resolved_content_slots: list[str] = []
    missing_assets: list[dict[str, str]] = []
    preview_assets: list[dict[str, str]] = []
    content_errors: list[dict[str, str]] = []
    content_inputs: list[dict[str, Any]] = []
    included_sources: list[dict[str, Any]] = []
    ambiguous_slots = sorted(set(figure_map) & set(content_map) & set(placeholders))

    for token in placeholders:
        if token in ambiguous_slots:
            unresolved_slots.append(token)
            continue
        if token in content_map:
            try:
                materialized = materialize_content_slot(
                    token, content_map[token], content_base, project_root
                )
            except Exception as exc:
                content_errors.append({"slot": token, "error": str(exc)})
                unresolved_slots.append(token)
                continue
            resolved_slots.append(token)
            resolved_content_slots.append(token)
            content_inputs.extend({"slot": token, **item} for item in materialized.inputs)
            included_sources.extend({"slot": token, **item} for item in materialized.sources)
            continue
        mapping = figure_map.get(token)
        if not isinstance(mapping, dict):
            unresolved_slots.append(token)
            continue
        candidate_values: list[str] = []
        if mapping.get("path"):
            candidate_values.append(str(mapping["path"]))
        for item in mapping.get("items", []):
            if isinstance(item, dict) and item.get("path"):
                candidate_values.append(str(item["path"]))
        if not candidate_values:
            unresolved_slots.append(token)
            continue
        token_ok = True
        for raw_path in candidate_values:
            path = resolve_figure_path(raw_path, figure_base)
            if not path.exists():
                missing_assets.append({"slot": token, "path": str(path)})
                token_ok = False
            if token.startswith("FIGURE_Q4") and "preview" in str(path).lower():
                preview_assets.append({"slot": token, "path": str(path)})
                token_ok = False
        if token_ok:
            resolved_slots.append(token)
            resolved_figure_slots.append(token)
        else:
            unresolved_slots.append(token)

    issues: list[str] = []
    if not abstract:
        issues.append("abstract_missing")
    if not keywords:
        issues.append("keywords_missing")
    if unresolved_slots:
        issues.append("unresolved_slots")
    if missing_assets:
        issues.append("missing_assets")
    if preview_assets:
        issues.append("preview_assets_forbidden")
    if ambiguous_slots:
        issues.append("ambiguous_slot_mappings")
    if content_errors:
        issues.append("invalid_content_slots")

    return {
        "schema_version": "1.0",
        "ready": not issues,
        "title": title,
        "markdown": str(markdown),
        "markdown_sha256": sha256_file(markdown),
        "metadata_file": str(metadata_file),
        "metadata_sha256": sha256_file(metadata_file),
        "figure_map": str(figure_map_file),
        "figure_map_sha256": sha256_file(figure_map_file),
        "content_map": str(content_map_file) if content_map_file else None,
        "content_map_sha256": sha256_file(content_map_file) if content_map_file else None,
        "abstract_present": bool(abstract),
        "keyword_count": len(keywords),
        "placeholder_count": len(placeholders),
        "resolved_slots": resolved_slots,
        "resolved_figure_slots": resolved_figure_slots,
        "resolved_content_slots": resolved_content_slots,
        "unresolved_slots": unresolved_slots,
        "missing_assets": missing_assets,
        "preview_assets": preview_assets,
        "ambiguous_slots": ambiguous_slots,
        "content_errors": content_errors,
        "content_inputs": content_inputs,
        "included_source_files": included_sources,
        "issues": issues,
    }


def defaults() -> tuple[Path, Path, Path, Path]:
    src = Path(__file__).resolve().parent
    project = src.parents[1]
    return (
        project / "论文" / "内容稿.md",
        src / "paper_metadata.yml",
        src / "figure_map.json",
        src / "content_map.json",
    )


def parse_args() -> argparse.Namespace:
    markdown, metadata, figure_map, content_map = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", default=str(markdown))
    parser.add_argument("--metadata-file", default=str(metadata))
    parser.add_argument("--figure-map", default=str(figure_map))
    parser.add_argument("--content-map", default=str(content_map))
    parser.add_argument("--json", dest="json_path", help="Write readiness report JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = check_readiness(
            Path(args.markdown).resolve(),
            Path(args.metadata_file).resolve(),
            Path(args.figure_map).resolve(),
            Path(args.content_map).resolve(),
        )
    except Exception as exc:
        print(json.dumps({"ready": False, "issues": ["readiness_error"], "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    if args.json_path:
        target = Path(args.json_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
