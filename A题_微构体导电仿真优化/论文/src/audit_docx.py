#!/usr/bin/env python3
"""Structural and privacy audit for a generated competition-paper DOCX."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
NS = {"w": W_NS, "m": M_NS, "rel": REL_NS, "cp": CP_NS, "dc": DC_NS}

EXPECTED_PAGE = (11906, 16838)
EXPECTED_MARGINS = (1418, 1418, 1418, 1418)
EXPECTED_HEADER_FOOTER = (851, 992)
EXPECTED_FIRST_TABLE_WIDTH = 9067
EXPECTED_FIRST_TABLE_GRID = [1728, 5760, 1579]
CM_PATTERN = re.compile(r"^CM[0-9]{7}$")
ALLOWED_GROUPS = {"研究生", "本科生", "专科生"}
QA_IDENTITY = "INTERNAL-QA"
FORBIDDEN_TEMPLATE_TEXT = ["此处换成", "看完后删除", "正文内容", "CM******", "可另起一页", "另起一页"]
FORBIDDEN_METADATA = ["xywxq", "沐欣然", "选择题目", "276067011"]


@dataclass
class Check:
    id: str
    ok: bool
    severity: str
    expected: Any
    actual: Any


class Auditor:
    def __init__(self, path: Path, internal_qa: bool):
        self.path = path
        self.internal_qa = internal_qa
        self.checks: list[Check] = []
        self.parts: dict[str, bytes] = {}
        self.document: etree._Element | None = None
        self.styles: etree._Element | None = None

    def add(self, check_id: str, ok: bool, expected: Any, actual: Any, severity: str = "blocker") -> None:
        self.checks.append(Check(check_id, bool(ok), severity, expected, actual))

    @staticmethod
    def attr(element: etree._Element | None, name: str) -> str | None:
        if element is None:
            return None
        return element.get(f"{{{W_NS}}}{name}")

    @staticmethod
    def text(element: etree._Element | None) -> str:
        if element is None:
            return ""
        return "".join(element.itertext())

    def load(self) -> bool:
        if not self.path.exists():
            self.add("package.exists", False, True, False)
            return False
        try:
            with zipfile.ZipFile(self.path, "r") as archive:
                bad = archive.testzip()
                self.parts = {info.filename: archive.read(info.filename) for info in archive.infolist()}
        except (OSError, zipfile.BadZipFile) as exc:
            self.add("package.valid_zip", False, "valid DOCX ZIP", str(exc))
            return False
        self.add("package.valid_zip", bad is None, None, bad)
        required = {"[Content_Types].xml", "word/document.xml", "word/styles.xml", "word/settings.xml"}
        missing = sorted(required - set(self.parts))
        self.add("package.required_parts", not missing, [], missing)
        if missing:
            return False
        self.document = etree.fromstring(self.parts["word/document.xml"])
        self.styles = etree.fromstring(self.parts["word/styles.xml"])
        return True

    def audit_geometry(self) -> None:
        assert self.document is not None
        sections = self.document.findall(".//w:sectPr", namespaces=NS)
        self.add("layout.section_count", len(sections) == 1, 1, len(sections))
        if not sections:
            return
        section = sections[-1]
        page = section.find("w:pgSz", namespaces=NS)
        page_actual = (int(self.attr(page, "w") or 0), int(self.attr(page, "h") or 0))
        self.add("layout.page_size", page_actual == EXPECTED_PAGE, EXPECTED_PAGE, page_actual)
        margins = section.find("w:pgMar", namespaces=NS)
        margin_actual = tuple(int(self.attr(margins, name) or 0) for name in ("top", "right", "bottom", "left"))
        self.add("layout.margins", margin_actual == EXPECTED_MARGINS, EXPECTED_MARGINS, margin_actual)
        distance_actual = (int(self.attr(margins, "header") or 0), int(self.attr(margins, "footer") or 0))
        self.add("layout.header_footer_distance", distance_actual == EXPECTED_HEADER_FOOTER, EXPECTED_HEADER_FOOTER, distance_actual)
        page_type = section.find("w:pgNumType", namespaces=NS)
        page_start = self.attr(page_type, "start")
        self.add("layout.page_number_start", page_start == "1", "1", page_start)

    def audit_first_table(self) -> None:
        assert self.document is not None
        body = self.document.find("w:body", namespaces=NS)
        children = list(body) if body is not None else []
        first_is_table = bool(children) and etree.QName(children[0]).localname == "tbl"
        self.add("template.first_component_is_table", first_is_table, True, first_is_table)
        if not first_is_table:
            return
        table = children[0]
        rows = table.findall("w:tr", namespaces=NS)
        cells_per_row = [len(row.findall("w:tc", namespaces=NS)) for row in rows]
        self.add("template.first_table_shape", len(rows) == 2 and cells_per_row == [3, 3], [3, 3], cells_per_row)
        width_node = table.find("w:tblPr/w:tblW", namespaces=NS)
        width = int(self.attr(width_node, "w") or 0)
        self.add("template.first_table_width", width == EXPECTED_FIRST_TABLE_WIDTH, EXPECTED_FIRST_TABLE_WIDTH, width)
        grid = [int(self.attr(node, "w") or 0) for node in table.findall("w:tblGrid/w:gridCol", namespaces=NS)]
        self.add("template.first_table_grid", grid == EXPECTED_FIRST_TABLE_GRID, EXPECTED_FIRST_TABLE_GRID, grid)
        merges: list[str | None] = []
        for row in rows:
            cells = row.findall("w:tc", namespaces=NS)
            merge = cells[1].find("w:tcPr/w:vMerge", namespaces=NS) if len(cells) > 1 else None
            merges.append(self.attr(merge, "val") if merge is not None else None)
        merge_ok = len(merges) == 2 and merges[0] == "restart" and merges[1] in (None, "continue")
        self.add("template.first_table_middle_merge", merge_ok, ["restart", "continue"], merges)

        if len(rows) == 2 and cells_per_row == [3, 3]:
            second_cells = rows[1].findall("w:tc", namespaces=NS)
            group = self.text(second_cells[0]).strip()
            competition_id = self.text(second_cells[2]).strip()
            if self.internal_qa:
                identity_ok = group == QA_IDENTITY and competition_id == QA_IDENTITY
                expected = [QA_IDENTITY, QA_IDENTITY]
            else:
                identity_ok = group in ALLOWED_GROUPS and bool(CM_PATTERN.fullmatch(competition_id))
                expected = ["研究生/本科生/专科生", "CM + seven digits"]
            self.add("identity.slots", identity_ok, expected, [group, competition_id])

    def audit_styles(self) -> None:
        assert self.styles is not None
        body = self.styles.find(".//w:style[@w:styleId='PaperBody']", namespaces=NS)
        self.add("style.paper_body_exists", body is not None, True, body is not None)
        if body is not None:
            fonts = body.find("w:rPr/w:rFonts", namespaces=NS)
            size = body.find("w:rPr/w:sz", namespaces=NS)
            spacing = body.find("w:pPr/w:spacing", namespaces=NS)
            indent = body.find("w:pPr/w:ind", namespaces=NS)
            alignment = body.find("w:pPr/w:jc", namespaces=NS)
            actual = {
                "eastAsia": self.attr(fonts, "eastAsia"),
                "ascii": self.attr(fonts, "ascii"),
                "size_half_points": self.attr(size, "val"),
                "line": self.attr(spacing, "line"),
                "lineRule": self.attr(spacing, "lineRule"),
                "firstLine": self.attr(indent, "firstLine"),
                "alignment": self.attr(alignment, "val"),
            }
            ok = (
                actual["eastAsia"] == "宋体"
                and actual["size_half_points"] == "24"
                and actual["line"] == "240"
                and actual["lineRule"] == "auto"
                and actual["firstLine"] == "480"
                and actual["alignment"] == "both"
            )
            self.add(
                "style.paper_body_baseline",
                ok,
                {"eastAsia": "宋体", "size_half_points": "24", "line": "240", "lineRule": "auto", "firstLine": "480", "alignment": "both"},
                actual,
            )
        red_nodes = self.styles.findall(".//w:color[@w:val='FF0000']", namespaces=NS)
        assert self.document is not None
        red_nodes += self.document.findall(".//w:color[@w:val='FF0000']", namespaces=NS)
        self.add("style.no_template_red", not red_nodes, 0, len(red_nodes))

    def audit_flow(self) -> None:
        assert self.document is not None
        body = self.document.find("w:body", namespaces=NS)
        paragraphs = body.findall("w:p", namespaces=NS) if body is not None else []
        records: list[tuple[str, etree._Element]] = [(self.text(paragraph).strip(), paragraph) for paragraph in paragraphs]
        all_text = "\n".join(text for text, _ in records)
        title_index = next((idx for idx, (text, _) in enumerate(records) if text and text != "摘要"), None)
        abstract_index = next((idx for idx, (text, _) in enumerate(records) if text == "摘要"), None)
        keyword_index = next((idx for idx, (text, _) in enumerate(records) if text.startswith("关键词：")), None)
        body_index = next((idx for idx, (text, _) in enumerate(records) if text == "一、问题重述"), None)
        ordered = None not in (title_index, abstract_index, keyword_index, body_index) and title_index < abstract_index < keyword_index < body_index
        self.add("flow.first_page_components_order", ordered, "title < 摘要 < 关键词 < 一、问题重述", [title_index, abstract_index, keyword_index, body_index])
        if body_index is not None:
            page_break = records[body_index][1].find("w:pPr/w:pageBreakBefore", namespaces=NS)
            self.add("flow.body_page_break", page_break is not None, True, page_break is not None)
        else:
            self.add("flow.body_page_break", False, True, "body heading missing")
        appendix_index = next((idx for idx, (text, _) in enumerate(records) if text == "附录"), None)
        if appendix_index is not None:
            appendix_break = records[appendix_index][1].find("w:pPr/w:pageBreakBefore", namespaces=NS)
            self.add("flow.appendix_page_break", appendix_break is not None, True, appendix_break is not None)

        forbidden = [value for value in FORBIDDEN_TEMPLATE_TEXT if value in all_text]
        self.add("content.no_template_placeholders", not forbidden, [], forbidden)
        qa_markers = sorted(set(re.findall(r"\[INTERNAL-QA:[^\]]+\]", all_text)))
        self.add(
            "content.no_unresolved_slots",
            not qa_markers,
            [],
            qa_markers,
            severity="warning" if self.internal_qa else "blocker",
        )
        toc_headings = [text for text, _ in records if re.sub(r"\s+", "", text) == "目录"]
        self.add("content.no_toc_heading", not toc_headings, [], toc_headings)
        toc_fields = [
            (node.text or "").strip()
            for node in self.document.findall(".//w:instrText", namespaces=NS)
            if re.search(r"(?:^|\s)TOC(?:\s|$)", (node.text or ""), flags=re.IGNORECASE)
        ]
        self.add("content.no_toc_field", not toc_fields, [], toc_fields)
        math_count = len(self.document.findall(".//m:oMath", namespaces=NS))
        self.add("content.native_omml_math", math_count > 0, ">0", math_count)
        image_count = len(self.document.findall(".//w:drawing", namespaces=NS))
        self.add("content.figures_present", image_count > 0, ">0 for current QA map", image_count, severity="warning" if self.internal_qa else "blocker")

        if body is not None:
            children = list(body)
            last = children[-2] if children and etree.QName(children[-1]).localname == "sectPr" and len(children) >= 2 else (children[-1] if children else None)
            last_blank = last is not None and etree.QName(last).localname == "p" and not self.text(last).strip() and not last.findall(".//w:drawing", namespaces=NS)
            self.add("flow.no_blank_trailing_paragraph", not last_blank, True, last_blank)

    def audit_numbering_and_tables(self) -> None:
        assert self.document is not None
        numbered = self.document.findall(".//w:numPr", namespaces=NS)
        has_part = "word/numbering.xml" in self.parts
        self.add("structure.real_numbering", (not numbered) or has_part, "numbering.xml when numPr exists", {"numPr": len(numbered), "numbering_part": has_part})
        malformed: list[dict[str, Any]] = []
        tables = self.document.findall(".//w:tbl", namespaces=NS)
        for table_index, table in enumerate(tables[1:], start=2):
            grid = [int(self.attr(node, "w") or 0) for node in table.findall("w:tblGrid/w:gridCol", namespaces=NS)]
            table_width = int(self.attr(table.find("w:tblPr/w:tblW", namespaces=NS), "w") or 0)
            row_geometry: list[list[dict[str, int]]] = []
            rows_ok = True
            for row in table.findall("w:tr", namespaces=NS):
                column = 0
                cells: list[dict[str, int]] = []
                for cell in row.findall("w:tc", namespaces=NS):
                    width = int(self.attr(cell.find("w:tcPr/w:tcW", namespaces=NS), "w") or 0)
                    span = int(self.attr(cell.find("w:tcPr/w:gridSpan", namespaces=NS), "val") or 1)
                    expected_width = sum(grid[column : column + span]) if span > 0 else 0
                    cells.append({"width": width, "span": span, "expected_width": expected_width})
                    if span <= 0 or column + span > len(grid) or width != expected_width:
                        rows_ok = False
                    column += span
                if column != len(grid):
                    rows_ok = False
                row_geometry.append(cells)
            ok = bool(grid) and sum(grid) == table_width and rows_ok
            if not ok:
                malformed.append({"table": table_index, "tblW": table_width, "grid": grid, "rows": row_geometry})
        self.add("structure.body_table_geometry", not malformed, [], malformed)

    def audit_fields_and_headers(self) -> None:
        assert self.document is not None
        header_parts = sorted(name for name in self.parts if re.fullmatch(r"word/header\d+\.xml", name))
        self.add("header.none", not header_parts, [], header_parts)
        footer_parts = sorted(name for name in self.parts if re.fullmatch(r"word/footer\d+\.xml", name))
        self.add("footer.exists", len(footer_parts) == 1, 1, len(footer_parts))
        page_field = False
        centered = False
        for name in footer_parts:
            root = etree.fromstring(self.parts[name])
            instructions = " ".join(self.text(node) for node in root.findall(".//w:instrText", namespaces=NS))
            page_field = page_field or bool(re.search(r"\bPAGE\b", instructions))
            for paragraph in root.findall(".//w:p", namespaces=NS):
                jc = paragraph.find("w:pPr/w:jc", namespaces=NS)
                if self.attr(jc, "val") == "center":
                    centered = True
        self.add("footer.page_field", page_field, True, page_field)
        self.add("footer.centered", centered, True, centered)
        settings = etree.fromstring(self.parts["word/settings.xml"])
        update = settings.find("w:updateFields", namespaces=NS)
        self.add("fields.update_on_open", self.attr(update, "val") in {"true", "1"}, "true", self.attr(update, "val"))

    def audit_privacy(self) -> None:
        custom = "docProps/custom.xml" in self.parts
        self.add("privacy.no_custom_properties", not custom, False, custom)
        metadata: dict[str, str] = {}
        if "docProps/core.xml" in self.parts:
            core = etree.fromstring(self.parts["docProps/core.xml"])
            for name, namespace in (("creator", DC_NS), ("lastModifiedBy", CP_NS), ("title", DC_NS), ("subject", DC_NS), ("description", DC_NS), ("keywords", CP_NS)):
                node = core.find(f"{{{namespace}}}{name}")
                metadata[name] = self.text(node).strip()
        dirty_metadata = {key: value for key, value in metadata.items() if value}
        self.add("privacy.core_properties_empty", not dirty_metadata, {}, dirty_metadata)

        rsid_count = 0
        for name, value in self.parts.items():
            if not re.fullmatch(r"word/(?:document|header\d+|footer\d+|footnotes|endnotes)\.xml", name):
                continue
            root = etree.fromstring(value)
            for element in root.iter():
                rsid_count += sum(1 for attr in element.attrib if etree.QName(attr).namespace == W_NS and etree.QName(attr).localname.startswith("rsid"))
        settings = etree.fromstring(self.parts["word/settings.xml"])
        settings_rsids = len(settings.findall("w:rsids", namespaces=NS))
        doc_vars = len(settings.findall("w:docVars", namespaces=NS))
        self.add("privacy.no_rsid", rsid_count == 0 and settings_rsids == 0, 0, {"attributes": rsid_count, "settings_rsids": settings_rsids})
        self.add("privacy.no_template_docvars", doc_vars == 0, 0, doc_vars)
        xml_text = "\n".join(value.decode("utf-8", errors="ignore") for name, value in self.parts.items() if name.endswith((".xml", ".rels")))
        leaked = [value for value in FORBIDDEN_METADATA if value in xml_text]
        self.add("privacy.no_known_template_identity", not leaked, [], leaked)

    def run(self) -> dict[str, Any]:
        if self.load():
            self.audit_geometry()
            self.audit_first_table()
            self.audit_styles()
            self.audit_flow()
            self.audit_numbering_and_tables()
            self.audit_fields_and_headers()
            self.audit_privacy()
        blocker_failures = [check.id for check in self.checks if check.severity == "blocker" and not check.ok]
        warnings = [check.id for check in self.checks if check.severity == "warning" and not check.ok]
        return {
            "schema_version": "1.0",
            "path": str(self.path),
            "mode": "internal_qa" if self.internal_qa else "formal",
            "passed": not blocker_failures,
            "blocker_failures": blocker_failures,
            "warnings": warnings,
            "checks": [asdict(check) for check in self.checks],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", help="Generated DOCX to audit")
    parser.add_argument("--json", dest="json_path", help="Write detailed JSON report")
    parser.add_argument("--internal-qa", action="store_true", help="Accept exact INTERNAL-QA identity and downgrade unresolved slots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Auditor(Path(args.docx).resolve(), args.internal_qa).run()
    if args.json_path:
        target = Path(args.json_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
