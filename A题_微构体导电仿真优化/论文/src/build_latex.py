from __future__ import annotations

import argparse
import hashlib
import re
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


PAPER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PAPER_DIR.parent
SOURCE = PAPER_DIR / "内容稿.md"
REFERENCES_BIB = PAPER_DIR / "references.bib"
GENERATED_DIR = PAPER_DIR / "generated"
MARKDOWN_OUT = GENERATED_DIR / "content_for_latex.md"
LATEX_OUT = PAPER_DIR / "content.tex"
SOURCE_ALLOWLIST = PAPER_DIR / "src" / "source_appendix_allowlist.json"
SOURCE_APPENDIX_DIR = PAPER_DIR / "source_appendix"
SOURCE_APPENDIX_TEX = PAPER_DIR / "source_appendix.tex"
BUILD_META_TEX = PAPER_DIR / "build_meta.tex"
Q4_RESULT_ROOT = (
    PROJECT_DIR / "问题" / "问题4" / "results" / "D_screen2000_confirm50000"
)
DEFAULT_Q4_SUMMARY = Q4_RESULT_ROOT / "q4_summary.json"
DEFAULT_Q4_ANALYSIS = Q4_RESULT_ROOT / "q4_confirmation_integer_domain_analysis.json"
DEFAULT_Q3_SUMMARY = (
    PROJECT_DIR / "问题" / "问题3" / "results" / "D_confirmation_n50000" / "q3_summary.json"
)
DEFAULT_Q4_FRONTIER_PDF = PAPER_DIR / "figures" / "generated" / "q4_cost_frontier.pdf"
DEFAULT_Q4_FRONTIER_PNG = PAPER_DIR / "figures" / "generated" / "q4_cost_frontier.png"
DEFAULT_Q4_FRONTIER_AUDIT = (
    PAPER_DIR / "figures" / "generated" / "q4_cost_frontier.audit.json"
)
ALLOWED_GROUPS = {"研究生", "本科生", "专科生"}
CM_PATTERN = re.compile(r"^CM[0-9]{7}$")
FINAL_RESULT_STATUSES = {
    "globally_certified_minimum_cost",
    "lowest_statistically_feasible_cost",
    "screening_candidate_not_confirmed",
}
FINAL_FORBIDDEN_TERMS = {
    "待填写": "封面仍有待填写项",
    "待回填": "正文仍有待回填项",
    "INTERNAL-QA": "仍含内部审阅标记",
    "尚未回填": "仍含尚未回填说明",
    "不得用于正式提交": "仍含内部禁提交警示",
}
PROCESS_NARRATIVE_PATTERNS = (
    re.compile(r"(?:题目|赛题).{0,10}(?:修改|更正|澄清)"),
    re.compile(r"(?:修改|更正|澄清).{0,10}(?:过程|题目|赛题)"),
    re.compile(r"组委会.{0,10}(?:确认|回复|说明)"),
    re.compile(r"(?:旧方案|旧题意|原题版本|历史诊断)"),
)
PRIVATE_HOME_PATTERN = re.compile(
    r"(?i)(?:[A-Z]:[\\/]Users[\\/][^\\/\s]+|/Users/[^/\s]+|/home/[^/\s]+)"
)
DOI_VALUE_PATTERN = re.compile(
    r"10\.\d{4,9}/[-._;()/:A-Z0-9]*[A-Z0-9]", re.IGNORECASE
)
BIB_ENTRY_PATTERN = re.compile(
    r"^\s*@(?!comment\b|string\b|preamble\b)[A-Z]+\s*[\{(]\s*[^,\s]+",
    re.IGNORECASE | re.MULTILINE,
)
BIB_DOI_ASSIGNMENT_PATTERN = re.compile(r"(?<![A-Z0-9_])doi\s*=", re.IGNORECASE)
BIB_DOI_FIELD_PATTERN = re.compile(
    r'(?<![A-Z0-9_])doi\s*=\s*(?:\{(?P<braced>[^{}\r\n]+)\}|"(?P<quoted>[^"\r\n]+)"|(?P<bare>[^,\s}]+))',
    re.IGNORECASE,
)


FIGURE_Q1 = r"""
```{=latex}
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{figures/generated/q1_groups_3d.pdf}
  \caption{附件三组微构体的三维接触结构。浅灰线为有限平底圆柱轴线，橙色粗线为程序恢复的电极间见证片段；为提高薄层结构的可读性，三轴按各自显示范围归一化，接触判定仍使用原始 nm 坐标。轴测投影中的线段交叉不等于三维接触。}
  \label{fig:q1-groups}
\end{figure}
```
"""


FIGURE_Q2_Q3 = r"""
```{=latex}
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.94\textwidth]{figures/generated/q2_q3_probability.pdf}
  \caption{全 A 介质导通概率曲线及问题三临界区间。颜色、点形和区间线共同区分可行、不可行与尚不能区分的整数点；全部数值来自冻结结果文件。}
  \label{fig:q2-q3-curve}
\end{figure}
```
"""


Q4_INTERIM = r"""
在 $2000$ 个固定探索样本中，对覆盖全部不高于候选成本设计的成本有界整数域进行扫描，描述性经验最低点为
$(N_A,N_B)=(613,0)$，其导通次数为 $1802/2000$；按预先给定的单侧
$95\%$ Clopper--Pearson 下限规则，冻结的保守候选为
$(619,0)$，导通次数为 $1825/2000$，单侧下限为 $90.1403\%$，成本为
$9.1884516534$ 元。该结果只用于在独立确认流启动前冻结候选：问题三已在其
Bonferroni--Clopper--Pearson 联合置信协议下确认更便宜的 $(616,0)$ 统计可行，故不能把
$(619,0)$ 表述为全局最低成本。最终结论须以 $50000$
个独立确认样本同时检验候选可行性和全部严格更低成本设计的排除结果为准。

表 \ref{tab:q4-interim} 中 $(616,0)$ 的下限采用问题三 Bonferroni 联合协议，$(619,0)$ 的下限
采用探索阶段名义单侧 $95\%$ 协议；后者只用于冻结候选，不作为最终推断证据。

```{=latex}
\begin{table}[htbp]
  \centering
  \small
  \caption{问题四探索证据与问题三交叉证据（置信下限按各自协议计算）}
  \label{tab:q4-interim}
  \begin{tabularx}{\textwidth}{c>{\raggedright\arraybackslash}Xcccc}
    \toprule
    设计 & 证据角色 & 成功数 & 经验概率 & 协议内 CP 下限 & 成本（元） \\
    \midrule
    $(613,0)$ & 探索流中描述性经验最低 & $1802/2000$ & $90.10\%$ & --- & $9.09938750$ \\
    $(616,0)$ & 问题三独立确认流的交叉可行证据 & $45226/50000$ & $90.452\%$ & $90.0821\%$ & $9.14391958$ \\
    $(619,0)$ & 探索流中的预冻结保守候选 & $1825/2000$ & $91.25\%$ & $90.1403\%$ & $9.18845165$ \\
    \bottomrule
  \end{tabularx}
\end{table}
```
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}的 JSON 顶层必须为对象：{path}")
    return payload


def normalize_doi(raw: str, label: str) -> str:
    value = raw.strip()
    value = re.sub(r"(?i)^doi\s*:\s*", "", value)
    value = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", value)
    if not DOI_VALUE_PATTERN.fullmatch(value):
        raise ValueError(f"{label}不是有效 DOI：{raw!r}")
    return value.casefold()


def require_unique_dois(dois: Sequence[str], label: str) -> frozenset[str]:
    duplicates = sorted({doi for doi in dois if dois.count(doi) > 1})
    if duplicates:
        raise ValueError(f"{label}存在重复 DOI：{', '.join(duplicates)}")
    return frozenset(dois)


def validate_reference_doi_consistency(
    bib_path: Path | None = None,
    markdown_path: Path | None = None,
) -> frozenset[str]:
    bib_path = REFERENCES_BIB if bib_path is None else bib_path
    markdown_path = SOURCE if markdown_path is None else markdown_path
    if not bib_path.is_file():
        raise FileNotFoundError(f"BibTeX 参考文献不存在：{bib_path}")
    if not markdown_path.is_file():
        raise FileNotFoundError(f"论文内容稿不存在：{markdown_path}")

    bib_text = bib_path.read_text(encoding="utf-8-sig")
    if not BIB_ENTRY_PATTERN.search(bib_text):
        raise ValueError("references.bib 未包含可识别的 BibTeX 条目")
    assignments = list(BIB_DOI_ASSIGNMENT_PATTERN.finditer(bib_text))
    fields = list(BIB_DOI_FIELD_PATTERN.finditer(bib_text))
    if len(assignments) != len(fields):
        raise ValueError("references.bib 含无法解析的 doi 字段")
    bib_doi_list = [
        normalize_doi(
            next(value for value in match.groupdict().values() if value is not None),
            "references.bib doi 字段",
        )
        for match in fields
    ]
    all_bib_tokens = [value.casefold() for value in DOI_VALUE_PATTERN.findall(bib_text)]
    if sorted(all_bib_tokens) != sorted(bib_doi_list):
        raise ValueError("references.bib 中的 DOI 必须且只能写在可解析的 doi 字段内")
    bib_dois = require_unique_dois(bib_doi_list, "references.bib")

    markdown_text = markdown_path.read_text(encoding="utf-8-sig")
    headings = list(re.finditer(r"^##\s+参考文献\s*$", markdown_text, re.MULTILINE))
    if len(headings) != 1:
        raise ValueError("内容稿必须且只能包含一个二级‘参考文献’标题")
    section_start = headings[0].end()
    next_heading = re.search(r"^##\s+", markdown_text[section_start:], re.MULTILINE)
    section_end = (
        len(markdown_text)
        if next_heading is None
        else section_start + next_heading.start()
    )
    reference_section = markdown_text[section_start:section_end]
    if not re.search(r"^\s*\[\d+\]\s+", reference_section, re.MULTILINE):
        raise ValueError("内容稿的参考文献区段未包含手工编号条目")
    manual_doi_list = [
        normalize_doi(value, "内容稿手工参考文献 DOI")
        for value in DOI_VALUE_PATTERN.findall(reference_section)
    ]
    doi_labels = re.findall(r"\bDOI\s*[:：]", reference_section, re.IGNORECASE)
    if len(doi_labels) != len(manual_doi_list):
        raise ValueError("内容稿手工参考文献含畸形 DOI 标注")
    manual_dois = require_unique_dois(manual_doi_list, "内容稿手工参考文献")

    if bib_dois != manual_dois:
        missing_manual = sorted(bib_dois - manual_dois)
        missing_bib = sorted(manual_dois - bib_dois)
        details: list[str] = []
        if missing_manual:
            details.append(f"内容稿缺少 {', '.join(missing_manual)}")
        if missing_bib:
            details.append(f"references.bib 缺少 {', '.join(missing_bib)}")
        raise ValueError("参考文献 DOI 集合不一致：" + "；".join(details))
    return bib_dois


def require_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label}必须位于项目目录内：{resolved}") from exc
    return resolved


def resolve_evidence_path(raw: Any, owner: Path, project_root: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label}缺少路径")
    raw_path = Path(raw).expanduser()
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [owner.parent / raw_path, project_root / raw_path]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(project_root.resolve())
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"{label}不存在或越出项目目录：{candidates[0]}")


def validate_identity(mode: str, group: str | None, competition_id: str | None) -> tuple[str, str]:
    if mode == "internal":
        if group is not None and group not in ALLOWED_GROUPS:
            raise ValueError(f"组别必须为 {sorted(ALLOWED_GROUPS)} 之一")
        if competition_id is not None and not CM_PATTERN.fullmatch(competition_id):
            raise ValueError("参赛编号必须为 CM 加 7 位数字")
        if competition_id is not None and competition_id == "CM0000000":
            raise ValueError("正式参赛编号不得使用全零占位符 CM0000000")
        return group or "内部审阅", competition_id or "INTERNAL-QA"
    if group not in ALLOWED_GROUPS:
        raise ValueError(f"正式模式必须提供组别，且为 {sorted(ALLOWED_GROUPS)} 之一")
    if competition_id is None or not CM_PATTERN.fullmatch(competition_id):
        raise ValueError("正式模式必须提供 CM 加 7 位数字的完整参赛编号")
    if competition_id == "CM0000000":
        raise ValueError("正式参赛编号不得使用全零占位符 CM0000000")
    return group, competition_id


def validate_final_evidence(
    summary_path: Path,
    analysis_path: Path,
    q3_summary_path: Path,
    frontier_pdf: Path,
    frontier_png: Path,
    frontier_audit_path: Path,
    *,
    project_root: Path = PROJECT_DIR,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    summary_path = require_inside(summary_path, project_root, "Q4 最终摘要")
    analysis_path = require_inside(analysis_path, project_root, "Q4 独立完整域审计")
    q3_summary_path = require_inside(q3_summary_path, project_root, "Q3 独立确认摘要")
    frontier_pdf = require_inside(frontier_pdf, project_root, "Q4 正式成本前沿 PDF")
    frontier_png = require_inside(frontier_png, project_root, "Q4 正式成本前沿 PNG")
    frontier_audit_path = require_inside(
        frontier_audit_path, project_root, "Q4 正式成本前沿审计"
    )
    summary = read_json_object(summary_path, "Q4 最终摘要")
    if summary.get("kind") != "q4_final_summary":
        raise ValueError("Q4 摘要尚不是 q4_final_summary")
    status = str(summary.get("result_status", ""))
    if status not in FINAL_RESULT_STATUSES:
        raise ValueError("Q4 尚未形成可写入正式论文的统计可行结论")
    contract = summary.get("boundary_contract")
    if not isinstance(contract, dict) or contract.get("mode") != "D":
        raise ValueError("Q4 最终摘要未采用截断片段独立的 D 几何合同")
    candidate_feasible = summary.get("candidate_statistically_feasible") is True
    if status == "screening_candidate_not_confirmed":
        if candidate_feasible:
            raise ValueError("Q4 未确认状态与 candidate_statistically_feasible=true 矛盾")
    elif not candidate_feasible:
        raise ValueError("Q4 可行状态与 candidate_statistically_feasible=false 矛盾")
    if status == "globally_certified_minimum_cost" and (
        summary.get("all_strictly_cheaper_maximal_designs_excluded") is not True
    ):
        raise ValueError("Q4 全局最低成本标签与更便宜设计排除结果矛盾")

    design = summary.get("reported_design")
    if status == "screening_candidate_not_confirmed":
        if design is not None:
            raise ValueError("Q4 候选未确认时不得给出 reported_design")
    elif not isinstance(design, dict):
        raise ValueError("Q4 最终摘要缺少 reported_design")
    records = summary.get("confirmation_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Q4 最终摘要缺少独立确认记录")
    candidates = [
        row for row in records if isinstance(row, dict) and row.get("role") == "candidate"
    ]
    if len(candidates) != 1:
        raise ValueError("Q4 最终摘要必须恰有一条 candidate 记录")
    candidate = candidates[0]
    expected_proof_status = (
        "candidate_not_confirmed"
        if status == "screening_candidate_not_confirmed"
        else "candidate_statistically_feasible"
    )
    if candidate.get("proof_status") != expected_proof_status:
        raise ValueError("Q4 candidate 记录与最终结论状态不一致")
    required_candidate_fields = {
        "n_a",
        "n_b",
        "cost_weight",
        "cost_yuan",
        "successes",
        "trials",
        "estimate",
        "clopper_pearson_one_sided_lower",
        "clopper_pearson_one_sided_upper",
    }
    missing = sorted(required_candidate_fields - set(candidate))
    if missing:
        raise ValueError(f"Q4 candidate 记录缺少字段：{missing}")
    if int(candidate["trials"]) != 50000:
        raise ValueError("Q4 正式独立确认样本数必须为 50000")
    if isinstance(design, dict):
        for key in ("n_a", "n_b", "cost_weight"):
            if int(design[key]) != int(candidate[key]):
                raise ValueError(f"Q4 reported_design 与 candidate 的 {key} 不一致")

    freeze_path = resolve_evidence_path(
        summary.get("freeze_path"), summary_path, project_root, "Q4 冻结协议"
    )
    expected_freeze_hash = str(summary.get("freeze_sha256", "")).upper()
    if not expected_freeze_hash or sha256(freeze_path) != expected_freeze_hash:
        raise ValueError("Q4 冻结协议 SHA-256 与最终摘要不一致")
    freeze = read_json_object(freeze_path, "Q4 冻结协议")
    if freeze.get("kind") != "q4_confirmation_freeze":
        raise ValueError("Q4 最终摘要引用的不是确认冻结协议")
    protocol = freeze.get("confirmation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Q4 冻结协议缺少 confirmation_protocol")
    if int(protocol.get("fixed_trial_count", -1)) != 50000:
        raise ValueError("Q4 冻结协议的正式确认样本数不是 50000")
    screening_stream_id = protocol.get("screening_stream_id")
    confirmation_stream_id = protocol.get("confirmation_stream_id")
    if (
        not isinstance(screening_stream_id, int)
        or isinstance(screening_stream_id, bool)
        or not isinstance(confirmation_stream_id, int)
        or isinstance(confirmation_stream_id, bool)
        or protocol.get("stream_ids_distinct") is not True
        or screening_stream_id == confirmation_stream_id
    ):
        raise ValueError("Q4 冻结协议未证明探索流与正式确认流相互独立")
    confirmation_config = protocol.get("configuration")
    if not isinstance(confirmation_config, dict):
        raise ValueError("Q4 冻结协议缺少正式确认配置")
    if int(confirmation_config.get("trial_count", -1)) != 50000:
        raise ValueError("Q4 冻结协议的确认配置样本数不是 50000")
    if int(confirmation_config.get("stream_id", -1)) != confirmation_stream_id:
        raise ValueError("Q4 冻结协议的确认配置与独立确认流不一致")
    statement_count = int(protocol.get("bonferroni_statement_count", -1))
    if statement_count != len(records):
        raise ValueError("Q4 Bonferroni 陈述数与最终确认记录数不一致")

    analysis = read_json_object(analysis_path, "Q4 独立完整域审计")
    if analysis.get("kind") != "q4_confirmation_integer_domain_analysis":
        raise ValueError("Q4 独立完整域审计类型无效")
    if analysis.get("audit_status") != "passed":
        raise ValueError("Q4 独立完整域审计尚未通过")
    if analysis.get("result_status") != status:
        raise ValueError("Q4 独立完整域审计与最终摘要的结论状态不一致")
    analysis_input = analysis.get("input_files", {}).get("final_summary", {})
    if str(analysis_input.get("sha256", "")).upper() != sha256(summary_path):
        raise ValueError("Q4 独立完整域审计与最终摘要 SHA-256 不一致")
    analysis_freeze_input = analysis.get("input_files", {}).get("freeze", {})
    if str(analysis_freeze_input.get("sha256", "")).upper() != expected_freeze_hash:
        raise ValueError("Q4 独立完整域审计与冻结协议 SHA-256 不一致")
    analysis_config = analysis.get("configuration")
    if not isinstance(analysis_config, dict):
        raise ValueError("Q4 独立完整域审计缺少配置")
    if int(analysis_config.get("trial_count", -1)) != 50000:
        raise ValueError("Q4 独立完整域审计样本数不是 50000")
    if int(analysis_config.get("stream_id", -1)) != confirmation_stream_id:
        raise ValueError("Q4 独立完整域审计使用的不是冻结确认流")
    if list(analysis_config.get("integer_domain_shape", [])) != [620, 5484]:
        raise ValueError("Q4 独立完整域审计的整数域形状不是 620×5484")
    statistical = analysis.get("statistical_results")
    if not isinstance(statistical, dict):
        raise ValueError("Q4 独立完整域审计缺少统计结果")
    analysis_candidate = statistical.get("candidate")
    if not isinstance(analysis_candidate, dict):
        raise ValueError("Q4 独立完整域审计缺少候选记录")
    for key in ("n_a", "n_b", "successes", "trials", "cost_weight"):
        if int(analysis_candidate[key]) != int(candidate[key]):
            raise ValueError(f"Q4 独立完整域审计候选的 {key} 与最终摘要不一致")
    if bool(statistical.get("candidate_statistically_feasible")) != candidate_feasible:
        raise ValueError("Q4 独立完整域审计与最终摘要的候选可行结论不一致")
    if bool(statistical.get("all_strictly_cheaper_maximal_designs_excluded")) != bool(
        summary.get("all_strictly_cheaper_maximal_designs_excluded")
    ):
        raise ValueError("Q4 独立完整域审计与最终摘要的更便宜设计排除结论不一致")

    q3_summary = read_json_object(q3_summary_path, "Q3 独立确认摘要")
    if q3_summary.get("result_scope") != "independent_fixed_confirmation":
        raise ValueError("Q3 参照证据不是独立固定样本确认")
    if int(q3_summary.get("fixed_trial_count", -1)) != 50000:
        raise ValueError("Q3 独立确认样本数不是 50000")
    q3_records = q3_summary.get("candidate_records")
    if not isinstance(q3_records, list):
        raise ValueError("Q3 独立确认摘要缺少候选记录")
    q3_candidates = [
        row
        for row in q3_records
        if isinstance(row, dict) and int(row.get("particle_count", -1)) == 616
    ]
    if len(q3_candidates) != 1:
        raise ValueError("Q3 独立确认摘要必须恰有一条 N_A=616 记录")
    q3_reference = q3_candidates[0]
    if q3_reference.get("classification_by_bonferroni_cp") != "statistically_feasible":
        raise ValueError("Q3 独立确认未证明 (616,0) 统计可行")
    analysis_q3_reference = statistical.get("q3_reference_design_in_confirmation_stream")
    if not isinstance(analysis_q3_reference, dict):
        raise ValueError("Q4 独立完整域审计缺少 (616,0) 交叉参照")
    if (int(analysis_q3_reference.get("n_a", -1)), int(analysis_q3_reference.get("n_b", -1))) != (616, 0):
        raise ValueError("Q4 独立完整域审计中的 Q3 交叉参照不是 (616,0)")

    for path, label in (
        (frontier_pdf, "Q4 正式成本前沿 PDF"),
        (frontier_png, "Q4 正式成本前沿 PNG"),
        (frontier_audit_path, "Q4 正式成本前沿审计"),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{label}不存在或为空：{path}")
        if "preview" in path.name.lower():
            raise ValueError(f"{label}不得使用 preview 资产")
    figure_audit = read_json_object(frontier_audit_path, "Q4 正式成本前沿审计")
    if figure_audit.get("kind") != "q4_cost_frontier_figure_audit":
        raise ValueError("Q4 成本前沿审计类型无效")
    if str(figure_audit.get("final_sha256", "")).upper() != sha256(summary_path):
        raise ValueError("Q4 成本前沿图与最终摘要 SHA-256 不一致")
    if str(figure_audit.get("freeze_sha256", "")).upper() != expected_freeze_hash:
        raise ValueError("Q4 成本前沿图与冻结协议 SHA-256 不一致")
    if str(figure_audit.get("output_pdf_sha256", "")).upper() != sha256(frontier_pdf):
        raise ValueError("Q4 成本前沿 PDF 与图件审计 SHA-256 不一致")
    if str(figure_audit.get("output_png_sha256", "")).upper() != sha256(frontier_png):
        raise ValueError("Q4 成本前沿 PNG 与图件审计 SHA-256 不一致")
    if figure_audit.get("result_status") != status:
        raise ValueError("Q4 成本前沿图的结论状态与最终摘要不一致")
    monotonicity = figure_audit.get("monotonicity")
    if not isinstance(monotonicity, dict) or monotonicity.get("passed") is not True:
        raise ValueError("Q4 成本前沿图未通过完整整数域单调性审计")
    for field, expected, label in (
        ("output_pdf", frontier_pdf, "PDF"),
        ("output_png", frontier_png, "PNG"),
    ):
        actual = resolve_evidence_path(
            figure_audit.get(field), frontier_audit_path, project_root, f"Q4 前沿{label}"
        )
        if actual != expected:
            raise ValueError(f"Q4 成本前沿审计引用的{label}不是正式门禁资产")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "summary_sha256": sha256(summary_path),
        "freeze": freeze,
        "freeze_path": freeze_path,
        "freeze_sha256": sha256(freeze_path),
        "analysis": analysis,
        "analysis_path": analysis_path,
        "analysis_sha256": sha256(analysis_path),
        "q3_summary": q3_summary,
        "q3_summary_path": q3_summary_path,
        "q3_reference": q3_reference,
        "q3_reference_cost_yuan": float(analysis_q3_reference["cost_yuan"]),
        "candidate": candidate,
        "frontier_pdf": frontier_pdf,
        "frontier_png": frontier_png,
        "frontier_audit": figure_audit,
        "frontier_audit_path": frontier_audit_path,
    }


def percent_text(value: Any, digits: int = 4) -> str:
    return f"{100.0 * float(value):.{digits}f}".rstrip("0").rstrip(".") + r"\%"


def decimal_text(value: Any, digits: int = 10) -> str:
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def build_final_q4_blocks(evidence: dict[str, Any]) -> dict[str, str]:
    summary = evidence["summary"]
    freeze = evidence["freeze"]
    candidate = evidence["candidate"]
    design = summary.get("reported_design") or candidate
    n_a, n_b = int(design["n_a"]), int(design["n_b"])
    successes, trials = int(candidate["successes"]), int(candidate["trials"])
    estimate = percent_text(candidate["estimate"], 3)
    cp_lower = percent_text(candidate["clopper_pearson_one_sided_lower"], 4)
    cp_upper = percent_text(candidate["clopper_pearson_one_sided_upper"], 4)
    cost = decimal_text(design["cost_yuan"], 10)
    protocol = freeze["confirmation_protocol"]
    statement_count = int(protocol["bonferroni_statement_count"])
    family_confidence = percent_text(protocol["familywise_confidence"], 1)
    status = str(summary["result_status"])
    excluded = int(summary.get("excluded_frontier_count", 0))
    not_excluded = int(summary.get("not_excluded_frontier_count", 0))
    equal_cost_count = len(summary.get("equal_cost_designs", []))
    q3 = evidence["q3_reference"]
    q3_successes = int(q3["successes"])
    q3_trials = int(q3["trials"])
    q3_estimate = percent_text(q3["estimate"], 3)
    q3_cp_lower = percent_text(
        q3["clopper_pearson_one_sided_bounds"]["lower"], 4
    )
    q3_cp_upper = percent_text(
        q3["clopper_pearson_one_sided_bounds"]["upper"], 4
    )
    q3_cost_value = float(evidence["q3_reference_cost_yuan"])
    q3_cost = decimal_text(q3_cost_value, 10)
    cross_question_cheaper = q3_cost_value < float(design["cost_yuan"]) - 1e-12
    globally_coherent = (
        status == "globally_certified_minimum_cost" and not cross_question_cheaper
    )

    if globally_coherent:
        decision_text = (
            f"其余 {excluded} 个严格更低成本极大设计的单侧 Clopper--Pearson 上限均低于 "
            f"$90\\%$。因此在预先冻结的 {statement_count} 个单侧陈述上，经 Bonferroni "
            f"校正，以联合 {family_confidence} 置信度确认上述成本为所建 D 边界模型下的最低成本。"
        )
        conclusion = (
            f"问题四在 {trials} 次独立确认和联合 {family_confidence} 置信度下，确认代表性配比 "
            f"$(N_A,N_B)=({n_a},{n_b})$ 的最低成本为 ${cost}$ 元；同成本配比未作唯一性声明。"
        )
        abstract = (
            f"\\textbf{{针对问题四，}}以二维 minimax 非支配路径前沿压缩成本有界整数域，"
            f"并对 {statement_count} 个预注册单侧陈述进行 Bonferroni--Clopper--Pearson 独立确认。"
            f"代表性配比为 $({n_a},{n_b})$，经验导通概率为 ${estimate}$，最低成本为 "
            f"$\\mathbf{{{cost}}}$ 元；该最低成本结论的联合置信度为 ${family_confidence}$。"
        )
        evidence_label = f"联合 {family_confidence} 最低成本证据"
    elif status == "lowest_statistically_feasible_cost" or (
        status == "globally_certified_minimum_cost" and cross_question_cheaper
    ):
        interval = summary.get("cost_uncertainty_interval")
        if status == "lowest_statistically_feasible_cost":
            if not isinstance(interval, dict):
                raise ValueError("Q4 未完全排除更便宜设计时必须给出成本不确定区间")
            lower_cost = decimal_text(interval["lower_cost_yuan"], 10)
            upper_cost = decimal_text(interval["upper_cost_yuan"], 10)
            family_note = (
                f"在 Q4 预注册的 {statement_count} 个陈述家族内，{excluded} 个严格更低成本"
                f"极大设计已被排除，仍有 {not_excluded} 个未被同时排除，对应成本证据区间为 "
                f"$[{lower_cost},{upper_cost}]$ 元。"
            )
            abstract_family_note = (
                f"严格更低成本极大设计中已有 \\textbf{{{excluded} 个}}被排除、仍有 "
                f"\\textbf{{{not_excluded} 个}}未决，成本证据区间为 "
                f"$\\mathbf{{[{lower_cost},{upper_cost}]}}$ 元。"
            )
        else:
            family_note = (
                "Q4 预注册陈述家族给出机器状态“已排除全部更便宜极大设计”，但该结论与"
                "下述独立 Q3 可行证据指向不同，故不据此作全局最低声明。"
            )
            abstract_family_note = family_note
        decision_text = (
            f"{family_note}另一个独立随机流已确认更便宜的全 A 设计 $(616,0)$ 统计可行，"
            f"其成本为 ${q3_cost}$ 元。两个随机流对应不同的预注册置信家族，不能把两项"
            "点态可行证据拼接成全局最优性证明。"
        )
        conclusion = (
            f"问题四的独立流确认配比 $(N_A,N_B)=({n_a},{n_b})$ 统计可行，成本为 "
            f"${cost}$ 元。{family_note}问题三的独立流同时确认更便宜的 $(616,0)$ 可行、"
            f"成本为 ${q3_cost}$ 元。综合两组证据，当前不宣称全局最低成本。"
        )
        abstract = (
            f"\\textbf{{针对问题四，}}以二维 minimax 非支配路径前沿压缩成本有界整数域，"
            f"并用 {trials} 个独立样本作 Bonferroni--Clopper--Pearson 确认。配比 "
            f"$({n_a},{n_b})$ 导通 ${successes}$ 次，经验概率为 $\\mathbf{{{estimate}}}$、校正后"
            f"单侧下限为 $\\mathbf{{{cp_lower}}}$，成本为 $\\mathbf{{{cost}}}$ 元。"
            f"{abstract_family_note}另一独立流确认更便宜的 $(616,0)$ 可行；结合该证据，"
            f"当前保守可行成本上界为 $\\mathbf{{{q3_cost}}}$ 元，并将上述 "
            f"\\textbf{{{not_excluded} 个}}未决极大点保留为待确认的更低成本边界集。"
        )
        evidence_label = "Q4 预注册家族内的统计可行证据"
    else:
        decision_text = (
            f"该探索候选的单侧 Clopper--Pearson 下限未达到 $90\\%$，因此不能把"
            f"$({n_a},{n_b})$ 作为已确认可行方案，更不能报告为最低成本。问题三的另一个"
            f"独立随机流确认全 A 设计 $(616,0)$ 统计可行，其成本为 ${q3_cost}$ 元；"
            "该交叉证据只能给出当前已确认可行的成本上界，尚不能确定 Q4 的最低成本。"
        )
        conclusion = (
            f"问题四的预注册候选 $(N_A,N_B)=({n_a},{n_b})$ 未通过独立确认；"
            f"问题三独立流确认 $(616,0)$ 可行、成本为 ${q3_cost}$ 元。故当前仅报告"
            "这一可行成本上界，不宣称问题四的最低成本。"
        )
        abstract = (
            f"\\textbf{{针对问题四，}}以二维 minimax 非支配路径前沿压缩成本有界整数域，"
            f"并用 {trials} 个独立样本作 Bonferroni--Clopper--Pearson 确认。预注册候选 "
            f"$({n_a},{n_b})$ 未通过可行性下限检验；另一独立流确认 $(616,0)$ 可行、"
            f"成本为 $\\mathbf{{{q3_cost}}}$ 元。本文据此报告可行成本上界，不作最低成本声明。"
        )
        evidence_label = "Q4 预注册候选未通过独立确认"

    result = rf"""
在 ${trials}$ 个独立确认样本中，代表性设计
$(N_A,N_B)=({n_a},{n_b})$ 导通 ${successes}$ 次，经验导通概率为 ${estimate}$；
其 Bonferroni 校正后的单侧 Clopper--Pearson 下限为 ${cp_lower}$、上限为
${cp_upper}$，成本为 ${cost}$ 元。{decision_text}
  同一整数成本权重对应 {equal_cost_count} 个配比；本文只报告已确认统计可行的方案，
  不作组成唯一性或全局最优性声明。
正式统计结果见表 \ref{{tab:q4-confirmation}}；成本有界整数域的探索结构见
图 \ref{{fig:q4-cost-frontier}}，独立确认判定另见第 6.3 节。

```{{=latex}}
\begin{{table}}[htbp]
  \centering
  \small
  \caption{{问题四正式独立确认结果（各行按对应预注册协议计算精确限）}}
  \label{{tab:q4-confirmation}}
  \begin{{tabularx}}{{\textwidth}}{{c>{{\raggedright\arraybackslash}}Xccccc}}
    \toprule
    配比 & 证据角色 & 成功数 & 经验概率 & 协议内 CP 下限 & 协议内 CP 上限 & 成本（元） \\
    \midrule
    $({n_a},{n_b})$ & {evidence_label} & ${successes}/{trials}$ & ${estimate}$ & ${cp_lower}$ & ${cp_upper}$ & ${cost}$ \\
    $(616,0)$ & 问题三独立流的交叉可行证据 & ${q3_successes}/{q3_trials}$ & ${q3_estimate}$ & ${q3_cp_lower}$ & ${q3_cp_upper}$ & ${q3_cost}$ \\
    \bottomrule
  \end{{tabularx}}
\end{{table}}
```
"""
    frontier = rf"""
```{{=latex}}
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{{figures/generated/q4_cost_frontier.pdf}}
  \caption{{问题四成本有界整数域的探索结构。左图给出完整整数域的经验导通概率、$90\%$ 等值线和候选等成本线；右图给出严格更便宜极大前沿的成本--经验概率分布。两图均基于每点 $2000$ 次探索样本，仅用于候选定位，不承担独立确认结论。}}
  \label{{fig:q4-cost-frontier}}
\end{{figure}}
```
"""
    return {
        "result": result.strip(),
        "frontier": frontier.strip(),
        "conclusion": conclusion,
        "abstract": abstract,
    }


FIGURE_Q4_3D = r"""
```{=latex}
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.72\textwidth]{figures/rendered/q4_final_na000619_nb000000_trial000001_axonometric.png}
  \caption{问题四确认流内已确认统计可行方案 $(619,0)$ 的 FreeCAD 全局轴测图。深灰面为左右电极，浅灰实体为 A 介质截断片段，橙色实体为程序恢复的贯通见证；该图只展示方案的空间结构。}
  \label{fig:q4-global-3d}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{figures/rendered/q4_final_na000619_nb000000_trial000001_witness_focus.pdf}
  \caption{问题四确认流内统计可行方案的局部三维贯通见证与有序接触链。橙色实体为见证片段，青色虚线为接触图边，不表示连续空间电流轨迹；链条按 $L\to1\to\cdots\to10\to R$ 给出 11 条经几何核验的接触边，同源内部边为 0。}
  \label{fig:q4-witness-3d}
\end{figure}
\FloatBarrier
```
"""


FINAL_Q4 = r"""
问题四的 $2000$ 样本成本有界域预筛选将 $(619,0)$ 冻结为保守确认候选，其探索样本单侧精确下限为 $90.1403\%$、成本为 $9.1884516534$ 元。在 $50000$ 个独立确认样本的成本有界域结果回填前，本文不宣称该点为全局最低成本。
"""


SOURCE_APPENDIX = r"""
```{=latex}
\input{source_appendix.tex}
```
"""


REPRODUCTION_COMMANDS = r"""
```{=latex}
\begin{itemize}
  \item \texttt{python -B -m pytest -q -p no:cacheprovider}
  \item \texttt{python run\_pipeline.py --from-step 1 --to-step 4 --dry-run}
  \item \texttt{python run\_pipeline.py --from-step 1 --to-step 4}
\end{itemize}
```

项目级流水线从 `项目配置.json` 读取各问正式运行组，依次执行问题一，问题二的 D 主结果与
A/B 边界敏感性，问题三独立确认，以及问题四的筛选、冻结和确认。随机种子、stream ID、
样本量、分批大小和输出目录的完整参数列表同时保存在支撑材料的复现说明中。
"""


def build_markdown(
    mode: str = "internal",
    evidence: dict[str, Any] | None = None,
    *,
    review: bool = False,
) -> str:
    if mode not in {"internal", "final"}:
        raise ValueError(f"unknown LaTeX build mode: {mode}")
    if review and mode != "internal":
        raise ValueError("便阅构建仅允许 internal 模式")
    if evidence is not None:
        q4_blocks = build_final_q4_blocks(evidence)
    elif mode == "final":
        raise ValueError("final mode requires validated Q4 evidence")
    else:
        q4_blocks = {
            "result": Q4_INTERIM.strip(),
            "frontier": "",
            "conclusion": FINAL_Q4.strip(),
        }
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    text = "\n".join(lines).lstrip()
    text = text.replace("<!-- 首页摘要在 Q3/Q4 结果冻结后生成。 -->", "")
    source_appendix_block = "### 附录 C：主要程序\n\n<!-- SOURCE_CODE_APPENDIX -->"
    if review:
        if text.count(source_appendix_block) != 1:
            raise RuntimeError("便阅构建无法唯一定位附录 C 标题与源码槽位")
        text = text.replace(source_appendix_block, "")
    replacements = {
        "<!-- FIGURE_Q1_GROUPS -->": FIGURE_Q1,
        "<!-- FIGURE_Q2_Q3_CURVE -->": FIGURE_Q2_Q3,
        "<!-- Q4_FINAL_RESULT -->": q4_blocks["result"],
        "<!-- FIGURE_Q4_FRONTIER -->": q4_blocks["frontier"],
        "<!-- FIGURE_Q4_3D -->": FIGURE_Q4_3D,
        "<!-- FINAL_Q3_Q4_CONCLUSION -->": q4_blocks["conclusion"],
    }
    if not review:
        replacements["<!-- SOURCE_CODE_APPENDIX -->"] = SOURCE_APPENDIX
    for marker, replacement in replacements.items():
        if marker not in text:
            raise RuntimeError(f"missing marker: {marker}")
        text = text.replace(marker, replacement.strip())

    appendix_transition = "```{=latex}\n\\clearpage\n```\n\n## 附录"
    if text.count(appendix_transition) != 1:
        raise RuntimeError("论文附录另起页标记缺失或不唯一")
    appendix_open = r"""```{=latex}
\endgroup
\clearpage
\TemplateMatterHeading{附录}
```"""
    if review:
        appendix_open = appendix_open.replace(
            r"\TemplateMatterHeading{附录}",
            "\\TemplateMatterHeading{附录}\n\\begingroup\n\\small",
        )
    text = text.replace(appendix_transition, appendix_open)
    matter_headings = {
        "## AI 工具使用声明": r"""```{=latex}
\TemplateMatterHeading{AI 工具使用声明}
```""",
        "## 参考文献": r"""```{=latex}
\TemplateMatterHeading{参考文献}
\begingroup
\zihao{5}
```""",
    }
    for heading, replacement in matter_headings.items():
        if text.count(heading) != 1:
            raise RuntimeError(f"模板式后置标题缺失或不唯一: {heading}")
        text = text.replace(heading, replacement)

    text = text.replace(
        "并在完整整数设计域中选择",
        "并在覆盖全部不高于候选成本设计的成本有界整数域中选择",
    )
    text = text.replace(
        "从而精确回答整个整数\n设计域",
        "从而在覆盖全部不高于候选成本设计的成本有界整数域内精确回答概率查询",
    )
    text = text.replace(r"\mathrm{元}", r"\text{元}")
    text = re.sub(
        r"```powershell\s+.*?\s+```",
        lambda _: REPRODUCTION_COMMANDS.strip(),
        text,
        count=1,
        flags=re.DOTALL,
    )
    if review:
        text = text.rstrip() + "\n\n```{=latex}\n\\endgroup\n```"
    return text.rstrip() + "\n"


def escape_latex(text: str) -> str:
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
    return "".join(replacements.get(char, char) for char in text)


def normalized_project_file(raw: Any, label: str) -> tuple[Path, str]:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"{label}必须为 POSIX 风格的项目相对路径")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"{label}不是安全的项目相对路径：{raw}")
    path = require_inside(PROJECT_DIR / Path(*relative.parts), PROJECT_DIR, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    return path, relative.as_posix()


def load_final_submission_sources(
    allowlist: dict[str, Any], entries: Sequence[dict[str, Any]]
) -> list[tuple[dict[str, Any], Path, str]]:
    manifest_path = PROJECT_DIR / "提交源码" / "source-manifest.json"
    manifest = read_json_object(manifest_path, "提交源码清单")
    expected_allowlist = SOURCE_ALLOWLIST.relative_to(PROJECT_DIR).as_posix()
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("status") != "frozen"
        or manifest.get("submission_root") != "提交源码"
        or manifest.get("allowlist") != expected_allowlist
        or str(manifest.get("allowlist_sha256", "")).upper() != sha256(SOURCE_ALLOWLIST)
    ):
        raise ValueError("提交源码清单与冻结白名单不一致")

    headers = [str(line) for line in allowlist.get("ai_header_lines", [])]
    if len(headers) != 2 or manifest.get("ai_header_lines") != headers:
        raise ValueError("提交源码清单中的 AI 注释头与冻结白名单不一致")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(entries):
        raise ValueError("提交源码清单的程序数量与冻结白名单不一致")

    prepared: list[tuple[dict[str, Any], Path, str]] = []
    seen_sources: set[str] = set()
    seen_submissions: set[str] = set()
    for index, (entry, item) in enumerate(zip(entries, files, strict=True), start=1):
        if not isinstance(entry, dict) or not isinstance(item, dict):
            raise ValueError(f"源码白名单或提交清单第 {index} 项不是对象")
        source_path, source_relative = normalized_project_file(
            item.get("source_path"), f"原源码第 {index} 项"
        )
        allowlist_relative = str(entry.get("path") or "")
        if source_relative != allowlist_relative or source_relative in seen_sources:
            raise ValueError(f"提交源码清单第 {index} 项与白名单路径或顺序不一致")
        seen_sources.add(source_relative)
        if (
            sha256(source_path) != str(item.get("source_sha256", "")).upper()
            or source_path.stat().st_size != int(item.get("source_bytes", -1))
        ):
            raise ValueError(f"冻结后原源码发生变化：{source_relative}")

        submission_path, submission_relative = normalized_project_file(
            item.get("submission_path"), f"提交源码第 {index} 项"
        )
        expected_submission = f"提交源码/{source_relative}"
        if submission_relative != expected_submission or submission_relative in seen_submissions:
            raise ValueError(f"提交源码清单第 {index} 项的副本路径不一致")
        seen_submissions.add(submission_relative)
        if (
            sha256(submission_path) != str(item.get("submission_sha256", "")).upper()
            or submission_path.stat().st_size != int(item.get("submission_bytes", -1))
        ):
            raise ValueError(f"提交源码副本哈希或字节数不一致：{submission_relative}")
        if str(item.get("title") or "") != str(entry.get("title") or ""):
            raise ValueError(f"提交源码清单第 {index} 项标题与白名单不一致")

        raw = submission_path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"提交源码副本不得含 UTF-8 BOM：{submission_relative}")
        code = raw.decode("utf-8")
        if code.splitlines()[:2] != headers:
            raise ValueError(f"提交源码副本缺少精确的两行 AI 注释头：{submission_relative}")
        if PRIVATE_HOME_PATTERN.search(code):
            raise ValueError(f"提交源码副本含用户主目录绝对路径：{submission_relative}")
        prepared.append((entry, source_path, code))
    return prepared


def build_source_appendix(mode: str = "internal") -> None:
    config = json.loads(SOURCE_ALLOWLIST.read_text(encoding="utf-8"))
    if mode == "final" and config.get("status") != "frozen":
        raise RuntimeError("正式 LaTeX 源码附录白名单尚未冻结")
    entries = config.get("files", [])
    if not entries:
        raise RuntimeError("source appendix allowlist is empty")
    appendix_entries = [entry for entry in entries if entry.get("include_in_appendix", True)]
    if not appendix_entries:
        raise RuntimeError("source appendix selection is empty")
    headers = [str(line) for line in config.get("ai_header_lines", [])]
    SOURCE_APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

    submission_manifest = PROJECT_DIR / "提交源码" / "source-manifest.json"
    use_frozen_submission = config.get("status") == "frozen" and submission_manifest.is_file()

    if mode == "final":
        appendix_notice = (
            r"\noindent\textbf{源码生成说明：} "
            r"本附录由冻结白名单自动生成，共列入 "
            + str(len(appendix_entries))
            + r" 个程序；程序版本与提交源码清单逐项核验。\par"
        )
    elif use_frozen_submission:
        appendix_notice = (
            r"\noindent\textbf{源码生成说明：} "
            r"本附录由冻结提交源码清单自动生成，共列入 "
            + str(len(appendix_entries))
            + r" 个程序；代码内容与正式提交源码逐字节一致。"
            r"当前仅因组别与参赛编号未填写而标记为 INTERNAL-QA。\par"
        )
    else:
        appendix_notice = (
            r"\noindent\textbf{源码生成说明：} "
            r"本附录由 \texttt{source\_appendix\_allowlist.json} 自动生成，"
            r"共列入 "
            + str(len(appendix_entries))
            + r" 个白名单程序。当前为 INTERNAL-QA 排版副本，"
            r"不修改任何原始求解程序；正式提交时再与冻结源码清单哈希逐项对账。\par"
        )
    tex_lines = [appendix_notice, r"\vspace{0.4em}"]

    prepared: list[tuple[dict[str, Any], Path, str]] = []
    if mode == "final" or use_frozen_submission:
        prepared_all = load_final_submission_sources(config, entries)
        prepared = [
            item for item in prepared_all if item[0].get("include_in_appendix", True)
        ]
    else:
        for entry in appendix_entries:
            source_path, _ = normalized_project_file(entry.get("path"), "源码白名单路径")
            code = source_path.read_text(encoding="utf-8")
            for replacement in entry.get("replacements", []):
                old = str(replacement["old"])
                new = str(replacement["new"])
                if old not in code:
                    raise RuntimeError(
                        f"allowlisted replacement missing in {entry.get('path')}: {old}"
                    )
                code = code.replace(old, new)
            if headers and not code.startswith(headers[0]):
                code = "\n".join(headers) + "\n" + code.lstrip("\ufeff")
            prepared.append((entry, source_path, code))

    for stale in SOURCE_APPENDIX_DIR.glob("*.*"):
        if stale.is_file():
            stale.unlink()

    for index, (entry, source_path, code) in enumerate(prepared, start=1):
        relative_path = Path(str(entry["path"]))
        for forbidden in entry.get("forbidden_terms", []):
            if str(forbidden) in code:
                raise RuntimeError(f"forbidden appendix term remains in {relative_path}: {forbidden}")
        if mode == "final" or use_frozen_submission:
            for pattern in PROCESS_NARRATIVE_PATTERNS:
                match = pattern.search(code)
                if match:
                    raise RuntimeError(
                        "正式源码附录含题目修改或澄清过程叙事："
                        f"{relative_path}（{match.group(0)}）"
                    )
        output_name = f"{index:03d}{source_path.suffix.lower()}"
        output_path = SOURCE_APPENDIX_DIR / output_name
        output_path.write_bytes(code.encode("utf-8"))

        title = escape_latex(str(entry["title"]))
        display_path = escape_latex(relative_path.as_posix())
        input_path = f"source_appendix/{output_name}"
        tex_lines.extend(
            [
                r"\Needspace{6\baselineskip}",
                rf"\subsection{{{title}}}",
                rf"\noindent{{}}原项目相对路径：\texttt{{{display_path}}}\par",
                r"\vspace{0.25em}",
                r"{\fontsize{6.4pt}{7.2pt}\selectfont",
                r"\VerbatimInput[breaklines=true,breakanywhere=true,numbers=left,numbersep=5pt,frame=single,framesep=2mm,rulecolor=\color{black!35}]{"
                + input_path
                + "}",
                r"}",
            ]
        )
    SOURCE_APPENDIX_TEX.write_text("\n".join(tex_lines) + "\n", encoding="utf-8", newline="\n")


def write_build_meta(
    mode: str, group: str, competition_id: str, q4_abstract: str | None
) -> None:
    if mode == "internal":
        notice = r"""\newcommand{\PaperBuildNotice}{%
\begin{center}
  {\color{danger}\small\bfseries INTERNAL-QA：参赛组别与编号尚未填写，本文件不得用于正式提交。}
\end{center}}
"""
        abstract = q4_abstract or (
            r"\textbf{针对问题四，}在成本有界整数域内建立二维 minimax 路径前沿模型；"
            r"未加载通过审计的最终证据时，不输出数值性最低成本结论。"
        )
    else:
        notice = r"\newcommand{\PaperBuildNotice}{}" + "\n"
        if not q4_abstract:
            raise ValueError("final mode requires a Q4 abstract")
        abstract = q4_abstract
    lines = [
        rf"\newcommand{{\PaperGroup}}{{{escape_latex(group)}}}",
        rf"\newcommand{{\PaperCompetitionId}}{{{escape_latex(competition_id)}}}",
        notice.rstrip(),
        rf"\newcommand{{\PaperQFourAbstract}}{{{abstract}}}",
    ]
    BUILD_META_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def scan_final_sources(paths: Sequence[Path]) -> None:
    failures: list[str] = []
    unresolved_marker = re.compile(
        r"<!--\s*(?:FIGURE|Q4|FINAL|SOURCE)[A-Z0-9_\- ]*-->", re.IGNORECASE
    )
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        for term, reason in FINAL_FORBIDDEN_TERMS.items():
            if term in text:
                failures.append(f"{path.name}: {reason}（{term}）")
        if unresolved_marker.search(text):
            failures.append(f"{path.name}: 仍含未解析内容或图件槽位")
        for pattern in PROCESS_NARRATIVE_PATTERNS:
            match = pattern.search(text)
            if match:
                failures.append(
                    f"{path.name}: 含题目修改/澄清过程叙事（{match.group(0)}）"
                )
    if failures:
        raise RuntimeError("正式 LaTeX 内容门禁失败：\n- " + "\n- ".join(failures))


def compile_pdf(
    mode: str, competition_id: str, output_dir: Path, *, review: bool = False
) -> Path:
    latexmk = shutil.which("latexmk")
    if not latexmk:
        raise RuntimeError("latexmk not found")
    build_dir_name = (
        "latex_internal_review"
        if review
        else "latex_internal" if mode == "internal" else f"latex_{competition_id}"
    )
    build_dir = GENERATED_DIR / build_dir_name
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            latexmk,
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={build_dir}",
            "main.tex",
        ],
        check=True,
        cwd=PAPER_DIR,
    )
    built = build_dir / "main.pdf"
    if not built.is_file() or built.stat().st_size == 0:
        raise RuntimeError("XeLaTeX 未生成有效 PDF")
    output_dir = require_inside(output_dir, PROJECT_DIR, "论文 PDF 输出目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        "A题论文_便阅稿.pdf"
        if review
        else (
            "A题论文.pdf"
            if mode == "internal"
            else f"A{competition_id}.pdf"
        )
    )
    output = output_dir / filename
    shutil.copy2(built, output)
    return output


def verify_anonymous_pdf_metadata(path: Path) -> None:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf not found; cannot audit formal PDF metadata") from exc
    metadata = PdfReader(str(path)).metadata or {}
    nonempty = {
        key: str(metadata.get(key, "")).strip()
        for key in ("/Author", "/Subject", "/Keywords")
        if str(metadata.get(key, "")).strip()
    }
    if nonempty:
        raise RuntimeError(f"正式 PDF 匿名元数据门禁失败：{nonempty}")


def write_manifest(
    mode: str,
    group: str,
    competition_id: str,
    output_pdf: Path | None,
    evidence: dict[str, Any] | None,
    *,
    review: bool = False,
) -> Path:
    manifest = {
        "schema_version": 1,
        "kind": "latex_paper_build_manifest",
        "mode": mode,
        "variant": "review_without_source_appendix" if review else "complete",
        "formal_submission_allowed": mode == "final" and output_pdf is not None,
        "group": group,
        "competition_id": competition_id,
        "anonymous_pdf_metadata": mode != "final" or output_pdf is not None,
        "source_markdown": SOURCE.relative_to(PROJECT_DIR).as_posix(),
        "main_tex": (PAPER_DIR / "main.tex").relative_to(PROJECT_DIR).as_posix(),
        "generated_content_tex": LATEX_OUT.relative_to(PROJECT_DIR).as_posix(),
        "output_pdf": (
            None if output_pdf is None else output_pdf.relative_to(PROJECT_DIR).as_posix()
        ),
        "output_pdf_sha256": None if output_pdf is None else sha256(output_pdf),
        "q4_final_evidence": (
            None
            if evidence is None
            else {
                "summary": evidence["summary_path"].relative_to(PROJECT_DIR).as_posix(),
                "summary_sha256": evidence["summary_sha256"],
                "independent_analysis": evidence["analysis_path"]
                .relative_to(PROJECT_DIR)
                .as_posix(),
                "independent_analysis_sha256": evidence["analysis_sha256"],
                "q3_cross_evidence": evidence["q3_summary_path"]
                .relative_to(PROJECT_DIR)
                .as_posix(),
                "frontier_audit": evidence["frontier_audit_path"]
                .relative_to(PROJECT_DIR)
                .as_posix(),
                "result_status": evidence["summary"]["result_status"],
            }
        ),
    }
    name = (
        "latex_internal_review_build_manifest.json"
        if review
        else "latex_internal_build_manifest.json"
        if mode == "internal"
        else "latex_final_build_manifest.json"
    )
    path = GENERATED_DIR / name
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从数学建模论文 Markdown 生成带严格正式化门禁的 XeLaTeX PDF"
    )
    parser.add_argument("--mode", choices=("internal", "final"), default="internal")
    parser.add_argument("--group", choices=sorted(ALLOWED_GROUPS))
    parser.add_argument("--competition-id")
    parser.add_argument("--q4-summary", type=Path, default=DEFAULT_Q4_SUMMARY)
    parser.add_argument("--q4-analysis", type=Path, default=DEFAULT_Q4_ANALYSIS)
    parser.add_argument("--q3-summary", type=Path, default=DEFAULT_Q3_SUMMARY)
    parser.add_argument("--q4-frontier-pdf", type=Path, default=DEFAULT_Q4_FRONTIER_PDF)
    parser.add_argument("--q4-frontier-png", type=Path, default=DEFAULT_Q4_FRONTIER_PNG)
    parser.add_argument(
        "--q4-frontier-audit", type=Path, default=DEFAULT_Q4_FRONTIER_AUDIT
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER_DIR)
    parser.add_argument(
        "--review",
        action="store_true",
        help="生成省略完整源码附录的 internal 便阅版，并同时移除附录 C 标题",
    )
    parser.add_argument(
        "--no-pdf", action="store_true", help="只生成并门禁 TeX 中间件，不运行 XeLaTeX"
    )
    return parser.parse_args(argv)


def snapshot_files(paths: Sequence[Path]) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in paths}


def restore_files(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(content)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.review and args.mode != "internal":
        raise ValueError("--review 仅允许与 --mode internal 同时使用")
    group, competition_id = validate_identity(
        args.mode, args.group, args.competition_id
    )
    validate_reference_doi_consistency()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    evidence = validate_final_evidence(
        args.q4_summary,
        args.q4_analysis,
        args.q3_summary,
        args.q4_frontier_pdf,
        args.q4_frontier_png,
        args.q4_frontier_audit,
    )
    if not args.review:
        build_source_appendix(args.mode)
    final_blocks = None if evidence is None else build_final_q4_blocks(evidence)
    write_build_meta(
        args.mode,
        group,
        competition_id,
        None if final_blocks is None else final_blocks["abstract"],
    )
    canonical_snapshot = (
        snapshot_files((MARKDOWN_OUT, LATEX_OUT)) if args.review else None
    )
    output_pdf = None
    manifest = None
    try:
        MARKDOWN_OUT.write_text(
            build_markdown(args.mode, evidence, review=args.review),
            encoding="utf-8",
            newline="\n",
        )
        pandoc = shutil.which("pandoc")
        if not pandoc:
            raise RuntimeError("pandoc not found")
        subprocess.run(
            [
                pandoc,
                str(MARKDOWN_OUT),
                "--from=markdown+raw_tex",
                "--to=latex",
                "--top-level-division=section",
                "--shift-heading-level-by=-1",
                "--wrap=none",
                "--no-highlight",
                f"--output={LATEX_OUT}",
            ],
            check=True,
            cwd=PAPER_DIR,
        )
        if args.mode == "final":
            scan_final_sources(
                (
                    PAPER_DIR / "main.tex",
                    BUILD_META_TEX,
                    MARKDOWN_OUT,
                    LATEX_OUT,
                    SOURCE_APPENDIX_TEX,
                )
            )
        if not args.no_pdf:
            output_pdf = compile_pdf(
                args.mode, competition_id, args.output_dir, review=args.review
            )
            if args.mode == "final":
                verify_anonymous_pdf_metadata(output_pdf)
        manifest = write_manifest(
            args.mode,
            group,
            competition_id,
            output_pdf,
            evidence,
            review=args.review,
        )
    finally:
        if canonical_snapshot is not None:
            restore_files(canonical_snapshot)
    if manifest is None:
        raise RuntimeError("LaTeX 构建清单未生成")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "variant": "review_without_source_appendix" if args.review else "complete",
                "tex": str(LATEX_OUT),
                "pdf": None if output_pdf is None else str(output_pdf),
                "manifest": str(manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
