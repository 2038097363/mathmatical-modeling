from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "论文" / "src" / "build_latex.py"
SPEC = importlib.util.spec_from_file_location("build_latex", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
latex = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latex)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_evidence_tree(
    root: Path, status: str, *, production_like: bool = False
) -> tuple[Path, Path, Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    candidate_feasible = status != "screening_candidate_not_confirmed"
    all_excluded = status == "globally_certified_minimum_cost"
    if production_like and candidate_feasible:
        successes = 45554
        estimate = 0.91108
        cp_lower = 0.9061863570173373
        cp_upper = 0.9158116422756992
    else:
        successes = 45200 if candidate_feasible else 44000
        estimate = 0.904 if candidate_feasible else 0.88
        cp_lower = 0.901 if candidate_feasible else 0.876
        cp_upper = 0.907 if candidate_feasible else 0.884
    candidate = {
        "role": "candidate",
        "n_a": 619,
        "n_b": 0,
        "cost_weight": 350973,
        "cost_yuan": 9.1884516534,
        "successes": successes,
        "trials": 50000,
        "estimate": estimate,
        "clopper_pearson_one_sided_lower": cp_lower,
        "clopper_pearson_one_sided_upper": cp_upper,
        "proof_status": (
            "candidate_statistically_feasible"
            if candidate_feasible
            else "candidate_not_confirmed"
        ),
    }
    frontier = {
        "role": "strictly_cheaper_maximal",
        "n_a": 618,
        "n_b": 8,
        "proof_status": (
            "strictly_cheaper_design_excluded"
            if all_excluded
            else "strictly_cheaper_design_not_excluded"
        ),
    }
    freeze = {
        "kind": "q4_confirmation_freeze",
        "confirmation_protocol": {
            "fixed_trial_count": 50000,
            "screening_stream_id": 4,
            "confirmation_stream_id": 5,
            "stream_ids_distinct": True,
            "configuration": {"trial_count": 50000, "stream_id": 5},
            "bonferroni_statement_count": 2,
            "familywise_confidence": 0.95,
        },
    }
    freeze_path = root / "freeze.json"
    write_json(freeze_path, freeze)
    summary = {
        "kind": "q4_final_summary",
        "result_status": status,
        "boundary_contract": {"mode": "D"},
        "candidate_statistically_feasible": candidate_feasible,
        "all_strictly_cheaper_maximal_designs_excluded": all_excluded,
        "excluded_frontier_count": (
            573
            if production_like and status == "lowest_statistically_feasible_cost"
            else (1 if all_excluded else 0)
        ),
        "not_excluded_frontier_count": (
            46
            if production_like and status == "lowest_statistically_feasible_cost"
            else (0 if all_excluded else 1)
        ),
        "reported_design": (
            {
                "n_a": 619,
                "n_b": 0,
                "cost_weight": 350973,
                "cost_yuan": 9.1884516534,
            }
            if candidate_feasible
            else None
        ),
        "cost_uncertainty_interval": (
            {
                "lower_cost_yuan": (
                    8.505626490145346 if production_like else 9.14
                ),
                "upper_cost_yuan": 9.1884516534,
            }
            if status == "lowest_statistically_feasible_cost"
            else None
        ),
        "equal_cost_designs": [[619, 0]],
        "confirmation_records": [candidate, frontier],
        "freeze_path": str(freeze_path),
        "freeze_sha256": latex.sha256(freeze_path),
    }
    summary_path = root / "q4_summary.json"
    write_json(summary_path, summary)

    analysis = {
        "kind": "q4_confirmation_integer_domain_analysis",
        "audit_status": "passed",
        "result_status": status,
        "configuration": {
            "trial_count": 50000,
            "stream_id": 5,
            "integer_domain_shape": [620, 5484],
        },
        "input_files": {
            "freeze": {"sha256": latex.sha256(freeze_path)},
            "final_summary": {"sha256": latex.sha256(summary_path)},
        },
        "statistical_results": {
            "candidate": candidate,
            "candidate_statistically_feasible": candidate_feasible,
            "all_strictly_cheaper_maximal_designs_excluded": all_excluded,
            "q3_reference_design_in_confirmation_stream": {
                "n_a": 616,
                "n_b": 0,
                "cost_yuan": 9.1439195775,
            },
        },
    }
    analysis_path = root / "analysis.json"
    write_json(analysis_path, analysis)
    q3_summary = {
        "result_scope": "independent_fixed_confirmation",
        "fixed_trial_count": 50000,
        "candidate_records": [
            {
                "particle_count": 616,
                "successes": 45226,
                "trials": 50000,
                "estimate": 0.90452,
                "clopper_pearson_one_sided_bounds": {
                    "lower": 0.9008213434,
                    "upper": 0.9081299842,
                },
                "classification_by_bonferroni_cp": "statistically_feasible",
            }
        ],
    }
    q3_path = root / "q3_summary.json"
    write_json(q3_path, q3_summary)

    pdf_path = root / "q4_cost_frontier.pdf"
    png_path = root / "q4_cost_frontier.png"
    pdf_path.write_bytes(b"%PDF-1.4\nsynthetic\n")
    png_path.write_bytes(b"synthetic-png")
    figure_audit = {
        "kind": "q4_cost_frontier_figure_audit",
        "final_sha256": latex.sha256(summary_path),
        "freeze_sha256": latex.sha256(freeze_path),
        "output_pdf": str(pdf_path),
        "output_pdf_sha256": latex.sha256(pdf_path),
        "output_png": str(png_path),
        "output_png_sha256": latex.sha256(png_path),
        "result_status": status,
        "monotonicity": {"passed": True},
    }
    audit_path = root / "q4_cost_frontier.audit.json"
    write_json(audit_path, figure_audit)
    return summary_path, analysis_path, q3_path, pdf_path, png_path, audit_path


def test_identity_modes_and_formal_filename_contract() -> None:
    assert latex.validate_identity("internal", None, None) == (
        "内部审阅",
        "INTERNAL-QA",
    )
    assert latex.validate_identity("final", "本科生", "CM1234567") == (
        "本科生",
        "CM1234567",
    )
    with pytest.raises(ValueError, match="CM"):
        latex.validate_identity("final", "本科生", "1234567")
    with pytest.raises(ValueError, match="组别"):
        latex.validate_identity("final", None, "CM1234567")


@pytest.mark.parametrize(
    "status",
    [
        "globally_certified_minimum_cost",
        "lowest_statistically_feasible_cost",
        "screening_candidate_not_confirmed",
    ],
)
def test_final_gate_accepts_all_honest_solver_outcomes(tmp_path: Path, status: str) -> None:
    summary, analysis, q3, pdf, png, audit = make_evidence_tree(tmp_path, status)
    evidence = latex.validate_final_evidence(
        summary,
        analysis,
        q3,
        pdf,
        png,
        audit,
        project_root=tmp_path,
    )
    blocks = latex.build_final_q4_blocks(evidence)
    combined = "\n".join(blocks.values())
    assert "(616,0)" in combined
    if status == "screening_candidate_not_confirmed":
        assert "未通过独立确认" in combined
        assert "不作最低成本声明" in combined
    else:
        assert "统计可行" in combined or "最低成本" in combined
    if status == "lowest_statistically_feasible_cost":
        assert "不宣称全局最低成本" in combined


def test_final_gate_binds_independent_analysis_to_summary(tmp_path: Path) -> None:
    summary, analysis, q3, pdf, png, audit = make_evidence_tree(
        tmp_path, "lowest_statistically_feasible_cost"
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert "final_evidence_available" not in payload
    payload["excluded_frontier_count"] = 2
    write_json(summary, payload)
    with pytest.raises(ValueError, match="独立完整域审计与最终摘要 SHA-256"):
        latex.validate_final_evidence(
            summary,
            analysis,
            q3,
            pdf,
            png,
            audit,
            project_root=tmp_path,
        )


def test_final_gate_rejects_reused_confirmation_stream(tmp_path: Path) -> None:
    summary, analysis, q3, pdf, png, audit = make_evidence_tree(
        tmp_path, "lowest_statistically_feasible_cost"
    )
    freeze_path = tmp_path / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["confirmation_protocol"]["confirmation_stream_id"] = 4
    freeze["confirmation_protocol"]["configuration"]["stream_id"] = 4
    write_json(freeze_path, freeze)

    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    summary_payload["freeze_sha256"] = latex.sha256(freeze_path)
    write_json(summary, summary_payload)

    analysis_payload = json.loads(analysis.read_text(encoding="utf-8"))
    analysis_payload["configuration"]["stream_id"] = 4
    analysis_payload["input_files"]["freeze"]["sha256"] = latex.sha256(freeze_path)
    analysis_payload["input_files"]["final_summary"]["sha256"] = latex.sha256(summary)
    write_json(analysis, analysis_payload)

    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    audit_payload["freeze_sha256"] = latex.sha256(freeze_path)
    audit_payload["final_sha256"] = latex.sha256(summary)
    write_json(audit, audit_payload)

    with pytest.raises(ValueError, match="探索流与正式确认流相互独立"):
        latex.validate_final_evidence(
            summary,
            analysis,
            q3,
            pdf,
            png,
            audit,
            project_root=tmp_path,
        )


def test_final_text_scan_rejects_placeholders_and_process_narrative(tmp_path: Path) -> None:
    clean = tmp_path / "clean.tex"
    clean.write_text("纯模型、求解与结论。", encoding="utf-8")
    latex.scan_final_sources([clean])
    dirty = tmp_path / "dirty.tex"
    dirty.write_text("待回填；这里记录题目澄清过程。", encoding="utf-8")
    with pytest.raises(RuntimeError, match="门禁失败"):
        latex.scan_final_sources([dirty])


def test_project_reference_doi_sets_are_identical() -> None:
    dois = latex.validate_reference_doi_consistency()
    assert len(dois) == 8
    assert "10.1109/56.2083" in dois
    assert "10.1214/20-aos1991" in dois


def test_reference_doi_gate_accepts_entries_without_doi(tmp_path: Path) -> None:
    bib = tmp_path / "references.bib"
    markdown = tmp_path / "content.md"
    bib.write_text(
        """@article{paper,
  title = {Paper},
  doi = {10.1234/ABC.1}
}

@misc{official,
  title = {Official Rules}
}

@misc{ai,
  title = {AI Tool Record}
}
""",
        encoding="utf-8",
    )
    markdown.write_text(
        """## 参考文献

[1] Paper. DOI:10.1234/abc.1.

[2] Official Rules.

[3] AI Tool Record.

## 附录
""",
        encoding="utf-8",
    )
    assert latex.validate_reference_doi_consistency(bib, markdown) == frozenset(
        {"10.1234/abc.1"}
    )

    bib.write_text(
        "@article{paper, title={Paper}, doi=\"10.1234/ABC.1\"}\n"
        "@misc{official, title={Official Rules}}\n",
        encoding="utf-8",
    )
    assert latex.validate_reference_doi_consistency(bib, markdown) == frozenset(
        {"10.1234/abc.1"}
    )


@pytest.mark.parametrize(
    ("bib_doi", "manual_doi"),
    [
        ("10.1234/one.1", "10.1234/two.2"),
        ("10.1234/one.1", None),
        (None, "10.1234/one.1"),
    ],
)
def test_reference_doi_gate_rejects_set_mismatch(
    tmp_path: Path, bib_doi: str | None, manual_doi: str | None
) -> None:
    bib = tmp_path / "references.bib"
    markdown = tmp_path / "content.md"
    bib_field = "" if bib_doi is None else f"\n  doi = {{{bib_doi}}}"
    manual_field = "" if manual_doi is None else f" DOI:{manual_doi}."
    bib.write_text(
        f"@article{{paper,\n  title = {{Paper}}{bib_field}\n}}\n",
        encoding="utf-8",
    )
    markdown.write_text(
        f"## 参考文献\n\n[1] Paper.{manual_field}\n\n## 附录\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DOI 集合不一致"):
        latex.validate_reference_doi_consistency(bib, markdown)


def test_reference_doi_gate_rejects_duplicate_and_malformed_values(
    tmp_path: Path,
) -> None:
    duplicate_bib = tmp_path / "duplicate.bib"
    malformed_bib = tmp_path / "malformed.bib"
    valid_bib = tmp_path / "valid.bib"
    markdown = tmp_path / "content.md"
    malformed_markdown = tmp_path / "malformed-content.md"
    markdown.write_text(
        "## 参考文献\n\n[1] Paper. DOI:10.1234/one.1.\n\n## 附录\n",
        encoding="utf-8",
    )
    duplicate_bib.write_text(
        """@article{one,
  doi = {10.1234/one.1}
}
@article{two,
  doi = {10.1234/one.1}
}
""",
        encoding="utf-8",
    )
    malformed_bib.write_text(
        """@article{one,
  doi = {not-a-doi}
}
""",
        encoding="utf-8",
    )
    valid_bib.write_text(
        """@article{one,
  doi = {10.1234/one.1}
}
""",
        encoding="utf-8",
    )
    malformed_markdown.write_text(
        "## 参考文献\n\n[1] Paper. DOI:not-a-doi.\n\n## 附录\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复 DOI"):
        latex.validate_reference_doi_consistency(duplicate_bib, markdown)
    with pytest.raises(ValueError, match="不是有效 DOI"):
        latex.validate_reference_doi_consistency(malformed_bib, markdown)
    with pytest.raises(ValueError, match="畸形 DOI 标注"):
        latex.validate_reference_doi_consistency(valid_bib, malformed_markdown)


def test_main_fails_closed_before_writing_when_reference_gate_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = tmp_path / "generated"
    monkeypatch.setattr(latex, "GENERATED_DIR", generated)

    def reject_references(*args: object, **kwargs: object) -> frozenset[str]:
        raise ValueError("synthetic DOI mismatch")

    def forbidden_downstream(*args: object, **kwargs: object) -> None:
        pytest.fail("reference rejection must stop the build")

    monkeypatch.setattr(latex, "validate_reference_doi_consistency", reject_references)
    monkeypatch.setattr(latex, "validate_final_evidence", forbidden_downstream)
    monkeypatch.setattr(latex, "build_source_appendix", forbidden_downstream)
    monkeypatch.setattr(latex, "write_build_meta", forbidden_downstream)
    monkeypatch.setattr(latex, "build_markdown", forbidden_downstream)

    with pytest.raises(ValueError, match="synthetic DOI mismatch"):
        latex.main(["--mode", "internal", "--no-pdf"])
    assert not generated.exists()


def test_project_main_uses_generated_metadata_without_hardcoded_placeholders() -> None:
    main = (PROJECT_ROOT / "论文" / "main.tex").read_text(encoding="utf-8")
    assert r"\PaperGroup" in main
    assert r"\PaperCompetitionId" in main
    assert r"\PaperQFourAbstract" in main
    assert "待填写" not in main
    assert "INTERNAL-QA" not in main


def test_review_markdown_removes_source_appendix_heading_and_slot_together() -> None:
    complete = latex.build_markdown("internal")
    review = latex.build_markdown("internal", review=True)
    assert "### 附录 C：主要程序" in complete
    assert r"\input{generated/source_appendix.tex}" in complete
    assert "### 附录 C：主要程序" not in review
    assert "SOURCE_CODE_APPENDIX" not in review
    assert r"\input{generated/source_appendix.tex}" not in review
    assert "### 附录 B：软件、环境与复现命令" in review
    assert "\\begingroup\n\\small" in review
    assert review.rstrip().endswith("\\endgroup\n```")


def test_internal_review_can_render_validated_final_q4_evidence(tmp_path: Path) -> None:
    summary, analysis, q3, pdf, png, audit = make_evidence_tree(
        tmp_path, "lowest_statistically_feasible_cost"
    )
    evidence = latex.validate_final_evidence(
        summary,
        analysis,
        q3,
        pdf,
        png,
        audit,
        project_root=tmp_path,
    )
    review = latex.build_markdown("internal", evidence=evidence, review=True)
    assert "问题四正式独立确认结果" in review
    assert "图 \\ref{fig:q4-cost-frontier}" in review
    assert "50000 次独立确认样本的成本有界域结果回填前" not in review


def test_internal_metadata_uses_validated_q4_abstract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = tmp_path / "build_meta.tex"
    monkeypatch.setattr(latex, "BUILD_META_TEX", meta)
    q4_abstract = r"\\textbf{针对问题四，}正式证据摘要。"
    latex.write_build_meta("internal", "内部审阅", "INTERNAL-QA", q4_abstract)
    text = meta.read_text(encoding="utf-8")
    assert q4_abstract in text
    assert "50000 次独立确认尚未回填" not in text
    assert "参赛组别与编号尚未填写" in text


@pytest.mark.parametrize("review", [False, True])
def test_internal_modes_use_final_q4_values_and_exclude_interim_values(
    tmp_path: Path, review: bool
) -> None:
    summary, analysis, q3, pdf, png, audit = make_evidence_tree(
        tmp_path,
        "lowest_statistically_feasible_cost",
        production_like=True,
    )
    evidence = latex.validate_final_evidence(
        summary,
        analysis,
        q3,
        pdf,
        png,
        audit,
        project_root=tmp_path,
    )
    blocks = latex.build_final_q4_blocks(evidence)
    rendered = latex.build_markdown("internal", evidence=evidence, review=review)
    combined = blocks["abstract"] + "\n" + rendered

    for expected in (
        "45554/50000",
        r"91.108\%",
        r"90.6186\%",
        "573 个严格更低成本",
        "46 个未被同时排除",
        "[8.5056264901,9.1884516534]",
        "(616,0)",
        r"\textbf{573 个}",
        r"\textbf{46 个}",
        r"\mathbf{[8.5056264901,9.1884516534]}",
        "保守可行成本上界",
        "待确认的更低成本边界集",
    ):
        assert expected in combined
    for forbidden in (
        "1825/2000",
        r"91.25\%",
        r"90.1403\%",
        "尚未回填",
        "结果回填前",
        "最低统计可行成本",
        "最低成本统计证据",
    ):
        assert forbidden not in combined


@pytest.mark.parametrize("review", [False, True])
def test_internal_main_fails_closed_when_final_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, review: bool
) -> None:
    monkeypatch.setattr(latex, "GENERATED_DIR", tmp_path / "generated")

    def reject_evidence(*args: object, **kwargs: object) -> dict:
        raise ValueError("synthetic evidence rejection")

    def forbidden_downstream(*args: object, **kwargs: object) -> None:
        pytest.fail("evidence rejection must stop the review build")

    monkeypatch.setattr(latex, "validate_final_evidence", reject_evidence)
    monkeypatch.setattr(latex, "build_final_q4_blocks", forbidden_downstream)
    monkeypatch.setattr(latex, "build_source_appendix", forbidden_downstream)
    monkeypatch.setattr(latex, "write_build_meta", forbidden_downstream)
    monkeypatch.setattr(latex, "build_markdown", forbidden_downstream)
    monkeypatch.setattr(latex, "compile_pdf", forbidden_downstream)

    argv = ["--mode", "internal", "--no-pdf"]
    if review:
        argv.append("--review")
    with pytest.raises(ValueError, match="synthetic evidence rejection"):
        latex.main(argv)


@pytest.mark.parametrize("pandoc_fails", [False, True])
def test_internal_review_main_passes_evidence_and_restores_canonical_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pandoc_fails: bool,
) -> None:
    paper = tmp_path / "paper"
    generated = paper / "generated"
    markdown_out = generated / "content_for_latex.md"
    latex_out = generated / "content.tex"
    generated.mkdir(parents=True)
    markdown_out.write_bytes(b"canonical-markdown")
    latex_out.write_bytes(b"canonical-latex")

    evidence = {"kind": "validated-final-evidence"}
    captured: dict[str, object] = {}
    manifest = generated / "review_manifest.json"

    monkeypatch.setattr(latex, "PAPER_DIR", paper)
    monkeypatch.setattr(latex, "GENERATED_DIR", generated)
    monkeypatch.setattr(latex, "MARKDOWN_OUT", markdown_out)
    monkeypatch.setattr(latex, "LATEX_OUT", latex_out)
    monkeypatch.setattr(latex, "validate_final_evidence", lambda *args: evidence)
    monkeypatch.setattr(
        latex,
        "build_final_q4_blocks",
        lambda value: {
            "abstract": "validated abstract",
            "result": "validated result",
            "frontier": "validated frontier",
            "conclusion": "validated conclusion",
        },
    )

    def capture_meta(mode: str, group: str, competition_id: str, abstract: str) -> None:
        captured["meta"] = (mode, group, competition_id, abstract)

    def capture_markdown(mode: str, value: dict, *, review: bool) -> str:
        captured["markdown"] = (mode, value, review)
        return "temporary-review-markdown"

    def fake_pandoc(*args: object, **kwargs: object) -> None:
        latex_out.write_bytes(b"temporary-review-latex")
        if pandoc_fails:
            raise RuntimeError("synthetic pandoc failure")

    monkeypatch.setattr(latex, "write_build_meta", capture_meta)
    monkeypatch.setattr(latex, "build_markdown", capture_markdown)
    monkeypatch.setattr(latex.shutil, "which", lambda _: "pandoc")
    monkeypatch.setattr(latex.subprocess, "run", fake_pandoc)
    monkeypatch.setattr(latex, "write_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        latex,
        "compile_pdf",
        lambda *args, **kwargs: pytest.fail("--no-pdf must skip PDF compilation"),
    )

    argv = ["--mode", "internal", "--review", "--no-pdf"]
    if pandoc_fails:
        with pytest.raises(RuntimeError, match="synthetic pandoc failure"):
            latex.main(argv)
    else:
        assert latex.main(argv) == 0

    assert captured["meta"] == (
        "internal",
        "内部审阅",
        "INTERNAL-QA",
        "validated abstract",
    )
    assert captured["markdown"] == ("internal", evidence, True)
    assert markdown_out.read_bytes() == b"canonical-markdown"
    assert latex_out.read_bytes() == b"canonical-latex"


def test_internal_complete_main_passes_final_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paper = tmp_path / "paper"
    generated = paper / "generated"
    markdown_out = generated / "content_for_latex.md"
    latex_out = generated / "content.tex"
    generated.mkdir(parents=True)
    markdown_out.write_bytes(b"stale-markdown")
    latex_out.write_bytes(b"stale-latex")

    evidence = {"kind": "validated-final-evidence"}
    captured: dict[str, object] = {}
    manifest = generated / "complete_manifest.json"

    monkeypatch.setattr(latex, "PAPER_DIR", paper)
    monkeypatch.setattr(latex, "GENERATED_DIR", generated)
    monkeypatch.setattr(latex, "MARKDOWN_OUT", markdown_out)
    monkeypatch.setattr(latex, "LATEX_OUT", latex_out)
    monkeypatch.setattr(latex, "validate_final_evidence", lambda *args: evidence)
    monkeypatch.setattr(
        latex,
        "build_source_appendix",
        lambda mode: captured.setdefault("appendix_mode", mode),
    )
    monkeypatch.setattr(
        latex,
        "build_final_q4_blocks",
        lambda value: {
            "abstract": "validated abstract",
            "result": "validated result",
            "frontier": "validated frontier",
            "conclusion": "validated conclusion",
        },
    )

    def capture_meta(mode: str, group: str, competition_id: str, abstract: str) -> None:
        captured["meta"] = (mode, group, competition_id, abstract)

    def capture_markdown(mode: str, value: dict, *, review: bool) -> str:
        captured["markdown"] = (mode, value, review)
        return "complete-final-evidence-markdown"

    def fake_pandoc(*args: object, **kwargs: object) -> None:
        latex_out.write_bytes(b"complete-final-evidence-latex")

    def capture_manifest(
        mode: str,
        group: str,
        competition_id: str,
        output_pdf: Path | None,
        value: dict,
        *,
        review: bool,
    ) -> Path:
        captured["manifest"] = (
            mode,
            group,
            competition_id,
            output_pdf,
            value,
            review,
        )
        return manifest

    monkeypatch.setattr(latex, "write_build_meta", capture_meta)
    monkeypatch.setattr(latex, "build_markdown", capture_markdown)
    monkeypatch.setattr(latex.shutil, "which", lambda _: "pandoc")
    monkeypatch.setattr(latex.subprocess, "run", fake_pandoc)
    monkeypatch.setattr(latex, "write_manifest", capture_manifest)
    monkeypatch.setattr(
        latex,
        "compile_pdf",
        lambda *args, **kwargs: pytest.fail("--no-pdf must skip PDF compilation"),
    )

    assert latex.main(["--mode", "internal", "--no-pdf"]) == 0
    assert captured["appendix_mode"] == "internal"
    assert captured["meta"] == (
        "internal",
        "内部审阅",
        "INTERNAL-QA",
        "validated abstract",
    )
    assert captured["markdown"] == ("internal", evidence, False)
    assert captured["manifest"] == (
        "internal",
        "内部审阅",
        "INTERNAL-QA",
        None,
        evidence,
        False,
    )
    assert markdown_out.read_text(encoding="utf-8") == "complete-final-evidence-markdown"
    assert latex_out.read_bytes() == b"complete-final-evidence-latex"


def test_generated_markdown_has_visible_table_titles_and_no_tilde_references() -> None:
    review = latex.build_markdown("internal", review=True)
    assert r"~\ref" not in review
    assert "Table: 主要符号及定义" in review
    assert "Table: 附件三组微构体的确定性导通判定" in review
    assert "Table: 四个题给 A 填充量的导通概率估计" in review


def test_review_mode_is_internal_only() -> None:
    with pytest.raises(ValueError, match="仅允许 internal"):
        latex.build_markdown("final", evidence={}, review=True)


def test_snapshot_restore_preserves_canonical_generated_files(tmp_path: Path) -> None:
    existing = tmp_path / "content.tex"
    missing = tmp_path / "content_for_latex.md"
    existing.write_bytes(b"canonical")
    snapshot = latex.snapshot_files((existing, missing))
    existing.write_bytes(b"review")
    missing.write_bytes(b"temporary")
    latex.restore_files(snapshot)
    assert existing.read_bytes() == b"canonical"
    assert not missing.exists()


def test_review_compile_uses_explicit_body_only_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    paper = project / "论文"
    generated = paper / "generated"
    paper.mkdir(parents=True)

    monkeypatch.setattr(latex, "PROJECT_DIR", project)
    monkeypatch.setattr(latex, "PAPER_DIR", paper)
    monkeypatch.setattr(latex, "GENERATED_DIR", generated)
    monkeypatch.setattr(latex.shutil, "which", lambda _: "latexmk")

    def fake_run(*args: object, **kwargs: object) -> None:
        built = generated / "latex_internal_review" / "main.pdf"
        built.parent.mkdir(parents=True, exist_ok=True)
        built.write_bytes(b"%PDF-1.7\nreview\n")

    monkeypatch.setattr(latex.subprocess, "run", fake_run)
    output = latex.compile_pdf("internal", "INTERNAL-QA", paper, review=True)
    body_only = paper / "A题论文_正文便阅稿_LaTeX.pdf"
    complete = paper / "A题论文_当前审阅稿_LaTeX.pdf"
    assert output == body_only
    assert output.read_bytes() == body_only.read_bytes()
    assert not complete.exists()


def test_complete_internal_compile_uses_current_review_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    paper = project / "论文"
    generated = paper / "generated"
    paper.mkdir(parents=True)

    monkeypatch.setattr(latex, "PROJECT_DIR", project)
    monkeypatch.setattr(latex, "PAPER_DIR", paper)
    monkeypatch.setattr(latex, "GENERATED_DIR", generated)
    monkeypatch.setattr(latex.shutil, "which", lambda _: "latexmk")

    def fake_run(*args: object, **kwargs: object) -> None:
        built = generated / "latex_internal" / "main.pdf"
        built.parent.mkdir(parents=True, exist_ok=True)
        built.write_bytes(b"%PDF-1.7\ncomplete\n")

    monkeypatch.setattr(latex.subprocess, "run", fake_run)
    output = latex.compile_pdf("internal", "INTERNAL-QA", paper)
    current = paper / "A题论文_当前审阅稿_LaTeX.pdf"
    legacy = paper / "A题_微构体导电仿真优化_INTERNAL_QA.pdf"
    assert output == current
    assert output.read_bytes() == current.read_bytes()
    assert not legacy.exists()


def test_final_source_appendix_requires_exact_frozen_submission_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    paper = project / "论文"
    source_dir = paper / "src"
    generated = paper / "generated"
    appendix_dir = generated / "source_appendix"
    submission_dir = project / "提交源码"
    source_dir.mkdir(parents=True)
    submission_dir.mkdir(parents=True)

    headers = ["# AI header 1", "# AI header 2"]
    original = project / "program.py"
    original.write_text("print('ok')\n", encoding="utf-8")
    submitted = submission_dir / "program.py"
    submitted.write_text("\n".join(headers) + "\nprint('ok')\n\n", encoding="utf-8")
    allowlist = {
        "schema_version": "1.0",
        "status": "frozen",
        "ai_header_lines": headers,
        "files": [{"path": "program.py", "title": "程序一"}],
    }
    allowlist_path = source_dir / "source_appendix_allowlist.json"
    write_json(allowlist_path, allowlist)
    manifest = {
        "schema_version": "1.0",
        "status": "frozen",
        "submission_root": "提交源码",
        "allowlist": "论文/src/source_appendix_allowlist.json",
        "allowlist_sha256": latex.sha256(allowlist_path),
        "ai_header_lines": headers,
        "files": [
            {
                "title": "程序一",
                "source_path": "program.py",
                "source_sha256": latex.sha256(original),
                "source_bytes": original.stat().st_size,
                "submission_path": "提交源码/program.py",
                "submission_sha256": latex.sha256(submitted),
                "submission_bytes": submitted.stat().st_size,
            }
        ],
    }
    write_json(submission_dir / "source-manifest.json", manifest)

    monkeypatch.setattr(latex, "PROJECT_DIR", project)
    monkeypatch.setattr(latex, "PAPER_DIR", paper)
    monkeypatch.setattr(latex, "SOURCE_ALLOWLIST", allowlist_path)
    monkeypatch.setattr(latex, "SOURCE_APPENDIX_DIR", appendix_dir)
    monkeypatch.setattr(latex, "SOURCE_APPENDIX_TEX", generated / "source_appendix.tex")
    latex.build_source_appendix("final")
    appendix_copy = appendix_dir / "001.py"
    assert appendix_copy.read_bytes() == submitted.read_bytes()
    assert latex.sha256(appendix_copy) == manifest["files"][0]["submission_sha256"]

    original.write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="冻结后原源码发生变化"):
        latex.build_source_appendix("final")
