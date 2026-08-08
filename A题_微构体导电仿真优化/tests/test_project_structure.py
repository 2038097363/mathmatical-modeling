from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "项目配置.json").read_text(encoding="utf-8-sig"))

    def test_question_ids_and_scripts(self) -> None:
        questions = self.config["questions"]
        ids = [int(item["id"]) for item in questions]
        self.assertEqual(ids, list(range(1, len(ids) + 1)))
        self.assertEqual(len(ids), len(set(ids)))
        for item in questions:
            self.assertTrue((ROOT / item["script"]).is_file())

    def test_pipeline_dry_run_uses_formal_configured_runs(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "run_pipeline.py"),
                "--from-step",
                "1",
                "--to-step",
                "4",
                "--dry-run",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        output = completed.stdout.replace("\\", "/")
        self.assertNotIn("results-smoke", output)
        for expected in (
            "问题/问题1/results",
            "问题/问题2/results/D_primary_n20000",
            "问题/问题2/results/B_sensitivity_n5000",
            "问题/问题2/results/A_sensitivity_n5000",
            "问题/问题3/results/D_confirmation_n50000",
            "问题/问题4/results/D_screen2000_confirm50000",
        ):
            self.assertIn(expected, output)
        self.assertIn("--trials 20000", output)
        self.assertIn("--confirmation-trials 50000", output)
        self.assertIn("--stage screen", output)
        self.assertIn("--stage freeze", output)
        self.assertIn("--stage confirm", output)
        self.assertEqual(output.count("[问题4/"), 3)

    def test_question_directories(self) -> None:
        for item in self.config["questions"]:
            question_root = ROOT / "问题" / f"问题{item['id']}"
            for name in ("src", "results"):
                self.assertTrue((question_root / name).is_dir())
            for name in ("data", "figures", "models"):
                optional = question_root / name
                self.assertFalse(optional.is_dir() and not any(optional.iterdir()))
            for name in (
                "问题分析.md",
                "数学推导.md",
                "计算模型与实现.md",
                "验证与拓展.md",
                "结果与结论.md",
            ):
                self.assertTrue((question_root / name).is_file())

    def test_evidence_structure(self) -> None:
        self.assertGreaterEqual(int(self.config.get("workflow_schema_version", 0)), 3)
        manifest_path = ROOT / self.config["artifact_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        self.assertEqual(manifest["policy"]["mode"], "adaptive")
        artifacts = manifest["artifacts"]
        ids = [item["id"] for item in artifacts]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(
            {
                "problem_map",
                "data_parameter_audit",
                "formulation",
                "model_decision",
                "implementation_trace",
                "validation",
                "claim_evidence",
                "result_registry",
                "figure_trace",
                "paper",
                "delivery",
            }.issubset(ids)
        )
        for item in artifacts:
            self.assertIn(
                item["applicability"],
                {"applicable", "conditional", "not_applicable"},
            )
            if item["applicability"] != "not_applicable":
                self.assertTrue(item.get("paths"), item["id"])

    def test_feedback_log_is_jsonl(self) -> None:
        path = ROOT / self.config["feedback_log"]
        self.assertTrue(path.is_file())
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                self.assertIsInstance(json.loads(line), dict)

    def test_registry_schema(self) -> None:
        registry = json.loads((ROOT / "结果注册表.json").read_text(encoding="utf-8-sig"))
        self.assertIsInstance(registry.get("results"), dict)

    def test_knowledge_and_code_policy(self) -> None:
        knowledge = self.config["knowledge_policy"]
        self.assertEqual(knowledge["role"], "candidate_and_pattern_source")
        self.assertTrue(knowledge["independent_validation_required"])
        self.assertFalse(knowledge["runtime_dependency_allowed"])
        style = self.config["code_style"]
        self.assertEqual(style["submission_programs"], "numbered_chinese_modules")
        self.assertEqual(style["small_helpers"], "minimal")
        self.assertEqual(style["narrative_language"], "zh-CN")

    def test_paper_has_no_table_of_contents(self) -> None:
        paper = (ROOT / self.config["paper_main"]).read_text(encoding="utf-8-sig")
        generated = (ROOT / "论文" / "generated" / "content.tex").read_text(
            encoding="utf-8-sig"
        )
        source_appendix = (
            ROOT / "论文" / "generated" / "source_appendix.tex"
        ).read_text(encoding="utf-8-sig")
        combined = paper + "\n" + generated
        self.assertNotIn(r"\tableofcontents", combined)
        self.assertTrue(
            r"\begin{abstract}" in paper or r"摘\quad{}要" in paper,
            "首页必须包含摘要标题或 abstract 环境",
        )
        self.assertRegex(paper, r"\\textbf\{针对问题一")
        self.assertIn(r"\pagenumbering{arabic}", paper)
        self.assertNotIn(r"\pagenumbering{gobble}", paper)
        self.assertIn(r"\input{generated/content.tex}", paper)
        self.assertTrue(
            r"\appendix" in generated
            or r"\section{附录}" in generated
            or r"\TemplateMatterHeading{附录}" in generated,
            "生成正文必须包含附录入口",
        )
        self.assertIn(r"\input{generated/source_appendix.tex}", generated)
        self.assertIn(r"\VerbatimInput", source_appendix)
        appendix_marker = next(
            marker
            for marker in (
                r"\appendix",
                r"\section{附录}",
                r"\TemplateMatterHeading{附录}",
            )
            if marker in generated
        )
        body = generated.split(appendix_marker, maxsplit=1)[0]
        ai_marker = (
            r"\TemplateMatterHeading{AI 工具使用声明}"
            if r"\TemplateMatterHeading{AI 工具使用声明}" in body
            else r"\section{AI 工具使用声明}"
        )
        modeling_body = body.split(ai_marker, maxsplit=1)[0]
        section_count = len(re.findall(r"(?m)^\s*\\section\{", modeling_body))
        self.assertGreaterEqual(section_count, 5)
        self.assertLessEqual(section_count, 8)


if __name__ == "__main__":
    unittest.main()
