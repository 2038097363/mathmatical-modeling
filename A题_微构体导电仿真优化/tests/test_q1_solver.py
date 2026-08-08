from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Q1_SOURCE_DIR = ROOT / "问题" / "问题1" / "src"
if str(Q1_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(Q1_SOURCE_DIR))

import solve as q1_solver  # noqa: E402


class Q1SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        base = Path(cls.temporary.name)
        args = q1_solver.parse_args(
            [
                "--output-dir",
                str(base / "results"),
                "--processed-dir",
                str(base / "processed"),
                "--skip-registry",
            ]
        )
        cls.payload = q1_solver.run(args)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_union_find_and_path_agree_on_toy_graph(self) -> None:
        edges = [
            {"edge_id": "left", "node_u": "LEFT", "node_v": "r3"},
            {"edge_id": "middle", "node_u": "r3", "node_v": "r4"},
            {"edge_id": "right", "node_u": "r4", "node_v": "RIGHT"},
        ]
        self.assertTrue(q1_solver.connected_by_union_find(edges))
        nodes, path_edges = q1_solver.shortest_path(edges)
        self.assertEqual(nodes, ["LEFT", "r3", "r4", "RIGHT"])
        self.assertEqual(len(path_edges), 3)

    def test_no_uncertain_or_independent_threshold_disagreement(self) -> None:
        for row in self.payload["screening"]:
            self.assertEqual(row["narrow_uncertain"], 0)
            self.assertEqual(row["slsqp_threshold_disagreements"], 0)

    def test_attachment_electrode_support_points_lie_on_finite_faces(self) -> None:
        segments = q1_solver.load_segments(ROOT / "00_赛题与附件" / "附件.xlsx")
        for rows in segments.values():
            for segment in rows:
                for direction, offset in (([-1.0, 0.0, 0.0], -5000.0), ([1.0, 0.0, 0.0], 5000.0)):
                    gap = q1_solver.shape_plane_distance(segment.cylinder, [1.0, 0.0, 0.0], offset)
                    if gap <= q1_solver.CONTACT_CUTOFF_NM:
                        support = segment.cylinder.support(direction)
                        self.assertLessEqual(abs(float(support[1])), 5000.0 + 1e-9)
                        self.assertLessEqual(abs(float(support[2])), 5000.0 + 1e-9)

    def test_fragment_level_results(self) -> None:
        indexed = {
            (row["scenario"], row["internal_mode"], row["sheet"]): row
            for row in self.payload["scenario_results"]
        }
        expected = {"组1": False, "组2": True, "组3": True}
        for sheet, conductive in expected.items():
            result = indexed[("A_row_literal", "disconnected_fragments", sheet)]
            self.assertEqual(result["conductive_definite"], conductive)

    def test_excluded_full_cube_diagnostic_results_remain_reproducible(self) -> None:
        indexed = {
            (row["scenario"], row["internal_mode"], row["sheet"]): row
            for row in self.payload["scenario_results"]
        }
        expected_paths = {
            "组1": ["LEFT", "r4", "r12", "RIGHT"],
            "组2": ["LEFT", "r12", "r41", "RIGHT"],
            "组3": ["LEFT", "r10", "r452", "RIGHT"],
        }
        for sheet, path in expected_paths.items():
            result = indexed[("B_full_cube_periodic", "connected_same_particle", sheet)]
            self.assertTrue(result["conductive_definite"])
            self.assertEqual(result["witness_row_nodes"], path)
            internal = [
                edge for edge in result["witness_edges"]
                if edge["edge_type"] == "same_particle_periodic_junction"
            ]
            self.assertEqual(len(internal), 1)
            self.assertEqual(internal[0]["component_status"], "unique")
            self.assertEqual(internal[0]["mapped_endpoint_residual_nm"], 0.0)

    def test_registry_contains_only_official_row_literal_results(self) -> None:
        base = Path(self.temporary.name)
        registry_path = base / "registry.json"
        latex_path = base / "results.tex"
        registry_path.write_text(
            json.dumps(
                {
                    "project": "test",
                    "results": {
                        "q1_same_particle_full_cube_group1_conductive": {
                            "question": 1,
                            "value": True,
                        },
                        "q1_unknown_legacy_key": {
                            "question": 1,
                            "value": "legacy",
                        },
                        "q2_keep": {"question": 2, "value": 17},
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        expected_q1_keys = {
            "q1_fragment_disconnected_group1_conductive",
            "q1_fragment_disconnected_group2_conductive",
            "q1_fragment_disconnected_group2_witness",
            "q1_fragment_disconnected_group3_conductive",
            "q1_fragment_disconnected_group3_witness",
            "q1_numerically_uncertain_edges",
            "q1_slsqp_threshold_disagreements",
            "q1_closest_connected_gap_upper_nm",
        }

        for _ in range(2):
            q1_solver.register_q1_results(
                self.payload,
                registry_path=registry_path,
                latex_path=latex_path,
            )
            parsed = json.loads(registry_path.read_text(encoding="utf-8"))
            q1_keys = {
                key for key in parsed["results"] if key.startswith("q1_")
            }
            self.assertEqual(q1_keys, expected_q1_keys)
            self.assertEqual(parsed["results"]["q2_keep"]["value"], 17)
            self.assertFalse(
                parsed["results"][
                    "q1_fragment_disconnected_group1_conductive"
                ]["value"]
            )
            self.assertEqual(
                parsed["results"][
                    "q1_fragment_disconnected_group2_witness"
                ]["value"],
                "LEFT -> r13 -> r14 -> r26 -> r41 -> RIGHT",
            )
            self.assertEqual(
                parsed["results"][
                    "q1_fragment_disconnected_group3_witness"
                ]["value"],
                "LEFT -> r65 -> r266 -> r218 -> r353 -> RIGHT",
            )

    def test_report_uses_only_the_final_problem_wording(self) -> None:
        report = q1_solver.build_report(
            self.payload["metadata"],
            self.payload["screening"],
            self.payload["scenario_results"],
        )
        self.assertIn(
            "正式口径：仅 `A_row_literal / disconnected_fragments`",
            report,
        )
        self.assertIn("仅用于程序一致性检查，不进入本题结果注册表", report)
        for forbidden in ("历史诊断", "澄清", "组委会", "不是本题答案"):
            self.assertNotIn(forbidden, report)

    def test_formal_sources_have_no_tmp_dependency(self) -> None:
        sources = [
            ROOT / "问题" / "问题1" / "src" / "solve.py",
            ROOT / "公共代码" / "geometry_kernel.py",
            ROOT / "02_数据与参数" / "src" / "reconstruct_segments.py",
        ]
        for source in sources:
            text = source.read_text(encoding="utf-8")
            self.assertNotIn(' / "tmp"', text)
            self.assertNotIn("benchmark_geometry_kernel", text)


if __name__ == "__main__":
    unittest.main()
