from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SRC = PROJECT_ROOT / "论文" / "figures" / "src"
sys.path.insert(0, str(FIGURE_SRC))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCENE_MODULE = _load_module(
    "build_q4_mixed_scene", FIGURE_SRC / "build_q4_mixed_scene.py"
)
RENDER_MODULE = _load_module(
    "render_q4_mixed_scene", FIGURE_SRC / "render_q4_mixed_scene.py"
)


def _final_source(
    config,
    *,
    n_a: int | None = None,
    n_b: int | None = None,
) -> object:
    selected_n_a = config.n_a if n_a is None else n_a
    selected_n_b = config.n_b if n_b is None else n_b
    selected = SCENE_MODULE.replace(config, n_a=selected_n_a, n_b=selected_n_b)
    return SCENE_MODULE.DesignSource(
        n_a=selected_n_a,
        n_b=selected_n_b,
        source_status="globally_certified_minimum_cost",
        publication_status=SCENE_MODULE.FINAL_PUBLICATION_STATUS,
        source_path=None,
        source_sha256=None,
        boundary_primary=SCENE_MODULE.PRIMARY_BOUNDARY,
        artifact_configuration=config,
        artifact_configuration_fingerprint=config.fingerprint,
        selected_configuration_fingerprint=selected.fingerprint,
        confirmation_proof_status="candidate_statistically_feasible",
    )


def _write_final_fixture(
    directory: Path,
    config,
    *,
    reported: tuple[int, int],
    frontiers: list[list[list[int]]],
) -> tuple[Path, Path]:
    artifact = directory / "mixed_pareto_frontier_samples.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "mixed_pareto_frontier_samples",
                "schema_version": 1,
                "configuration": config.to_dict(),
                "configuration_fingerprint": config.fingerprint,
                "boundary_contract": SCENE_MODULE.BOUNDARY_CONTRACT,
                "records": [
                    {"trial_id": trial_id, "connectivity_frontier": frontier}
                    for trial_id, frontier in enumerate(frontiers)
                ],
                "trials": config.trial_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_hash = SCENE_MODULE.sha256(artifact)
    summary = directory / "q4_summary.json"
    summary.write_text(
        json.dumps(
            {
                "kind": "q4_final_summary",
                "schema_version": 1,
                "question": 4,
                "result_status": "globally_certified_minimum_cost",
                "boundary_contract": SCENE_MODULE.BOUNDARY_CONTRACT,
                "reported_design": {"n_a": reported[0], "n_b": reported[1]},
                "confirmation_records": [
                    {
                        "role": "candidate",
                        "proof_status": "candidate_statistically_feasible",
                        "n_a": reported[0],
                        "n_b": reported[1],
                        "configuration": config.to_dict(),
                        "configuration_fingerprint": config.fingerprint,
                        "artifact_path": str(artifact.resolve()),
                        "artifact_sha256": artifact_hash,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary, artifact


class Q4MixedSceneTests(unittest.TestCase):
    def test_only_publishable_final_summary_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q4_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "kind": "q4_stage_summary",
                        "result_status": "screening_complete",
                        "final_evidence_available": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "q4_final_summary"):
                SCENE_MODULE.load_confirmed_design(path)

    def test_final_summary_uses_artifact_stream_and_first_connected_trial(self) -> None:
        config = SCENE_MODULE.MixedSimulationConfig(
            n_a=3,
            n_b=3,
            trial_count=2,
            master_seed=917,
            stream_id=5,
            box_length_nm=10.0,
            a_length_nm=4.0,
            a_radius_nm=0.5,
            b_radius_nm=1.0,
            contact_cutoff_nm=10.0,
            cell_size_nm=2.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            summary, _ = _write_final_fixture(
                Path(directory),
                config,
                reported=(2, 2),
                frontiers=[[], [[0, 1]]],
            )
            source = SCENE_MODULE.load_confirmed_design(summary)
            self.assertEqual(SCENE_MODULE.select_conductive_trial(source), 1)
            with self.assertRaisesRegex(ValueError, "不导通"):
                SCENE_MODULE.select_conductive_trial(source, 0)
            selected_config, selected_geometry = SCENE_MODULE.generate_selected_geometry(
                source, 1
            )
            maximum_geometry = SCENE_MODULE.generate_mixed_trial(config, 1)
            np.testing.assert_array_equal(
                selected_geometry.a_centers, maximum_geometry.a_centers[:2]
            )
            np.testing.assert_array_equal(
                selected_geometry.a_directions, maximum_geometry.a_directions[:2]
            )
            np.testing.assert_array_equal(
                selected_geometry.b_centers, maximum_geometry.b_centers[:2]
            )
            scene = SCENE_MODULE.build_scene_from_geometry(
                source, selected_config, selected_geometry, 1
            )

        trace = scene["traceability"]
        self.assertEqual(scene["publication_status"], "final_random_trial_geometry")
        self.assertEqual(
            scene["visible_banner"],
            "N_A=2, N_B=2 | 随机导通样本 000001",
        )
        self.assertEqual(scene["electrodes"]["transparency"], 70)
        serialized = json.dumps(scene, ensure_ascii=False).lower()
        for forbidden in SCENE_MODULE.FINAL_FORBIDDEN_TEXT:
            self.assertNotIn(forbidden.lower(), serialized)
        self.assertEqual(trace["random_stream"]["trial_id"], 1)
        self.assertEqual(trace["random_stream"]["master_seed"], 917)
        self.assertEqual(trace["maximum_static_graph_design"], {"n_a": 3, "n_b": 3})
        self.assertEqual(trace["design_counts"], {"n_a": 2, "n_b": 2})
        self.assertEqual(
            {row["source_index"] for row in scene["cylinders"]}, {0, 1}
        )
        self.assertEqual({row["source_index"] for row in scene["spheres"]}, {0, 1})
        counts = trace["geometry_counts"]
        self.assertEqual(
            counts["all_fragments"], len(scene["cylinders"]) + len(scene["spheres"])
        )
        witness = trace["mixed_witness"]
        self.assertEqual(witness["status"], "actual_conductive_trial")
        self.assertEqual(witness["nodes"][0], "electrode_left")
        self.assertEqual(witness["nodes"][-1], "electrode_right")
        self.assertTrue(witness["all_edges_geometry_verified"])
        self.assertTrue(all(edge["connected"] for edge in witness["edges"]))
        self.assertTrue(all(not edge["same_source_pair"] for edge in witness["edges"]))
        self.assertTrue(trace["cross_validation"]["fixed_design_solver_conductive"])

    def test_hand_geometry_extracts_real_mixed_a_b_a_path(self) -> None:
        config = SCENE_MODULE.MixedSimulationConfig(
            n_a=2,
            n_b=1,
            trial_count=1,
            master_seed=1,
            stream_id=5,
            box_length_nm=10.0,
            a_length_nm=4.0,
            a_radius_nm=0.5,
            b_radius_nm=1.0,
            contact_cutoff_nm=0.2,
            cell_size_nm=2.0,
        )
        geometry = SCENE_MODULE.MixedTrialGeometry(
            np.asarray([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0]]),
        )
        scene = SCENE_MODULE.build_scene_from_geometry(
            _final_source(config), config, geometry, 0
        )
        witness = scene["traceability"]["mixed_witness"]
        self.assertEqual(
            witness["nodes"],
            [
                "electrode_left",
                "A_s000001_f001",
                "B_s000001_f001",
                "A_s000002_f001",
                "electrode_right",
            ],
        )
        self.assertEqual(
            [edge["contact_type"] for edge in witness["edges"]],
            ["electrode-A", "A-B", "A-B", "electrode-A"],
        )
        self.assertEqual(
            sum(row["role"] == "witness" for row in scene["cylinders"]), 2
        )
        self.assertEqual(sum(row["role"] == "witness" for row in scene["spheres"]), 1)

    def test_same_source_boundary_fragments_are_never_connected(self) -> None:
        config = SCENE_MODULE.MixedSimulationConfig(
            n_a=1,
            n_b=1,
            trial_count=1,
            master_seed=2,
            stream_id=5,
            box_length_nm=10.0,
            a_length_nm=8.0,
            a_radius_nm=0.5,
            b_radius_nm=1.0,
            contact_cutoff_nm=10.0,
            cell_size_nm=2.0,
        )
        geometry = SCENE_MODULE.MixedTrialGeometry(
            np.asarray([[4.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0]]),
        )
        scene = SCENE_MODULE.build_scene_from_geometry(
            _final_source(config), config, geometry, 0
        )
        trace = scene["traceability"]
        self.assertEqual(len(scene["cylinders"]), 2)
        self.assertGreaterEqual(trace["contact_graph"]["same_source_skips"], 1)
        cylinder_ids = {row["id"] for row in scene["cylinders"]}
        for edge in trace["contact_graph"]["all_connected_edges"]:
            self.assertFalse(set(edge["nodes"]).issubset(cylinder_ids))
        self.assertEqual(trace["mixed_witness"]["same_source_edges"], 0)

    def test_artifact_hash_tampering_is_rejected(self) -> None:
        config = SCENE_MODULE.MixedSimulationConfig(n_a=1, n_b=1, trial_count=1)
        with tempfile.TemporaryDirectory() as directory:
            summary, artifact = _write_final_fixture(
                Path(directory), config, reported=(1, 1), frontiers=[[[0, 1]]]
            )
            artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact_sha256"):
                SCENE_MODULE.load_confirmed_design(summary)

    def test_frozen_confirmation_shard_selects_first_connected_trial(self) -> None:
        config = SCENE_MODULE.MixedSimulationConfig(
            n_a=3,
            n_b=4,
            trial_count=2,
            master_seed=20260801,
            stream_id=5,
            box_length_nm=10.0,
            a_length_nm=4.0,
            a_radius_nm=0.5,
            b_radius_nm=1.0,
            contact_cutoff_nm=0.2,
            cell_size_nm=2.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screening_json = root / "q4_screening.json"
            screening_csv = root / "q4_screening.csv"
            screening_artifact = root / "screening_artifact.json"
            for path, content in (
                (screening_json, "{}"),
                (screening_csv, "n_a,n_b\n"),
                (screening_artifact, "{}"),
            ):
                path.write_text(content, encoding="utf-8")
            freeze_path = root / "q4_confirmation_freeze.json"
            freeze_path.write_text(
                json.dumps(
                    {
                        "kind": "q4_confirmation_freeze",
                        "source_screening": {
                            "json_path": str(screening_json),
                            "json_sha256": SCENE_MODULE.sha256(screening_json),
                            "csv_path": str(screening_csv),
                            "csv_sha256": SCENE_MODULE.sha256(screening_csv),
                            "pareto_artifact_path": str(screening_artifact),
                            "pareto_artifact_sha256": SCENE_MODULE.sha256(
                                screening_artifact
                            ),
                        },
                        "candidate_freeze": {"n_a": 2, "n_b": 0},
                        "confirmation_protocol": {
                            "boundary_contract": SCENE_MODULE.BOUNDARY_CONTRACT,
                            "configuration": config.to_dict(),
                            "configuration_fingerprint": config.fingerprint,
                        },
                    }
                ),
                encoding="utf-8",
            )
            shard_path = root / "shard_000000_000001.json"
            shard_path.write_text(
                json.dumps(
                    {
                        "kind": "mixed_pareto_frontier_shard",
                        "schema_version": 1,
                        "configuration": config.to_dict(),
                        "configuration_fingerprint": config.fingerprint,
                        "trial_ids": [0, 1],
                        "records": [
                            {"trial_id": 0, "connectivity_frontier": [[1, 1]]},
                            {"trial_id": 1, "connectivity_frontier": [[2, 0]]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source = SCENE_MODULE.load_frozen_shard_design(
                freeze_path, shard_path
            )
            self.assertEqual((source.n_a, source.n_b), (2, 0))
            self.assertEqual(source.artifact_sha256, SCENE_MODULE.sha256(shard_path))
            self.assertEqual(SCENE_MODULE.select_conductive_trial(source), 1)
            with self.assertRaisesRegex(ValueError, "不导通"):
                SCENE_MODULE.select_conductive_trial(source, 0)

    def test_preview_remains_explicit_and_deterministic(self) -> None:
        first = SCENE_MODULE.build_preview_scene(SCENE_MODULE.preview_source())
        second = SCENE_MODULE.build_preview_scene(SCENE_MODULE.preview_source())
        self.assertEqual(first, second)
        self.assertEqual(first["publication_status"], "preview_not_optimal")
        self.assertIn("NOT AN OPTIMAL DESIGN", first["visible_banner"])

    def test_final_overlay_and_top_view_are_audited(self) -> None:
        config = SCENE_MODULE.MixedSimulationConfig(
            n_a=2,
            n_b=1,
            trial_count=1,
            box_length_nm=10.0,
            a_length_nm=4.0,
            a_radius_nm=0.5,
            b_radius_nm=1.0,
            contact_cutoff_nm=0.2,
            cell_size_nm=2.0,
        )
        geometry = SCENE_MODULE.MixedTrialGeometry(
            np.asarray([[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0]]),
        )
        scene = SCENE_MODULE.build_scene_from_geometry(
            _final_source(config), config, geometry, 0
        )
        image = Image.new("RGB", (1200, 900), "white")
        overlay = RENDER_MODULE.overlay_status_banner(image, scene)
        self.assertIn("橙色：实际导通见证", overlay["legend_labels"])
        self.assertNotIn("橙色：示意见证链", overlay["legend_labels"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "actual.FCStd"
            scene_path = root / "actual_scene.json"
            output = root / "actual_top.png"
            audit_path = root / "actual_top.audit.json"
            source.write_bytes(b"FCStd-test")
            scene_path.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")

            def fake_render(args):
                render_calls.append(args)
                raw = Image.new("RGB", (args.width, args.height), "white")
                draw = ImageDraw.Draw(raw)
                draw.rectangle((260, 250, 420, 330), fill=(175, 178, 181))
                draw.rectangle((450, 250, 610, 330), fill=(55, 135, 205))
                draw.rectangle((640, 250, 800, 330), fill=(232, 88, 29))
                draw.rectangle((830, 250, 990, 330), fill=(55, 58, 61))
                raw.save(args.output)
                return {
                    "freecad_executable": "test-freecad",
                    "freecad_executable_sha256": "0" * 64,
                    "process": {"return_code": 0},
                    "freecad_log": "test",
                    "macro": "test",
                    "pixels": {"sha256": RENDER_MODULE.sha256(args.output)},
                }

            args = argparse.Namespace(
                source=source,
                scene=scene_path,
                output=output,
                audit=audit_path,
                freecad_exe=None,
                width=1200,
                height=900,
                zoom=0.82,
                timeout=30.0,
                view="top",
                focus_witness=False,
            )
            render_calls = []
            with mock.patch.object(RENDER_MODULE, "render_base", side_effect=fake_render):
                audit = RENDER_MODULE.render_q4(args)
            self.assertEqual(audit["parameters"]["view"], "top")
            self.assertEqual(audit["parameters"]["hidden_styles"], [])
            self.assertEqual(audit["parameters"]["electrode_transparency"], 70)
            self.assertFalse(audit["parameters"]["electrode_wireframe"])
            self.assertTrue(audit["scene_color_audit"]["passed"])
            self.assertTrue(output.is_file())

            witness_output = root / "actual_witness_axonometric.png"
            witness_audit_path = root / "actual_witness_axonometric.audit.json"
            witness_args = argparse.Namespace(
                **{
                    **vars(args),
                    "output": witness_output,
                    "audit": witness_audit_path,
                    "view": "axonometric",
                    "focus_witness": True,
                }
            )
            with mock.patch.object(RENDER_MODULE, "render_base", side_effect=fake_render):
                witness_audit = RENDER_MODULE.render_q4(witness_args)
            self.assertEqual(
                witness_audit["parameters"]["hidden_styles"],
                ["background_a", "background_b"],
            )
            self.assertIn("见证聚焦", witness_audit["overlay"]["text"])
            self.assertEqual(
                witness_audit["parameters"]["electrode_transparency"], 84
            )
            self.assertTrue(witness_audit["parameters"]["electrode_wireframe"])
            self.assertEqual(render_calls[-1].view, "axonometric")
            self.assertTrue(witness_output.is_file())


if __name__ == "__main__":
    unittest.main()
