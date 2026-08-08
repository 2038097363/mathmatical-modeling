from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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


MODULE = _load_module(
    "build_q4_final_assets", FIGURE_SRC / "build_q4_final_assets.py"
)


class Q4FinalAssetsTests(unittest.TestCase):
    def test_paths_are_semantic_and_exclude_preview(self) -> None:
        root = Path("figures")
        paths = MODULE._paths(root, 619, 0, 1)
        self.assertTrue(paths["scene"].name.startswith("q4_final_na000619_nb000000"))
        self.assertTrue(paths["model"].name.endswith("trial000001.FCStd"))
        self.assertTrue(all("preview" not in path.name for path in paths.values()))

    def test_build_assets_is_serial_and_hashes_every_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "q4_summary.json"
            artifact = root / "mixed_pareto_frontier_samples.json"
            summary.write_text("{}", encoding="utf-8")
            artifact.write_text("{}", encoding="utf-8")
            source = argparse.Namespace(
                n_a=619,
                n_b=0,
                source_status="frozen_candidate_confirmed_trial_geometry",
                source_path=summary,
                artifact_path=artifact,
                artifact_configuration_fingerprint="A" * 64,
                selected_configuration_fingerprint="B" * 64,
            )
            scene = {
                "publication_status": "final_random_trial_geometry",
                "spheres": [],
                "traceability": {
                    "boundary_primary": "D_truncated_fragments_independent",
                    "random_stream": {
                        "master_seed": 20260801,
                        "stream_id": 5,
                        "trial_count": 50000,
                        "trial_id": 1,
                    },
                    "geometry_counts": {
                        "a_source_particles": 619,
                        "b_source_particles": 0,
                        "a_fragments": 1100,
                        "b_fragments": 0,
                        "clipped_b_fragments": 0,
                        "all_fragments": 1100,
                        "witness_fragments": 10,
                    },
                    "mixed_witness": {
                        "nodes": ["electrode_left", "A", "electrode_right"],
                        "edge_count": 2,
                        "same_source_edges": 0,
                        "all_edges_geometry_verified": True,
                    },
                },
            }
            render_order = []

            def fake_build(scene_path, model_path, audit_path, _exe, _timeout):
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(b"FCStd")
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.write_text("{}", encoding="utf-8")
                return {"status": "passed"}

            def fake_render(render_args):
                render_order.append(render_args.view)
                render_args.output.parent.mkdir(parents=True, exist_ok=True)
                render_args.output.write_bytes(b"PNG")
                render_args.audit.write_text("{}", encoding="utf-8")
                return {"status": "passed"}

            def fake_witness(_scene, png, pdf, audit):
                png.parent.mkdir(parents=True, exist_ok=True)
                png.write_bytes(b"PNG-WITNESS")
                pdf.write_bytes(b"PDF-WITNESS")
                audit.write_text("{}", encoding="utf-8")
                return {"status": "passed"}

            args = argparse.Namespace(
                design_json=summary,
                output_root=root / "figures",
                manifest=None,
                freecadcmd_exe=None,
                freecad_exe=None,
                width=2400,
                height=1800,
                axonometric_zoom=0.92,
                top_zoom=1.20,
                timeout=30.0,
            )
            with (
                mock.patch.object(MODULE.scene_builder, "discover_design", return_value=source),
                mock.patch.object(MODULE.scene_builder, "select_conductive_trial", return_value=1),
                mock.patch.object(MODULE.scene_builder, "build_verified_trial_scene", return_value=scene),
                mock.patch.object(MODULE.scene_builder, "build_fcstd", side_effect=fake_build),
                mock.patch.object(MODULE.scene_renderer, "render_q4", side_effect=fake_render),
                mock.patch.object(MODULE.witness_builder, "build_figure", side_effect=fake_witness),
            ):
                result = MODULE.build_assets(args)

            self.assertEqual(render_order, ["axonometric", "top"])
            self.assertEqual(result["selection_rule"], MODULE.SELECTION_RULE)
            self.assertTrue(all(result["checks"].values()))
            self.assertTrue(Path(result["manifest"]["path"]).is_file())
            self.assertEqual(set(result["artifacts"]), {
                "scene",
                "model",
                "build_audit",
                "axonometric",
                "axonometric_audit",
                "top",
                "top_audit",
                "witness_png",
                "witness_pdf",
                "witness_audit",
            })

    def test_rejects_a_design_other_than_the_frozen_candidate(self) -> None:
        args = argparse.Namespace(design_json=None)
        source = argparse.Namespace(n_a=618, n_b=0)
        with mock.patch.object(MODULE.scene_builder, "discover_design", return_value=source):
            with self.assertRaisesRegex(ValueError, "冻结候选"):
                MODULE.build_assets(args)


if __name__ == "__main__":
    unittest.main()
