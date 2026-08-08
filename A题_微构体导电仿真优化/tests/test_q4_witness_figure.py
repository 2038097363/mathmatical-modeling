from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


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
    "build_q4_witness_figure", FIGURE_SRC / "build_q4_witness_figure.py"
)


class Q4WitnessFigureTests(unittest.TestCase):
    def test_builds_focus_figure_without_inventing_b_geometry(self) -> None:
        scene = {
            "publication_status": "final_random_trial_geometry",
            "box": {"length_nm": 10.0},
            "cylinders": [
                {
                    "id": "A_s000001_f001",
                    "role": "witness",
                    "start_nm": [-5.0, 0.0, 0.0],
                    "end_nm": [0.2, 0.0, 0.0],
                },
                {
                    "id": "A_s000002_f001",
                    "role": "witness",
                    "start_nm": [-0.2, 0.0, 0.0],
                    "end_nm": [5.0, 0.0, 0.0],
                },
            ],
            "spheres": [],
            "traceability": {
                "boundary_primary": "D_truncated_fragments_independent",
                "design_counts": {"n_a": 619, "n_b": 0},
                "random_stream": {"trial_id": 1},
                "mixed_witness": {
                    "nodes": [
                        "electrode_left",
                        "A_s000001_f001",
                        "A_s000002_f001",
                        "electrode_right",
                    ],
                    "edge_count": 3,
                    "same_source_edges": 0,
                    "all_edges_geometry_verified": True,
                    "edges": [
                        {"edge_id": "E1", "nodes": ["electrode_left", "A_s000001_f001"], "contact_type": "electrode-A", "connected": True, "same_source_pair": False},
                        {"edge_id": "E2", "nodes": ["A_s000001_f001", "A_s000002_f001"], "contact_type": "A-A", "connected": True, "same_source_pair": False},
                        {"edge_id": "E3", "nodes": ["A_s000002_f001", "electrode_right"], "contact_type": "electrode-A", "connected": True, "same_source_pair": False},
                    ],
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path = root / "scene.json"
            png = root / "witness.png"
            pdf = root / "witness.pdf"
            audit_path = root / "witness.audit.json"
            scene_path.write_text(json.dumps(scene), encoding="utf-8")
            audit = MODULE.build_figure(scene_path, png, pdf, audit_path)
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["design"], {"n_a": 619, "n_b": 0})
            self.assertIn("no B geometry invented", audit["render_contract"]["medium_b"])
            self.assertEqual(audit["rendered_topology_edge_count"], 3)
            self.assertTrue(audit["checks"]["focus_bounds_include_all_witness_geometry"])
            self.assertTrue(audit["checks"]["rendered_topology_edge_count_matches_witness"])
            self.assertTrue(audit["pixels"]["checks"]["topology_edges_visible_in_3d_panel"])
            y_limits = audit["render_contract"]["focus_limits_nm"]["y"]
            z_limits = audit["render_contract"]["focus_limits_nm"]["z"]
            self.assertLess(y_limits[1] - y_limits[0], 10.0)
            self.assertLess(z_limits[1] - z_limits[0], 10.0)
            self.assertTrue(audit["pixels"]["passed"])
            self.assertTrue(png.is_file())
            self.assertTrue(pdf.is_file())


if __name__ == "__main__":
    unittest.main()
