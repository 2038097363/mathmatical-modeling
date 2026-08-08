from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "02_数据与参数" / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import reconstruct_segments  # noqa: E402


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = ROOT / "00_赛题与附件" / "附件.xlsx"
        cls.before_hash = file_hash(cls.source)
        cls.result = reconstruct_segments.reconstruct_workbook(cls.source)

    @classmethod
    def tearDownClass(cls) -> None:
        if file_hash(cls.source) != cls.before_hash:
            raise AssertionError("原始附件在重建测试中被修改")

    def test_source_hash_and_row_counts(self) -> None:
        self.assertEqual(
            self.result["source_sha256"],
            "6DC68DD49356AEAB483906A524FF79855B454FCACCD168F81FF3A3AD989C4C51",
        )
        literal = {
            row["sheet"]: row["input_segments"]
            for row in self.result["scenario_summary"]
            if row["scenario"] == "row_literal"
        }
        self.assertEqual(literal, {"组1": 12, "组2": 49, "组3": 535})

    def test_expected_component_counts(self) -> None:
        counts = {
            (row["sheet"], row["scenario"]): row["potential_particle_chains"]
            for row in self.result["scenario_summary"]
        }
        self.assertEqual(counts[("组1", "full_cube_periodic")], 9)
        self.assertEqual(counts[("组2", "full_cube_periodic")], 39)
        self.assertEqual(counts[("组3", "full_cube_periodic")], 357)
        self.assertEqual(counts[("组1", "thin_prism_periodic")], 7)
        self.assertEqual(counts[("组2", "thin_prism_periodic")], 28)

    def test_all_selected_mappings_are_numerically_unique(self) -> None:
        for row in self.result["scenario_summary"]:
            self.assertEqual(row["ambiguous_endpoints"], 0)
            self.assertEqual(row["maximum_matching_solutions"], 1)

    def test_maps_partition_every_scenario(self) -> None:
        expected = {"组1": 12, "组2": 49, "组3": 535}
        grouped: dict[tuple[str, str], list[int]] = {}
        for row in self.result["row_identity_map"]:
            grouped.setdefault((row["sheet"], row["scenario"]), []).append(row["sheet_row"])
        for (sheet, _scenario), rows in grouped.items():
            self.assertEqual(len(rows), expected[sheet])
            self.assertEqual(len(rows), len(set(rows)))

    def test_processed_outputs_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            reconstruct_segments.write_processed_outputs(self.result, output)
            required = {
                "scenario_summary.csv",
                "component_results.csv",
                "row_identity_map.csv",
                "junctions.csv",
                "boundary_evidence.csv",
                "scaled_yz_test.csv",
                "reconstruction_results.json",
                "reconstruction_report.md",
            }
            self.assertTrue(required.issubset(path.name for path in output.iterdir()))
            loaded = json.loads((output / "reconstruction_results.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["validation"], "passed")
            with (output / "row_identity_map.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 1253)


if __name__ == "__main__":
    unittest.main()
