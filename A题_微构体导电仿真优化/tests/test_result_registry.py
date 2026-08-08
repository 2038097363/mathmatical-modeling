from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "公共代码" / "result_registry.py"
SPEC = importlib.util.spec_from_file_location("result_registry", MODULE_PATH)
assert SPEC and SPEC.loader
result_registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(result_registry)


class ResultRegistryTests(unittest.TestCase):
    def test_register_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "results.json"
            tex = root / "results.tex"
            result_registry.register_result(
                "q1_error",
                question=1,
                value=0.0123,
                formatted="1.23",
                unit="%",
                source_script="问题/问题1/src/solve.py",
                source_artifact="问题/问题1/results/metrics.csv",
                validation="整水平留出",
                latex_macro="QOneError",
                registry_path=registry,
            )
            count = result_registry.export_latex(registry, tex)
            self.assertEqual(count, 1)
            parsed = json.loads(registry.read_text(encoding="utf-8"))
            self.assertEqual(parsed["results"]["q1_error"]["question"], 1)
            self.assertIn(r"\providecommand{\QOneError}{1.23\%}", tex.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
