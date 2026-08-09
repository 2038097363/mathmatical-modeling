from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = PROJECT_ROOT / "问题" / "问题4" / "results" / "v23_refined_geometry"
V23_CODE_ROOT = PROJECT_ROOT / "问题" / "问题4" / "src" / "v23_refined_geometry"
SUMMARY_PATH = RESULT_ROOT / "q4_v23_final_summary.json"
MANIFEST_PATH = RESULT_ROOT / "v23_import_manifest.csv"
sys.path.insert(0, str(PROJECT_ROOT / "公共代码"))
from result_registry import read_registry, write_registry


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"数值不一致: {actual} != {expected}")


def build_manifest() -> list[dict]:
    paths = []
    paths.extend(("q4_data", path) for path in RESULT_ROOT.glob("q4_v230*"))
    paths.extend(
        ("package_record", RESULT_ROOT / name)
        for name in (
            "00_README_先读此文件.md",
            "PACKAGE_RECEIPT.json",
            "SHA256SUMS.txt",
            "source_file_manifest.csv",
        )
    )
    paths.extend(
        ("conservative_evidence", path)
        for path in (RESULT_ROOT / "v23_conservative_evidence").rglob("*.csv")
    )
    paths.extend(("v23_source", path) for path in V23_CODE_ROOT.glob("*"))
    rows = []
    for role, path in sorted(paths, key=lambda item: item[1].as_posix()):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "role": role,
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_summary(manifest: list[dict]) -> dict:
    local_path = RESULT_ROOT / "q4_v230_local_grid_A612_B1_8_pooled150000_summary.json"
    branch_path = RESULT_ROOT / "q4_v230_branch_below_A612B2_A605_611_pooled150000_audit.json"
    a_only_path = RESULT_ROOT / "v23_conservative_evidence" / "pooled_D_v230_150000_prefix.csv"
    receipt_path = RESULT_ROOT / "PACKAGE_RECEIPT.json"
    local = load_json(local_path)
    branch = load_json(branch_path)
    receipt = load_json(receipt_path)
    empirical = local["lowest_cost_point_feasible"]
    a_only = next(row for row in load_csv(a_only_path) if int(row["N_A"]) == 613)
    conservative = receipt["q4_lower_bound_conservative"]

    if (int(empirical["N_A"]), int(empirical["N_B"])) != (612, 2):
        raise ValueError("v23 经验方案不是 (612,2)")
    if int(empirical["successes"]) != 135007 or int(empirical["trials"]) != 150000:
        raise ValueError("v23 经验方案计数不一致")
    assert_close(float(empirical["probability"]), 0.9000466666666667)
    assert_close(float(empirical["CP_one_sided95_lower"]), 0.8987637812739432)
    if (int(conservative["count_A"]), int(conservative["count_B"])) != (613, 1):
        raise ValueError("v23 保守方案不是 (613,1)")
    if int(a_only["successes"]) != 135242 or int(a_only["trials"]) != 150000:
        raise ValueError("v23 A-only 保守证据计数不一致")
    assert_close(float(a_only["CP95_one_sided_lower"]), 0.9003393482431642)
    assert_close(float(branch["PAVA_maximum_absolute_adjustment"]), 2e-5)
    if any(float(row["probability"]) >= 0.9 for row in branch["rows"]):
        raise ValueError("严格更低成本临界分支存在经验可行点")

    return {
        "schema_version": "1.0",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "adoption": {
            "question_3": "retain_current_project_result",
            "question_4": "adopt_v23_refined_geometry_two_criterion_result",
            "geometry_scope": "Q4 actual-cylinder clipping refinement; Q3 registry is unchanged",
        },
        "primary_point_estimate_solution": empirical,
        "confidence_conservative_solution": {
            "N_A": int(conservative["count_A"]),
            "N_B": int(conservative["count_B"]),
            "cost_yuan": float(conservative["cost_yuan"]),
            "direct_mixed_successes": None,
            "direct_mixed_trials": None,
            "inherited_A_only_successes": int(a_only["successes"]),
            "inherited_A_only_trials": int(a_only["trials"]),
            "inherited_probability_floor": float(a_only["probability"]),
            "inherited_CP_one_sided95_lower": float(a_only["CP95_one_sided_lower"]),
            "evidence_type": "monotonic inheritance from v23 A-only N_A=613; adding B removes no existing path",
            "confidence_scope": "nominal one-sided 95% CP for the selected v23 A-only candidate",
        },
        "strictly_cheaper_critical_branch": {
            "candidate_count": int(branch["branch_candidate_count"]),
            "N_A_range": [605, 611],
            "maximum_probability": branch["maximum_probability"],
            "all_point_estimates_below_0_90": True,
            "PAVA_maximum_absolute_adjustment": float(branch["PAVA_maximum_absolute_adjustment"]),
        },
        "integrity": {
            "imported_file_count": len(manifest),
            "q4_data_file_count": sum(row["role"] == "q4_data" for row in manifest),
            "conservative_evidence_file_count": sum(
                row["role"] == "conservative_evidence" for row in manifest
            ),
            "source_file_count": sum(row["role"] == "v23_source" for row in manifest),
            "manifest": MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "key_source_sha256": {
                local_path.relative_to(PROJECT_ROOT).as_posix(): sha256(local_path),
                branch_path.relative_to(PROJECT_ROOT).as_posix(): sha256(branch_path),
                a_only_path.relative_to(PROJECT_ROOT).as_posix(): sha256(a_only_path),
                receipt_path.relative_to(PROJECT_ROOT).as_posix(): sha256(receipt_path),
            },
        },
    }


def update_registry(summary: dict) -> None:
    registry = read_registry()
    q3_before = {
        key: value for key, value in registry["results"].items() if key.startswith("q3_")
    }
    for key in [key for key in registry["results"] if key.startswith("q4_")]:
        del registry["results"][key]

    empirical = summary["primary_point_estimate_solution"]
    conservative = summary["confidence_conservative_solution"]
    branch = summary["strictly_cheaper_critical_branch"]
    source_script = "问题/问题4/src/import_v23_results.py"
    source_artifact = SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix()
    empirical_validation = (
        "v23真实圆柱裁剪精化模型；3个种子合并150000次；固定样本点估计判据；"
        "严格更低成本临界分支7点均低于0.90；PAVA最大调整2e-5"
    )
    conservative_validation = (
        "v23真实圆柱裁剪A-only证据135242/150000；增加B不删除既有导通路径；"
        "选中候选的名义单侧95% Clopper-Pearson下界，不替代当前问题三结论"
    )
    timestamp = summary["frozen_at"]

    def entry(value, formatted, unit, macro, validation):
        return {
            "question": 4,
            "value": value,
            "formatted": formatted,
            "unit": unit,
            "source_script": source_script,
            "source_artifact": source_artifact,
            "validation": validation,
            "latex_macro": macro,
            "updated_at": timestamp,
        }

    entries = {
        "q4_reported_n_a": entry(612, "612", "个", "QFourNA", empirical_validation),
        "q4_reported_n_b": entry(2, "2", "个", "QFourNB", empirical_validation),
        "q4_cost_weight": entry(347132, "347132", "", "QFourCostWeight", empirical_validation),
        "q4_reported_cost_yuan": entry(
            empirical["cost_yuan"], "9.0879", "元", "QFourCost", empirical_validation
        ),
        "q4_candidate_probability": entry(
            empirical["probability"], "90.0047%", "", "QFourProbability", empirical_validation
        ),
        "q4_candidate_cp_lower": entry(
            empirical["CP_one_sided95_lower"], "89.8764%", "", "QFourLower", empirical_validation
        ),
        "q4_empirical_n_a": entry(612, "612", "个", "QFourEmpiricalNA", empirical_validation),
        "q4_empirical_n_b": entry(2, "2", "个", "QFourEmpiricalNB", empirical_validation),
        "q4_empirical_cost_yuan": entry(
            empirical["cost_yuan"], "9.0879", "元", "QFourEmpiricalCost", empirical_validation
        ),
        "q4_empirical_successes": entry(
            empirical["successes"], "135007", "次", None, empirical_validation
        ),
        "q4_empirical_trials": entry(empirical["trials"], "150000", "次", None, empirical_validation),
        "q4_empirical_probability": entry(
            empirical["probability"], "90.0047%", "", None, empirical_validation
        ),
        "q4_empirical_wilson95": entry(
            [empirical["Wilson95_low"], empirical["Wilson95_high"]],
            "[89.8519%, 90.1554%]",
            "",
            None,
            empirical_validation,
        ),
        "q4_empirical_cp_lower": entry(
            empirical["CP_one_sided95_lower"], "89.8764%", "", None, empirical_validation
        ),
        "q4_conservative_n_a": entry(
            conservative["N_A"], "613", "个", "QFourConservativeNA", conservative_validation
        ),
        "q4_conservative_n_b": entry(
            conservative["N_B"], "1", "个", "QFourConservativeNB", conservative_validation
        ),
        "q4_conservative_cost_yuan": entry(
            conservative["cost_yuan"], "9.1011", "元", "QFourConservativeCost", conservative_validation
        ),
        "q4_conservative_probability_floor": entry(
            conservative["inherited_probability_floor"],
            "至少90.1613%",
            "",
            "QFourConservativeProbability",
            conservative_validation,
        ),
        "q4_conservative_cp_lower": entry(
            conservative["inherited_CP_one_sided95_lower"],
            "至少90.0339%",
            "",
            "QFourConservativeLower",
            conservative_validation,
        ),
        "q4_strict_cheaper_max_probability": entry(
            branch["maximum_probability"]["probability"], "89.9427%", "", None, empirical_validation
        ),
        "q4_pava_max_adjustment": entry(
            branch["PAVA_maximum_absolute_adjustment"], "0.000020", "", None, empirical_validation
        ),
    }
    registry["results"].update(entries)
    q3_after = {
        key: value for key, value in registry["results"].items() if key.startswith("q3_")
    }
    if q3_after != q3_before:
        raise RuntimeError("Q3结果注册项发生变化")
    write_registry(registry)


def main() -> None:
    manifest = build_manifest()
    summary = build_summary(manifest)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    update_registry(summary)
    print(
        json.dumps(
            {
                "summary": SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "manifest_rows": len(manifest),
                "primary": [612, 2],
                "conservative": [613, 1],
                "q3_unchanged": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
