from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_q4_mixed_scene as scene_builder
import render_q4_mixed_scene as scene_renderer
import build_q4_witness_figure as witness_builder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURE_ROOT = PROJECT_ROOT / "论文" / "figures"
EXPECTED_N_A = 619
EXPECTED_N_B = 0
SELECTION_RULE = "minimum_trial_id_connected_in_confirmation_artifact"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从问题4最终确认摘要串行生成可发布的三维模型、视图和总审计"
    )
    parser.add_argument("--design-json", type=Path)
    parser.add_argument("--freeze-json", type=Path)
    parser.add_argument("--confirmation-shard", type=Path)
    parser.add_argument("--output-root", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--freecadcmd-exe", type=Path)
    parser.add_argument("--freecad-exe", type=Path)
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument("--axonometric-zoom", type=float, default=0.92)
    parser.add_argument("--top-zoom", type=float, default=1.20)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": scene_builder.sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _paths(output_root: Path, n_a: int, n_b: int, trial_id: int) -> dict[str, Path]:
    stem = f"q4_final_na{n_a:06d}_nb{n_b:06d}_trial{trial_id:06d}"
    root = output_root.expanduser().resolve()
    return {
        "scene": root / "data" / f"{stem}_scene.json",
        "model": root / "models" / f"{stem}.FCStd",
        "build_audit": root / "models" / f"{stem}.build.audit.json",
        "axonometric": root / "rendered" / f"{stem}_axonometric.png",
        "axonometric_audit": root
        / "rendered"
        / f"{stem}_axonometric.audit.json",
        "top": root / "rendered" / f"{stem}_top.png",
        "top_audit": root / "rendered" / f"{stem}_top.audit.json",
        "witness_png": root / "rendered" / f"{stem}_witness_focus.png",
        "witness_pdf": root / "rendered" / f"{stem}_witness_focus.pdf",
        "witness_audit": root / "rendered" / f"{stem}_witness_focus.audit.json",
        "manifest": root / "generated" / f"{stem}_assets.audit.json",
    }


def _render_args(
    args: argparse.Namespace,
    *,
    model: Path,
    scene: Path,
    output: Path,
    audit: Path,
    view: str,
    zoom: float,
) -> argparse.Namespace:
    return argparse.Namespace(
        source=model,
        scene=scene,
        output=output,
        audit=audit,
        freecad_exe=args.freecad_exe,
        width=args.width,
        height=args.height,
        zoom=zoom,
        timeout=args.timeout,
        view=view,
        focus_witness=False,
    )


def _discover_source(args: argparse.Namespace) -> scene_builder.DesignSource:
    freeze_json = getattr(args, "freeze_json", None)
    confirmation_shard = getattr(args, "confirmation_shard", None)
    design_json = getattr(args, "design_json", None)
    if freeze_json is not None or confirmation_shard is not None:
        if design_json is not None or freeze_json is None or confirmation_shard is None:
            raise ValueError("冻结分片模式要求同时给出 freeze/shard，且不使用 final summary")
        return scene_builder.load_frozen_shard_design(freeze_json, confirmation_shard)
    return scene_builder.discover_design(design_json)


def build_assets(args: argparse.Namespace) -> dict[str, Any]:
    source = _discover_source(args)
    if (source.n_a, source.n_b) != (EXPECTED_N_A, EXPECTED_N_B):
        raise ValueError(
            "最终确认设计与冻结候选不一致："
            f"expected=({EXPECTED_N_A},{EXPECTED_N_B}), "
            f"actual=({source.n_a},{source.n_b})"
        )
    trial_id = scene_builder.select_conductive_trial(source)
    scene = scene_builder.build_verified_trial_scene(source, trial_id)
    counts = scene["traceability"]["geometry_counts"]
    if counts["a_source_particles"] != EXPECTED_N_A:
        raise RuntimeError("正式场景未覆盖全部 619 个 A 源粒子")
    if counts["b_source_particles"] != 0 or scene.get("spheres"):
        raise RuntimeError("N_B=0 的正式场景不得包含 B 几何")
    if scene["traceability"]["random_stream"]["trial_id"] != trial_id:
        raise RuntimeError("场景随机流 trial_id 与选择结果不一致")

    paths = _paths(args.output_root, source.n_a, source.n_b, trial_id)
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else paths["manifest"]
    )
    for path in (*paths.values(), manifest_path):
        if "preview" in path.name.casefold():
            raise ValueError(f"正式产物路径不得含 preview：{path}")

    scene_builder.atomic_write_json(paths["scene"], scene)
    build_audit = scene_builder.build_fcstd(
        paths["scene"],
        paths["model"],
        paths["build_audit"],
        args.freecadcmd_exe,
        args.timeout,
    )
    axonometric_audit = scene_renderer.render_q4(
        _render_args(
            args,
            model=paths["model"],
            scene=paths["scene"],
            output=paths["axonometric"],
            audit=paths["axonometric_audit"],
            view="axonometric",
            zoom=args.axonometric_zoom,
        )
    )
    top_audit = scene_renderer.render_q4(
        _render_args(
            args,
            model=paths["model"],
            scene=paths["scene"],
            output=paths["top"],
            audit=paths["top_audit"],
            view="top",
            zoom=args.top_zoom,
        )
    )
    witness_audit = witness_builder.build_figure(
        paths["scene"],
        paths["witness_png"],
        paths["witness_pdf"],
        paths["witness_audit"],
    )

    witness = scene["traceability"]["mixed_witness"]
    manifest = {
        "kind": "q4_final_3d_assets_audit",
        "schema_version": 1,
        "status": "passed",
        "publication_status": scene["publication_status"],
        "selection_rule": SELECTION_RULE,
        "design": {"n_a": source.n_a, "n_b": source.n_b},
        "trial_id": trial_id,
        "source_design_evidence": _file_record(source.source_path),
        "source_design_evidence_status": source.source_status,
        "source_confirmation_artifact": _file_record(source.artifact_path),
        "configuration": {
            "maximum_fingerprint": source.artifact_configuration_fingerprint,
            "selected_fingerprint": source.selected_configuration_fingerprint,
            "random_stream": scene["traceability"]["random_stream"],
            "boundary_primary": scene["traceability"]["boundary_primary"],
        },
        "geometry_counts": counts,
        "witness": {
            "node_count": len(witness["nodes"]),
            "edge_count": witness["edge_count"],
            "same_source_edges": witness["same_source_edges"],
            "all_edges_geometry_verified": witness["all_edges_geometry_verified"],
        },
        "artifacts": {
            key: _file_record(paths[key])
            for key in (
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
            )
        },
        "checks": {
            "candidate_matches_frozen_counts": True,
            "no_b_geometry_when_n_b_zero": True,
            "minimum_connected_trial_selected": True,
            "freecad_build_passed": build_audit["status"] == "passed",
            "axonometric_render_passed": axonometric_audit["status"] == "passed",
            "top_render_passed": top_audit["status"] == "passed",
            "witness_focus_passed": witness_audit["status"] == "passed",
            "formal_paths_exclude_preview": True,
        },
    }
    if not all(manifest["checks"].values()):
        raise RuntimeError("问题4正式三维产物总审计未通过")
    scene_builder.atomic_write_json(manifest_path, manifest)
    manifest["manifest"] = _file_record(manifest_path)
    return manifest


def main() -> int:
    result = build_assets(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
