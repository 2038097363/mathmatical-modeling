# AI 工具：OpenAI Codex；模型/版本：GPT-5 系列；开发机构：OpenAI。
# 版本发布日期：2025-08-07（GPT-5 系列公开快照日期）；本程序由参赛队逐行复核并对结果负责。
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import FreeCAD as App
import Part


ROLE_STYLE = {
    "background_a": ((0.72, 0.75, 0.78), 62),
    "background_b": ((0.20, 0.55, 0.82), 25),
    "witness": ((0.93, 0.36, 0.12), 0),
    "boundary_fragment": ((0.84, 0.20, 0.26), 5),
    "candidate": ((0.18, 0.64, 0.45), 10),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="由统一场景文件生成 FreeCAD 三维实体模型")
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step", type=Path)
    return parser.parse_args()


def vector(values: list[float]) -> App.Vector:
    if len(values) != 3 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"三维坐标非法：{values}")
    return App.Vector(*(float(value) for value in values))


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return cleaned[:48] or "Object"


def add_feature(
    document: App.Document,
    name: str,
    label: str,
    shape: Part.Shape,
    color: tuple[float, float, float],
    transparency: int,
    source_ids: list[str] | None = None,
) -> App.DocumentObject:
    feature = document.addObject("Part::Feature", safe_name(name))
    feature.Label = label
    feature.Shape = shape
    if source_ids:
        feature.addProperty("App::PropertyStringList", "SourceIds", "Traceability")
        feature.SourceIds = source_ids
    view = getattr(feature, "ViewObject", None)
    if view is not None:
        view.ShapeColor = color
        view.LineColor = tuple(max(0.0, component * 0.72) for component in color)
        view.Transparency = transparency
    return feature


def cylinder_shape(record: dict) -> Part.Shape:
    start = vector(record["start_nm"])
    end = vector(record["end_nm"])
    axis = end.sub(start)
    length = axis.Length
    radius = float(record["radius_nm"])
    if length <= 1e-9 or radius <= 0.0:
        raise ValueError(f"圆柱 {record.get('id', '')} 的长度或半径非法")
    return Part.makeCylinder(radius, length, start, axis)


def sphere_shape(record: dict) -> Part.Shape:
    radius = float(record["radius_nm"])
    if radius <= 0.0:
        raise ValueError(f"球体 {record.get('id', '')} 的半径非法")
    sphere = Part.makeSphere(radius, vector(record["center_nm"]))
    has_lower = "clip_box_lower_nm" in record
    has_upper = "clip_box_upper_nm" in record
    if has_lower != has_upper:
        raise ValueError(
            f"截球 {record.get('id', '')} 必须同时给出 clip_box_lower_nm/clip_box_upper_nm"
        )
    if not has_lower:
        return sphere

    lower_values = [float(value) for value in record["clip_box_lower_nm"]]
    upper_values = [float(value) for value in record["clip_box_upper_nm"]]
    if len(lower_values) != 3 or len(upper_values) != 3:
        raise ValueError(f"截球 {record.get('id', '')} 的裁剪盒必须是三维盒")
    if not all(math.isfinite(value) for value in lower_values + upper_values):
        raise ValueError(f"截球 {record.get('id', '')} 的裁剪盒坐标非法")
    lengths = [upper - lower for lower, upper in zip(lower_values, upper_values)]
    if not all(length > 0.0 for length in lengths):
        raise ValueError(f"截球 {record.get('id', '')} 的裁剪盒必须有正体积")
    clip_box = Part.makeBox(*lengths, vector(lower_values))
    clipped = sphere.common(clip_box)
    if clipped.isNull() or float(clipped.Volume) <= 0.0:
        raise ValueError(f"截球 {record.get('id', '')} 与基础盒没有正体积交集")
    return clipped


def _source_index(record: dict, kind: str) -> int | None:
    if "source_index" not in record:
        return None
    value = record["source_index"]
    if isinstance(value, bool):
        raise ValueError(f"{kind} 的 source_index 不能是布尔值")
    numeric = float(value)
    index = int(numeric)
    if not math.isfinite(numeric) or numeric != index or index < 0:
        raise ValueError(f"{kind} 的 source_index 必须是非负整数")
    return index


def build_scene(scene: dict, output: Path, step_path: Path | None) -> dict:
    title = str(scene.get("name", "Microstructure"))
    document = App.newDocument(safe_name(title))
    exported_objects: list[App.DocumentObject] = []

    box = scene.get("box", {})
    length = float(box.get("length_nm", 10_000.0))
    if length <= 0.0:
        raise ValueError("立方体边长必须为正")
    half = length / 2.0
    if bool(box.get("show", True)):
        box_shape = Part.makeBox(
            length,
            length,
            length,
            App.Vector(-half, -half, -half),
        )
        box_object = add_feature(
            document,
            "Domain",
            "微构体边界",
            box_shape,
            (0.78, 0.82, 0.86),
            int(box.get("transparency", 92)),
        )
        exported_objects.append(box_object)

    electrodes = scene.get("electrodes", {})
    if bool(electrodes.get("show", True)):
        thickness = float(electrodes.get("thickness_nm", max(12.0, length * 0.002)))
        if thickness <= 0.0:
            raise ValueError("电极厚度必须为正")
        left_shape = Part.makeBox(
            thickness,
            length,
            length,
            App.Vector(-half - thickness, -half, -half),
        )
        right_shape = Part.makeBox(
            thickness,
            length,
            length,
            App.Vector(half, -half, -half),
        )
        electrode_color = (0.16, 0.18, 0.20)
        electrode_transparency = int(electrodes.get("transparency", 8))
        if not 0 <= electrode_transparency <= 100:
            raise ValueError("电极透明度必须位于 0 到 100")
        exported_objects.append(
            add_feature(
                document,
                "LeftElectrode",
                "左带电面",
                left_shape,
                electrode_color,
                electrode_transparency,
            )
        )
        exported_objects.append(
            add_feature(
                document,
                "RightElectrode",
                "右带电面",
                right_shape,
                electrode_color,
                electrode_transparency,
            )
        )

    publication_status = str(scene.get("publication_status", "unspecified"))
    if exported_objects:
        metadata_host = exported_objects[0]
        metadata_host.addProperty(
            "App::PropertyString", "PublicationStatus", "Traceability"
        )
        metadata_host.PublicationStatus = publication_status
        metadata_host.addProperty(
            "App::PropertyString", "SceneMetadataJSON", "Traceability"
        )
        metadata_host.SceneMetadataJSON = json.dumps(
            scene.get("traceability", {}), ensure_ascii=True, sort_keys=True
        )

    grouped_shapes: dict[
        tuple[str, str], list[tuple[str, Part.Shape]]
    ] = defaultdict(list)
    identifiers: set[str] = set()
    source_indices: dict[str, set[int]] = {"A": set(), "B": set()}
    fragment_keys: set[tuple[str, int, int]] = set()
    clipped_sphere_count = 0
    witness_fragment_count = 0
    domain_tolerance = max(1e-7, 1e-10 * length)

    def register_record(record: dict, kind: str) -> tuple[str, int | None]:
        nonlocal witness_fragment_count
        identifier = str(record.get("id", "")).strip()
        if not identifier:
            raise ValueError(f"{kind} 记录缺少非空 id")
        if identifier in identifiers:
            raise ValueError(f"场景含重复几何 id：{identifier}")
        identifiers.add(identifier)
        source_index = _source_index(record, kind)
        if source_index is not None:
            source_indices[kind].add(source_index)
            if "fragment_index" in record:
                fragment_value = record["fragment_index"]
                if isinstance(fragment_value, bool):
                    raise ValueError(f"{identifier} 的 fragment_index 不能是布尔值")
                numeric = float(fragment_value)
                fragment_index = int(numeric)
                if (
                    not math.isfinite(numeric)
                    or numeric != fragment_index
                    or fragment_index < 0
                ):
                    raise ValueError(f"{identifier} 的 fragment_index 必须是非负整数")
                key = (kind, source_index, fragment_index)
                if key in fragment_keys:
                    raise ValueError(f"场景含重复源片段键：{key}")
                fragment_keys.add(key)
            if "cell_shift" in record:
                shifts = record["cell_shift"]
                if not isinstance(shifts, (list, tuple)) or len(shifts) != 3:
                    raise ValueError(f"{identifier} 的 cell_shift 必须是三个整数")
                for value in shifts:
                    if isinstance(value, bool):
                        raise ValueError(f"{identifier} 的 cell_shift 不能含布尔值")
                    numeric = float(value)
                    if not math.isfinite(numeric) or numeric != int(numeric):
                        raise ValueError(f"{identifier} 的 cell_shift 必须是三个整数")
        if str(record.get("role", "")) == "witness":
            witness_fragment_count += 1
        return identifier, source_index

    for record in scene.get("cylinders", []):
        identifier, _ = register_record(record, "A")
        role = str(record.get("role", "background_a"))
        shape = cylinder_shape(record)
        start = [float(value) for value in record["start_nm"]]
        end = [float(value) for value in record["end_nm"]]
        if not all(
            -half - domain_tolerance <= value <= half + domain_tolerance
            for value in (*start, *end)
        ):
            raise ValueError(f"圆柱片段 {identifier} 的中心线端点超出基础盒")
        if shape.isNull() or float(shape.Volume) <= 0.0:
            raise ValueError(f"圆柱片段 {identifier} 不是正体积实体")
        grouped_shapes[("cylinder", role)].append((identifier, shape))
    for record in scene.get("spheres", []):
        identifier, _ = register_record(record, "B")
        role = str(record.get("role", "background_b"))
        clipped = "clip_box_lower_nm" in record
        clipped_sphere_count += int(clipped)
        shape = sphere_shape(record)
        if clipped:
            clip_lower = [float(value) for value in record["clip_box_lower_nm"]]
            clip_upper = [float(value) for value in record["clip_box_upper_nm"]]
            expected_lower = [-half, -half, -half]
            expected_upper = [half, half, half]
            if any(
                abs(value - expected) > domain_tolerance
                for value, expected in zip(clip_lower, expected_lower)
            ) or any(
                abs(value - expected) > domain_tolerance
                for value, expected in zip(clip_upper, expected_upper)
            ):
                raise ValueError(f"截球 {identifier} 的裁剪盒不是基础盒")
        else:
            center = [float(value) for value in record["center_nm"]]
            radius = float(record["radius_nm"])
            if any(
                value - radius < -half - domain_tolerance
                or value + radius > half + domain_tolerance
                for value in center
            ):
                raise ValueError(f"完整球 {identifier} 的实体边界超出基础盒")
        grouped_shapes[("sphere", role)].append((identifier, shape))

    cylinders = list(scene.get("cylinders", []))
    spheres = list(scene.get("spheres", []))
    if publication_status == "final_random_trial_geometry":
        required_trace_fields = ("source_index", "fragment_index", "cell_shift")
        if any(
            any(field not in record for field in required_trace_fields)
            for record in cylinders + spheres
        ):
            raise ValueError(
                "正式随机试验场景的每个介质片段都必须记录 "
                "source_index/fragment_index/cell_shift"
            )
        design_counts = scene.get("traceability", {}).get("design_counts", {})
        expected_a = int(design_counts.get("n_a", -1))
        expected_b = int(design_counts.get("n_b", -1))
        if expected_a < 0 or expected_b < 0:
            raise ValueError("正式随机试验场景缺少 design_counts")
        if witness_fragment_count < 1:
            raise ValueError("正式随机试验场景缺少导通见证片段")
        expected_a_indices = set(range(expected_a))
        expected_b_indices = set(range(expected_b))
        if (
            source_indices["A"] != expected_a_indices
            or source_indices["B"] != expected_b_indices
        ):
            raise ValueError(
                "正式随机试验场景的 source_index 覆盖与 design_counts 不一致："
                f"A={len(source_indices['A'])}/{expected_a}, "
                f"B={len(source_indices['B'])}/{expected_b}"
            )

    counts: dict[str, int] = {}
    for (kind, role), items in sorted(grouped_shapes.items()):
        color, transparency = ROLE_STYLE.get(role, ROLE_STYLE["candidate"])
        role_label = {
            "background_a": "介质A",
            "background_b": "介质B",
            "witness": "导通见证",
            "boundary_fragment": "边界搬移片段",
            "candidate": "候选介质",
        }.get(role, role)
        shape = Part.makeCompound([item[1] for item in items])
        object_name = f"{kind}_{role}"
        feature = add_feature(
            document,
            object_name,
            f"{role_label}（{len(items)}个）",
            shape,
            color,
            transparency,
            [item[0] for item in items],
        )
        exported_objects.append(feature)
        counts[object_name] = len(items)

    document.recompute()
    output.parent.mkdir(parents=True, exist_ok=True)
    document.recompute()
    document.saveAs(str(output.resolve()))
    if step_path is not None:
        step_path.parent.mkdir(parents=True, exist_ok=True)
        Part.export(exported_objects, str(step_path.resolve()))

    return {
        "scene": title,
        "scene_sha256": hashlib.sha256(
            json.dumps(scene, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest().upper(),
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "output_size_bytes": output.stat().st_size,
        "step": str(step_path.resolve()) if step_path is not None else None,
        "counts": counts,
        "geometry_audit": {
            "cylinder_fragment_count": len(cylinders),
            "sphere_fragment_count": len(spheres),
            "total_fragment_count": len(cylinders) + len(spheres),
            "clipped_sphere_fragment_count": clipped_sphere_count,
            "full_sphere_fragment_count": len(spheres) - clipped_sphere_count,
            "witness_fragment_count": witness_fragment_count,
            "unique_source_particles": {
                "A": len(source_indices["A"]),
                "B": len(source_indices["B"]),
            },
            "unique_geometry_ids": len(identifiers),
            "duplicate_geometry_ids": 0,
            "positive_volume_shapes": True,
            "sphere_shapes_inside_base_box": True,
            "cylinder_centerlines_inside_base_box": True,
            "domain_tolerance_nm": domain_tolerance,
        },
        "freecad_version": ".".join(App.Version()[:3]),
    }


def main() -> None:
    args = parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8-sig"))
    metadata = build_scene(scene, args.output, args.step)
    metadata["scene_file_sha256"] = file_sha256(args.scene)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
