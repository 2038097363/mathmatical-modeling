from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT / "公共代码"
DEFAULT_SHARD = (
    ROOT
    / "问题"
    / "问题4"
    / "results"
    / "D_screen2000_confirm50000"
    / "confirmation"
    / "pareto_max_A000619_B005483"
    / "shards"
    / "shard_000000_000099.json"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_case(shard_path: Path, trial_id: int):
    if str(COMMON_DIR) not in sys.path:
        sys.path.insert(0, str(COMMON_DIR))
    import mixed_microstructure_sim as mixed

    payload = json.loads(shard_path.read_text(encoding="utf-8-sig"))
    records = {int(record["trial_id"]): record for record in payload["records"]}
    if trial_id not in records:
        raise ValueError(f"trial {trial_id} is not present in {shard_path}")
    config = mixed.MixedSimulationConfig.from_dict(payload["configuration"])
    seconds_per_trial = float(payload["runtime_seconds"]) / len(payload["records"])
    return mixed, config, records[trial_id], seconds_per_trial


def _ab_full_pair(first: object, second: object, cylinder_type, sphere_type):
    if isinstance(first, cylinder_type) and isinstance(second, sphere_type):
        return first, second
    if isinstance(first, sphere_type) and isinstance(second, cylinder_type):
        return second, first
    return None


def _array_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest().upper()


def capture_phase(shard_path: Path, trial_id: int, exchange: Path) -> dict[str, Any]:
    mixed, config, stored, stored_cpu_seconds = _load_case(shard_path, trial_id)
    from geometry_kernel import Cylinder, Sphere

    original = mixed.evaluate_exact_contact
    pairs = []

    def capture(first, second, current):
        result = original(first, second, current)
        normalized = _ab_full_pair(first, second, Cylinder, Sphere)
        if normalized is not None:
            pairs.append((*normalized, result))
        return result

    mixed.evaluate_exact_contact = capture
    started = time.perf_counter()
    try:
        record = mixed.run_one_pareto_trial(config, trial_id)
    finally:
        mixed.evaluate_exact_contact = original
    capture_seconds = time.perf_counter() - started
    if record != stored:
        raise RuntimeError("CPU recomputation differs from the completed shard")

    pack_started = time.perf_counter()
    arrays = {
        "center": np.stack([pair[0].center for pair in pairs]),
        "axis": np.stack([pair[0].axis for pair in pairs]),
        "half_length": np.asarray([pair[0].half_length for pair in pairs]),
        "cylinder_radius": np.asarray([pair[0].radius for pair in pairs]),
        "sphere_center": np.stack([pair[1].center for pair in pairs]),
        "sphere_radius": np.asarray([pair[1].radius for pair in pairs]),
        "expected_broad": np.asarray(
            [pair[2].broad_phase_rejected for pair in pairs], dtype=np.bool_
        ),
        "expected_connected": np.asarray(
            [pair[2].connected for pair in pairs], dtype=np.bool_
        ),
        "expected_narrow": np.asarray(
            [pair[2].narrow_phase_calls for pair in pairs], dtype=np.int64
        ),
        "contact_cutoff_nm": np.asarray(config.contact_cutoff_nm),
        "broad_phase_guard_nm": np.asarray(config.broad_phase_guard_nm),
    }
    pack_seconds = time.perf_counter() - pack_started
    exchange.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(exchange / "ab_pairs.npz", **arrays)
    payload = {
        "phase": "capture",
        "trial_id": trial_id,
        "configuration_fingerprint": config.fingerprint,
        "stored_cpu_seconds_per_trial": stored_cpu_seconds,
        "capture_seconds": capture_seconds,
        "cpu_recomputation_full_record_match": True,
        "ab_full_pair_count": len(pairs),
        "pack_seconds": pack_seconds,
        "pair_fingerprint": _array_fingerprint(arrays),
    }
    _write_json(exchange / "capture.json", payload)
    return payload


def _cuda_classify(torch, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    device = torch.device("cuda")
    values = {
        name: torch.as_tensor(arrays[name], dtype=torch.float64, device=device)
        for name in (
            "center",
            "axis",
            "half_length",
            "cylinder_radius",
            "sphere_center",
            "sphere_radius",
        )
    }
    center = values["center"]
    axis = values["axis"]
    half_length = values["half_length"]
    cylinder_radius = values["cylinder_radius"]
    sphere_center = values["sphere_center"]
    sphere_radius = values["sphere_radius"]
    cutoff = float(arrays["contact_cutoff_nm"])
    broad_guard = float(arrays["broad_phase_guard_nm"])

    endpoint_a = center - half_length[:, None] * axis
    direction = 2.0 * half_length[:, None] * axis
    length_sq = torch.sum(direction * direction, dim=1)
    parameter = torch.clamp(
        torch.sum((sphere_center - endpoint_a) * direction, dim=1) / length_sq,
        0.0,
        1.0,
    )
    closest = endpoint_a + parameter[:, None] * direction
    capsule_gap = torch.clamp(
        torch.linalg.vector_norm(sphere_center - closest, dim=1)
        - cylinder_radius
        - sphere_radius,
        min=0.0,
    )
    broad = capsule_gap > (cutoff + broad_guard)

    offset = sphere_center - center
    axial = torch.sum(offset * axis, dim=1)
    radial = offset - axial[:, None] * axis
    axial_gap = torch.clamp(torch.abs(axial) - half_length, min=0.0)
    radial_gap = torch.clamp(
        torch.linalg.vector_norm(radial, dim=1) - cylinder_radius,
        min=0.0,
    )
    exact = torch.clamp(torch.hypot(axial_gap, radial_gap) - sphere_radius, min=0.0)
    scale = torch.maximum(torch.ones_like(exact), exact)
    scale = torch.maximum(scale, torch.full_like(scale, cutoff))
    scale = torch.maximum(scale, torch.hypot(half_length, cylinder_radius))
    scale = torch.maximum(scale, sphere_radius)
    scale = torch.maximum(scale, torch.linalg.vector_norm(center, dim=1))
    scale = torch.maximum(scale, torch.linalg.vector_norm(sphere_center, dim=1))
    floating_guard = 64.0 * torch.finfo(torch.float64).eps * scale
    connected = (~broad) & (exact <= cutoff + floating_guard)
    narrow = (~broad).to(torch.int64)
    torch.cuda.synchronize()
    return {
        "broad": broad.cpu().numpy(),
        "connected": connected.cpu().numpy(),
        "narrow": narrow.cpu().numpy(),
        "capsule_gap": capsule_gap.cpu().numpy(),
        "exact": exact.cpu().numpy(),
    }


def cuda_phase(exchange: Path, repeats: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is not available")
    with np.load(exchange / "ab_pairs.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    gpu = _cuda_classify(torch, arrays)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        gpu = _cuda_classify(torch, arrays)
        samples.append(time.perf_counter() - started)
    mismatches = {
        "broad_phase_rejected": int(
            np.count_nonzero(gpu["broad"] != arrays["expected_broad"])
        ),
        "connected": int(
            np.count_nonzero(gpu["connected"] != arrays["expected_connected"])
        ),
        "narrow_phase_calls": int(
            np.count_nonzero(gpu["narrow"] != arrays["expected_narrow"])
        ),
    }
    np.savez_compressed(exchange / "cuda_results.npz", **gpu)
    payload = {
        "phase": "cuda",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "repeats": repeats,
        "cuda_batch_seconds_including_transfers": statistics.median(samples),
        "cuda_pair_mismatches": mismatches,
    }
    _write_json(exchange / "cuda.json", payload)
    if any(mismatches.values()):
        raise RuntimeError(f"CUDA A-B classifications differ: {mismatches}")
    return payload


def replay_phase(shard_path: Path, trial_id: int, exchange: Path) -> dict[str, Any]:
    mixed, config, stored, _ = _load_case(shard_path, trial_id)
    from geometry_kernel import Cylinder, Sphere

    with np.load(exchange / "cuda_results.npz", allow_pickle=False) as archive:
        gpu = {name: archive[name] for name in archive.files}
    original = mixed.evaluate_exact_contact
    cursor = 0

    def replay(first, second, current):
        nonlocal cursor
        normalized = _ab_full_pair(first, second, Cylinder, Sphere)
        if normalized is None:
            return original(first, second, current)
        index = cursor
        cursor += 1
        broad = bool(gpu["broad"][index])
        if broad:
            lower = max(
                0.0,
                float(gpu["capsule_gap"][index]) - current.broad_phase_guard_nm,
            )
            return mixed.ExactContactResult(
                False,
                "A-B",
                "cuda_capsule_lower_bound",
                True,
                0,
                None,
                lower,
                None,
            )
        distance = float(gpu["exact"][index])
        return mixed.ExactContactResult(
            bool(gpu["connected"][index]),
            "A-B",
            "cuda_flat_cylinder_sphere_closed_form",
            False,
            1,
            distance,
            distance,
            True,
        )

    mixed.evaluate_exact_contact = replay
    started = time.perf_counter()
    try:
        record = mixed.run_one_pareto_trial(config, trial_id)
    finally:
        mixed.evaluate_exact_contact = original
    replay_seconds = time.perf_counter() - started
    payload = {
        "phase": "replay",
        "cuda_replay_seconds": replay_seconds,
        "cuda_replay_pair_count": cursor,
        "cuda_replay_full_record_match": record == stored,
    }
    _write_json(exchange / "replay.json", payload)
    if not payload["cuda_replay_full_record_match"]:
        raise RuntimeError("CUDA replay differs from the completed shard")
    return payload


def _run_child(executable: Path, arguments: list[str]) -> None:
    completed = subprocess.run(
        [str(executable), str(Path(__file__).resolve()), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeError(
            f"child failed ({completed.returncode}): {completed.stderr.strip()}"
        )


def all_phases(args: argparse.Namespace) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    exchange = ROOT / "tmp" / f"q4_gpu_parity_trial{args.trial_id:06d}_{stamp}"
    exchange.mkdir(parents=True, exist_ok=False)
    cpu_python = args.cpu_python
    common = [
        "--shard",
        str(args.shard.resolve()),
        "--trial-id",
        str(args.trial_id),
        "--exchange",
        str(exchange.resolve()),
    ]
    _run_child(cpu_python, ["--phase", "capture", *common])
    _run_child(
        Path(sys.executable),
        ["--phase", "cuda", "--repeats", str(args.repeats), *common],
    )
    _run_child(cpu_python, ["--phase", "replay", *common])
    capture = json.loads((exchange / "capture.json").read_text(encoding="utf-8"))
    cuda = json.loads((exchange / "cuda.json").read_text(encoding="utf-8"))
    replay = json.loads((exchange / "replay.json").read_text(encoding="utf-8"))
    optimistic_hybrid = (
        replay["cuda_replay_seconds"]
        + capture["pack_seconds"]
        + cuda["cuda_batch_seconds_including_transfers"]
    )
    speedup = capture["stored_cpu_seconds_per_trial"] / optimistic_hybrid
    return {
        "kind": "q4_torch_cuda_ab_feasibility_benchmark",
        **{key: value for key, value in capture.items() if key != "phase"},
        **{key: value for key, value in cuda.items() if key != "phase"},
        **{key: value for key, value in replay.items() if key != "phase"},
        "optimistic_hybrid_seconds": optimistic_hybrid,
        "optimistic_end_to_end_speedup": speedup,
        "advance_to_large_run": bool(
            replay["cuda_replay_full_record_match"] and speedup > 2.0
        ),
        "exchange_directory": str(exchange.resolve()),
    }


def default_cpu_python() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    candidate = local / "Programs" / "Python" / "Python313" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("all", "capture", "cuda", "replay"), default="all"
    )
    parser.add_argument("--shard", type=Path, default=DEFAULT_SHARD)
    parser.add_argument("--trial-id", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--exchange", type=Path)
    parser.add_argument("--cpu-python", type=Path, default=default_cpu_python())
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trial_id < 0 or args.repeats < 1:
        raise ValueError("trial-id must be nonnegative and repeats must be positive")
    if args.phase != "all" and args.exchange is None:
        raise ValueError("exchange is required for a worker phase")
    if args.phase == "capture":
        payload = capture_phase(args.shard, args.trial_id, args.exchange)
    elif args.phase == "cuda":
        payload = cuda_phase(args.exchange, args.repeats)
    elif args.phase == "replay":
        payload = replay_phase(args.shard, args.trial_id, args.exchange)
    else:
        payload = all_phases(args)
    output = json.dumps(payload, ensure_ascii=True, indent=2)
    print(output)
    if args.output is not None:
        _write_json(args.output, payload)


if __name__ == "__main__":
    main()
