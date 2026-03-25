from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evaluate.datasets import (
    find_calib_path,
    get_scenes,
    normalize_dataset_type,
    resolve_scene_root,
    safe_scene_id,
)
from evaluate.public_config import LEANGATE_POLICY_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_ROOT = REPO_ROOT / "outputs" / "predictions"
DEFAULT_SLAM_OUTPUT_ROOT = REPO_ROOT / "outputs" / "slam"
DEFAULT_STAGING_ROOT = REPO_ROOT / "outputs" / "mast3r_sparse_inputs"
DEFAULT_MAST3R_CONFIG_NO_CALIB = REPO_ROOT / "configs" / "mast3r_slam_public_no_calib.yaml"
DEFAULT_MAST3R_CONFIG_CALIB = REPO_ROOT / "configs" / "mast3r_slam_public_calib.yaml"


@dataclass(frozen=True)
class ManifestEntry:
    timestamp: str
    relative_path: str
    absolute_path: Path


@dataclass(frozen=True)
class CalibrationSpec:
    width: int
    height: int
    calibration: list[float]
    source_path: Path


@dataclass(frozen=True)
class SceneRunResult:
    dataset_type: str
    dataset_slug: str
    scene_id: str
    scene_safe: str
    returncode: int
    status: str
    predictions_manifest: Path
    staged_scene_root: Path
    output_dir: Path
    trajectory_path: Path | None
    reconstruction_path: Path | None
    calibration_path: Path | None


def _read_prediction_manifest(predictions_manifest: Path, scene_root: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with predictions_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed manifest line in {predictions_manifest}: {stripped!r}")
            timestamp, rel_path = parts
            absolute_path = (scene_root / rel_path).resolve(strict=False)
            if not absolute_path.exists():
                raise FileNotFoundError(
                    f"Manifest entry {rel_path!r} does not exist on disk for scene {scene_root}"
                )
            entries.append(
                ManifestEntry(
                    timestamp=timestamp,
                    relative_path=rel_path,
                    absolute_path=absolute_path,
                )
            )
    if not entries:
        raise ValueError(f"No usable frames found in predictions manifest {predictions_manifest}")
    return entries


def _sequence_resolution(scene_root: Path) -> tuple[int, int]:
    from slam_prefilter.utils.data_loader import SceneFrameSequence

    sequence, meta = SceneFrameSequence.from_scene_root(scene_root)
    del sequence
    width, height = meta.resolution
    return int(width), int(height)


def _as_float_list(raw: object) -> list[float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            return [float(x) for x in raw]
        except Exception:
            return None
    return None


def _find_scalar(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in mapping:
            try:
                return int(mapping[key])
            except Exception:
                pass
    for value in mapping.values():
        if isinstance(value, dict):
            found = _find_scalar(value, keys)
            if found is not None:
                return found
    return None


def _extract_calibration_from_mapping(mapping: dict[str, Any]) -> list[float] | None:
    fx = mapping.get("fx")
    fy = mapping.get("fy")
    cx = mapping.get("cx")
    cy = mapping.get("cy")
    if all(v is not None for v in (fx, fy, cx, cy)):
        try:
            return [float(fx), float(fy), float(cx), float(cy)]
        except Exception:
            pass

    for key in ("calibration", "intrinsics"):
        values = _as_float_list(mapping.get(key))
        if values is not None:
            return values

    for key in ("camera_matrix", "K"):
        value = mapping.get(key)
        if isinstance(value, dict):
            data = _as_float_list(value.get("data"))
            if data is not None and len(data) >= 9:
                return [float(data[0]), float(data[4]), float(data[2]), float(data[5])]
        values = _as_float_list(value)
        if values is not None and len(values) >= 9:
            return [float(values[0]), float(values[4]), float(values[2]), float(values[5])]

    for value in mapping.values():
        if isinstance(value, dict):
            nested = _extract_calibration_from_mapping(value)
            if nested is not None:
                return nested
    return None


def _parse_camera_txt(calib_path: Path) -> list[float] | None:
    raw = calib_path.read_text(encoding="utf-8").split()
    try:
        values = [float(x) for x in raw]
    except Exception:
        return None
    if len(values) >= 9:
        return [float(values[0]), float(values[4]), float(values[2]), float(values[5])]
    if len(values) >= 4:
        return values[:4]
    return None


def _resolve_calibration_spec(scene_root: Path, dataset_type: str) -> CalibrationSpec | None:
    calib_path = find_calib_path(scene_root, dataset_type)
    if calib_path is None or not calib_path.exists():
        return None

    width: int | None = None
    height: int | None = None
    calibration: list[float] | None = None

    if calib_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(calib_path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            width = _find_scalar(payload, ("width", "image_width"))
            height = _find_scalar(payload, ("height", "image_height"))
            calibration = _extract_calibration_from_mapping(payload)
    else:
        calibration = _parse_camera_txt(calib_path)

    if calibration is None:
        return None
    if width is None or height is None:
        width, height = _sequence_resolution(scene_root)

    return CalibrationSpec(
        width=int(width),
        height=int(height),
        calibration=[float(v) for v in calibration],
        source_path=calib_path,
    )


def _write_intrinsics_yaml(target_path: Path, spec: CalibrationSpec) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "width": int(spec.width),
        "height": int(spec.height),
        "calibration": [float(v) for v in spec.calibration],
        "source_path": str(spec.source_path),
    }
    target_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target_path


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path, copy_mode: str) -> None:
    _replace_path(dst)
    if copy_mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _stage_sparse_scene(
    entries: list[ManifestEntry],
    staging_scene_root: Path,
    copy_mode: str,
) -> Path:
    if staging_scene_root.exists():
        shutil.rmtree(staging_scene_root)
    staging_scene_root.mkdir(parents=True, exist_ok=True)

    staged_manifest = staging_scene_root / "sparse_rgb.txt"
    with staged_manifest.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp staged_filename source_relpath\n")
        for idx, entry in enumerate(entries):
            staged_name = f"frame_{idx:06d}.png"
            staged_path = staging_scene_root / staged_name
            _link_or_copy(entry.absolute_path, staged_path, copy_mode=copy_mode)
            handle.write(f"{entry.timestamp} {staged_name} {entry.relative_path}\n")
    return staged_manifest


def _mast3r_pythonpath() -> str:
    roots = [
        REPO_ROOT / "third_party" / "MASt3R-SLAM",
        REPO_ROOT / "third_party" / "MASt3R-SLAM" / "thirdparty" / "mast3r",
        REPO_ROOT / "third_party" / "MASt3R-SLAM" / "thirdparty" / "in3d",
    ]
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(p) for p in roots]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _default_mast3r_config(calibration_mode: str, has_calibration: bool) -> Path:
    if calibration_mode == "always":
        return DEFAULT_MAST3R_CONFIG_CALIB
    if calibration_mode == "never":
        return DEFAULT_MAST3R_CONFIG_NO_CALIB
    return DEFAULT_MAST3R_CONFIG_CALIB if has_calibration else DEFAULT_MAST3R_CONFIG_NO_CALIB


def _predictions_manifest_path(
    predictions_root: Path,
    dataset_slug: str,
    policy: str,
    scene_safe: str,
) -> Path:
    return predictions_root / dataset_slug / policy / f"{scene_safe}.txt"


def _staging_scene_root(staging_root: Path, dataset_slug: str, policy: str, scene_safe: str) -> Path:
    bucket = f"dataset_{dataset_slug}"
    return staging_root / bucket / policy / scene_safe


def _logs_save_as(dataset_slug: str, policy: str, scene_safe: str) -> str:
    return f"public_slam/{dataset_slug}/{policy}/{scene_safe}"


def _scene_output_dir(output_root: Path, dataset_slug: str, policy: str, scene_safe: str) -> Path:
    return output_root / dataset_slug / policy / scene_safe


def _run_mast3r_command(
    dataset_dir: Path,
    config_path: Path,
    save_as: str,
    python_bin: str,
    calib_path: Path | None,
    no_viz: bool,
) -> int:
    command = [
        python_bin,
        str(REPO_ROOT / "third_party" / "MASt3R-SLAM" / "main.py"),
        "--dataset",
        str(dataset_dir),
        "--config",
        str(config_path),
        "--save-as",
        save_as,
    ]
    if no_viz:
        command.append("--no-viz")
    if calib_path is not None:
        command.extend(["--calib", str(calib_path)])

    env = os.environ.copy()
    env["PYTHONPATH"] = _mast3r_pythonpath()
    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
    return int(completed.returncode)


def _copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _run_scene(
    dataset_root: Path,
    dataset_type: str,
    scene_id: str,
    predictions_root: Path,
    policy: str,
    output_root: Path,
    staging_root: Path,
    python_bin: str,
    mast3r_config: Path | None,
    calibration_mode: str,
    copy_mode: str,
    no_viz: bool,
    cleanup_staging: bool,
) -> SceneRunResult:
    dataset_type, dataset_slug = normalize_dataset_type(dataset_type)
    scene_root = resolve_scene_root(dataset_root, dataset_type, scene_id)
    scene_safe = safe_scene_id(scene_id)
    predictions_manifest = _predictions_manifest_path(predictions_root, dataset_slug, policy, scene_safe)
    if not predictions_manifest.exists():
        raise FileNotFoundError(
            f"Predictions manifest not found: {predictions_manifest}. "
            "Run `python3 scripts/generate_rgb_lists.py ...` first."
        )

    entries = _read_prediction_manifest(predictions_manifest, scene_root)
    staged_scene_root = _staging_scene_root(staging_root, dataset_slug, policy, scene_safe)
    _stage_sparse_scene(entries, staged_scene_root, copy_mode=copy_mode)

    output_dir = _scene_output_dir(output_root, dataset_slug, policy, scene_safe)
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_spec = _resolve_calibration_spec(scene_root, dataset_type)
    generated_calib_path: Path | None = None
    if calibration_mode != "never" and calib_spec is not None:
        generated_calib_path = _write_intrinsics_yaml(staged_scene_root / "intrinsics.yaml", calib_spec)
    elif calibration_mode == "always":
        raise FileNotFoundError(
            f"Calibration requested but no supported calibration file was found under {scene_root}"
        )

    config_path = Path(mast3r_config) if mast3r_config is not None else _default_mast3r_config(
        calibration_mode=calibration_mode,
        has_calibration=generated_calib_path is not None,
    )
    save_as = _logs_save_as(dataset_slug, policy, scene_safe)
    returncode = _run_mast3r_command(
        dataset_dir=staged_scene_root,
        config_path=config_path,
        save_as=save_as,
        python_bin=python_bin,
        calib_path=generated_calib_path,
        no_viz=no_viz,
    )

    logs_dir = REPO_ROOT / "logs" / save_as
    trajectory_src = logs_dir / f"{scene_safe}.txt"
    reconstruction_src = logs_dir / f"{scene_safe}.ply"

    trajectory_path = _copy_if_exists(trajectory_src, output_dir / "trajectory_keyframes.tum")
    reconstruction_path = _copy_if_exists(reconstruction_src, output_dir / "reconstruction.ply")

    metadata = {
        "dataset_type": dataset_type,
        "dataset_slug": dataset_slug,
        "scene_id": scene_id,
        "scene_safe": scene_safe,
        "scene_root": str(scene_root),
        "predictions_manifest": str(predictions_manifest),
        "staged_scene_root": str(staged_scene_root),
        "output_dir": str(output_dir),
        "mast3r_config": str(config_path),
        "calibration_mode": calibration_mode,
        "generated_calibration_path": None if generated_calib_path is None else str(generated_calib_path),
        "returncode": int(returncode),
        "trajectory_path": None if trajectory_path is None else str(trajectory_path),
        "reconstruction_path": None if reconstruction_path is None else str(reconstruction_path),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if cleanup_staging and staged_scene_root.exists():
        shutil.rmtree(staged_scene_root)

    status = "ok" if returncode == 0 and trajectory_path is not None else "failed"
    return SceneRunResult(
        dataset_type=dataset_type,
        dataset_slug=dataset_slug,
        scene_id=scene_id,
        scene_safe=scene_safe,
        returncode=returncode,
        status=status,
        predictions_manifest=predictions_manifest,
        staged_scene_root=staged_scene_root,
        output_dir=output_dir,
        trajectory_path=trajectory_path,
        reconstruction_path=reconstruction_path,
        calibration_path=generated_calib_path,
    )


def _write_dataset_summary(results: list[SceneRunResult], output_root: Path, dataset_slug: str, policy: str) -> None:
    dataset_dir = output_root / dataset_slug / policy
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "scene_id": item.scene_id,
            "status": item.status,
            "returncode": int(item.returncode),
            "trajectory_path": "" if item.trajectory_path is None else str(item.trajectory_path),
            "reconstruction_path": "" if item.reconstruction_path is None else str(item.reconstruction_path),
        }
        for item in results
    ]

    summary_csv = dataset_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scene_id", "status", "returncode", "trajectory_path", "reconstruction_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_json = dataset_dir / "summary.json"
    summary_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Prepared dataset root.")
    parser.add_argument("--dataset-type", type=str, required=True, help="TUM, 7SCENES, or EUROC.")
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=DEFAULT_PREDICTIONS_ROOT,
        help="Root containing sparse RGB manifests from generate_rgb_lists.py.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=LEANGATE_POLICY_NAME,
        help="Sparse policy name. Public release supports only leangate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SLAM_OUTPUT_ROOT,
        help="Root directory for copied MASt3R-SLAM outputs.",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=DEFAULT_STAGING_ROOT,
        help="Temporary/public sparse scene export root passed into MASt3R-SLAM.",
    )
    parser.add_argument(
        "--mast3r-config",
        type=Path,
        default=None,
        help="Optional explicit MASt3R-SLAM config path. Defaults to the public configs in ./configs.",
    )
    parser.add_argument(
        "--calibration-mode",
        type=str,
        choices=("auto", "always", "never"),
        default="auto",
        help="Whether to pass generated intrinsics.yaml into MASt3R-SLAM.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize sparse frames for MASt3R-SLAM.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable or "python3",
        help="Python executable used to launch third_party/MASt3R-SLAM/main.py.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable MASt3R-SLAM visualization window. Default is headless.",
    )
    parser.add_argument(
        "--cleanup-staging",
        action="store_true",
        help="Delete the staged sparse scene after a successful run.",
    )
    return parser


def run_scene_cli(argv: list[str] | None = None) -> int:
    parser = _base_parser("Run MASt3R-SLAM on one sparse RGB manifest produced by this repo.")
    parser.add_argument("--scene-id", type=str, required=True, help="Scene identifier relative to dataset root.")
    args = parser.parse_args(argv)

    result = _run_scene(
        dataset_root=args.dataset_root,
        dataset_type=args.dataset_type,
        scene_id=args.scene_id,
        predictions_root=args.predictions_root,
        policy=args.policy,
        output_root=args.output_root,
        staging_root=args.staging_root,
        python_bin=args.python_bin,
        mast3r_config=args.mast3r_config,
        calibration_mode=args.calibration_mode,
        copy_mode=args.copy_mode,
        no_viz=not bool(args.viz),
        cleanup_staging=bool(args.cleanup_staging),
    )
    print(
        f"[{result.status.upper()}] {result.scene_id} -> "
        f"{result.trajectory_path if result.trajectory_path is not None else result.output_dir}"
    )
    return 0 if result.status == "ok" else 1


def run_dataset_cli(argv: list[str] | None = None) -> int:
    parser = _base_parser("Run MASt3R-SLAM over all sparse RGB manifests for one prepared dataset root.")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional limit on how many scenes to run in canonical/discovered order.",
    )
    args = parser.parse_args(argv)

    dataset_type, dataset_slug = normalize_dataset_type(args.dataset_type)
    scene_ids = list(get_scenes(args.dataset_root, dataset_type, max_scenes=args.max_scenes))
    results: list[SceneRunResult] = []

    for scene_id in scene_ids:
        result = _run_scene(
            dataset_root=args.dataset_root,
            dataset_type=dataset_type,
            scene_id=scene_id,
            predictions_root=args.predictions_root,
            policy=args.policy,
            output_root=args.output_root,
            staging_root=args.staging_root,
            python_bin=args.python_bin,
            mast3r_config=args.mast3r_config,
            calibration_mode=args.calibration_mode,
            copy_mode=args.copy_mode,
            no_viz=not bool(args.viz),
            cleanup_staging=bool(args.cleanup_staging),
        )
        results.append(result)
        print(
            f"[{result.status.upper()}] {result.scene_id} -> "
            f"{result.trajectory_path if result.trajectory_path is not None else result.output_dir}"
        )

    _write_dataset_summary(results, args.output_root, dataset_slug, args.policy)
    failed = [item for item in results if item.status != "ok"]
    return 0 if not failed else 1


def main_scene() -> None:
    raise SystemExit(run_scene_cli())


def main_dataset() -> None:
    raise SystemExit(run_dataset_cli())


__all__ = [
    "CalibrationSpec",
    "ManifestEntry",
    "SceneRunResult",
    "main_dataset",
    "main_scene",
    "run_dataset_cli",
    "run_scene_cli",
]
