from __future__ import annotations

from pathlib import Path

from evaluate.datasets import find_calib_path, find_gt_path, get_scenes, normalize_dataset_type, resolve_scene_root


def _write_manifest(path: Path, rel_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"0.0 {rel_path}\n", encoding="utf-8")


def test_normalize_dataset_type_public_only() -> None:
    assert normalize_dataset_type("TUM") == ("TUM", "tum")
    assert normalize_dataset_type("7scenes") == ("7SCENES", "7scenes")
    assert normalize_dataset_type("euroc") == ("EUROC", "euroc")


def test_tum_scene_discovery(tmp_path: Path) -> None:
    scene_root = tmp_path / "rgbd_dataset_freiburg1_desk"
    _write_manifest(scene_root / "rgb.txt", "rgb/000000.png")
    (scene_root / "groundtruth.txt").write_text("0 0 0 0 0 0 0 1\n", encoding="utf-8")
    (scene_root / "camera_640.yaml").write_text("width: 640\n", encoding="utf-8")

    scenes = get_scenes(tmp_path, "TUM")
    assert scenes == ["rgbd_dataset_freiburg1_desk"]
    assert resolve_scene_root(tmp_path, "TUM", scenes[0]) == scene_root
    assert find_gt_path(scene_root, "TUM") == scene_root / "groundtruth.txt"
    assert find_calib_path(scene_root, "TUM") == scene_root / "camera_640.yaml"


def test_7scenes_scene_discovery(tmp_path: Path) -> None:
    scene_root = tmp_path / "chess" / "seq-01"
    _write_manifest(scene_root / "rgb.txt", "rgb/frame-000000.color.png")
    (scene_root / "groundtruth.txt").write_text("0 0 0 0 0 0 0 1\n", encoding="utf-8")
    (scene_root / "calib.yaml").write_text("width: 640\n", encoding="utf-8")

    scenes = get_scenes(tmp_path, "7SCENES")
    assert scenes == ["chess/seq-01"]
    assert resolve_scene_root(tmp_path, "7SCENES", "chess/seq-01") == scene_root


def test_euroc_scene_discovery_and_resolution(tmp_path: Path) -> None:
    scene_root = tmp_path / "machine_hall" / "MH_01_easy" / "mav0" / "cam0"
    _write_manifest(scene_root / "rgb.txt", "data/1403636579763555584.png")
    (scene_root / "groundtruth.txt").write_text("0 0 0 0 0 0 0 1\n", encoding="utf-8")
    (scene_root / "calib.yaml").write_text("width: 752\n", encoding="utf-8")

    scenes = get_scenes(tmp_path, "EUROC")
    assert scenes == ["MH_01_easy"]
    resolved = resolve_scene_root(tmp_path, "EUROC", "MH_01_easy")
    assert resolved == tmp_path / "machine_hall" / "MH_01_easy"
    assert find_gt_path(resolved, "EUROC") == scene_root / "groundtruth.txt"
    assert find_calib_path(resolved, "EUROC") == scene_root / "calib.yaml"
