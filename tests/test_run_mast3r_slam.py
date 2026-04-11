from __future__ import annotations

from pathlib import Path

from evaluate.run_mast3r_slam import ManifestEntry
from evaluate.run_mast3r_slam import (
    _read_prediction_manifest,
    _resolve_calibration_spec,
    _stage_sparse_scene,
)


def test_read_prediction_manifest_resolves_scene_relative_paths(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    rgb_dir = scene_root / "rgb"
    rgb_dir.mkdir(parents=True)
    frame = rgb_dir / "000001.png"
    frame.write_text("png", encoding="utf-8")

    manifest = tmp_path / "predictions.txt"
    manifest.write_text("# timestamp filename\n1.0 rgb/000001.png\n", encoding="utf-8")

    entries = _read_prediction_manifest(manifest, scene_root)
    assert len(entries) == 1
    assert entries[0].timestamp == "1.0"
    assert entries[0].relative_path == "rgb/000001.png"
    assert entries[0].absolute_path == frame


def test_stage_sparse_scene_materializes_png_filenames(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_text("png", encoding="utf-8")

    entries = [ManifestEntry(timestamp="1.0", relative_path="rgb/000001.png", absolute_path=source)]
    staging_root = tmp_path / "staging"
    staged_manifest = _stage_sparse_scene(entries, staging_root, copy_mode="copy")

    assert staged_manifest.exists()
    assert (staging_root / "frame_000000.png").exists()
    assert "frame_000000.png" in staged_manifest.read_text(encoding="utf-8")


def test_resolve_calibration_spec_from_yaml(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    scene_root.mkdir()
    calib = scene_root / "calib.yaml"
    calib.write_text(
        "width: 640\nheight: 480\nfx: 500.0\nfy: 501.0\ncx: 320.0\ncy: 240.0\n",
        encoding="utf-8",
    )

    spec = _resolve_calibration_spec(scene_root, "7SCENES")
    assert spec is not None
    assert spec.width == 640
    assert spec.height == 480
    assert spec.calibration == [500.0, 501.0, 320.0, 240.0]
    assert spec.source_path == calib
