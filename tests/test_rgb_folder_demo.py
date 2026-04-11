from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from evaluate.produce_policy_rgb_lists import _write_manifest
from slam_prefilter.utils.data_loader import SceneFrameSequence


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 10, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


def test_from_rgb_folder_uses_sorted_filenames_and_folder_relative_manifest(tmp_path: Path) -> None:
    rgb_folder = tmp_path / "rgb_frames"
    _write_image(rgb_folder / "000010.png", 10)
    _write_image(rgb_folder / "000002.png", 20)
    _write_image(rgb_folder / "000001.png", 30)

    sequence, _meta = SceneFrameSequence.from_rgb_folder(rgb_folder)

    out_path = tmp_path / "outputs" / "demo" / "leangate" / "rgb_frames.txt"
    _write_manifest(out_path, rgb_folder, sequence, [0, 2])

    assert [sequence.record(i).path.name for i in range(len(sequence))] == [
        "000001.png",
        "000002.png",
        "000010.png",
    ]
    assert out_path.read_text(encoding="utf-8").splitlines() == [
        "# timestamp filename",
        "0 000001.png",
        "2 000010.png",
    ]
