from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evaluate.produce_policy_rgb_lists import _write_manifest


@dataclass(frozen=True)
class _Record:
    path: Path
    timestamp: float


class _FakeSequence:
    def __init__(self, paths: list[Path], timestamps: list[float]) -> None:
        self._records = [_Record(path=p, timestamp=t) for p, t in zip(paths, timestamps)]

    def record(self, index: int) -> _Record:
        return self._records[index]


def test_write_manifest_preserves_original_timestamp_strings(tmp_path: Path) -> None:
    scene_root = tmp_path / "scene"
    rgb_dir = scene_root / "rgb"
    rgb_dir.mkdir(parents=True)
    rgb_txt = scene_root / "rgb.txt"
    rgb_txt.write_text(
        "# timestamp filename\n"
        "1305031102.175304 rgb/000001.png\n"
        "1305031102.211214 rgb/000002.png\n",
        encoding="utf-8",
    )
    (rgb_dir / "000001.png").write_text("", encoding="utf-8")
    (rgb_dir / "000002.png").write_text("", encoding="utf-8")

    sequence = _FakeSequence(
        [rgb_dir / "000001.png", rgb_dir / "000002.png"],
        [1305031102.0, 1305031102.0],
    )
    out_path = tmp_path / "predictions" / "tum" / "leangate" / "scene.txt"
    _write_manifest(out_path, scene_root, sequence, [0, 1])

    assert out_path.read_text(encoding="utf-8").splitlines() == [
        "# timestamp filename",
        "1305031102.175304 rgb/000001.png",
        "1305031102.211214 rgb/000002.png",
    ]
