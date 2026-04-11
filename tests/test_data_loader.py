from __future__ import annotations

from pathlib import Path

import pytest

from slam_prefilter.utils import data_loader as dl


def test_create_default_sequence_requires_explicit_scene_root() -> None:
    with pytest.raises(ValueError, match="scene_root must be provided explicitly"):
        dl.create_default_sequence()


def test_create_default_sequence_uses_explicit_scene_root(monkeypatch, tmp_path: Path) -> None:
    calls: list[Path] = []

    def fake_from_scene_root(path: Path):
        calls.append(path)
        return "sequence", None

    monkeypatch.setattr(dl.RgbFrameSequence, "from_scene_root", staticmethod(fake_from_scene_root))

    sequence = dl.create_default_sequence(tmp_path)
    assert sequence == "sequence"
    assert calls == [tmp_path]
