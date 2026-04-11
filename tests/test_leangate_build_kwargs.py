from __future__ import annotations

import sys
import types
from pathlib import Path

from evaluate.produce_policy_rgb_lists import _infer_leangate_build_kwargs


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def test_infer_leangate_build_kwargs_uses_public_defaults_without_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "leangate.pt"
    checkpoint_path.write_bytes(b"not-a-real-torch-zip")

    raw = {
        "head.overlap_proj.weight": _FakeTensor((768, 8)),
    }
    fake_torch = types.SimpleNamespace(load=lambda *args, **kwargs: raw)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    kwargs = _infer_leangate_build_kwargs(checkpoint_path)

    assert kwargs == {
        "overlap_head_type": 2,
        "overlap_dim": 8,
        "overlap_iters": 4,
        "decoder_depth": 6,
    }


def test_infer_leangate_build_kwargs_prefers_checkpoint_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "leangate.pt"
    checkpoint_path.write_bytes(b"not-a-real-torch-zip")

    raw = {
        "state_dict": {
            "head.overlap_proj.weight": _FakeTensor((768, 16)),
        },
        "overlap_iters": 7,
        "decoder_depth": 9,
    }
    fake_torch = types.SimpleNamespace(load=lambda *args, **kwargs: raw)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    kwargs = _infer_leangate_build_kwargs(checkpoint_path)

    assert kwargs == {
        "overlap_head_type": 2,
        "overlap_dim": 16,
        "overlap_iters": 7,
        "decoder_depth": 9,
    }
