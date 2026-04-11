from __future__ import annotations

from student.flare import _extract_backbone_state_dict


def test_extract_backbone_state_dict_from_full_leangate_checkpoint() -> None:
    ckpt = {
        "state_dict": {
            "module.backbone.encoder.weight": object(),
            "module.backbone.decoder.bias": object(),
            "module.head.readout.weight": object(),
        }
    }

    backbone = _extract_backbone_state_dict(ckpt)

    assert set(backbone.keys()) == {"encoder.weight", "decoder.bias"}


def test_extract_backbone_state_dict_keeps_plain_backbone_checkpoint() -> None:
    ckpt = {
        "model": {
            "encoder.weight": object(),
            "decoder.bias": object(),
        }
    }

    backbone = _extract_backbone_state_dict(ckpt)

    assert set(backbone.keys()) == {"encoder.weight", "decoder.bias"}
