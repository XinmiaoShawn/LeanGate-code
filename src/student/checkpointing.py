"""Checkpoint helpers for student models.

We separate checkpoints into:
- FLARE backbone checkpoints (e.g. geometry_pose.pth), loaded by FlareStudent via `flare_ckpt`.
- Lightweight overlap checkpoints ("overlap checkpoint") that store only the overlap parts we train:
  the overlap `head` and (optionally trained) `trunk` (pose_head.trunk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


OVERLAP_CHECKPOINT_FORMAT = "overlap_checkpoint_v1"


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _looks_like_state_dict(obj: object) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    return all(isinstance(k, str) for k in obj.keys()) and all(isinstance(v, torch.Tensor) for v in obj.values())


def _extract_head_trunk(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    base = _unwrap_module(model)
    head = getattr(base, "head", None)
    trunk = getattr(base, "trunk", None)
    if not isinstance(head, nn.Module):
        raise AttributeError("Student model has no nn.Module attribute `head`.")
    if not isinstance(trunk, nn.Module):
        raise AttributeError("Student model has no nn.Module attribute `trunk`.")
    return head, trunk


def save_overlap_checkpoint(model: nn.Module, path: str | Path, *, meta: Optional[Dict[str, Any]] = None) -> Path:
    """Save overlap checkpoint (head + trunk)."""
    out_path = Path(path)
    head, trunk = _extract_head_trunk(model)
    payload: Dict[str, Any] = {
        "format": OVERLAP_CHECKPOINT_FORMAT,
        "head": head.state_dict(),
        "trunk": trunk.state_dict(),
        "meta": {} if meta is None else dict(meta),
    }
    torch.save(payload, out_path)
    return out_path


def load_overlap_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict_head: bool = True,
    strict_trunk: bool = True,
) -> Dict[str, Any]:
    """Load overlap checkpoint into an existing student model.

    Supported layouts:
    - {"format": "overlap_checkpoint_v1", "head": <sd>, "trunk": <sd>, ...}
    - {"head": <sd>, "trunk": <sd>} (legacy/custom)
    - {"model": <sd>} / {"state_dict": <sd>} / {"model_state_dict": <sd>} (common wrappers)
    - <full model state_dict> (fallback; loaded with strict=False)
    """
    ckpt_path = Path(path)
    ckpt = torch.load(ckpt_path, map_location=map_location)
    head, trunk = _extract_head_trunk(model)

    report: Dict[str, Any] = {"path": str(ckpt_path), "loaded": []}

    if isinstance(ckpt, dict):
        # Wrapper forms.
        for key in ("model", "state_dict", "model_state_dict"):
            maybe_sd = ckpt.get(key)
            if _looks_like_state_dict(maybe_sd):
                # Strip 'module.' prefix from DDP-saved checkpoints
                state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in maybe_sd.items()}
                missing, unexpected = _unwrap_module(model).load_state_dict(state_dict, strict=False)
                report.update({"format": f"wrapped:{key}", "missing": missing, "unexpected": unexpected})
                report["loaded"].append("model")
                return report

        head_sd = ckpt.get("head")
        trunk_sd = ckpt.get("trunk")
        if isinstance(head_sd, dict) and isinstance(trunk_sd, dict):
            head_missing, head_unexpected = head.load_state_dict(head_sd, strict=bool(strict_head))
            trunk_missing, trunk_unexpected = trunk.load_state_dict(trunk_sd, strict=bool(strict_trunk))
            report.update(
                {
                    "format": str(ckpt.get("format", "head+trunk")),
                    "head_missing": head_missing,
                    "head_unexpected": head_unexpected,
                    "trunk_missing": trunk_missing,
                    "trunk_unexpected": trunk_unexpected,
                }
            )
            report["loaded"].extend(["head", "trunk"])
            return report

    # Full state_dict fallback.
    if _looks_like_state_dict(ckpt):
        # Strip 'module.' prefix from DDP-saved checkpoints
        state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ckpt.items()}
        missing, unexpected = _unwrap_module(model).load_state_dict(state_dict, strict=False)
        report.update({"format": "full_state_dict", "missing": missing, "unexpected": unexpected})
        report["loaded"].append("model")
        return report

    raise TypeError(f"Unsupported overlap checkpoint type: {type(ckpt)} ({ckpt_path})")

