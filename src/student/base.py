"""Base interfaces for overlap/pose student models."""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclasses.dataclass
class StudentBatch:
    ref_image: torch.Tensor
    cur_image: torch.Tensor
    intrinsics: Optional[torch.Tensor] = None
    meta: Dict[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class StudentPred:
    overlap_score: Optional[torch.Tensor] = None
    pose_ij: Optional[torch.Tensor] = None
    extras: Dict[str, object] = dataclasses.field(default_factory=dict)


class OverlapStudent(nn.Module):
    """Abstract student model that consumes image pairs and predicts overlap/pose."""

    def forward(self, batch: StudentBatch) -> StudentPred:  # type: ignore[override]
        raise NotImplementedError

