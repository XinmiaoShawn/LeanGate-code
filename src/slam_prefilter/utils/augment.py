"""Deterministic augmentation utilities for fixed-size decision frames."""

from __future__ import annotations

import dataclasses
import hashlib
import random

import torch


@dataclasses.dataclass(frozen=True)
class AugmentParams:
    brightness_factor: float
    contrast_factor: float
    temperature_shift: float
    noise_std: float
    noise_seed: int


def _seed_from_components(scene_id: str, start_idx: int, base_seed: int) -> int:
    digest = hashlib.sha1()
    digest.update(scene_id.encode("utf-8"))
    digest.update(start_idx.to_bytes(8, "little", signed=False))
    digest.update(base_seed.to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def sample_augment_params(scene_id: str, start_idx: int, base_seed: int) -> AugmentParams:
    seed = _seed_from_components(scene_id, start_idx, base_seed)
    rng = random.Random(seed)
    brightness = 1.0 + rng.uniform(-0.1, 0.1)
    contrast = 1.0 + rng.uniform(-0.1, 0.1)
    temperature = rng.uniform(-0.1, 0.1)
    noise_std = max(0.0, rng.uniform(0.0, 0.02))
    noise_seed = rng.randrange(1 << 31)
    return AugmentParams(
        brightness_factor=brightness,
        contrast_factor=contrast,
        temperature_shift=temperature,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )


def apply_augment_224(image: torch.Tensor, params: AugmentParams) -> torch.Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected image tensor of shape (3,H,W)")
    img = image.clone()
    mean = img.mean(dim=(1, 2), keepdim=True)
    img = (img - mean) * params.contrast_factor + mean
    img = torch.clamp(img * params.brightness_factor, 0.0, 1.0)

    temp = params.temperature_shift
    if abs(temp) > 1e-6:
        r_scale = 1.0 + temp
        b_scale = 1.0 - temp
        scales = torch.tensor([r_scale, 1.0, b_scale], dtype=img.dtype, device=img.device)
        img = torch.clamp(img * scales[:, None, None], 0.0, 1.0)

    if params.noise_std > 0.0:
        generator = torch.Generator(device=img.device)
        generator.manual_seed(params.noise_seed)
        noise = torch.normal(0.0, params.noise_std, size=img.shape, generator=generator, device=img.device)
        img = torch.clamp(img + noise, 0.0, 1.0)

    return img
