"""Public RGB-only scene loader used by sparse-frame inference pipelines."""

from __future__ import annotations

import dataclasses
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .augment import AugmentParams, apply_augment_224, sample_augment_params


LOGGER = logging.getLogger(__name__)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclasses.dataclass(frozen=True)
class FrameRecord:
    """Lightweight metadata for a single RGB frame."""

    scene_id: str
    index: int
    path: Path
    # Optional timestamp (seconds or milliseconds). When absent, downstream
    # components fall back to a fixed dt for time-based rules.
    timestamp: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class FrameMeta:
    """Per-sequence metadata (resolution + intrinsics approximation)."""

    intrinsics_full: np.ndarray
    resolution: Tuple[int, int]
    calibration_path: Optional[Path]
    scene_root: Path
    # Kept for API compatibility; not used in overlap pipelines.
    slam_dataset_root: Optional[Path] = None


@dataclasses.dataclass
class LoadedFrame:
    """Decoded frame with fixed-size decision image.

    Note: historically this tensor was always 256x256 and named `image_256`.
    It is now a configurable square decision image (e.g., 256 or 512).
    """

    record: FrameRecord
    image_256: torch.Tensor


@dataclasses.dataclass
class DatasetSlice:
    """Contiguous slice of a frame sequence."""

    sequence: "RgbFrameSequence"
    start_idx: int
    length: int
    augment: bool = True
    base_seed: int = 42

    def __post_init__(self) -> None:
        if self.start_idx < 0 or self.length <= 0:
            raise ValueError("Invalid slice indices")
        if self.start_idx + self.length > len(self.sequence):
            raise ValueError("Slice extends beyond sequence length")

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[LoadedFrame]:
        params: Optional[AugmentParams]
        if self.augment:
            params = sample_augment_params(
                self.sequence.scene_id,
                self.start_idx,
                self.base_seed,
            )
        else:
            params = None
        for offset in range(self.length):
            yield self.sequence.load(
                self.start_idx + offset,
                augment=self.augment,
                augment_params=params,
            )


def _load_rgb_manifest(scene_root: Path) -> Optional[tuple[List[Path], List[Optional[float]]]]:
    """Load an optional rgb.txt manifest with timestamps."""
    manifest = scene_root / "rgb.txt"
    if not manifest.exists():
        # EuRoC-style layout keeps rgb.txt under mav0/cam0/.
        euroc_manifest = scene_root / "mav0" / "cam0" / "rgb.txt"
        if euroc_manifest.exists():
            manifest = euroc_manifest
        else:
            return None

    paths: List[Path] = []
    timestamps: List[Optional[float]] = []
    skipped = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                LOGGER.warning("Skipping malformed rgb.txt line in %s: %s", manifest, stripped)
                skipped += 1
                continue
            try:
                timestamp = float(parts[0])
            except ValueError:
                LOGGER.warning("Skipping rgb.txt line with invalid timestamp in %s: %s", manifest, stripped)
                skipped += 1
                continue
            rel_path = Path(" ".join(parts[1:]))
            # Resolve relative paths against the manifest directory, not scene_root.
            abs_path = (manifest.parent / rel_path).resolve(strict=False)
            if not abs_path.exists():
                LOGGER.warning(
                    "Scene %s: rgb manifest entry %s missing on disk; skipping",
                    scene_root.name,
                    rel_path,
                )
                skipped += 1
                continue
            paths.append(abs_path)
            timestamps.append(timestamp)

    if not paths:
        LOGGER.warning(
            "Scene %s: rgb manifest %s had no usable entries; falling back to directory scan",
            scene_root.name,
            manifest,
        )
        return None

    if skipped:
        LOGGER.info(
            "Scene %s: loaded %d rgb timestamps (skipped %d malformed/missing entries)",
            scene_root.name,
            len(paths),
            skipped,
        )
    return paths, timestamps


def _discover_frame_paths(scene_root: Path) -> List[Path]:
    """Discover RGB frame paths under a scene root.

    Preferred layout for scenes:
      scene_root/rgb/*.png|jpg

    Compatibility fallbacks also accept:
      scene_root/resized_images/*.png|jpg
      scene_root/rgb_resize/*.png|jpg
    """
    rgb_dir_name = os.environ.get("SCANNETPP_RGB_DIR_NAME")
    if rgb_dir_name:
        explicit_dir = scene_root / rgb_dir_name
        if not explicit_dir.is_dir():
            raise FileNotFoundError(
                f"Explicit RGB directory '{rgb_dir_name}' not found under {scene_root}"
            )
        search_dir = explicit_dir
    else:
        rgb_resize_dir = scene_root / "rgb_resize"
        rgb_dir = scene_root / "rgb"
        alt_dir = scene_root / "resized_images"
        if rgb_resize_dir.exists():
            search_dir = rgb_resize_dir
        elif rgb_dir.exists():
            search_dir = rgb_dir
        elif alt_dir.exists():
            search_dir = alt_dir
        else:
            raise FileNotFoundError(
                f"No RGB directory found under {scene_root} "
                "(expected 'rgb_resize', 'rgb', or 'resized_images')"
            )
    candidates = [
        path
        for path in sorted(search_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    ]
    if not candidates:
        raise RuntimeError(f"No RGB frames found under {search_dir}")
    return candidates


class RgbFrameSequence:
    """RGB-only frame sequence with fixed-size decision images."""

    def __init__(self, scene_root: Path, *, decision_img_size: int = 256) -> None:
        scene_root = Path(scene_root)
        if not scene_root.exists():
            raise FileNotFoundError(f"Scene root {scene_root} missing")
        if int(decision_img_size) <= 0:
            raise ValueError(f"decision_img_size must be positive, got {decision_img_size}")
        self.decision_img_size = int(decision_img_size)

        rgb_manifest = _load_rgb_manifest(scene_root)
        if rgb_manifest is not None:
            rgb_paths, rgb_timestamps = rgb_manifest
        else:
            rgb_paths = _discover_frame_paths(scene_root)
            rgb_timestamps = [None] * len(rgb_paths)

        if not rgb_paths:
            raise RuntimeError(f"No RGB frames discovered under {scene_root}")

        self.scene_root = scene_root
        self.scene_id = scene_root.name

        self._records: List[FrameRecord] = []
        for idx, path in enumerate(rgb_paths):
            ts = rgb_timestamps[idx] if idx < len(rgb_timestamps) else None
            self._records.append(
                FrameRecord(
                    scene_id=self.scene_id,
                    index=idx,
                    path=path,
                    timestamp=ts,
                )
            )

        # Infer resolution + dummy intrinsics from the first frame.
        base = self._load_rgb_base(self._records[0].path)
        height, width = base.shape[:2]
        intrinsics = np.array(
            [
                [float(width), 0.0, float(width) / 2.0],
                [0.0, float(height), float(height) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.meta = FrameMeta(
            intrinsics_full=intrinsics,
            resolution=(int(width), int(height)),
            calibration_path=None,
            scene_root=self.scene_root,
            slam_dataset_root=None,
        )

    @classmethod
    def from_scene_root(
        cls,
        scene_root: Path,
        dense_index_mode: bool = True,  # kept for API compatibility; unused
        decision_img_size: int = 256,
    ) -> tuple["RgbFrameSequence", FrameMeta]:
        """Factory mirroring the legacy SceneFrameSequence.from_scene_root API."""
        sequence = cls(scene_root, decision_img_size=decision_img_size)
        return sequence, sequence.meta

    def __len__(self) -> int:
        return len(self._records)

    def record(self, index: int) -> FrameRecord:
        if index < 0 or index >= len(self._records):
            raise IndexError(f"Frame index {index} out of range for {self.scene_id}")
        return self._records[index]

    def records(self) -> Iterable[FrameRecord]:
        return list(self._records)

    def timestamp(self, index: int) -> Optional[float]:
        rec = self.record(index)
        return rec.timestamp

    def iter_loaded(
        self,
        augment: bool = True,
        augment_params: Optional[AugmentParams] = None,
    ) -> Iterator[LoadedFrame]:
        for idx in range(len(self)):
            yield self.load(idx, augment=augment, augment_params=augment_params)

    @lru_cache(maxsize=64)
    def _load_rgb_base(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    def _build_decision_image(
        self,
        base: np.ndarray,
        augment: bool,
        augment_params: Optional[AugmentParams],
        img_size: int,
    ) -> torch.Tensor:
        """Resize + center-crop to a square decision image of size `img_size`."""
        height, width = base.shape[:2]
        target = float(int(img_size))
        scale = target / float(min(height, width))
        new_w = max(int(target), int(round(width * scale)))
        new_h = max(int(target), int(round(height * scale)))
        resized = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = max(0, (new_h - int(target)) // 2)
        left = max(0, (new_w - int(target)) // 2)
        crop = resized[top : top + int(target), left : left + int(target)]
        if crop.shape[0] != int(target) or crop.shape[1] != int(target):
            crop = cv2.resize(crop, (int(target), int(target)), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.transpose(crop, (2, 0, 1))).contiguous()
        if augment and augment_params is not None:
            tensor = apply_augment_224(tensor, augment_params)
        return tensor

    def load(
        self,
        index: int,
        augment: bool = True,
        augment_params: Optional[AugmentParams] = None,
        img_size: Optional[int] = None,
    ) -> LoadedFrame:
        if index < 0 or index >= len(self._records):
            raise IndexError(f"Frame index {index} out of range for {self.scene_id}")
        record = self._records[index]
        base = self._load_rgb_base(record.path)
        size = int(self.decision_img_size if img_size is None else img_size)
        decision = self._build_decision_image(base, augment, augment_params, size)
        return LoadedFrame(
            record=record,
            image_256=decision,
        )


def enumerate_slices(
    sequence: RgbFrameSequence,
    slice_length: int = 600,
    slice_overlap: int = 100,
) -> List[DatasetSlice]:
    """Create a list of overlapping DatasetSlice definitions for a sequence."""
    if slice_length <= 0:
        raise ValueError("slice_length must be positive")
    if slice_overlap < 0 or slice_overlap >= slice_length:
        raise ValueError("slice_overlap must satisfy 0 <= overlap < slice_length")
    slices: List[DatasetSlice] = []
    stride = slice_length - slice_overlap
    if len(sequence) < slice_length:
        LOGGER.warning(
            "Scene %s shorter than slice length (%d < %d); skipping",
            sequence.scene_id,
            len(sequence),
            slice_length,
        )
        return slices
    start = 0
    while start + slice_length <= len(sequence):
        slices.append(
            DatasetSlice(
                sequence=sequence,
                start_idx=start,
                length=slice_length,
            )
        )
        start += stride
    return slices


def create_default_sequence(scene_root: Optional[Path] = None) -> "RgbFrameSequence":
    """Convenience helper for quick local testing."""
    base = scene_root or Path("/ssd1/zhiwen/wenyan/scans/scene0021_00")
    sequence, _ = RgbFrameSequence.from_scene_root(base)
    return sequence


# Backwards compatibility: keep the old class name as an alias so that
# existing imports continue to work. New code should prefer RgbFrameSequence.
SceneFrameSequence = RgbFrameSequence


__all__ = [
    "FrameRecord",
    "FrameMeta",
    "LoadedFrame",
    "DatasetSlice",
    "RgbFrameSequence",
    "SceneFrameSequence",
    "enumerate_slices",
    "create_default_sequence",
]
