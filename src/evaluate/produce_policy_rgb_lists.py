from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_imports() -> None:
    """Allow running as `python src/evaluate/produce_policy_rgb_lists.py ...` from repo root."""
    src = REPO_ROOT / "src"
    if str(src) in sys.path:
        sys.path.remove(str(src))
    sys.path.insert(0, str(src))


_ensure_repo_imports()

from evaluate.datasets import get_scenes, normalize_dataset_type, resolve_scene_root, safe_scene_id
from evaluate.public_config import (
    DEFAULT_PREDICTIONS_ROOT,
    LEANGATE_DECODER_DEPTH,
    LEANGATE_ENABLE_CUROPE2D,
    LEANGATE_OVERLAP_ITERS,
    LEANGATE_POLICY_NAME,
    LEANGATE_THRESHOLD,
    LEANGATE_WARMUP_KEPT,
    LEGACY_LEANGATE_CHECKPOINT_PATH,
    PUBLIC_LEANGATE_CHECKPOINT_PATH,
)

@dataclass
class TimingStats:
    """Per-stream timing statistics (IO/model-load excluded from inference_ms)."""

    inference_ms: float = 0.0
    inference_count: int = 0
    setup_ms: float = 0.0
    io_ms: float = 0.0
    model_load_ms: float = 0.0
    timing_start_frame: int = 2


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


class SubsampledSequenceView:
    """Wrapper that presents a subsampled view of a SceneFrameSequence.
    
    Presents a virtual dense sequence (0, 1, 2, 3...) that maps to actual
    subsampled indices (0, stride, 2*stride, 3*stride...).
    """
    
    def __init__(self, base_sequence: SceneFrameSequence, subsample_stride: int):
        self.base_sequence = base_sequence
        self.subsample_stride = subsample_stride
        # Build mapping: virtual_idx -> real_idx
        self.real_indices = list(range(0, len(base_sequence), subsample_stride))
        # Copy metadata
        self.meta = base_sequence.meta
        self.scene_root = base_sequence.scene_root
        self.scene_id = base_sequence.scene_id
        # Cache the subsampled records
        all_records = list(base_sequence.records())
        self._records = [all_records[i] for i in self.real_indices]
    
    def __len__(self) -> int:
        return len(self.real_indices)
    
    def __getitem__(self, virtual_idx: int):
        """Map virtual index to real index and access base sequence."""
        if not 0 <= virtual_idx < len(self.real_indices):
            raise IndexError(f"Virtual index {virtual_idx} out of range [0, {len(self.real_indices)})")
        real_idx = self.real_indices[virtual_idx]
        return self.base_sequence[real_idx]
    
    def records(self):
        """Return iterator over subsampled records."""
        return iter(self._records)
    
    def _load_rgb_base(self, path):
        """Delegate to base sequence's cached loader."""
        return self.base_sequence._load_rgb_base(path)

    def load(self, virtual_idx: int, augment: bool = True, augment_params=None, img_size=None):
        """Load a frame from the subsampled view.

        Mirrors SceneFrameSequence.load() but takes a virtual index (0..len(view)-1).
        """
        if not 0 <= int(virtual_idx) < len(self.real_indices):
            raise IndexError(f"Virtual index {virtual_idx} out of range [0, {len(self.real_indices)})")
        real_idx = int(self.real_indices[int(virtual_idx)])
        return self.base_sequence.load(real_idx, augment=augment, augment_params=augment_params, img_size=img_size)
    
    def get_mapped_indices(self, virtual_indices: List[int]) -> List[int]:
        """Convert virtual indices back to real sequence indices."""
        return [self.real_indices[vi] for vi in virtual_indices]

@dataclass(frozen=True)
class StudentScoreConfig:
    student_model: str
    checkpoint_path: Path
    threshold: float
    warmup_kept: int
    enable_curope2d: bool


def _resolve_leangate_checkpoint() -> Path:
    if PUBLIC_LEANGATE_CHECKPOINT_PATH.exists():
        return PUBLIC_LEANGATE_CHECKPOINT_PATH
    if LEGACY_LEANGATE_CHECKPOINT_PATH.exists():
        return LEGACY_LEANGATE_CHECKPOINT_PATH
    raise FileNotFoundError(
        "LeanGate checkpoint not found.\n"
        f"Expected one of:\n  - {PUBLIC_LEANGATE_CHECKPOINT_PATH}\n  - {LEGACY_LEANGATE_CHECKPOINT_PATH}\n"
        "Run `python scripts/download_checkpoints.py --output-root checkpoints` first."
    )


def _build_leangate_student_cfg() -> StudentScoreConfig:
    return StudentScoreConfig(
        student_model="flare",
        checkpoint_path=_resolve_leangate_checkpoint(),
        threshold=float(LEANGATE_THRESHOLD),
        warmup_kept=int(LEANGATE_WARMUP_KEPT),
        enable_curope2d=bool(LEANGATE_ENABLE_CUROPE2D),
    )


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _unwrap_checkpoint_state_dict(obj: object) -> dict:
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
    for key in ("model", "state_dict", "model_state_dict"):
        maybe = obj.get(key)
        if isinstance(maybe, dict) and maybe:
            return maybe
    return obj


def _infer_checkpoint_int(raw: object, field_names: tuple[str, ...]) -> int | None:
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in field_names:
                try:
                    return int(value)
                except Exception:
                    pass
        for value in raw.values():
            found = _infer_checkpoint_int(value, field_names)
            if found is not None:
                return found
    if isinstance(raw, (list, tuple)):
        for value in raw:
            found = _infer_checkpoint_int(value, field_names)
            if found is not None:
                return found
    return None


def _infer_checkpoint_archive_int(checkpoint_path: Path, *, pattern: str) -> int | None:
    try:
        with zipfile.ZipFile(checkpoint_path) as archive:
            names = archive.namelist()
    except (FileNotFoundError, OSError, zipfile.BadZipFile):
        return None

    roots = []
    for name in names:
        if "/" in name:
            roots.append(name.split("/", 1)[0])
    for root in roots:
        match = re.search(pattern, root)
        if match is not None:
            return int(match.group(1))
    return None


def _infer_leangate_build_kwargs(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_module_prefix(_unwrap_checkpoint_state_dict(raw))

    build_kwargs: dict[str, Any] = {}

    if "head.overlap_proj.weight" in state_dict:
        build_kwargs["overlap_head_type"] = 2
        weight = state_dict["head.overlap_proj.weight"]
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            build_kwargs["overlap_dim"] = int(weight.shape[1])
        overlap_iters = _infer_checkpoint_int(raw, ("overlap_iters", "num_iters"))
        if overlap_iters is None:
            overlap_iters = _infer_checkpoint_archive_int(checkpoint_path, pattern=r"iter(\d+)")
        if overlap_iters is None:
            match = re.search(r"iter(\d+)", checkpoint_path.as_posix())
            if match is not None:
                overlap_iters = int(match.group(1))
        if overlap_iters is None:
            overlap_iters = int(LEANGATE_OVERLAP_ITERS)
        build_kwargs["overlap_iters"] = int(overlap_iters)
    else:
        build_kwargs["overlap_head_type"] = 1

    decoder_depth = _infer_checkpoint_int(raw, ("decoder_depth",))
    if decoder_depth is None:
        decoder_depth = _infer_checkpoint_archive_int(checkpoint_path, pattern=r"dec(\d+)")
    if decoder_depth is None:
        decoder_indices: list[int] = []
        for key in state_dict.keys():
            match = re.match(r"backbone\.dec_blocks\.(\d+)\.", str(key))
            if match is not None:
                decoder_indices.append(int(match.group(1)))
        if decoder_indices:
            decoder_depth = max(decoder_indices) + 1
    if decoder_depth is None and build_kwargs.get("overlap_head_type") == 2:
        decoder_depth = int(LEANGATE_DECODER_DEPTH)
    if decoder_depth is not None:
        build_kwargs["decoder_depth"] = int(decoder_depth)

    return build_kwargs


def _parse_gpus_arg(raw: str | None) -> list[str]:
    if raw is None:
        return []
    devices: list[str] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        if t.startswith("cuda:"):
            devices.append(t)
        elif t.isdigit():
            devices.append(f"cuda:{t}")
        else:
            devices.append(t)
    return devices


def _build_keep_indices_for_leangate(
    sequence: SceneFrameSequence,
    device: str,
    score_log_path: Path | None = None,
    timing_log_path: Path | None = None,
    timing_start_frame: int = 2,
    input_subsample: int = 1,
    student_cfg: StudentScoreConfig | None = None,
) -> tuple[List[int], TimingStats | None]:
    """Build keep indices for the public LeanGate policy.
    
    Args:
        input_subsample: If > 1, creates a subsampled view of the sequence before
                        applying the policy.
    
    Returns:
        (keep_indices, timing_stats) where keep_indices are in the ORIGINAL sequence
        coordinates and timing_stats is written to `timing_log_path` when provided.
    """
    import torch

    timing_start_frame = int(timing_start_frame)
    if timing_start_frame < 1:
        raise ValueError(f"timing_start_frame must be >= 1, got {timing_start_frame}")
    start_count_idx = timing_start_frame - 1  # 0-based frame index

    def _maybe_write_timing(stats: TimingStats | None, *, extra: dict[str, Any] | None = None) -> None:
        if timing_log_path is None or stats is None:
            return
        payload: dict[str, Any] = {
            "policy": LEANGATE_POLICY_NAME,
            "policy_type": "student",
            "device": str(device),
            "input_subsample": int(input_subsample),
            "scene_id": str(getattr(sequence, "scene_id", "")),
            **asdict(stats),
        }
        if extra:
            payload.update(extra)
        _write_json_atomic(timing_log_path, payload)

    def _is_cuda_device(device_str: str) -> bool:
        return str(device_str).startswith("cuda") and torch.cuda.is_available()

    def _cuda_sync(device_str: str) -> None:
        if _is_cuda_device(device_str):
            torch.cuda.synchronize(torch.device(device_str))

    def _get_student_for_device(device_str: str) -> torch.nn.Module:
        # Local import keeps startup lighter for metadata-only operations.
        from student import build_student, load_overlap_checkpoint

        cache = getattr(_get_student_for_device, "_cache", None)
        if cache is None:
            cache = {}
            setattr(_get_student_for_device, "_cache", cache)

        if student_cfg is None:
            raise ValueError("student_cfg is required for LeanGate inference")
        checkpoint_path = Path(student_cfg.checkpoint_path)

        key = (
            str(device_str),
            str(student_cfg.student_model),
            bool(student_cfg.enable_curope2d),
            str(checkpoint_path),
        )
        if key in cache:
            return cache[key]

        build_kwargs = dict(
            device=str(device_str),
            enable_curope2d=bool(student_cfg.enable_curope2d),
            flare_ckpt=str(checkpoint_path),
        )
        build_kwargs.update(_infer_leangate_build_kwargs(checkpoint_path))

        model = build_student(
            student_cfg.student_model,
            **build_kwargs,
        )

        raw = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _strip_module_prefix(_unwrap_checkpoint_state_dict(raw))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"[warn] student ckpt load had mismatches: missing={len(missing)} unexpected={len(unexpected)} "
                f"(ckpt={checkpoint_path})",
                flush=True,
            )

        model.eval()
        cache[key] = model
        return model

    def _student_score_from_tensors(
        model: torch.nn.Module,
        ref: torch.Tensor,
        cur: torch.Tensor,
    ) -> float:
        from student import StudentBatch

        with torch.inference_mode():
            pred = model(StudentBatch(ref_image=ref, cur_image=cur))
            if pred.overlap_score is None:
                raise RuntimeError("Student model did not produce overlap_score")
            return float(pred.overlap_score.squeeze().item())

    def _select_frames_student_cached(
        base_sequence: SceneFrameSequence,
        idx_map: list[int],
    ) -> tuple[List[int], TimingStats]:
        """Student selection with optional FLARE ref-feature caching + per-stream timing."""
        stats = TimingStats(timing_start_frame=timing_start_frame)

        t0 = time.perf_counter()
        model = _get_student_for_device(device)
        stats.model_load_ms = (time.perf_counter() - t0) * 1000.0

        base_model = model.module if hasattr(model, "module") else model
        supports_frame_cache = all(
            hasattr(base_model, name) for name in ("create_frame", "encode_frame", "forward_frames")
        )

        total = len(idx_map)
        if total == 0:
            return [], stats

        kept: List[int] = [0]
        last_keep = 0
        log_rows: List[tuple[int, int, float, float, bool]] = []
        tau_keep = float(student_cfg.threshold) if student_cfg is not None else 0.0

        if supports_frame_cache:
            # Initialize ref frame (index 0) and pre-encode outside timed region.
            io_t0 = time.perf_counter()
            ref_img = base_sequence.load(int(idx_map[0]), augment=False).image_256
            ref_frame = base_model.create_frame(ref_img)
            stats.io_ms += (time.perf_counter() - io_t0) * 1000.0

            setup_t0 = time.perf_counter()
            base_model.encode_frame(ref_frame)
            _cuda_sync(str(device))
            stats.setup_ms += (time.perf_counter() - setup_t0) * 1000.0

            for curr in range(1, total):
                key_idx = last_keep
                io_t0 = time.perf_counter()
                cur_img = base_sequence.load(int(idx_map[curr]), augment=False).image_256
                cur_frame = base_model.create_frame(cur_img)
                stats.io_ms += (time.perf_counter() - io_t0) * 1000.0

                _cuda_sync(str(device))
                inf_t0 = time.perf_counter()
                pred = base_model.forward_frames(ref_frame, cur_frame)
                _cuda_sync(str(device))
                inf_ms = (time.perf_counter() - inf_t0) * 1000.0

                if pred.overlap_score is None:
                    raise RuntimeError("Student model did not produce overlap_score")
                score = float(pred.overlap_score.squeeze().item())

                if curr >= start_count_idx:
                    stats.inference_ms += inf_ms
                    stats.inference_count += 1

                if student_cfg is not None and len(kept) < int(student_cfg.warmup_kept):
                    keep_flag = True
                else:
                    keep_flag = score >= tau_keep
                if keep_flag:
                    kept.append(curr)
                    last_keep = curr
                    ref_frame = cur_frame
                if score_log_path is not None:
                    log_rows.append((curr, key_idx, float(score), tau_keep, keep_flag))

        else:
            # Fallback: no encoder caching, but still exclude IO by loading tensors outside timing.
            io_t0 = time.perf_counter()
            ref_img = base_sequence.load(int(idx_map[0]), augment=False).image_256
            stats.io_ms += (time.perf_counter() - io_t0) * 1000.0
            ref = ref_img.unsqueeze(0).to(device)

            for curr in range(1, total):
                key_idx = last_keep
                io_t0 = time.perf_counter()
                cur_img = base_sequence.load(int(idx_map[curr]), augment=False).image_256
                stats.io_ms += (time.perf_counter() - io_t0) * 1000.0
                cur = cur_img.unsqueeze(0).to(device)

                _cuda_sync(str(device))
                inf_t0 = time.perf_counter()
                score = _student_score_from_tensors(model, ref=ref, cur=cur)
                _cuda_sync(str(device))
                inf_ms = (time.perf_counter() - inf_t0) * 1000.0

                if curr >= start_count_idx:
                    stats.inference_ms += inf_ms
                    stats.inference_count += 1

                if student_cfg is not None and len(kept) < int(student_cfg.warmup_kept):
                    keep_flag = True
                else:
                    keep_flag = score >= tau_keep
                if keep_flag:
                    kept.append(curr)
                    last_keep = curr
                    ref = cur
                if score_log_path is not None:
                    log_rows.append((curr, key_idx, float(score), tau_keep, keep_flag))

        if score_log_path is not None and log_rows:
            score_log_path.parent.mkdir(parents=True, exist_ok=True)
            with score_log_path.open("w", encoding="utf-8") as handle:
                handle.write("frame_idx,key_idx,score,tau_keep,keep\n")
                for frame_idx, key_idx, score, tau_keep, keep_flag in log_rows:
                    handle.write(
                        f"{int(frame_idx)},{int(key_idx)},{float(score):.6f},{float(tau_keep):.6f},{int(keep_flag)}\n"
                    )

        return kept, stats

    if input_subsample > 1:
        view = SubsampledSequenceView(sequence, input_subsample)
        kept_virtual, stats = _select_frames_student_cached(
            base_sequence=sequence,
            idx_map=list(view.real_indices),
        )
        _maybe_write_timing(
            stats,
            extra={"student_model": str(student_cfg.student_model) if student_cfg is not None else ""},
        )
        return view.get_mapped_indices(kept_virtual), stats

    kept, stats = _select_frames_student_cached(
        base_sequence=sequence,
        idx_map=list(range(len(sequence))),
    )
    _maybe_write_timing(
        stats,
        extra={"student_model": str(student_cfg.student_model) if student_cfg is not None else ""},
    )
    return kept, stats


def _write_manifest(
    out_path: Path,
    scene_root: Path,
    sequence: SceneFrameSequence,
    keep_indices: List[int],
) -> None:
    """Write RGB list with original timestamps from rgb.txt to avoid precision loss."""
    # Load original timestamps from rgb.txt to preserve precision.
    # Support both TUM-style (scene_root/rgb.txt) and EuRoC-style (scene_root/mav0/cam0/rgb.txt).
    original_timestamps: dict[str, str] = {}
    rgb_txt = scene_root / "rgb.txt"
    if not rgb_txt.exists():
        rgb_txt = scene_root / "mav0" / "cam0" / "rgb.txt"
    if rgb_txt.exists():
        with rgb_txt.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                # Preserve original timestamp string (avoid float round-trip).
                ts_str = parts[0]
                rel_path_raw = Path(" ".join(parts[1:]))
                abs_path = (rgb_txt.parent / rel_path_raw).resolve(strict=False)
                try:
                    rel_path_norm = abs_path.relative_to(scene_root).as_posix()
                except Exception:
                    rel_path_norm = rel_path_raw.as_posix()
                original_timestamps[rel_path_norm] = ts_str
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp filename\n")
        for idx in keep_indices:
            rec = sequence.record(idx)
            rel_path = rec.path.relative_to(scene_root)
            rel_path_str = rel_path.as_posix()
            
            # Use original timestamp string if available, otherwise use processed timestamp
            if rel_path_str in original_timestamps:
                ts_str = original_timestamps[rel_path_str]
            else:
                ts = rec.timestamp if rec.timestamp is not None else idx
                ts_str = str(ts)
            
            handle.write(f"{ts_str} {rel_path_str}\n")


def _load_canonical_indices(
    scene_root: Path,
    sequence: SceneFrameSequence,
    dataset_slug: str,
    scene_safe: str,
    canonical_root: Path,
) -> List[int]:
    """Load canonical keyframe indices from keyframe_only manifest, if available."""
    candidates = [
        canonical_root / dataset_slug / "keyframe_only" / f"{scene_safe}.txt",
        canonical_root / "keyframe_only" / f"{scene_safe}.txt",
    ]
    manifest_path = None
    for cand in candidates:
        if cand.exists():
            manifest_path = cand
            break
    if manifest_path is None:
        return []

    canonical_rel_paths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            rel_path = parts[0] if len(parts) == 1 else parts[1]
            canonical_rel_paths.add(rel_path)

    indices: List[int] = []
    for rec in sequence.records():
        rel = rec.path.relative_to(scene_root).as_posix()
        if rel in canonical_rel_paths:
            indices.append(rec.index)
    return indices


def _generate_for_scene(
    dataset_slug: str,
    scene_id: str,
    scene_root: Path,
    output_root: Path,
    device: str,
    timing_root: Path | None = None,
    timing_start_frame: int = 2,
    input_subsample: int = 1,
    canonical_root: Path | None = None,
    student_cfg: StudentScoreConfig | None = None,
) -> None:
    from slam_prefilter.utils.data_loader import SceneFrameSequence

    sequence, _meta = SceneFrameSequence.from_scene_root(scene_root)
    scene_safe = safe_scene_id(scene_id)
    canonical_indices: List[int] = []
    if canonical_root is not None:
        canonical_indices = _load_canonical_indices(
            scene_root=scene_root,
            sequence=sequence,
            dataset_slug=dataset_slug,
            scene_safe=scene_safe,
            canonical_root=canonical_root,
        )
    score_log_path = (
        output_root
        / dataset_slug
        / LEANGATE_POLICY_NAME
        / "scores"
        / f"{scene_safe}_scores.csv"
    )
    timing_base = output_root if timing_root is None else timing_root
    timing_log_path = (
        timing_base
        / dataset_slug
        / LEANGATE_POLICY_NAME
        / "timings"
        / f"{scene_safe}.json"
    )
    keep_indices, _timing = _build_keep_indices_for_leangate(
        sequence=sequence,
        device=device,
        score_log_path=score_log_path,
        timing_log_path=timing_log_path,
        timing_start_frame=int(timing_start_frame),
        input_subsample=input_subsample,
        student_cfg=student_cfg,
    )
    if canonical_indices:
        keep_indices = sorted(set(keep_indices) | set(canonical_indices))
    out_dir = output_root / dataset_slug / LEANGATE_POLICY_NAME
    out_path = out_dir / f"{scene_safe}.txt"
    _write_manifest(out_path, scene_root, sequence, keep_indices)
    print(f"[OK] {scene_id} / {LEANGATE_POLICY_NAME} -> {out_path}")


def _worker_process(
    task_queue: mp.Queue,
    dataset_root: Path,
    dataset_type: str,
    dataset_slug: str,
    device: str,
    output_root: Path,
    timing_root: Path | None = None,
    timing_start_frame: int = 2,
    input_subsample: int = 1,
    canonical_root: Path | None = None,
    student_cfg: StudentScoreConfig | None = None,
) -> None:
    while True:
        scene_id = task_queue.get()
        if scene_id is None:
            break
        scene_root = resolve_scene_root(dataset_root, dataset_type, scene_id)
        try:
            from slam_prefilter.utils.data_loader import SceneFrameSequence

            sequence, _meta = SceneFrameSequence.from_scene_root(scene_root)
        except Exception as exc:  # pragma: no cover - diagnostics
            print(f"[{device}] [ERR] {scene_id}: failed to load scene ({exc})")
            continue

        scene_safe = safe_scene_id(scene_id)
        canonical_indices: List[int] = []
        if canonical_root is not None:
            canonical_indices = _load_canonical_indices(
                scene_root=scene_root,
                sequence=sequence,
                dataset_slug=dataset_slug,
                scene_safe=scene_safe,
                canonical_root=canonical_root,
            )
        try:
            score_log_path = (
                output_root
                / dataset_slug
                / LEANGATE_POLICY_NAME
                / "scores"
                / f"{scene_safe}_scores.csv"
            )
            timing_base = output_root if timing_root is None else timing_root
            timing_log_path = (
                timing_base
                / dataset_slug
                / LEANGATE_POLICY_NAME
                / "timings"
                / f"{scene_safe}.json"
            )
            keep_indices, _timing = _build_keep_indices_for_leangate(
                sequence=sequence,
                device=device,
                score_log_path=score_log_path,
                timing_log_path=timing_log_path,
                timing_start_frame=int(timing_start_frame),
                input_subsample=input_subsample,
                student_cfg=student_cfg,
            )
            if canonical_indices:
                keep_indices = sorted(set(keep_indices) | set(canonical_indices))
            out_dir = output_root / dataset_slug / LEANGATE_POLICY_NAME
            out_path = out_dir / f"{scene_safe}.txt"
            _write_manifest(out_path, scene_root, sequence, keep_indices)
            print(f"[{device}] [OK] {scene_id} / {LEANGATE_POLICY_NAME} -> {out_path}")
        except Exception as exc:  # pragma: no cover - diagnostics
            print(f"[{device}] [ERR] {scene_id} / {LEANGATE_POLICY_NAME}: {exc}")


def _run_parallel_generation(
    scene_ids: list[str],
    dataset_root: Path,
    dataset_type: str,
    dataset_slug: str,
    output_root: Path,
    timing_root: Path | None,
    timing_start_frame: int,
    devices: list[str],
    input_subsample: int = 1,
    canonical_root: Path | None = None,
    student_cfg: StudentScoreConfig | None = None,
) -> None:
    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()

    for scene_id in scene_ids:
        task_queue.put(scene_id)
    for _ in devices:
        task_queue.put(None)

    workers: list[mp.Process] = []
    for device in devices:
        proc = ctx.Process(
                target=_worker_process,
                args=(
                    task_queue,
                    dataset_root,
                    dataset_type,
                    dataset_slug,
                    device,
                    output_root,
                    timing_root,
                    int(timing_start_frame),
                    input_subsample,
                    canonical_root,
                    student_cfg,
                ),
            )
        proc.start()
        workers.append(proc)

    for proc, device in zip(workers, devices):
        proc.join()
        if proc.exitcode not in (0, None):
            print(f"[WARN] Worker on {device} exited with code {proc.exitcode}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-scene RGB manifest files with the public LeanGate model."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Prepared dataset root. Public release supports TUM, 7SCENES, and EUROC only.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        help="Dataset type: TUM, 7SCENES, or EUROC.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PREDICTIONS_ROOT,
        help="Root directory to store generated manifests.",
    )
    parser.add_argument(
        "--timing-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory to store per-scene timing JSON files. "
            "Default writes to <output-root>/<dataset_slug>/leangate/timings/<scene>.json."
        ),
    )
    parser.add_argument(
        "--timing-start-frame",
        type=int,
        default=2,
        help="1-based frame index to start counting inference time (e.g., 2 excludes reference init).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional limit on number of scenes to process (in fixed order).",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing canonical keyframe_only manifests: "
            "root/<dataset_slug>/keyframe_only/<scene>.txt. "
            "If provided, generated keep indices are unioned with these keyframes."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for student inference.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated CUDA devices for parallel execution (e.g., cuda:0,cuda:1 or 0,1,2,3). "
        "If provided, overrides --device and uses multi-process worker queue.",
    )
    parser.add_argument(
        "--input-subsample",
        type=int,
        default=2,
        help="Subsample input sequence before applying policy (e.g., 2 = use only 0,2,4,6...).",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    os.environ["ENABLE_CUROPE2D"] = "1" if LEANGATE_ENABLE_CUROPE2D else "0"
    dataset_type, dataset_slug = normalize_dataset_type(args.dataset_type)
    dataset_root: Path = args.dataset_root
    output_root: Path = args.output_root
    timing_root: Path | None = args.timing_root
    timing_start_frame: int = int(args.timing_start_frame)
    canonical_root: Path | None = args.canonical_root
    device: str = args.device
    gpu_devices = _parse_gpus_arg(args.gpus)
    output_root.mkdir(parents=True, exist_ok=True)

    scene_ids = list(get_scenes(dataset_root, dataset_type, max_scenes=args.max_scenes))
    student_cfg = _build_leangate_student_cfg()

    for scene_id in scene_ids:
        scene_root = resolve_scene_root(dataset_root, dataset_type, scene_id)
        if not scene_root.exists():
            raise FileNotFoundError(f"Scene root not found: {scene_root}")

    if gpu_devices:
        _run_parallel_generation(
            scene_ids=scene_ids,
            dataset_root=dataset_root,
            dataset_type=dataset_type,
            dataset_slug=dataset_slug,
            output_root=output_root,
            timing_root=timing_root,
            timing_start_frame=timing_start_frame,
            devices=gpu_devices,
            input_subsample=args.input_subsample,
            canonical_root=canonical_root,
            student_cfg=student_cfg,
        )
    else:
        for scene_id in scene_ids:
            scene_root = resolve_scene_root(dataset_root, dataset_type, scene_id)
            _generate_for_scene(
                dataset_slug=dataset_slug,
                scene_id=scene_id,
                scene_root=scene_root,
                output_root=output_root,
                device=device,
                timing_root=timing_root,
                timing_start_frame=timing_start_frame,
                input_subsample=args.input_subsample,
                canonical_root=canonical_root,
                student_cfg=student_cfg,
            )


if __name__ == "__main__":
    _main()
