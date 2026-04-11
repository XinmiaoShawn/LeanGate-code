from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from evaluate.public_config import SUPPORTED_PUBLIC_DATASETS

# Canonical TUM scene list used in paper tables.
TUM_SCENES: List[str] = [
    "rgbd_dataset_freiburg1_360",
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_desk2",
    "rgbd_dataset_freiburg1_floor",
    "rgbd_dataset_freiburg1_plant",
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg1_rpy",
    "rgbd_dataset_freiburg1_teddy",
    "rgbd_dataset_freiburg1_xyz",
]

# Canonical EuRoC sequence names (as used by MASt3R-SLAM scripts).
EUROC_SCENES: List[str] = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    # Optional fixed scene ids; if None we will discover by scanning for rgb.txt.
    scenes: List[str] | None
    # Candidate GT and calib filenames expected under each scene root.
    gt_candidates: Tuple[str, ...]
    calib_candidates: Tuple[str, ...]


_ALIASES = {
    "TUM": "TUM",
    "7SCENES": "7SCENES",
    "SEVEN_SCENES": "7SCENES",
    "EUROC": "EUROC",
}

DATASETS: Dict[str, DatasetSpec] = {
    "TUM": DatasetSpec(
        slug="tum",
        scenes=TUM_SCENES,
        gt_candidates=("groundtruth.txt", "gt.txt", "traj_gt_tum.txt"),
        calib_candidates=("camera_640.yaml", "calib.yaml", "camera.txt"),
    ),
    "7SCENES": DatasetSpec(
        slug="7scenes",
        scenes=None,  # discover seq-* containing rgb.txt
        gt_candidates=("groundtruth.txt", "gt.txt", "traj_gt_tum.txt"),
        calib_candidates=("calib.yaml", "camera_640.yaml", "camera.txt"),
    ),
    "EUROC": DatasetSpec(
        slug="euroc",
        scenes=None,  # discover subdirs with rgb.txt
        gt_candidates=("groundtruth.txt", "gt.txt", "traj_gt_tum.txt"),
        calib_candidates=("calib.yaml", "calibration.txt", "camera.txt"),
    ),
}


def safe_scene_id(scene_id: str) -> str:
    """Return a filename-safe version of a scene identifier."""
    s = str(scene_id).strip()
    # If discovery returns the dataset root (relative "."), avoid writing hidden files like ".txt".
    if s in {"", ".", "./"}:
        return "root"
    return s.replace("\\", "_").replace("/", "_")


def normalize_dataset_type(raw: str) -> Tuple[str, str]:
    """Return normalized dataset_type (upper) and slug."""
    key = raw.upper()
    if key not in _ALIASES:
        supported = ", ".join(SUPPORTED_PUBLIC_DATASETS)
        raise ValueError(
            f"Unsupported dataset_type {raw!r}. Public release supports only: {supported}."
        )
    norm = _ALIASES[key]
    spec = DATASETS[norm]
    return norm, spec.slug


@lru_cache(maxsize=32)
def _euroc_label_to_scene_root_rel(dataset_root: Path) -> Dict[str, Path]:
    """Return a mapping from EuRoC sequence label -> sequence directory (relative to dataset_root).

    EuRoC scenes are identified by their sequence folder name (e.g. "MH_01_easy"),
    while the required manifests (e.g. rgb.txt) may live under mav0/cam0/.
    """
    dataset_root = Path(dataset_root)
    mapping: Dict[str, Path] = {}
    for rgb_txt in sorted(dataset_root.rglob("rgb.txt")):
        parent = rgb_txt.parent
        try:
            # Common EuRoC layout: <seq>/mav0/cam0/rgb.txt  -> label is <seq>.
            if parent.name == "cam0" and parent.parent.name == "mav0":
                seq_dir = parent.parent.parent
            # Flat prepared layout: <seq>/rgb.txt + <seq>/data/*
            elif (parent / "data").is_dir():
                seq_dir = parent
            # Alternative prepared layout: <seq>/rgb.txt (but still has mav0/cam0 underneath).
            elif (parent / "mav0" / "cam0").is_dir():
                seq_dir = parent
            else:
                continue
            label = seq_dir.name
            rel_seq = seq_dir.relative_to(dataset_root)
        except Exception:
            continue
        if not label:
            continue
        if label in mapping and mapping[label] != rel_seq:
            # Allow both nested and flattened layouts to coexist temporarily:
            # prefer the shorter relative path (usually <label>/ over machine_hall/<label>/).
            prev = mapping[label]
            choose = rel_seq
            if len(prev.parts) < len(rel_seq.parts):
                choose = prev
            elif len(prev.parts) == len(rel_seq.parts):
                choose = min(prev, rel_seq)
            mapping[label] = choose
        else:
            mapping[label] = rel_seq
    return mapping


def resolve_scene_root(dataset_root: Path, dataset_type: str, scene_id: str) -> Path:
    """Resolve a scene identifier to an on-disk scene root directory.

    For EuRoC, `scene_id` is a short sequence label (e.g. "MH_01_easy") while
    the actual directory is usually nested (e.g. machine_hall/MH_01_easy/).
    For other datasets, `scene_id` is already a relative path under dataset_root.
    """
    dataset_root = Path(dataset_root)
    dataset_type = dataset_type.upper()

    # Backwards-compatible: if scene_id is already a valid path, accept it.
    direct = dataset_root / scene_id
    if direct.exists():
        return direct

    if dataset_type == "EUROC":
        mapping = _euroc_label_to_scene_root_rel(dataset_root)
        rel = mapping.get(scene_id)
        if rel is None:
            raise FileNotFoundError(
                f"EuRoC scene {scene_id!r} not found under {dataset_root}. "
                f"Known scenes: {sorted(mapping.keys())}"
            )
        return dataset_root / rel

    return direct


def _candidate_scene_roots(scene_root: Path) -> List[Path]:
    """Return candidate roots where per-scene files may live.

    Supports EuRoC-style nested manifests under scene_root/mav0/cam0/.
    """
    scene_root = Path(scene_root)
    roots = [scene_root]
    euroc_cam0 = scene_root / "mav0" / "cam0"
    if euroc_cam0.is_dir():
        roots.append(euroc_cam0)
    return roots


def find_rgb_txt(scene_root: Path) -> Path | None:
    """Find rgb.txt under scene_root (TUM) or scene_root/mav0/cam0 (EuRoC)."""
    for root in _candidate_scene_roots(scene_root):
        cand = root / "rgb.txt"
        if cand.exists():
            return cand
    return None


def find_gt_path(scene_root: Path, dataset_type: str) -> Path | None:
    """Find a ground-truth file under scene_root (or nested EuRoC cam0 root)."""
    dataset_type = dataset_type.upper()
    spec = DATASETS[dataset_type]
    for root in _candidate_scene_roots(scene_root):
        for name in spec.gt_candidates:
            cand = root / name
            if cand.exists():
                return cand
    return None


def find_calib_path(scene_root: Path, dataset_type: str) -> Path | None:
    """Find a calibration file under scene_root (or nested EuRoC cam0 root)."""
    dataset_type = dataset_type.upper()
    spec = DATASETS[dataset_type]
    extra = ("camera.yaml",)
    for root in _candidate_scene_roots(scene_root):
        for name in spec.calib_candidates + extra:
            cand = root / name
            if cand.exists():
                return cand
    return None


def _discover_scene_ids(dataset_root: Path) -> List[str]:
    """Discover scene ids by locating rgb.txt under dataset_root."""
    ids: List[str] = []
    for rgb_txt in sorted(dataset_root.rglob("rgb.txt")):
        try:
            scene_id = str(rgb_txt.parent.relative_to(dataset_root))
        except ValueError:
            continue
        ids.append(scene_id)
    return ids


def get_scenes(dataset_root: Path, dataset_type: str, max_scenes: int | None = None) -> List[str]:
    """Return scene identifiers (relative paths) for a dataset."""
    dataset_type = dataset_type.upper()
    if dataset_type not in DATASETS:
        supported = ", ".join(SUPPORTED_PUBLIC_DATASETS)
        raise ValueError(
            f"Unsupported dataset_type {dataset_type!r}. Public release supports only: {supported}."
        )
    spec = DATASETS[dataset_type]

    if spec.scenes is not None:
        # Prefer canonical scene ordering when it matches the on-disk layout, but
        # fall back to discovery for "single-scene" or repacked datasets where
        # rgb.txt lives directly under dataset_root (common for ad-hoc exports).
        canonical_existing = [s for s in spec.scenes if (Path(dataset_root) / s).exists()]
        scene_ids = canonical_existing if canonical_existing else _discover_scene_ids(Path(dataset_root))
    else:
        # Dataset-specific discovery for layouts that embed rgb.txt deeper than
        # the logical sequence root.
        if dataset_type == "EUROC":
            mapping = _euroc_label_to_scene_root_rel(dataset_root)
            # Prefer canonical ordering when present, then append any extra discovered scenes.
            scene_ids = [s for s in EUROC_SCENES if s in mapping]
            extras = sorted(set(mapping.keys()) - set(scene_ids))
            scene_ids.extend(extras)
        else:
            scene_ids = _discover_scene_ids(dataset_root)

    if not scene_ids:
        raise ValueError(f"No scenes found for {dataset_type} under {dataset_root}")
    if max_scenes is not None:
        scene_ids = scene_ids[:max_scenes]
    return scene_ids


def validate_scene_root(scene_root: Path, dataset_type: str) -> None:
    """Lightweight validation: ensure scene_root exists and has rgb frames and GT."""
    dataset_type = dataset_type.upper()
    spec = DATASETS[dataset_type]
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene root does not exist: {scene_root}")

    rgb_ok = find_rgb_txt(scene_root) is not None or (scene_root / "rgb").is_dir()
    if not rgb_ok:
        raise FileNotFoundError(f"Scene {scene_root} missing rgb.txt or rgb/ directory")

    if find_gt_path(scene_root, dataset_type) is None:
        raise FileNotFoundError(
            f"Scene {scene_root} missing ground truth (looked for {spec.gt_candidates})"
        )

    if find_calib_path(scene_root, dataset_type) is None:
        raise FileNotFoundError(
            f"Scene {scene_root} missing calibration file (looked for {spec.calib_candidates})"
        )
