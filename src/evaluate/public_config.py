from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_PUBLIC_DATASETS = ("TUM", "7SCENES", "EUROC")

LEANGATE_POLICY_NAME = "leangate"
LEANGATE_CHECKPOINT_FILENAME = "leangate.pt"
LEANGATE_THRESHOLD = 0.66
LEANGATE_WARMUP_KEPT = 5
LEANGATE_ENABLE_CUROPE2D = False

PUBLIC_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
PUBLIC_LEANGATE_CHECKPOINT_PATH = PUBLIC_CHECKPOINT_DIR / LEANGATE_CHECKPOINT_FILENAME
LEGACY_LEANGATE_CHECKPOINT_PATH = (
    REPO_ROOT
    / "checkpoints"
    / "iterative"
    / "flare_iter4_dec6_train_decoder_lr5e-5_img256_bs256x4_20ep"
    / "overlap_iterativemodel_dec6_full.pt"
)

LEANGATE_HF_REPO = os.environ.get("LEANGATE_HF_REPO", "")
LEANGATE_HF_FILENAME = os.environ.get("LEANGATE_HF_FILENAME", LEANGATE_CHECKPOINT_FILENAME)

FLARE_GEOMETRY_CHECKPOINT_PATH = REPO_ROOT / "third_party" / "FLARE" / "checkpoints" / "geometry_pose.pth"
FLARE_GEOMETRY_CHECKPOINT_URL = (
    "https://huggingface.co/AntResearch/FLARE/resolve/main/geometry_pose.pth"
)

DEFAULT_PREDICTIONS_ROOT = REPO_ROOT / "outputs" / "predictions"
