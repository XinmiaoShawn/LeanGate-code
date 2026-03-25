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
LEANGATE_OVERLAP_ITERS = 4
LEANGATE_DECODER_DEPTH = 6

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

DEFAULT_PREDICTIONS_ROOT = REPO_ROOT / "outputs" / "predictions"
