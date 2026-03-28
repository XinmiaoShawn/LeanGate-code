#!/usr/bin/env python3
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from evaluate.datasets import safe_scene_id
from evaluate.produce_policy_rgb_lists import (
    _build_keep_indices_for_leangate,
    _build_leangate_student_cfg,
    _write_manifest,
)
from evaluate.public_config import LEANGATE_ENABLE_CUROPE2D, LEANGATE_POLICY_NAME
from slam_prefilter.utils.data_loader import SceneFrameSequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LeanGate on one plain RGB folder and write a filtered RGB list."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Directory containing RGB frames to process in sorted filename order.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "demo",
        help="Root directory for the filtered manifest and score CSV.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Optional output scene name. Defaults to the folder basename.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for LeanGate inference.",
    )
    parser.add_argument(
        "--input-subsample",
        type=int,
        default=2,
        help="Subsample input sequence before applying the policy.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    os.environ["ENABLE_CUROPE2D"] = "1" if LEANGATE_ENABLE_CUROPE2D else "0"

    rgb_folder = args.folder.resolve()
    if not rgb_folder.is_dir():
        raise NotADirectoryError(f"--folder must point to an existing directory: {rgb_folder}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scene_name = args.scene_name or rgb_folder.name
    scene_safe = safe_scene_id(scene_name)
    sequence, _meta = SceneFrameSequence.from_rgb_folder(rgb_folder)

    score_log_path = output_root / LEANGATE_POLICY_NAME / "scores" / f"{scene_safe}_scores.csv"

    keep_indices = _build_keep_indices_for_leangate(
        sequence=sequence,
        device=args.device,
        score_log_path=score_log_path,
        input_subsample=int(args.input_subsample),
        student_cfg=_build_leangate_student_cfg(),
    )

    out_path = output_root / LEANGATE_POLICY_NAME / f"{scene_safe}.txt"
    _write_manifest(out_path, rgb_folder, sequence, keep_indices)
    print(f"[OK] {rgb_folder} / {LEANGATE_POLICY_NAME} -> {out_path}")


if __name__ == "__main__":
    _main()
