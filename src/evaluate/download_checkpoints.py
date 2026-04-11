from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from evaluate.public_config import (
    LEANGATE_CHECKPOINT_FILENAME,
    LEANGATE_HF_FILENAME,
    LEANGATE_HF_REPO,
    LEGACY_LEANGATE_CHECKPOINT_PATH,
    PUBLIC_LEANGATE_CHECKPOINT_PATH,
)


def _ensure_repo_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _existing_local_leangate_checkpoint() -> Path | None:
    for candidate in (PUBLIC_LEANGATE_CHECKPOINT_PATH, LEGACY_LEANGATE_CHECKPOINT_PATH):
        if candidate.exists():
            return candidate
    return None


def _download_leangate_checkpoint(output_root: Path, repo_id: str | None = None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / LEANGATE_CHECKPOINT_FILENAME
    if out_path.exists():
        return out_path

    local_source = _existing_local_leangate_checkpoint()
    if local_source is not None and local_source.resolve() != out_path.resolve():
        shutil.copy2(local_source, out_path)
        return out_path

    resolved_repo_id = LEANGATE_HF_REPO if repo_id is None else str(repo_id).strip()
    if not resolved_repo_id:
        raise RuntimeError(
            "LeanGate Hugging Face repo is not configured. "
            "Pass --repo-id, set LEANGATE_HF_REPO, or update evaluate.public_config.LEANGATE_HF_REPO."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for checkpoint download. "
            "Install it via `pip install huggingface_hub`."
        ) from exc

    downloaded = hf_hub_download(
        repo_id=resolved_repo_id,
        filename=LEANGATE_HF_FILENAME,
        local_dir=str(output_root),
        local_dir_use_symlinks=False,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != out_path.resolve():
        shutil.copy2(downloaded_path, out_path)
    return out_path


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the public LeanGate checkpoint."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where leangate.pt will be stored.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Optional Hugging Face repo id override for the public LeanGate checkpoint.",
    )
    args = parser.parse_args(argv)

    try:
        leangate_path = _download_leangate_checkpoint(Path(args.output_root), repo_id=args.repo_id)
    except Exception as exc:
        print(f"[error] Failed to prepare LeanGate checkpoint: {exc}", file=sys.stderr)
        return 1

    print(f"[ok] LeanGate checkpoint: {leangate_path}")
    print("[done] LeanGate checkpoint is ready.")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
