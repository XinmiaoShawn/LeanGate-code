from __future__ import annotations

from pathlib import Path
import sys
import types

from evaluate import download_checkpoints as dc


def test_download_checkpoint_success(monkeypatch, tmp_path: Path, capsys) -> None:
    def fake_download(output_root: Path, repo_id: str | None = None) -> Path:
        output_root.mkdir(parents=True, exist_ok=True)
        path = output_root / "leangate.pt"
        path.write_text("weights", encoding="utf-8")
        return path

    monkeypatch.setattr(dc, "_download_leangate_checkpoint", fake_download)

    rc = dc.run(["--output-root", str(tmp_path / "checkpoints")])
    captured = capsys.readouterr()
    assert rc == 0
    assert "LeanGate checkpoint is ready." in captured.out


def test_download_checkpoint_accepts_repo_override(monkeypatch, tmp_path: Path) -> None:
    calls: list[str | None] = []

    def fake_download(output_root: Path, repo_id: str | None = None) -> Path:
        calls.append(repo_id)
        output_root.mkdir(parents=True, exist_ok=True)
        path = output_root / "leangate.pt"
        path.write_text("weights", encoding="utf-8")
        return path

    monkeypatch.setattr(dc, "_download_leangate_checkpoint", fake_download)

    rc = dc.run(["--output-root", str(tmp_path / "checkpoints"), "--repo-id", "org/repo"])
    assert rc == 0
    assert calls == ["org/repo"]


def test_download_checkpoint_uses_public_default_repo(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(dc, "PUBLIC_LEANGATE_CHECKPOINT_PATH", tmp_path / "public" / "leangate.pt")
    monkeypatch.setattr(dc, "LEGACY_LEANGATE_CHECKPOINT_PATH", tmp_path / "legacy" / "leangate.pt")
    monkeypatch.setattr(dc, "LEANGATE_HF_REPO", "ShawnX98/LeanGate")
    monkeypatch.setattr(dc, "LEANGATE_HF_FILENAME", "leangate.pt")

    def fake_hf_hub_download(*, repo_id: str, filename: str, local_dir: str, local_dir_use_symlinks: bool):
        calls.append((repo_id, filename))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("weights", encoding="utf-8")
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    out_path = dc._download_leangate_checkpoint(tmp_path / "checkpoints")
    assert out_path == tmp_path / "checkpoints" / "leangate.pt"
    assert calls == [("ShawnX98/LeanGate", "leangate.pt")]
