from __future__ import annotations

from pathlib import Path

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
