"""Shared pytest fixtures for dtst tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Chdir into an empty temp directory for the duration of the test.

    Functions under test that read/write relative paths (cache, sidecars,
    working-dir commands) should use this so they don't touch the repo.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def make_image(tmp_path: Path):
    """Factory that writes a small real image to disk and returns its Path."""

    def _make(
        name: str = "img.jpg", size: tuple[int, int] = (8, 8), color=(255, 0, 0)
    ) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        mode = "RGB"
        if suffix == ".png":
            fmt = "PNG"
        elif suffix == ".webp":
            fmt = "WEBP"
        elif suffix in (".tif", ".tiff"):
            fmt = "TIFF"
        elif suffix == ".bmp":
            fmt = "BMP"
        else:
            fmt = "JPEG"
        Image.new(mode, size, color=color).save(path, fmt)
        return path

    return _make


@pytest.fixture
def clean_dtst_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove dtst-specific env vars so tests see deterministic defaults."""
    for var in ("DTST_USER_AGENT",):
        monkeypatch.delenv(var, raising=False)
    # Also neutralize any XDG/HOME influence on cache paths
    monkeypatch.setenv("HOME", os.environ.get("HOME", "/tmp"))
