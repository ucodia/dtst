"""Tests for dtst.files pure-logic helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from dtst.errors import InputError
from dtst.files import (
    _split_csv,
    build_save_kwargs,
    copy_image,
    find_images,
    find_videos,
    format_elapsed,
    gather_images,
    gather_videos,
    move_image,
    resolve_dirs,
    resolve_workers,
)
from dtst.sidecar import sidecar_path


# ---------------------------------------------------------------------------
# resolve_dirs
# ---------------------------------------------------------------------------


class TestResolveDirs:
    def test_absolute_path_stays_absolute(self, tmp_path: Path) -> None:
        d = tmp_path / "a"
        d.mkdir()
        result = resolve_dirs([str(d)])
        assert result == [d.resolve()]

    def test_relative_path_resolved(self, isolated_cwd: Path) -> None:
        (isolated_cwd / "sub").mkdir()
        result = resolve_dirs(["sub"])
        assert result == [(isolated_cwd / "sub").resolve()]
        assert result[0].is_absolute()

    def test_duplicates_deduplicated_preserving_order(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        result = resolve_dirs([str(a), str(b), str(a)])
        assert result == [a.resolve(), b.resolve()]

    def test_glob_expands_to_existing_dirs_only(self, isolated_cwd: Path) -> None:
        (isolated_cwd / "images").mkdir()
        (isolated_cwd / "images" / "one").mkdir()
        (isolated_cwd / "images" / "two").mkdir()
        # Create a file (not a dir) matching the glob
        (isolated_cwd / "images" / "three.txt").write_text("hi")
        result = resolve_dirs(["images/*"])
        names = {p.name for p in result}
        assert names == {"one", "two"}
        assert all(p.is_dir() for p in result)

    def test_nonexistent_nonglob_path_still_included(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        result = resolve_dirs([str(missing)])
        assert result == [missing.resolve()]

    def test_tilde_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "fakehome"
        home.mkdir()
        (home / "foo").mkdir()
        monkeypatch.setenv("HOME", str(home))
        result = resolve_dirs(["~/foo"])
        assert result == [(home / "foo").resolve()]


# ---------------------------------------------------------------------------
# find_images / find_videos
# ---------------------------------------------------------------------------


class TestFindImages:
    def test_finds_all_supported_extensions(self, tmp_path: Path, make_image) -> None:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"):
            make_image(name=f"img{ext}")
        result = find_images(tmp_path)
        assert len(result) == 7

    def test_case_insensitive(self, tmp_path: Path, make_image) -> None:
        make_image(name="upper.JPG")
        make_image(name="mixed.PnG")
        result = find_images(tmp_path)
        assert len(result) == 2

    def test_non_recursive_does_not_descend(self, tmp_path: Path, make_image) -> None:
        make_image(name="top.jpg")
        make_image(name="sub/nested.jpg")
        result = find_images(tmp_path, recursive=False)
        assert [p.name for p in result] == ["top.jpg"]

    def test_recursive_descends(self, tmp_path: Path, make_image) -> None:
        make_image(name="top.jpg")
        make_image(name="sub/nested.jpg")
        result = find_images(tmp_path, recursive=True)
        assert {p.name for p in result} == {"top.jpg", "nested.jpg"}

    def test_excludes_non_image_files(self, tmp_path: Path, make_image) -> None:
        make_image(name="real.jpg")
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "video.mp4").write_bytes(b"fake")
        result = find_images(tmp_path)
        assert [p.name for p in result] == ["real.jpg"]

    def test_results_sorted(self, tmp_path: Path, make_image) -> None:
        for name in ("c.jpg", "a.jpg", "b.jpg"):
            make_image(name=name)
        result = find_images(tmp_path)
        assert [p.name for p in result] == ["a.jpg", "b.jpg", "c.jpg"]


class TestFindVideos:
    def test_finds_supported_video_extensions(self, tmp_path: Path) -> None:
        for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"):
            (tmp_path / f"clip{ext}").write_bytes(b"fake")
        result = find_videos(tmp_path)
        assert len(result) == 8

    def test_excludes_images(self, tmp_path: Path, make_image) -> None:
        make_image(name="photo.jpg")
        (tmp_path / "clip.mp4").write_bytes(b"fake")
        result = find_videos(tmp_path)
        assert [p.name for p in result] == ["clip.mp4"]

    def test_case_insensitive(self, tmp_path: Path) -> None:
        (tmp_path / "clip.MP4").write_bytes(b"fake")
        result = find_videos(tmp_path)
        assert len(result) == 1

    def test_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.mp4").write_bytes(b"fake")
        (tmp_path / "top.mp4").write_bytes(b"fake")
        non_rec = find_videos(tmp_path, recursive=False)
        rec = find_videos(tmp_path, recursive=True)
        assert {p.name for p in non_rec} == {"top.mp4"}
        assert {p.name for p in rec} == {"top.mp4", "nested.mp4"}


# ---------------------------------------------------------------------------
# build_save_kwargs
# ---------------------------------------------------------------------------


class TestBuildSaveKwargs:
    @pytest.mark.parametrize("suffix", [".jpg", ".jpeg", ".JPEG", ".webp", ".WEBP"])
    def test_jpeg_webp_gets_quality(self, suffix: str) -> None:
        result = build_save_kwargs(Path(f"img{suffix}"))
        assert result == {"quality": 95}

    def test_png_gets_compress_level(self) -> None:
        result = build_save_kwargs(Path("img.png"))
        assert result == {"compress_level": 6}

    def test_png_uppercase(self) -> None:
        result = build_save_kwargs(Path("img.PNG"))
        assert result == {"compress_level": 6}

    def test_other_format_empty(self) -> None:
        assert build_save_kwargs(Path("img.gif")) == {}
        assert build_save_kwargs(Path("img.bmp")) == {}

    def test_custom_quality(self) -> None:
        result = build_save_kwargs(Path("img.jpg"), quality=70)
        assert result == {"quality": 70}

    def test_custom_compress_level(self) -> None:
        result = build_save_kwargs(Path("img.png"), compress_level=9)
        assert result == {"compress_level": 9}


# ---------------------------------------------------------------------------
# move_image / copy_image
# ---------------------------------------------------------------------------


class TestMoveImage:
    def test_moves_image_without_sidecar(self, tmp_path: Path, make_image) -> None:
        src = make_image(name="a.jpg")
        dest = tmp_path / "b.jpg"
        move_image(src, dest)
        assert not src.exists()
        assert dest.exists()

    def test_moves_image_and_sidecar(self, tmp_path: Path, make_image) -> None:
        src = make_image(name="a.jpg")
        sidecar_path(src).write_text(json.dumps({"meta": 1}))
        dest = tmp_path / "b.jpg"
        move_image(src, dest)
        assert not src.exists()
        assert not sidecar_path(src).exists()
        assert dest.exists()
        assert sidecar_path(dest).exists()
        assert json.loads(sidecar_path(dest).read_text()) == {"meta": 1}

    def test_no_error_when_sidecar_absent(self, tmp_path: Path, make_image) -> None:
        src = make_image(name="a.jpg")
        dest = tmp_path / "b.jpg"
        # Should not raise
        move_image(src, dest)
        assert dest.exists()
        assert not sidecar_path(dest).exists()


class TestCopyImage:
    def test_copies_image_without_sidecar(self, tmp_path: Path, make_image) -> None:
        src = make_image(name="a.jpg")
        dest = tmp_path / "b.jpg"
        copy_image(src, dest)
        assert src.exists()
        assert dest.exists()

    def test_copies_image_and_sidecar(self, tmp_path: Path, make_image) -> None:
        src = make_image(name="a.jpg")
        sidecar_path(src).write_text(json.dumps({"meta": 2}))
        dest = tmp_path / "b.jpg"
        copy_image(src, dest)
        assert src.exists()
        assert sidecar_path(src).exists()
        assert dest.exists()
        assert sidecar_path(dest).exists()


# ---------------------------------------------------------------------------
# resolve_workers
# ---------------------------------------------------------------------------


class TestResolveWorkers:
    @pytest.mark.parametrize("value", [1, 4, 64, 1024])
    def test_explicit_int_returned_asis(self, value: int) -> None:
        assert resolve_workers(value) == value

    def test_none_returns_positive_int_from_cpu_count(self) -> None:
        result = resolve_workers(None)
        assert isinstance(result, int)
        assert result > 0

    def test_fallback_used_when_cpu_count_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("dtst.files.cpu_count", lambda: 0)
        assert resolve_workers(None) == 4

    def test_custom_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("dtst.files.cpu_count", lambda: 0)
        assert resolve_workers(None, fallback=12) == 12


# ---------------------------------------------------------------------------
# format_elapsed
# ---------------------------------------------------------------------------


class TestFormatElapsed:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "0m 0s"),
            (59, "0m 59s"),
            (60, "1m 0s"),
            (125.7, "2m 5s"),
            (3600, "60m 0s"),
        ],
    )
    def test_format(self, seconds: float, expected: str) -> None:
        assert format_elapsed(seconds) == expected


# ---------------------------------------------------------------------------
# _split_csv
# ---------------------------------------------------------------------------


class TestSplitCsv:
    def test_empty_string(self) -> None:
        assert _split_csv("") == []

    def test_basic_split_with_whitespace(self) -> None:
        assert _split_csv("a, b ,c") == ["a", "b", "c"]

    def test_empty_fields_ignored(self) -> None:
        assert _split_csv(",,a,,") == ["a"]

    def test_leading_trailing_whitespace(self) -> None:
        assert _split_csv("  x  ,  y  ") == ["x", "y"]


# ---------------------------------------------------------------------------
# gather_images / gather_videos
# ---------------------------------------------------------------------------


class TestGatherImages:
    def test_two_dirs(self, isolated_cwd: Path, make_image) -> None:
        a = isolated_cwd / "a"
        b = isolated_cwd / "b"
        a.mkdir()
        b.mkdir()
        make_image(name="a/one.jpg")
        make_image(name="b/two.jpg")
        input_dirs, items = gather_images("a,b")
        assert len(input_dirs) == 2
        assert len(items) == 2
        assert {p.name for p in items} == {"one.jpg", "two.jpg"}

    def test_missing_dir_logged_and_skipped(
        self,
        isolated_cwd: Path,
        make_image,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        a = isolated_cwd / "a"
        a.mkdir()
        make_image(name="a/one.jpg")
        with caplog.at_level(logging.WARNING, logger="dtst.files"):
            input_dirs, items = gather_images("a,missing")
        assert len(items) == 1
        assert any("does not exist" in rec.getMessage() for rec in caplog.records)

    def test_all_missing_raises(self, isolated_cwd: Path) -> None:
        with pytest.raises(InputError):
            gather_images("missing1,missing2")

    def test_empty_dirs_raises(self, isolated_cwd: Path) -> None:
        (isolated_cwd / "a").mkdir()
        with pytest.raises(InputError):
            gather_images("a")

    def test_recursive_flag_propagates(self, isolated_cwd: Path, make_image) -> None:
        a = isolated_cwd / "a"
        a.mkdir()
        make_image(name="a/top.jpg")
        make_image(name="a/sub/nested.jpg")
        _, items_non = gather_images("a", recursive=False)
        _, items_rec = gather_images("a", recursive=True)
        assert {p.name for p in items_non} == {"top.jpg"}
        assert {p.name for p in items_rec} == {"top.jpg", "nested.jpg"}

    def test_returns_tuple_shape(self, isolated_cwd: Path, make_image) -> None:
        (isolated_cwd / "a").mkdir()
        make_image(name="a/one.jpg")
        result = gather_images("a")
        assert isinstance(result, tuple)
        assert len(result) == 2
        input_dirs, items = result
        assert isinstance(input_dirs, list)
        assert isinstance(items, list)


class TestGatherVideos:
    def test_basic(self, isolated_cwd: Path) -> None:
        (isolated_cwd / "vids").mkdir()
        (isolated_cwd / "vids" / "clip.mp4").write_bytes(b"fake")
        input_dirs, items = gather_videos("vids")
        assert len(input_dirs) == 1
        assert [p.name for p in items] == ["clip.mp4"]

    def test_empty_raises(self, isolated_cwd: Path) -> None:
        (isolated_cwd / "vids").mkdir()
        with pytest.raises(InputError):
            gather_videos("vids")
