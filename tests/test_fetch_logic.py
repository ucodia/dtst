"""Tests for dtst.core.fetch pure-logic helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dtst.core.fetch import (
    CONTENT_TYPE_TO_EXT,
    IMAGE_EXTENSIONS,
    UNSUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    YTDLP_DOMAINS,
    _is_ytdlp_url,
    _load_urls_from_jsonl,
    _load_urls_from_txt,
)


# ---------------------------------------------------------------------------
# _is_ytdlp_url
# ---------------------------------------------------------------------------


class TestIsYtdlpUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://youtube.com/watch?v=abc",
            "https://www.youtube.com/watch?v=abc",
            "https://m.youtube.com/watch?v=abc",
            "https://youtu.be/abc",
            "https://vimeo.com/12345",
            "https://www.vimeo.com/12345",
            "https://player.vimeo.com/video/12345",
            "https://twitch.tv/somechannel",
            "https://www.twitch.tv/somechannel",
            "https://clips.twitch.tv/abc",
        ],
    )
    def test_true_for_canonical_video_hosts(self, url: str) -> None:
        assert _is_ytdlp_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/img.jpg",
            "https://flickr.com/photos/abc/123",
            "https://upload.wikimedia.org/wikipedia/commons/a/ab/foo.jpg",
            "http://images.google.com/image.png",
        ],
    )
    def test_false_for_image_hosts(self, url: str) -> None:
        assert _is_ytdlp_url(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            "https://YOUTUBE.com/watch?v=abc",
            "https://YouTu.Be/abc",
            "https://Www.Vimeo.COM/12345",
            "https://TWITCH.TV/channel",
        ],
    )
    def test_case_insensitive_hostname(self, url: str) -> None:
        assert _is_ytdlp_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "not-a-url",
            "///no-scheme-no-host",
            "youtube.com/watch?v=abc",  # no scheme -> urlparse has no hostname
        ],
    )
    def test_malformed_urls_return_false(self, url: str) -> None:
        # Must not raise
        assert _is_ytdlp_url(url) is False


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_image_extensions_membership(self) -> None:
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"):
            assert ext in IMAGE_EXTENSIONS
        # All lowercase, dot-prefixed
        assert all(e.startswith(".") and e == e.lower() for e in IMAGE_EXTENSIONS)

    def test_video_extensions_membership(self) -> None:
        for ext in (".mp4", ".mkv", ".mov", ".webm"):
            assert ext in VIDEO_EXTENSIONS
        assert all(e.startswith(".") and e == e.lower() for e in VIDEO_EXTENSIONS)

    def test_image_and_video_are_disjoint(self) -> None:
        assert IMAGE_EXTENSIONS.isdisjoint(VIDEO_EXTENSIONS)

    def test_unsupported_extensions_membership(self) -> None:
        assert ".djvu" in UNSUPPORTED_EXTENSIONS
        assert ".svg" in UNSUPPORTED_EXTENSIONS
        assert ".gif" in UNSUPPORTED_EXTENSIONS

    def test_content_type_maps_to_known_image_extensions(self) -> None:
        assert CONTENT_TYPE_TO_EXT["image/jpeg"] == ".jpg"
        assert CONTENT_TYPE_TO_EXT["image/png"] == ".png"
        assert CONTENT_TYPE_TO_EXT["image/webp"] == ".webp"
        # All mapped extensions are either image or gif (unsupported but mappable)
        known = IMAGE_EXTENSIONS | UNSUPPORTED_EXTENSIONS
        for ext in CONTENT_TYPE_TO_EXT.values():
            assert ext in known

    def test_ytdlp_domains_membership(self) -> None:
        for host in (
            "youtube.com",
            "www.youtube.com",
            "youtu.be",
            "vimeo.com",
            "twitch.tv",
        ):
            assert host in YTDLP_DOMAINS
        # Hostnames stored lowercase, no scheme/path
        assert all(
            "/" not in d and d == d.lower() and not d.startswith("http")
            for d in YTDLP_DOMAINS
        )


# ---------------------------------------------------------------------------
# _load_urls_from_jsonl
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestLoadUrlsFromJsonl:
    def test_reads_urls_happy_path(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        _write_jsonl(
            f,
            [
                {"url": "https://a.example.com/img.jpg", "engine": "flickr"},
                {"url": "https://b.example.com/pic.png", "engine": "brave"},
            ],
        )
        urls, skipped, sidecars = _load_urls_from_jsonl(
            f, min_size=0, license_filter=None
        )
        assert urls == [
            "https://a.example.com/img.jpg",
            "https://b.example.com/pic.png",
        ]
        assert skipped == 0
        assert sidecars["https://a.example.com/img.jpg"]["source"] == "flickr"
        assert sidecars["https://b.example.com/pic.png"]["source"] == "brave"
        # Default license when missing
        assert sidecars["https://a.example.com/img.jpg"]["license"] == "none"

    def test_urls_sorted_and_deduplicated(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        _write_jsonl(
            f,
            [
                {"url": "https://c.example.com/x.jpg"},
                {"url": "https://a.example.com/x.jpg"},
                {"url": "https://b.example.com/x.jpg"},
                {"url": "https://a.example.com/x.jpg"},  # dup
            ],
        )
        urls, _skipped, _sidecars = _load_urls_from_jsonl(
            f, min_size=0, license_filter=None
        )
        assert urls == [
            "https://a.example.com/x.jpg",
            "https://b.example.com/x.jpg",
            "https://c.example.com/x.jpg",
        ]

    def test_unsupported_extensions_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        _write_jsonl(
            f,
            [
                {"url": "https://a.example.com/img.jpg"},
                {"url": "https://b.example.com/scan.djvu"},
                {"url": "https://c.example.com/icon.svg"},
                {"url": "https://d.example.com/anim.gif"},
            ],
        )
        urls, skipped, _sidecars = _load_urls_from_jsonl(
            f, min_size=0, license_filter=None
        )
        assert urls == ["https://a.example.com/img.jpg"]
        assert skipped == 3

    def test_min_size_filter(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        _write_jsonl(
            f,
            [
                {"url": "https://a.example.com/small.jpg", "width": 100, "height": 100},
                {"url": "https://b.example.com/big.jpg", "width": 2000, "height": 1000},
                {"url": "https://c.example.com/tall.jpg", "width": 100, "height": 900},
            ],
        )
        urls, _skipped, _sidecars = _load_urls_from_jsonl(
            f, min_size=512, license_filter=None
        )
        # 'small' is dropped, 'big' (2000 max) and 'tall' (900 max) kept
        assert urls == [
            "https://b.example.com/big.jpg",
            "https://c.example.com/tall.jpg",
        ]

    def test_missing_dimensions_not_filtered(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        _write_jsonl(
            f,
            [
                {"url": "https://a.example.com/nosize.jpg"},
                {"url": "https://b.example.com/partial.jpg", "width": 100},
            ],
        )
        urls, _skipped, _sidecars = _load_urls_from_jsonl(
            f, min_size=99999, license_filter=None
        )
        # Both kept since dims are missing/incomplete
        assert urls == [
            "https://a.example.com/nosize.jpg",
            "https://b.example.com/partial.jpg",
        ]

    def test_license_filter_prefix_match(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        _write_jsonl(
            f,
            [
                {"url": "https://a.example.com/x.jpg", "license": "cc-by-4.0"},
                {"url": "https://b.example.com/x.jpg", "license": "cc-by-sa-4.0"},
                {"url": "https://c.example.com/x.jpg", "license": "public-domain"},
                {"url": "https://d.example.com/x.jpg"},  # no license
            ],
        )
        urls, _skipped, sidecars = _load_urls_from_jsonl(
            f, min_size=0, license_filter="cc-by"
        )
        assert urls == [
            "https://a.example.com/x.jpg",
            "https://b.example.com/x.jpg",
        ]
        assert sidecars["https://a.example.com/x.jpg"]["license"] == "cc-by-4.0"

    def test_skips_blank_lines_and_malformed_json(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        f.write_text(
            "\n"
            '{"url": "https://a.example.com/x.jpg"}\n'
            "not-json-at-all\n"
            "   \n"
            '{"url": "https://b.example.com/y.jpg"}\n'
        )
        urls, skipped, _sidecars = _load_urls_from_jsonl(
            f, min_size=0, license_filter=None
        )
        assert urls == [
            "https://a.example.com/x.jpg",
            "https://b.example.com/y.jpg",
        ]
        assert skipped == 0

    def test_records_without_url_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        _write_jsonl(
            f,
            [
                {"engine": "flickr"},
                {"url": "", "engine": "flickr"},
                {"url": "https://a.example.com/x.jpg"},
            ],
        )
        urls, _skipped, _sidecars = _load_urls_from_jsonl(
            f, min_size=0, license_filter=None
        )
        assert urls == ["https://a.example.com/x.jpg"]

    def test_sidecar_defaults_when_engine_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "results.jsonl"
        _write_jsonl(f, [{"url": "https://a.example.com/x.jpg"}])
        _urls, _skipped, sidecars = _load_urls_from_jsonl(
            f, min_size=0, license_filter=None
        )
        sc = sidecars["https://a.example.com/x.jpg"]
        assert sc["source"] == "unknown"
        assert sc["origin"] == "https://a.example.com/x.jpg"
        assert sc["license"] == "none"


# ---------------------------------------------------------------------------
# _load_urls_from_txt
# ---------------------------------------------------------------------------


class TestLoadUrlsFromTxt:
    def test_one_url_per_line(self, tmp_path: Path) -> None:
        f = tmp_path / "urls.txt"
        f.write_text("https://a.example.com/x.jpg\nhttps://b.example.com/y.png\n")
        urls, sidecars = _load_urls_from_txt(f)
        assert urls == [
            "https://a.example.com/x.jpg",
            "https://b.example.com/y.png",
        ]
        assert set(sidecars.keys()) == set(urls)

    def test_blank_lines_and_comments_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "urls.txt"
        f.write_text(
            "\n"
            "# a comment\n"
            "https://a.example.com/x.jpg\n"
            "   \n"
            "# another comment\n"
            "https://b.example.com/y.jpg\n"
        )
        urls, _sidecars = _load_urls_from_txt(f)
        assert urls == [
            "https://a.example.com/x.jpg",
            "https://b.example.com/y.jpg",
        ]

    def test_results_sorted_and_deduplicated(self, tmp_path: Path) -> None:
        f = tmp_path / "urls.txt"
        f.write_text(
            "https://c.example.com/x.jpg\n"
            "https://a.example.com/x.jpg\n"
            "https://b.example.com/x.jpg\n"
            "https://a.example.com/x.jpg\n"
        )
        urls, _sidecars = _load_urls_from_txt(f)
        assert urls == [
            "https://a.example.com/x.jpg",
            "https://b.example.com/x.jpg",
            "https://c.example.com/x.jpg",
        ]

    def test_sidecar_source_is_root_domain(self, tmp_path: Path) -> None:
        f = tmp_path / "urls.txt"
        f.write_text(
            "https://www.flickr.com/photos/abc.jpg\n"
            "https://upload.wikimedia.org/wiki/foo.jpg\n"
        )
        _urls, sidecars = _load_urls_from_txt(f)
        assert sidecars["https://www.flickr.com/photos/abc.jpg"]["source"] == "flickr"
        assert (
            sidecars["https://upload.wikimedia.org/wiki/foo.jpg"]["source"] == "upload"
        )

    def test_sidecar_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / "urls.txt"
        url = "https://a.example.com/x.jpg"
        f.write_text(url + "\n")
        _urls, sidecars = _load_urls_from_txt(f)
        sc = sidecars[url]
        assert sc["origin"] == url
        assert sc["license"] == "none"

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "urls.txt"
        f.write_text("")
        urls, sidecars = _load_urls_from_txt(f)
        assert urls == []
        assert sidecars == {}
