"""Unit tests for the pure helpers and per-image worker in frame."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dtst.core.frame import _content_bbox, _resize_image, _resolve_margin
from dtst.core.frame import frame as core_frame
from dtst.errors import InputError


def _white(size: tuple[int, int] = (20, 20)) -> Image.Image:
    return Image.new("RGB", size, (255, 255, 255))


def test_content_bbox_finds_offset_content():
    img = _white()
    img.paste((255, 0, 0), (4, 6, 10, 14))
    assert _content_bbox(img, 8) == (4, 6, 10, 14)


def test_content_bbox_uses_alpha_when_present():
    img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (4, 6, 10, 14))
    assert _content_bbox(img, 8) == (4, 6, 10, 14)


def test_content_bbox_detects_background_from_corners_not_hardcoded_white():
    img = Image.new("RGB", (20, 20), (0, 0, 0))
    img.paste((255, 255, 255), (4, 6, 10, 14))
    assert _content_bbox(img, 8) == (4, 6, 10, 14)


def test_content_bbox_detects_blue_only_difference():
    # A luminance-weighted difference gives blue a 0.114 weight, so a 55-level
    # blue-only delta would read as 6.3 and fall under the tolerance. The
    # per-channel max must see the full 55.
    img = _white()
    img.paste((255, 255, 200), (4, 6, 10, 14))
    assert _content_bbox(img, 8) == (4, 6, 10, 14)


def test_content_bbox_spans_full_image_when_content_touches_every_edge():
    img = _white((16, 16))
    img.paste((0, 0, 0), (0, 7, 16, 9))  # full-width bar
    img.paste((0, 0, 0), (7, 0, 9, 16))  # full-height bar
    assert _content_bbox(img, 8) == (0, 0, 16, 16)


def test_content_bbox_uses_the_median_of_the_corners_not_a_single_one():
    # The corners are not all alike: one is 240, the other three 255, so the
    # median is 255 and the 225 blob sits 30 levels out, beyond the tolerance
    # of 20. Keying on the first corner alone would put the blob 15 levels
    # out — inside the tolerance — and find nothing at all.
    img = _white()
    img.putpixel((0, 0), (240, 240, 240))
    img.paste((225, 225, 225), (4, 6, 10, 14))
    assert _content_bbox(img, 20) == (4, 6, 10, 14)


def test_content_bbox_uses_trns_transparency_without_an_alpha_channel():
    # A P-mode image flagging index 0 transparent via tRNS. Its mode is not
    # RGBA/LA/PA, so only the ``img.info`` check routes it to the alpha
    # branch. Both palette entries are the same red, so the corner-median
    # branch would see a uniform image and return None.
    img = Image.new("P", (20, 20), 0)
    img.putpalette([255, 0, 0, 255, 0, 0] + [0, 0, 0] * 254)
    img.info["transparency"] = 0
    img.paste(1, (4, 6, 10, 14))
    assert _content_bbox(img, 8) == (4, 6, 10, 14)


def test_content_bbox_inverts_when_content_reaches_all_four_corners():
    # Documented limitation: corner-median detection cannot distinguish
    # full-bleed content from a background. Here the corners ARE the subject,
    # so the median is the subject colour and the mask selects the interior
    # blob instead. This asserts the actual behaviour — the bbox is inverted,
    # it is not None — so that --trim's assumption stays visible.
    img = Image.new("RGB", (20, 20), (255, 0, 0))
    img.paste((255, 255, 255), (6, 6, 14, 14))
    assert _content_bbox(img, 8) == (6, 6, 14, 14)


def test_content_bbox_returns_none_for_uniform_image():
    assert _content_bbox(_white(), 8) is None


def test_content_bbox_ignores_noise_within_tolerance():
    img = _white()
    img.paste((250, 250, 250), (4, 6, 10, 14))  # delta of 5, under tolerance 8
    assert _content_bbox(img, 8) is None


@pytest.mark.parametrize(
    "value, width, height, expected",
    [
        (None, 1024, 1024, 0),
        ("48", 1024, 1024, 48),
        (48, 1024, 1024, 48),  # workflow steps bypass Click's str conversion
        ("0", 1024, 1024, 0),
        ("5%", 1024, 1024, 51),  # 51.2 rounds to 51
        ("5%", 1024, 512, 26),  # 5% of the shorter side, 25.6 rounds to 26
        ("5%", 8, 8, 0),  # 0.4 rounds to 0
        ("10%", 100, 100, 10),
        (" 48 ", 1024, 1024, 48),
    ],
)
def test_resolve_margin_valid(value, width, height, expected):
    assert _resolve_margin(value, width, height) == expected


@pytest.mark.parametrize("value", ["abc", "", "5 %", "%", "1.5", "-5", "-5%"])
def test_resolve_margin_rejects_malformed(value):
    with pytest.raises(InputError):
        _resolve_margin(value, 1024, 1024)


@pytest.mark.parametrize("value", ["60%", "50%", "512", "600"])
def test_resolve_margin_rejects_margins_that_swallow_the_canvas(value):
    with pytest.raises(InputError):
        _resolve_margin(value, 1024, 1024)


def _run(
    src: Path,
    out_dir: Path,
    *,
    width=None,
    height=None,
    mode="pad",
    gravity="center",
    fill="color",
    fill_color="#ffffff",
    trim=False,
    trim_tolerance=8,
    margin=0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    status, name, error = _resize_image(
        (
            str(src),
            str(out_dir),
            width,
            height,
            mode,
            gravity,
            fill,
            fill_color,
            95,
            0,
            trim,
            trim_tolerance,
            margin,
        )
    )
    return status, name, error


def _subject(path: Path, canvas, box, color=(255, 0, 0)):
    """Write a PNG with a solid rectangle on a white canvas."""
    img = Image.new("RGB", canvas, (255, 255, 255))
    img.paste(color, box)
    img.save(path)
    return path


def test_trim_and_margin_normalize_content_across_source_scales(tmp_path):
    # The same square subject, at 40px in one source and 120px in the other,
    # framed identically. This is the requirement the feature exists for.
    _subject(tmp_path / "small.png", (200, 200), (10, 20, 50, 60))
    _subject(tmp_path / "large.png", (200, 200), (60, 30, 180, 150))

    out = tmp_path / "out"
    for name in ("small.png", "large.png"):
        status, _, error = _run(
            tmp_path / name, out, width=100, height=100, trim=True, margin=10
        )
        assert status == "ok", error

    for name in ("small.png", "large.png"):
        box = _content_bbox(Image.open(out / name), 8)
        # Lanczos softens the edge by up to a pixel, so allow 1px of slack.
        # The point is that both sources land on the same 80x80 inset box.
        for coord, expected in zip(box, (10, 10, 90, 90)):
            assert abs(coord - expected) <= 1, f"{name}: {box}"


def test_trim_reports_failure_for_a_blank_image(tmp_path):
    src = tmp_path / "blank.png"
    Image.new("RGB", (32, 32), (255, 255, 255)).save(src)

    status, name, error = _run(src, tmp_path / "out", width=16, height=16, trim=True)
    assert status == "failed"
    assert name == "blank.png"
    assert "content" in error


def test_margin_is_honoured_under_non_centre_gravity(tmp_path):
    # 200x100 subject, no trim, into a 100x100 canvas with a 10px margin.
    # Inner box is 80x80, scale is 0.4, so the subject becomes 80x40 and
    # left-gravity puts it at x=10 (the margin), vertically centred at y=30.
    src = tmp_path / "wide.png"
    Image.new("RGB", (200, 100), (255, 0, 0)).save(src)

    status, name, error = _run(
        src, tmp_path / "out", width=100, height=100, gravity="left", margin=10
    )
    assert status == "ok", error

    out = Image.open(tmp_path / "out" / name).convert("RGB")
    assert out.size == (100, 100)
    assert out.getpixel((15, 50)) == (255, 0, 0)
    assert out.getpixel((4, 50)) == (255, 255, 255)  # inside the margin
    assert out.getpixel((50, 5)) == (255, 255, 255)  # above the subject
    # Discriminates y=30 from y=40: filling the full canvas and only then
    # adding the margin would push the subject's top edge down to y=40.
    assert out.getpixel((50, 35)) == (255, 0, 0)


def test_rgba_source_keeps_the_fill_colour_inside_the_content_box(tmp_path):
    # An L-shaped opaque subject leaves a transparent square inside its own
    # bounding box. Pasting without an alpha mask turns that square black
    # while the margin band stays white.
    src = tmp_path / "l.png"
    img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (10, 10, 30, 20))
    img.paste((255, 0, 0, 255), (10, 10, 20, 30))
    img.save(src)

    status, name, error = _run(
        src, tmp_path / "out", width=100, height=100, trim=True, margin=10
    )
    assert status == "ok", error

    out = Image.open(tmp_path / "out" / name).convert("RGB")
    assert out.getpixel((70, 70)) == (255, 255, 255)  # transparent, inside bbox
    assert out.getpixel((2, 2)) == (255, 255, 255)  # margin band
    assert out.getpixel((30, 30)) == (255, 0, 0)  # opaque subject


def test_no_trim_and_no_margin_leaves_existing_behaviour_untouched(tmp_path):
    src = tmp_path / "square.png"
    Image.new("RGB", (200, 200), (255, 0, 0)).save(src)

    status, name, error = _run(src, tmp_path / "out", width=100, height=100)
    assert status == "ok", error

    out = Image.open(tmp_path / "out" / name).convert("RGB")
    assert out.size == (100, 100)
    assert out.getpixel((0, 0)) == (255, 0, 0)
    assert out.getpixel((99, 99)) == (255, 0, 0)


def test_trim_applies_on_the_proportional_single_dimension_path(tmp_path):
    # Only --width given: the trimmed crop is what gets scaled, so a 40x20
    # subject inside a 200x200 canvas comes out 80x40, not 80x80.
    _subject(tmp_path / "wide.png", (200, 200), (10, 10, 50, 30))

    status, name, error = _run(
        tmp_path / "wide.png", tmp_path / "out", width=80, trim=True
    )
    assert status == "ok", error
    assert Image.open(tmp_path / "out" / name).size == (80, 40)


def test_trimmed_image_matching_the_target_is_still_inset_by_the_margin(tmp_path):
    # Guards the equal-dimensions fast path: a crop that already measures
    # exactly 100x100 must not skip the resize when a margin is set.
    _subject(tmp_path / "exact.png", (200, 200), (20, 20, 120, 120))

    status, name, error = _run(
        tmp_path / "exact.png",
        tmp_path / "out",
        width=100,
        height=100,
        trim=True,
        margin=10,
    )
    assert status == "ok", error

    out = Image.open(tmp_path / "out" / name).convert("RGB")
    assert out.getpixel((2, 2)) == (255, 255, 255)  # margin, not subject


def test_frame_rejects_margin_without_both_dimensions():
    with pytest.raises(InputError, match="both width and height"):
        core_frame(from_dirs="src", to="dst", width=100, margin="5%")


def test_frame_rejects_margin_outside_pad_mode():
    with pytest.raises(InputError, match="pad"):
        core_frame(
            from_dirs="src",
            to="dst",
            width=100,
            height=100,
            mode="crop",
            margin="5%",
        )


def test_frame_rejects_a_malformed_margin():
    with pytest.raises(InputError, match="Invalid margin"):
        core_frame(
            from_dirs="src",
            to="dst",
            width=100,
            height=100,
            mode="pad",
            margin="wide",
        )
