"""Unit tests for the per-image worker in format."""

from pathlib import Path

import pytest
from PIL import Image

from dtst.core.format import _format_image, _has_alpha

# Palette where index 0 is black and index 1 is blue, so an image whose
# transparent pixels are left uncomposited shows up as one of those colours.
PALETTE = [0, 0, 0, 0, 0, 255] + [0, 0, 0] * 254
FAKE_ICC = b"\x00\x00\x02\x0cfake-icc-profile" + b"\x00" * 500


def _run(src: Path, out_dir: Path, *, fmt=None, channels=None, background="white"):
    out_dir.mkdir(parents=True, exist_ok=True)
    status, name, error = _format_image(
        (str(src), str(out_dir), fmt, 95, 0, False, channels, background)
    )
    assert status == "ok", error
    return Image.open(out_dir / name)


@pytest.fixture
def palette_png(tmp_path: Path):
    """A P-mode PNG whose only colour is flagged transparent via tRNS."""

    def _make(name: str = "p.png", index: int = 0) -> Path:
        path = tmp_path / name
        img = Image.new("P", (8, 8), index)
        img.putpalette(PALETTE)
        img.info["transparency"] = index
        img.save(path)
        return path

    return _make


def test_palette_transparency_composites_onto_background(palette_png, tmp_path):
    # Regression: P-mode images were not detected as having alpha, so the
    # transparent pixels were baked in as their palette colour (black).
    out = _run(palette_png(), tmp_path / "out", fmt="jpg", channels="rgb")
    assert out.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_palette_transparency_honours_background_option(palette_png, tmp_path):
    src = palette_png(index=1)  # blue palette entry, so black must be the background
    out = _run(src, tmp_path / "out", fmt="jpg", channels="rgb", background="black")
    assert out.convert("RGB").getpixel((0, 0)) == (0, 0, 0)


def test_palette_transparency_composites_for_jpeg_without_channels(
    palette_png, tmp_path
):
    # No --channels: the jpeg fallback path must still composite.
    out = _run(palette_png(), tmp_path / "out", fmt="jpg")
    assert out.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_palette_transparency_composites_for_grayscale(palette_png, tmp_path):
    out = _run(palette_png(), tmp_path / "out", fmt="png", channels="grayscale")
    assert out.mode == "L"
    assert out.getpixel((0, 0)) == 255


def test_colorkey_transparency_composites_onto_background(tmp_path):
    src = tmp_path / "rgb_trns.png"
    Image.new("RGB", (8, 8), (0, 0, 255)).save(src, transparency=(0, 0, 255))
    out = _run(src, tmp_path / "out", fmt="jpg", channels="rgb")
    assert out.convert("RGB").getpixel((0, 0)) == (255, 255, 255)


def test_opaque_palette_image_keeps_its_colour(tmp_path):
    src = tmp_path / "p_opaque.png"
    img = Image.new("P", (8, 8), 1)
    img.putpalette(PALETTE)
    img.save(src)
    out = _run(src, tmp_path / "out", fmt="png", channels="rgb")
    assert out.convert("RGB").getpixel((0, 0)) == (0, 0, 255)


def test_rgba_alpha_is_preserved_when_no_channel_conversion(tmp_path):
    src = tmp_path / "rgba.png"
    Image.new("RGBA", (8, 8), (0, 255, 0, 0)).save(src)
    out = _run(src, tmp_path / "out", fmt="png")
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[3] == 0


def test_icc_profile_survives_compositing(palette_png, tmp_path):
    src = tmp_path / "p_icc.png"
    img = Image.new("P", (8, 8), 0)
    img.putpalette(PALETTE)
    img.info["transparency"] = 0
    img.save(src, icc_profile=FAKE_ICC)

    out = _run(src, tmp_path / "out", fmt="jpg", channels="rgb")
    assert out.info.get("icc_profile") == FAKE_ICC


def test_strip_metadata_drops_icc_profile(tmp_path):
    src = tmp_path / "rgb_icc.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(src, icc_profile=FAKE_ICC)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    status, name, error = _format_image(
        (str(src), str(out_dir), "jpg", 95, 0, True, None, "white")
    )
    assert status == "ok", error
    assert Image.open(out_dir / name).info.get("icc_profile") is None


@pytest.mark.parametrize("mode", ["RGBA", "LA", "PA"])
def test_has_alpha_detects_alpha_bands(mode):
    assert _has_alpha(Image.new(mode, (4, 4)))


@pytest.mark.parametrize("mode", ["RGB", "L", "P", "CMYK"])
def test_has_alpha_is_false_without_transparency(mode):
    assert not _has_alpha(Image.new(mode, (4, 4)))


@pytest.mark.parametrize("mode", ["P", "RGB", "L"])
def test_has_alpha_detects_trns_transparency(mode):
    img = Image.new(mode, (4, 4))
    img.info["transparency"] = 0
    assert _has_alpha(img)
