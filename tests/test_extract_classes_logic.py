"""Unit tests for pure geometry helpers in extract_classes."""

from dtst.core.extract_classes import _clamp_crop, _expand_box


def _crop_dims(box, square, img_w, img_h):
    x1, y1, x2, y2 = _clamp_crop(box, square, img_w, img_h)
    return x2 - x1, y2 - y1


def test_square_crop_is_exactly_square_with_fractional_box():
    # Box whose expanded edges land on fractions that would round/truncate
    # the width and height to different integers if handled independently.
    box = _expand_box((10.2, 10.7, 269.6, 270.9), margin=0.0, square=True)
    w, h = _crop_dims(box, square=True, img_w=1000, img_h=1000)
    assert w == h


def test_square_crop_stays_square_when_box_exceeds_image_bounds():
    # Tightly-framed subject: the square box + margin spills past every edge,
    # so naive clamping would yield a non-square crop.
    box = _expand_box((5.0, 5.0, 295.0, 295.0), margin=0.1, square=True)
    w, h = _crop_dims(box, square=True, img_w=300, img_h=300)
    assert w == h


def test_square_crop_within_image_bounds():
    box = _expand_box((5.0, 5.0, 295.0, 295.0), margin=0.1, square=True)
    x1, y1, x2, y2 = _clamp_crop(box, square=True, img_w=300, img_h=300)
    assert 0 <= x1 and 0 <= y1
    assert x2 <= 300 and y2 <= 300


def test_expand_box_squares_a_rectangle():
    # A wide box should become square (side == longer edge) before margin.
    x_min, y_min, x_max, y_max = _expand_box(
        (0.0, 0.0, 100.0, 40.0), margin=0.0, square=True
    )
    assert round(x_max - x_min) == round(y_max - y_min) == 100


def test_non_square_crop_clamps_to_image_bounds():
    box = _expand_box((-10.0, -10.0, 310.0, 310.0), margin=0.0, square=False)
    x1, y1, x2, y2 = _clamp_crop(box, square=False, img_w=300, img_h=300)
    assert (x1, y1, x2, y2) == (0, 0, 300, 300)
