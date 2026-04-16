import numpy as np
import PIL.Image
import pytest

from dtst.face_align import align_face


def _make_image(
    size: int = 256, color: tuple[int, int, int] = (128, 128, 128)
) -> PIL.Image.Image:
    return PIL.Image.new("RGB", (size, size), color=color)


def _dlib_landmarks(
    eye_left: tuple[float, float],
    eye_right: tuple[float, float],
    mouth_left: tuple[float, float],
    mouth_right: tuple[float, float],
) -> list[tuple[int, int]]:
    """Build a 68-point landmark list with the dlib indices populated.

    Only indices 36-41 (left eye), 42-47 (right eye), 48 and 54 (mouth
    corners) are read by ``align_face`` for 68-point input. The rest can
    be anything; we fill with zeros.
    """
    pts = [(0, 0)] * 68
    for i in range(36, 42):
        pts[i] = (int(eye_left[0]), int(eye_left[1]))
    for i in range(42, 48):
        pts[i] = (int(eye_right[0]), int(eye_right[1]))
    pts[48] = (int(mouth_left[0]), int(mouth_left[1]))
    pts[54] = (int(mouth_right[0]), int(mouth_right[1]))
    return pts


def _mp_landmarks(
    eye_left: tuple[float, float],
    eye_right: tuple[float, float],
    mouth_left: tuple[float, float],
    mouth_right: tuple[float, float],
    count: int = 468,
) -> list[tuple[int, int]]:
    """Build a MediaPipe-style landmark list (468 or 478 points)."""
    from dtst.face_align import MP_EYE_LEFT_IDX, MP_EYE_RIGHT_IDX, MP_MOUTH_IDX

    pts = [(0, 0)] * count
    for i in MP_EYE_LEFT_IDX:
        pts[i] = (int(eye_left[0]), int(eye_left[1]))
    for i in MP_EYE_RIGHT_IDX:
        pts[i] = (int(eye_right[0]), int(eye_right[1]))
    pts[MP_MOUTH_IDX[0]] = (int(mouth_left[0]), int(mouth_left[1]))
    pts[MP_MOUTH_IDX[1]] = (int(mouth_right[0]), int(mouth_right[1]))
    return pts


def _symmetric_dlib(cx: float, cy: float, eye_dx: float, mouth_dy: float):
    return _dlib_landmarks(
        eye_left=(cx - eye_dx, cy),
        eye_right=(cx + eye_dx, cy),
        mouth_left=(cx - eye_dx * 0.6, cy + mouth_dy),
        mouth_right=(cx + eye_dx * 0.6, cy + mouth_dy),
    )


class TestAlignFaceDlib:
    def test_returns_pil_rgb_image(self):
        img = _make_image(256)
        lm = _symmetric_dlib(cx=128, cy=120, eye_dx=20, mouth_dy=40)
        out = align_face(img, lm, max_size=128)
        assert isinstance(out, PIL.Image.Image)
        assert out.mode == "RGB"

    def test_output_is_square(self):
        img = _make_image(256)
        lm = _symmetric_dlib(cx=128, cy=120, eye_dx=20, mouth_dy=40)
        out = align_face(img, lm, max_size=128)
        assert out.size[0] == out.size[1]

    def test_max_size_caps_output_side(self):
        img = _make_image(512)
        lm = _symmetric_dlib(cx=256, cy=240, eye_dx=40, mouth_dy=80)
        out = align_face(img, lm, max_size=64)
        assert max(out.size) <= 64

    @pytest.mark.parametrize("max_size", [32, 64, 128])
    def test_varying_max_size_changes_output(self, max_size):
        img = _make_image(512)
        lm = _symmetric_dlib(cx=256, cy=240, eye_dx=40, mouth_dy=80)
        out = align_face(img, lm, max_size=max_size)
        assert out.size[0] <= max_size
        assert out.size[1] <= max_size

    def test_max_size_none_returns_natural_size(self):
        img = _make_image(512)
        lm = _symmetric_dlib(cx=256, cy=240, eye_dx=40, mouth_dy=80)
        out = align_face(img, lm, max_size=None)
        assert isinstance(out, PIL.Image.Image)
        assert out.size[0] == out.size[1]
        assert out.size[0] > 0

    def test_landmarks_near_edge_does_not_raise(self):
        img = _make_image(256)
        lm = _symmetric_dlib(cx=20, cy=20, eye_dx=8, mouth_dy=12)
        out = align_face(img, lm, max_size=64)
        assert isinstance(out, PIL.Image.Image)

    def test_landmarks_outside_image_does_not_raise(self):
        img = _make_image(256)
        lm = _symmetric_dlib(cx=250, cy=250, eye_dx=20, mouth_dy=30)
        out = align_face(img, lm, max_size=64)
        assert isinstance(out, PIL.Image.Image)

    def test_skip_partial_returns_none_when_face_exceeds_bounds(self):
        img = _make_image(256)
        # Put face well outside so the aligned crop must go far past edges
        lm = _symmetric_dlib(cx=5, cy=5, eye_dx=30, mouth_dy=50)
        out = align_face(img, lm, max_size=64, skip_partial=True)
        assert out is None

    def test_skip_partial_false_keeps_padded_face(self):
        img = _make_image(256)
        lm = _symmetric_dlib(cx=5, cy=5, eye_dx=30, mouth_dy=50)
        out = align_face(img, lm, max_size=64, skip_partial=False)
        assert isinstance(out, PIL.Image.Image)

    def test_enable_padding_false_does_not_raise(self):
        img = _make_image(256)
        lm = _symmetric_dlib(cx=128, cy=120, eye_dx=20, mouth_dy=40)
        out = align_face(img, lm, max_size=128, enable_padding=False)
        assert isinstance(out, PIL.Image.Image)

    def test_debug_flag_returns_image(self):
        img = _make_image(256)
        lm = _symmetric_dlib(cx=128, cy=120, eye_dx=20, mouth_dy=40)
        out = align_face(img, lm, max_size=128, debug=True)
        assert isinstance(out, PIL.Image.Image)

    def test_accepts_numpy_array_landmarks(self):
        img = _make_image(256)
        lm = np.array(_symmetric_dlib(cx=128, cy=120, eye_dx=20, mouth_dy=40))
        out = align_face(img, lm, max_size=64)
        assert isinstance(out, PIL.Image.Image)


class TestAlignFaceMediaPipe:
    def test_468_landmarks_supported(self):
        img = _make_image(256)
        lm = _mp_landmarks(
            eye_left=(108, 120),
            eye_right=(148, 120),
            mouth_left=(118, 160),
            mouth_right=(138, 160),
            count=468,
        )
        out = align_face(img, lm, max_size=64)
        assert isinstance(out, PIL.Image.Image)
        assert out.size == (64, 64)

    def test_478_landmarks_supported(self):
        img = _make_image(256)
        lm = _mp_landmarks(
            eye_left=(108, 120),
            eye_right=(148, 120),
            mouth_left=(118, 160),
            mouth_right=(138, 160),
            count=478,
        )
        out = align_face(img, lm, max_size=64)
        assert isinstance(out, PIL.Image.Image)


class TestAlignFaceErrors:
    @pytest.mark.parametrize("bad_count", [0, 5, 67, 69, 100, 467, 469, 477, 479, 500])
    def test_unsupported_landmark_count_raises(self, bad_count):
        img = _make_image(256)
        lm = [(0, 0)] * bad_count
        with pytest.raises(ValueError, match="Unsupported number of landmarks"):
            align_face(img, lm, max_size=64)

    def test_does_not_mutate_input_image(self):
        img = _make_image(256, color=(50, 100, 150))
        before = np.array(img).copy()
        lm = _symmetric_dlib(cx=128, cy=120, eye_dx=20, mouth_dy=40)
        align_face(img, lm, max_size=64, debug=True)
        after = np.array(img)
        assert np.array_equal(before, after)
