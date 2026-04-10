from __future__ import annotations

import bz2
import logging
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import scipy.ndimage
from PIL import ImageDraw

logger = logging.getLogger(__name__)

VALID_ENGINES = frozenset({"mediapipe", "dlib"})

MP_EYE_LEFT_IDX = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
]
MP_EYE_RIGHT_IDX = [
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
MP_MOUTH_IDX = [61, 291]

DLIB_EYE_LEFT_IDX = list(range(36, 42))
DLIB_EYE_RIGHT_IDX = list(range(42, 48))
DLIB_MOUTH_IDX = [48, 54]


def _get_data_dir() -> Path:
    if sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Application Support" / "dtst"
    elif sys.platform == "win32":
        base_dir = Path.home() / "AppData" / "Roaming" / "dtst"
    else:
        base_dir = Path.home() / ".local" / "share" / "dtst"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_FACE_LANDMARKER_FILENAME = "face_landmarker.task"

_DLIB_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
_DLIB_PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"


def _ensure_model(filename: str, url: str) -> Path:
    """Return the local path for *filename*, downloading from *url* if absent."""
    path = _get_data_dir() / filename
    if path.exists():
        return path
    logger.info("Downloading %s from %s …", filename, url)
    if url.endswith(".bz2"):
        bz2_path = path.with_suffix(path.suffix + ".bz2")
        urllib.request.urlretrieve(url, str(bz2_path))
        with bz2.BZ2File(str(bz2_path), "rb") as src, open(path, "wb") as dst:
            dst.write(src.read())
        bz2_path.unlink()
    else:
        urllib.request.urlretrieve(url, str(path))
    logger.info("Saved to %s", path)
    return path


def align_face(
    img: PIL.Image.Image,
    face_landmarks: list[tuple[int, int]],
    max_size: int | None = None,
    enable_padding: bool = True,
    skip_partial: bool = False,
    debug: bool = False,
) -> PIL.Image.Image | None:
    """Align and crop a single face from *img* using the given landmarks.

    *max_size* caps the output side length. Faces smaller than *max_size*
    are kept at their natural size (no upscale). When *max_size* is
    ``None`` the output is always at natural size.

    When *skip_partial* is ``True``, faces whose aligned crop extends
    beyond the image boundary are skipped (returns ``None``) instead of
    being padded or truncated.

    Returns a square RGB PIL Image, or ``None`` when *skip_partial*
    rejects the face.
    """
    img = img.copy()
    lm = np.array(face_landmarks)

    if len(lm) in (468, 478):
        eye_left_idx = MP_EYE_LEFT_IDX
        eye_right_idx = MP_EYE_RIGHT_IDX
        mouth_idx = MP_MOUTH_IDX
    elif len(lm) == 68:
        eye_left_idx = DLIB_EYE_LEFT_IDX
        eye_right_idx = DLIB_EYE_RIGHT_IDX
        mouth_idx = DLIB_MOUTH_IDX
    else:
        raise ValueError(f"Unsupported number of landmarks: {len(lm)}")

    eye_left_mean = np.mean(lm[eye_left_idx], axis=0)
    eye_right_mean = np.mean(lm[eye_right_idx], axis=0)
    mouth_mean = np.mean(lm[mouth_idx], axis=0)

    eye_avg = (eye_left_mean + eye_right_mean) * 0.5
    eye_to_eye = eye_right_mean - eye_left_mean
    eye_to_mouth = mouth_mean - eye_avg

    if debug:
        draw = ImageDraw.Draw(img)
        key_points = eye_left_idx + eye_right_idx + mouth_idx
        radius = int(np.linalg.norm(eye_to_eye) * 0.02)
        for idx in key_points:
            px, py = lm[idx]
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius), fill=(0, 255, 0)
            )

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c0 = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    qsize = np.hypot(*x) * 2

    natural = int(qsize)
    if max_size is not None:
        output_size = min(natural, max_size)
        transform_size = 4096
    else:
        output_size = natural
        transform_size = max(natural, 512)

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if skip_partial and max(pad) > border - 4:
        return None
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img_arr = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        h, w, _ = img_arr.shape
        yy, xx, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(xx) / pad[0], np.float32(w - 1 - xx) / pad[2]),
            1.0 - np.minimum(np.float32(yy) / pad[1], np.float32(h - 1 - yy) / pad[3]),
        )
        blur_sigma = qsize * 0.02
        img_arr += (
            scipy.ndimage.gaussian_filter(img_arr, [blur_sigma, blur_sigma, 0])
            - img_arr
        ) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img_arr += (np.median(img_arr, axis=(0, 1)) - img_arr) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img_arr), 0, 255)), "RGB")
        quad += pad[:2]

    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.Transform.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.Resampling.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)

    return img


class FaceAligner:
    """Detect faces and produce aligned crops using MediaPipe or dlib."""

    def __init__(
        self,
        engine: str = "mediapipe",
        max_faces: int = 1,
        refine_landmarks: bool = False,
    ) -> None:
        self.engine = engine.lower()
        if self.engine not in VALID_ENGINES:
            raise ValueError(
                f"Unsupported engine: {self.engine!r}; valid: {sorted(VALID_ENGINES)}"
            )

        if self.engine == "mediapipe":
            import mediapipe as mp

            model_path = _ensure_model(_FACE_LANDMARKER_FILENAME, _FACE_LANDMARKER_URL)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                num_faces=max_faces,
                min_face_detection_confidence=0.5,
            )
            self._mp = mp
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        else:
            try:
                import dlib
            except ImportError:
                raise ImportError(
                    "dlib is not installed. Install it with: pip install dlib "
                    "(requires cmake and a C++ compiler)"
                ) from None
            model_path = _ensure_model(_DLIB_PREDICTOR_FILENAME, _DLIB_PREDICTOR_URL)
            self._dlib_detector = dlib.get_frontal_face_detector()
            self._dlib_predictor = dlib.shape_predictor(str(model_path))

    def detect_landmarks(self, image: np.ndarray) -> list[list[tuple[int, int]]]:
        """Return per-face landmark lists from a BGR numpy image."""
        if self.engine == "mediapipe":
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)
            if not result.face_landmarks:
                return []
            h, w = image.shape[:2]
            return [
                [(int(lm.x * w), int(lm.y * h)) for lm in face]
                for face in result.face_landmarks
            ]
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dets = self._dlib_detector(gray, 1)
            if not dets:
                return []
            return [
                [(p.x, p.y) for p in self._dlib_predictor(gray, det).parts()]
                for det in dets
            ]

    def get_aligned_faces(
        self,
        image: np.ndarray,
        *,
        max_size: int | None = None,
        max_faces: int | None = None,
        enable_padding: bool = True,
        skip_partial: bool = False,
        debug: bool = False,
    ) -> list[PIL.Image.Image]:
        """Detect faces in a BGR image and return aligned RGB PIL images."""
        landmarks_list = self.detect_landmarks(image)
        if not landmarks_list:
            return []
        if max_faces is not None:
            landmarks_list = landmarks_list[:max_faces]

        pil_img = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = [
            align_face(
                pil_img,
                landmarks,
                max_size=max_size,
                enable_padding=enable_padding,
                skip_partial=skip_partial,
                debug=debug,
            )
            for landmarks in landmarks_list
        ]
        return [r for r in results if r is not None]
