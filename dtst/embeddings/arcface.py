from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm

from dtst.embeddings.base import EmbeddingBackend

logger = logging.getLogger(__name__)

# ArcFace recognition model input size
INPUT_SIZE = (112, 112)


def _load_and_preprocess(path: Path) -> tuple[Path, np.ndarray | None]:
    """Load an image and resize to 112x112 for ArcFace recognition.

    Returns (path, bgr_112x112) or (path, None) on failure.
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            return path, None
        resized = cv2.resize(img, INPUT_SIZE)
        return path, resized
    except Exception:
        return path, None


class ArcFaceBackend(EmbeddingBackend):
    """Face identity embeddings via insightface ArcFace.

    Uses the recognition model (w600k_r50) directly, bypassing face
    detection.  This assumes input images are already cropped faces
    (e.g. output of ``extract-faces``).  Each image is resized to
    112x112 and fed straight into the ArcFace embedding network.
    """

    def __init__(self) -> None:
        self._rec_model = None

    def load(self, device: str) -> None:
        import insightface

        providers: list[str]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps":
            # CoreML provider is unreliable with insightface; fall back to CPU
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Ensure model pack is downloaded
        model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
        rec_path = model_dir / "w600k_r50.onnx"

        if not rec_path.exists():
            # Trigger download via FaceAnalysis, then discard it
            from insightface.app import FaceAnalysis

            logger.info("Downloading buffalo_l model pack...")
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=-1)
            del app

        self._rec_model = insightface.model_zoo.get_model(
            str(rec_path), providers=providers
        )
        self._rec_model.prepare(ctx_id=0 if device == "cuda" else -1)
        logger.info("ArcFace recognition model loaded (device=%s)", device)

    def embed(
        self,
        image_paths: list[Path],
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> tuple[np.ndarray, list[Path]]:
        if self._rec_model is None:
            raise RuntimeError("Model not loaded -- call .load(device) first")

        embeddings: list[np.ndarray] = []
        valid_paths: list[Path] = []
        skipped = 0

        with ThreadPoolExecutor(max_workers=num_workers) as loader:
            with tqdm(
                total=len(image_paths), desc="Embedding (arcface)", unit="image"
            ) as pbar:
                for batch_start in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[batch_start : batch_start + batch_size]
                    loaded = list(loader.map(_load_and_preprocess, batch_paths))

                    # Separate successes from failures
                    batch_imgs = []
                    batch_valid = []
                    for path, img in loaded:
                        if img is None:
                            logger.warning("Could not read image: %s", path.name)
                            skipped += 1
                            pbar.update(1)
                            continue
                        batch_imgs.append(img)
                        batch_valid.append(path)

                    if batch_imgs:
                        # Use get_feat for batch embedding via blobFromImages
                        feats = self._rec_model.get_feat(batch_imgs)
                        embeddings.append(feats)
                        valid_paths.extend(batch_valid)
                        pbar.update(len(batch_valid))

                    pbar.set_postfix(embedded=len(valid_paths), skipped=skipped)

        if not embeddings:
            return np.empty((0, 512), dtype=np.float32), []

        matrix = np.vstack(embeddings).astype(np.float32)
        matrix = normalize(matrix)
        return matrix, valid_paths
