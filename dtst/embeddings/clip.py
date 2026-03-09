from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
from tqdm import tqdm

from dtst.embeddings.base import EmbeddingBackend

logger = logging.getLogger(__name__)

# Default model -- good speed/quality tradeoff
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


class CLIPBackend(EmbeddingBackend):
    """Visual similarity embeddings via OpenCLIP."""

    def __init__(self) -> None:
        self._model = None
        self._preprocess = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        import open_clip
        import torch

        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=device,
        )
        self._model.eval()
        logger.info("CLIP model loaded (%s, device=%s)", MODEL_NAME, device)

    def embed(
        self,
        image_paths: list[Path],
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> tuple[np.ndarray, list[Path]]:
        import torch

        if self._model is None or self._preprocess is None:
            raise RuntimeError("Model not loaded -- call .load(device) first")

        preprocess = self._preprocess

        def _load_and_preprocess(path: Path) -> tuple[Path, torch.Tensor | None]:
            try:
                img = Image.open(path).convert("RGB")
                return path, preprocess(img)
            except Exception:
                return path, None

        embeddings: list[np.ndarray] = []
        valid_paths: list[Path] = []
        skipped = 0

        with ThreadPoolExecutor(max_workers=num_workers) as loader:
            with tqdm(total=len(image_paths), desc="Embedding (clip)", unit="image") as pbar:
                for batch_start in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[batch_start : batch_start + batch_size]
                    loaded = list(loader.map(_load_and_preprocess, batch_paths))

                    tensors = []
                    batch_valid = []
                    for path, tensor in loaded:
                        if tensor is None:
                            logger.warning("Could not read image: %s", path.name)
                            skipped += 1
                            pbar.update(1)
                            continue
                        tensors.append(tensor)
                        batch_valid.append(path)

                    if tensors:
                        images = torch.stack(tensors).to(self._device)
                        with torch.no_grad():
                            features = self._model.encode_image(images)
                        embeddings.append(features.cpu().numpy())
                        valid_paths.extend(batch_valid)

                    pbar.update(len(batch_valid))
                    pbar.set_postfix(embedded=len(valid_paths), skipped=skipped)

        if not embeddings:
            return np.empty((0, 512), dtype=np.float32), []

        matrix = np.vstack(embeddings).astype(np.float32)
        matrix = normalize(matrix)
        return matrix, valid_paths

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        """Encode text prompts into the same embedding space as images.

        Returns an L2-normalized array of shape (len(prompts), D).
        """
        import open_clip
        import torch

        if self._model is None:
            raise RuntimeError("Model not loaded -- call .load(device) first")

        tokens = open_clip.tokenize(prompts).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)

        matrix = features.cpu().numpy().astype(np.float32)
        matrix = normalize(matrix)
        return matrix
