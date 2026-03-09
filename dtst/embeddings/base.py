from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingBackend(ABC):
    """Base class for image embedding backends."""

    @abstractmethod
    def load(self, device: str) -> None:
        """Load model onto the given device (cuda, mps, or cpu)."""

    @abstractmethod
    def embed(
        self,
        image_paths: list[Path],
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> tuple[np.ndarray, list[Path]]:
        """Compute embeddings for a list of images.

        Returns a tuple of (embeddings, valid_paths) where valid_paths
        contains only the images that were successfully embedded.
        Images that fail (e.g. no face detected) are excluded from both.

        Parameters
        ----------
        image_paths:
            Paths to images to embed.
        batch_size:
            Number of images per inference batch.
        num_workers:
            Number of threads for image preloading.

        Returns
        -------
        embeddings:
            Array of shape (N, D) with L2-normalized embeddings.
        valid_paths:
            Paths corresponding to each row in embeddings.
        """


def detect_device() -> str:
    """Auto-detect the best available compute device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
