"""Embedding cache keyed by model name and sorted input filenames."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = ".dtst" / Path("cache")


def _cache_key(model: str, image_paths: list[Path]) -> str:
    """Compute a deterministic cache key from model name and sorted filenames."""
    h = hashlib.sha256()
    h.update(model.encode())
    for resolved in sorted(str(p.resolve()) for p in image_paths):
        h.update(resolved.encode())
    return h.hexdigest()


def load_embeddings(
    model: str,
    image_paths: list[Path],
) -> tuple[np.ndarray, list[Path]] | None:
    """Load cached embeddings if they exist and match the current image set.

    The cache lives under ``.dtst/cache/``.  Returns
    (embeddings, valid_paths) on hit, or None on miss.
    """
    key = _cache_key(model, image_paths)
    cache_path = CACHE_DIR / f"{key}.npz"

    if not cache_path.exists():
        return None

    data = np.load(cache_path, allow_pickle=False)
    embeddings = data["embeddings"]
    filenames = list(data["filenames"])

    # Reconstruct full paths from cached filenames
    # Build a lookup from filename to original path
    path_lookup = {p.name: p for p in image_paths}
    valid_paths = []
    for name in filenames:
        if name in path_lookup:
            valid_paths.append(path_lookup[name])
        else:
            # A cached filename no longer exists in the input set; cache is stale
            logger.warning("Cache stale: %s no longer in input set", name)
            return None

    logger.info(
        "Loaded cached embeddings from %s (%d images)", cache_path, len(valid_paths)
    )
    return embeddings, valid_paths


def save_embeddings(
    model: str,
    image_paths: list[Path],
    embeddings: np.ndarray,
    valid_paths: list[Path],
) -> None:
    """Save embeddings to the ``.dtst/cache/`` directory."""
    key = _cache_key(model, image_paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{key}.npz"

    filenames = np.array([p.name for p in valid_paths])
    np.savez(cache_path, embeddings=embeddings, filenames=filenames)
    logger.info("Cached embeddings to %s", cache_path)
