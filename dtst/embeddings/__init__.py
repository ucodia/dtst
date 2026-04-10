from __future__ import annotations

from dtst.embeddings.base import EmbeddingBackend, detect_device

VALID_MODELS = frozenset({"arcface", "clip"})


def get_backend(name: str) -> EmbeddingBackend:
    """Return an embedding backend instance by name.

    The model is not loaded yet -- call ``.load(device)`` before use.
    """
    if name == "arcface":
        from dtst.embeddings.arcface import ArcFaceBackend

        return ArcFaceBackend()
    if name == "clip":
        from dtst.embeddings.clip import CLIPBackend

        return CLIPBackend()
    raise ValueError(
        f"Unknown embedding model: {name!r}; valid: {sorted(VALID_MODELS)}"
    )


__all__ = ["VALID_MODELS", "detect_device", "get_backend", "EmbeddingBackend"]
