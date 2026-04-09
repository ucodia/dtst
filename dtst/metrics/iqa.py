from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
NUM_WORKERS = 4


def validate_iqa_metrics(metric_names: list[str]) -> list[str]:
    """Check that all requested IQA metrics are no-reference (NR).

    Returns a list of metric names that are full-reference (FR) and
    therefore unsupported by the analyze command.
    """
    import pyiqa

    fr_metrics = []
    for name in metric_names:
        model = pyiqa.create_metric(name, device="cpu")
        mode = getattr(model, "metric_mode", None)
        if mode == "FR":
            fr_metrics.append(name)
        del model
    return fr_metrics


def compute_iqa_metrics(
    metric_names: list[str],
    image_paths: list[Path],
    batch_size: int = BATCH_SIZE,
    device: str | None = None,
) -> dict[Path, dict[str, float]]:
    """Score images with one or more IQA-PyTorch metrics.

    Loads each model sequentially to limit VRAM usage. Images are
    preprocessed in parallel via a thread pool and scored in batches.

    Returns a dict mapping image path to {metric_name: score}.
    """
    import pyiqa
    import torch
    from PIL import Image

    from dtst.embeddings.base import detect_device

    if device is None:
        device = detect_device()

    results: dict[Path, dict[str, float]] = {p: {} for p in image_paths}

    for metric_name in metric_names:
        logger.info("Loading IQA model: %s (device=%s)", metric_name, device)
        model = pyiqa.create_metric(metric_name, device=device)
        model.eval()

        def _load_image(path: Path) -> tuple[Path, torch.Tensor | None]:
            try:
                img = Image.open(path).convert("RGB")
                import torchvision.transforms.functional as TF

                tensor = TF.to_tensor(img)
                return path, tensor
            except Exception:
                return path, None

        errors = 0
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as loader:
            with tqdm(total=len(image_paths), desc=f"Scoring {metric_name}", unit="image") as pbar:
                for batch_start in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[batch_start : batch_start + batch_size]
                    loaded = list(loader.map(_load_image, batch_paths))

                    tensors = []
                    batch_valid: list[Path] = []
                    for path, tensor in loaded:
                        if tensor is None:
                            logger.error("%s: could not read image: %s", metric_name, path.name)
                            errors += 1
                            pbar.update(1)
                            continue
                        tensors.append(tensor)
                        batch_valid.append(path)

                    if tensors:
                        try:
                            batch_tensor = torch.stack(tensors).to(device)
                            with torch.no_grad():
                                scores = model(batch_tensor)
                            scores = scores.cpu().squeeze()
                            if scores.dim() == 0:
                                scores = scores.unsqueeze(0)
                            for path, score in zip(batch_valid, scores):
                                results[path][metric_name] = round(score.item(), 4)
                        except Exception as exc:
                            logger.error("%s: batch scoring failed: %s", metric_name, exc)
                            errors += len(batch_valid)

                    pbar.update(len(batch_valid))
                    pbar.set_postfix(scored=sum(1 for r in results.values() if metric_name in r), errors=errors)

        del model
        if device != "cpu":
            torch.cuda.empty_cache() if device == "cuda" else None

        logger.info("Finished %s: %d scored, %d errors", metric_name, len(image_paths) - errors, errors)

    return results
