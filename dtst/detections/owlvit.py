from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_ID = "google/owlv2-base-patch16-ensemble"

Detection = dict  # {"score": float, "box": [x_min, y_min, x_max, y_max]}


class OwlViT2Backend:
    """Open-vocabulary object detection via OWL-ViT 2."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._device = device
        self._processor = Owlv2Processor.from_pretrained(MODEL_ID)
        self._model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
        self._model.to(device)
        self._model.eval()
        logger.info("OWL-ViT 2 loaded (%s, device=%s)", MODEL_ID, device)

    def detect(
        self,
        image_paths: list[Path],
        classes: list[str],
        threshold: float = 0.2,
        max_instances: int = 1,
        num_workers: int = 4,
    ) -> Iterator[tuple[Path, dict[str, list[Detection]] | None]]:
        """Detect objects in images using open-vocabulary detection.

        Yields (path, detections) per image as it is processed. `detections`
        is None for images that could not be read.
        """
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded -- call .load(device) first")

        class_names = [cls.lower().strip() for cls in classes]
        text_queries = [[f"a {cls}" for cls in class_names]]

        def _load_image(path: Path) -> tuple[Path, Image.Image | None]:
            try:
                return path, Image.open(path).convert("RGB")
            except Exception:
                return path, None

        detected = 0
        skipped = 0

        with ThreadPoolExecutor(max_workers=num_workers) as loader:
            with tqdm(
                total=len(image_paths), desc="Detecting (owlv2)", unit="image"
            ) as pbar:
                futures = {loader.submit(_load_image, p): p for p in image_paths}
                for future in futures:
                    path, image = future.result()
                    if image is None:
                        logger.warning("Could not read image: %s", path.name)
                        skipped += 1
                        pbar.update(1)
                        pbar.set_postfix(detected=detected, skipped=skipped)
                        yield path, None
                        continue

                    inputs = self._processor(
                        text=text_queries, images=image, return_tensors="pt"
                    ).to(self._device)

                    with torch.no_grad():
                        outputs = self._model(**inputs)

                    target_sizes = torch.tensor([image.size[::-1]])
                    proc_results = (
                        self._processor.post_process_grounded_object_detection(
                            outputs,
                            threshold=threshold,
                            target_sizes=target_sizes,
                        )
                    )

                    detections = self._parse_detections(
                        proc_results[0], class_names, max_instances
                    )
                    detected += 1
                    pbar.update(1)
                    pbar.set_postfix(detected=detected, skipped=skipped)
                    yield path, detections

    def _parse_detections(
        self,
        result: dict,
        class_names: list[str],
        max_instances: int = 1,
    ) -> dict[str, list[Detection]]:
        """Extract all detections per class, sorted by score descending."""
        all_dets: dict[str, list[Detection]] = {cls: [] for cls in class_names}

        boxes = result["boxes"]
        scores = result["scores"]
        label_indices = result["labels"]

        for i in range(len(scores)):
            idx = int(label_indices[i])
            if idx >= len(class_names):
                continue
            cls = class_names[idx]
            score = float(scores[i])
            box = [int(x) for x in boxes[i].tolist()]
            all_dets[cls].append({"score": score, "box": box})

        parsed: dict[str, list[Detection]] = {}
        for cls in class_names:
            if all_dets[cls]:
                parsed[cls] = sorted(
                    all_dets[cls], key=lambda d: d["score"], reverse=True
                )[:max_instances]
            else:
                parsed[cls] = []

        return parsed
