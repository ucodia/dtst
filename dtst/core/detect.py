"""Library-layer implementation of ``dtst detect``."""

from __future__ import annotations

import json
import logging
import time

from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.errors import InputError
from dtst.files import gather_images
from dtst.results import DetectResult
from dtst.sidecar import read_sidecar, sidecar_path, write_sidecar

logger = logging.getLogger(__name__)


def detect(
    *,
    from_dirs: str,
    classes: str | None = None,
    threshold: float = 0.2,
    workers: int | None = None,
    max_instances: int = 1,
    clear: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> DetectResult:
    """Detect objects in images using OWL-ViT 2."""
    t0 = time.time()

    if not from_dirs:
        raise InputError("from_dirs is required")

    classes_list = (
        [c.strip() for c in classes.split(",") if c.strip()] if classes else None
    )

    _input_dirs, all_images = gather_images(from_dirs)

    if clear:
        modified = 0
        for img in all_images:
            sc = sidecar_path(img)
            if not sc.exists():
                continue
            sidecar = read_sidecar(img)
            if "classes" not in sidecar:
                continue
            if dry_run:
                modified += 1
                continue
            del sidecar["classes"]
            if sidecar:
                with open(sc, "w") as f:
                    json.dump(sidecar, f, indent=2)
                    f.write("\n")
            else:
                sc.unlink()
            modified += 1

        return DetectResult(
            processed=0,
            failed=0,
            class_counts={},
            valid=0,
            cleared=modified,
            dry_run=dry_run,
            elapsed=time.time() - t0,
        )

    if not classes_list:
        raise InputError("classes is required")

    needs_work = all_images

    if dry_run:
        return DetectResult(
            processed=len(needs_work),
            failed=0,
            class_counts={cls: 0 for cls in classes_list},
            valid=0,
            cleared=0,
            dry_run=True,
            elapsed=time.time() - t0,
        )

    logger.info(
        "Detecting %d images for %d classes",
        len(needs_work),
        len(classes_list),
    )

    num_workers = workers if workers is not None else 4

    from dtst.detections.owlvit import OwlViT2Backend
    from dtst.embeddings.base import detect_device

    device = detect_device()
    backend = OwlViT2Backend()
    backend.load(device)

    written = 0
    class_counts = {cls: 0 for cls in classes_list}
    valid = 0

    with logging_redirect_tqdm():
        for img_path, img_detections in backend.detect(
            needs_work,
            classes_list,
            threshold=threshold,
            max_instances=max_instances,
            num_workers=num_workers,
        ):
            if img_detections is None:
                continue
            write_sidecar(img_path, {"classes": img_detections})
            written += 1
            valid += 1
            for cls in classes_list:
                if img_detections.get(cls):
                    class_counts[cls] += 1

    return DetectResult(
        processed=written,
        failed=len(needs_work) - valid,
        class_counts=class_counts,
        valid=valid,
        cleared=0,
        dry_run=False,
        elapsed=time.time() - t0,
    )
