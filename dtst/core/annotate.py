"""Library-layer implementation of ``dtst annotate``."""

from __future__ import annotations

import logging
import time

from dtst.errors import InputError
from dtst.files import gather_images
from dtst.results import AnnotateResult
from dtst.sidecar import read_sidecar, write_sidecar

logger = logging.getLogger(__name__)


def annotate(
    *,
    from_dirs: str,
    source: str | None = None,
    license: str | None = None,
    origin: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> AnnotateResult:
    """Write source/license/origin metadata into per-image sidecars."""
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not source and not license and not origin:
        raise InputError("At least one of source, license, or origin is required.")

    t0 = time.time()
    _input_dirs, all_images = gather_images(from_dirs)
    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]

    annotation: dict[str, str] = {}
    if source:
        annotation["source"] = source
    if license:
        annotation["license"] = license
    if origin:
        annotation["origin"] = origin

    logger.info(
        "Found %d images in %s, annotating: %s",
        len(all_images),
        ", ".join(dirs_list),
        ", ".join(f"{k}={v}" for k, v in annotation.items()),
    )

    annotated = 0
    skipped = 0

    for img in all_images:
        existing = read_sidecar(img)
        if not overwrite:
            new_data = {k: v for k, v in annotation.items() if k not in existing}
        else:
            new_data = dict(annotation)

        if not new_data:
            skipped += 1
            continue

        if dry_run:
            fields = ", ".join(f"{k}={v}" for k, v in new_data.items())
            logger.debug("%s: %s", img.name, fields)
            annotated += 1
            continue

        write_sidecar(img, new_data)
        annotated += 1

    return AnnotateResult(
        annotated=annotated,
        skipped=skipped,
        dry_run=dry_run,
        elapsed=time.time() - t0,
    )
