"""Library-layer implementation of ``dtst rename``."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from dtst.errors import InputError
from dtst.files import gather_images, move_image
from dtst.results import RenameResult

logger = logging.getLogger(__name__)


def rename(
    *,
    working_dir: Path | None,
    from_dirs: str,
    prefix: str = "",
    digits: int | None = None,
    dry_run: bool = False,
) -> RenameResult:
    """Sequentially rename images in-place with a prefix + zero-padded number.

    Parameters mirror the CLI flags.  ``from_dirs`` is a comma-separated
    list of folder names (relative to ``working_dir``) and may contain
    globs.  Sidecar JSON files travel with their images.

    Raises :class:`InputError` if ``from_dirs`` is missing or no images
    are found.
    """
    if not from_dirs:
        raise InputError("from_dirs is required")

    t0 = time.time()
    _working, _input_dirs, all_images = gather_images(working_dir, from_dirs)
    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]

    pad = digits if digits is not None else len(str(len(all_images)))

    logger.info(
        "Found %d images in %s, renaming with prefix='%s', digits=%d",
        len(all_images),
        ", ".join(dirs_list),
        prefix,
        pad,
    )

    if dry_run:
        for i, img in enumerate(all_images, start=1):
            new_name = f"{prefix}{i:0{pad}d}{img.suffix}"
            logger.debug("%s -> %s", img.name, new_name)
        return RenameResult(
            renamed=len(all_images), dry_run=True, elapsed=time.time() - t0
        )

    # Phase 1: move to temporary names to avoid collisions
    tag = uuid.uuid4().hex[:8]
    temp_pairs: list[tuple[Path, Path]] = []
    for i, img in enumerate(all_images, start=1):
        tmp_name = f"__rename_tmp_{tag}_{i}{img.suffix}"
        tmp_path = img.parent / tmp_name
        move_image(img, tmp_path)
        final_name = f"{prefix}{i:0{pad}d}{img.suffix}"
        temp_pairs.append((tmp_path, img.parent / final_name))

    # Phase 2: move from temporary to final names
    renamed = 0
    for tmp_path, final_path in temp_pairs:
        move_image(tmp_path, final_path)
        renamed += 1

    return RenameResult(renamed=renamed, dry_run=False, elapsed=time.time() - t0)
