"""Library-layer implementation of ``dtst augment``."""

from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dtst.errors import InputError
from dtst.executor import run_pool
from dtst.files import (
    build_save_kwargs,
    find_images,
    resolve_dirs,
    resolve_workers,
)
from dtst.results import AugmentResult
from dtst.sidecar import EXCLUDE_METRICS_AND_CLASSES, copy_sidecar

logger = logging.getLogger(__name__)


def _transform_image(args: tuple) -> tuple[str, str, list[str], str | None]:
    (
        input_path_s,
        output_dir_s,
        flip_x,
        flip_y,
        flip_xy,
        copy_original,
    ) = args
    input_path = Path(input_path_s)
    output_dir = Path(output_dir_s)
    name = input_path.name
    stem = input_path.stem
    ext = input_path.suffix

    created: list[str] = []

    try:
        from PIL import Image

        img = Image.open(input_path)
        save_kw = build_save_kwargs(input_path)

        if copy_original:
            dest = output_dir / name
            shutil.copy2(input_path, dest)
            created.append(name)

        if flip_x:
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            out_name = f"{stem}_flipX{ext}"
            flipped.save(output_dir / out_name, **save_kw)
            created.append(out_name)

        if flip_y:
            flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
            out_name = f"{stem}_flipY{ext}"
            flipped.save(output_dir / out_name, **save_kw)
            created.append(out_name)

        if flip_xy:
            flipped = img.transpose(Image.ROTATE_180)
            out_name = f"{stem}_flipXY{ext}"
            flipped.save(output_dir / out_name, **save_kw)
            created.append(out_name)

        img.close()
        return "ok", name, created, None

    except Exception as e:
        return "failed", name, created, str(e)


def augment(
    *,
    from_dirs: str,
    to: str,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_xy: bool = False,
    no_copy: bool = False,
    workers: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> AugmentResult:
    """Augment a dataset with image flips."""
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not to:
        raise InputError("to is required")
    if not flip_x and not flip_y and not flip_xy:
        raise InputError(
            "At least one transform flag is required (flip_x, flip_y, flip_xy)"
        )

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    input_dirs = resolve_dirs(dirs_list)
    output_dir = Path(to).expanduser().resolve()

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise InputError(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: {', '.join(missing)}"
        )

    images: list[Path] = []
    for input_dir in input_dirs:
        found = find_images(input_dir)
        logger.info("Found %d images in %s", len(found), input_dir)
        images.extend(found)

    if not images:
        raise InputError(f"No images found in: {', '.join(str(d) for d in input_dirs)}")

    transforms = []
    if flip_x:
        transforms.append("flipX")
    if flip_y:
        transforms.append("flipY")
    if flip_xy:
        transforms.append("flipXY")

    copies_per_image = len(transforms) + (0 if no_copy else 1)
    total_output = len(images) * copies_per_image

    from_label = ", ".join(str(d) for d in input_dirs)
    num_workers = resolve_workers(workers)
    logger.info(
        "Augmenting %d images from [%s] with transforms [%s] (copy_original=%s, workers=%d expected output=%d)",
        len(images),
        from_label,
        ", ".join(transforms),
        not no_copy,
        num_workers,
        total_output,
    )

    if dry_run:
        return AugmentResult(
            ok=0,
            failed=0,
            files_written=0,
            transforms=transforms,
            copy_originals=not no_copy,
            total_output_estimate=total_output,
            output_dir=output_dir,
            dry_run=True,
            elapsed=0.0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_original = not no_copy

    work = [
        (
            str(img_path),
            str(output_dir),
            flip_x,
            flip_y,
            flip_xy,
            copy_original,
        )
        for img_path in images
    ]

    start_time = time.monotonic()
    total_files = 0

    def handle(result, work_item):
        nonlocal total_files
        status, name, created, error = result
        total_files += len(created)
        if status == "ok":
            src_path = Path(work_item[0])
            for out_name in created:
                copy_sidecar(
                    src_path,
                    output_dir / out_name,
                    exclude=EXCLUDE_METRICS_AND_CLASSES,
                )
            return "ok"
        logger.error("Failed to process %s: %s", name, error)
        return "fail"

    counts = run_pool(
        ProcessPoolExecutor,
        _transform_image,
        work,
        max_workers=num_workers,
        desc="Augmenting",
        unit="image",
        on_result=handle,
        postfix_keys=("ok", "fail"),
        progress=progress,
    )

    return AugmentResult(
        ok=counts.get("ok", 0),
        failed=counts.get("fail", 0),
        files_written=total_files,
        transforms=transforms,
        copy_originals=copy_original,
        total_output_estimate=total_output,
        output_dir=output_dir,
        dry_run=False,
        elapsed=time.monotonic() - start_time,
    )
