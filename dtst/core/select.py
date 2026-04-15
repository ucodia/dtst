"""Library-layer implementation of ``dtst select``."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.errors import InputError
from dtst.executor import run_pool
from dtst.files import (
    copy_image,
    find_images,
    move_image,
    resolve_dirs,
    resolve_workers,
)
from dtst.results import SelectResult
from dtst.sidecar import read_sidecar

logger = logging.getLogger(__name__)


def _check_image_dimensions(args: tuple) -> tuple[str, str, int, int, str | None]:
    input_path_s, min_side, max_side, min_width, max_width, min_height, max_height = (
        args
    )
    input_path = Path(input_path_s)
    name = input_path.name

    try:
        from PIL import Image

        with Image.open(input_path) as img:
            w, h = img.size

        largest = max(w, h)
        if min_side is not None and largest < min_side:
            return (
                "reject",
                name,
                w,
                h,
                f"side too small ({w}x{h}, min_side={min_side})",
            )
        if max_side is not None and largest > max_side:
            return (
                "reject",
                name,
                w,
                h,
                f"side too large ({w}x{h}, max_side={max_side})",
            )
        if min_width is not None and w < min_width:
            return "reject", name, w, h, f"width too small ({w}, min_width={min_width})"
        if max_width is not None and w > max_width:
            return "reject", name, w, h, f"width too large ({w}, max_width={max_width})"
        if min_height is not None and h < min_height:
            return (
                "reject",
                name,
                w,
                h,
                f"height too small ({h}, min_height={min_height})",
            )
        if max_height is not None and h > max_height:
            return (
                "reject",
                name,
                w,
                h,
                f"height too large ({h}, max_height={max_height})",
            )
        return "keep", name, w, h, None

    except Exception as e:
        return "failed", name, 0, 0, str(e)


def select(
    *,
    working_dir: Path | None,
    from_dirs: str,
    to: str,
    move: bool = False,
    min_side: int | None = None,
    max_side: int | None = None,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    max_height: int | None = None,
    min_metric: list[tuple[str, float]] | None = None,
    max_metric: list[tuple[str, float]] | None = None,
    max_detect: list[tuple[str, float]] | None = None,
    min_detect: list[tuple[str, float]] | None = None,
    source: list[str] | None = None,
    license_filter: list[str] | None = None,
    workers: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> SelectResult:
    """Select images from source folders into a destination folder."""
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not to:
        raise InputError("to is required")

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    working = (working_dir or Path(".")).resolve()
    source_list = [s.strip().lower() for s in source] if source else None
    license_list = (
        [lf.strip().lower() for lf in license_filter] if license_filter else None
    )
    min_metric_list = list(min_metric) if min_metric else None
    max_metric_list = list(max_metric) if max_metric else None
    min_detect_list = list(min_detect) if min_detect else None
    max_detect_list = list(max_detect) if max_detect else None

    input_dirs = resolve_dirs(working, dirs_list)
    output_dir = working / to

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise InputError(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: "
            f"{', '.join(missing)}"
        )

    images: list[Path] = []
    for input_dir in input_dirs:
        found = find_images(input_dir)
        logger.info("Found %d images in %s", len(found), input_dir)
        images.extend(found)

    if not images:
        raise InputError(f"No images found in: {', '.join(str(d) for d in input_dirs)}")

    num_workers = resolve_workers(workers)
    has_dimension_criteria = any(
        v is not None
        for v in (min_side, max_side, min_width, max_width, min_height, max_height)
    )
    has_metric_criteria = min_metric_list is not None or max_metric_list is not None
    has_sidecar_criteria = source_list is not None or license_list is not None
    has_criteria = (
        has_dimension_criteria
        or has_metric_criteria
        or max_detect_list
        or min_detect_list
        or has_sidecar_criteria
    )
    transfer_label = "move" if move else "copy"

    rejects: dict[Path, str] = {}
    kept_set: set[Path] = set(images)
    failed = 0

    if has_criteria:
        if has_dimension_criteria:
            work = [
                (
                    str(p),
                    min_side,
                    max_side,
                    min_width,
                    max_width,
                    min_height,
                    max_height,
                )
                for p in images
            ]

            def handle(result, work_item):
                status, name, w, h, reason = result
                if status == "reject":
                    original_path = Path(work_item[0])
                    rejects[original_path] = reason
                    kept_set.discard(original_path)
                    return "reject"
                if status == "failed":
                    logger.error("Failed to check %s: %s", name, reason)
                    return "fail"
                return "keep"

            counts = run_pool(
                ProcessPoolExecutor,
                _check_image_dimensions,
                work,
                max_workers=num_workers,
                desc="Checking dimensions",
                unit="image",
                on_result=handle,
                progress=progress,
            )
            failed += counts.get("fail", 0)

        if has_metric_criteria:
            remaining = sorted(kept_set)
            for img_path in remaining:
                sidecar = read_sidecar(img_path)
                metrics = sidecar.get("metrics", {})

                rejected = False
                if min_metric_list:
                    for metric_name, threshold in min_metric_list:
                        score = metrics.get(metric_name)
                        if score is None:
                            rejects[img_path] = f"missing '{metric_name}' metric data"
                            kept_set.discard(img_path)
                            rejected = True
                            break
                        if score < threshold:
                            rejects[img_path] = (
                                f"{metric_name} too low ({score} < {threshold})"
                            )
                            kept_set.discard(img_path)
                            rejected = True
                            break

                if not rejected and max_metric_list:
                    for metric_name, threshold in max_metric_list:
                        score = metrics.get(metric_name)
                        if score is None:
                            rejects[img_path] = f"missing '{metric_name}' metric data"
                            kept_set.discard(img_path)
                            break
                        if score > threshold:
                            rejects[img_path] = (
                                f"{metric_name} too high ({score} > {threshold})"
                            )
                            kept_set.discard(img_path)
                            break

        if max_detect_list or min_detect_list:
            remaining = sorted(kept_set)
            for img_path in remaining:
                sidecar = read_sidecar(img_path)
                detect_data = sidecar.get("classes")
                if detect_data is None:
                    rejects[img_path] = "missing detection data"
                    kept_set.discard(img_path)
                    continue

                rejected = False
                if max_detect_list:
                    for cls, threshold in max_detect_list:
                        entry = detect_data.get(cls, [])
                        if not entry:
                            continue
                        score = entry[0]["score"]
                        if score >= threshold:
                            rejects[img_path] = (
                                f"detection '{cls}' score {score:.3f} >= {threshold}"
                            )
                            kept_set.discard(img_path)
                            rejected = True
                            break

                if not rejected and min_detect_list:
                    for cls, threshold in min_detect_list:
                        entry = detect_data.get(cls, [])
                        if not entry:
                            rejects[img_path] = f"'{cls}' not detected"
                            kept_set.discard(img_path)
                            break
                        score = entry[0]["score"]
                        if score < threshold:
                            rejects[img_path] = (
                                f"detection '{cls}' score {score:.3f} < {threshold}"
                            )
                            kept_set.discard(img_path)
                            break

        if has_sidecar_criteria:
            remaining = sorted(kept_set)
            for img_path in remaining:
                sidecar = read_sidecar(img_path)

                if source_list is not None:
                    img_source = sidecar.get("source")
                    if img_source is None:
                        rejects[img_path] = "missing source data"
                        kept_set.discard(img_path)
                        continue
                    if str(img_source).lower() not in source_list:
                        rejects[img_path] = (
                            f"source '{img_source}' not in {source_list}"
                        )
                        kept_set.discard(img_path)
                        continue

                if license_list is not None:
                    img_license = sidecar.get("license")
                    if img_license is None:
                        rejects[img_path] = "missing license data"
                        kept_set.discard(img_path)
                        continue
                    if str(img_license).lower() not in license_list:
                        rejects[img_path] = (
                            f"license '{img_license}' not in {license_list}"
                        )
                        kept_set.discard(img_path)
                        continue

    selected = sorted(kept_set)
    from_label = ", ".join(str(d) for d in input_dirs)

    logger.info(
        "Selecting %d / %d images from [%s] to %s (%s)",
        len(selected),
        len(images),
        from_label,
        output_dir,
        transfer_label,
    )

    rejects_preview = [(p.name, r) for p, r in list(rejects.items())[:10]]

    if dry_run:
        return SelectResult(
            ok=0,
            skipped=0,
            excluded=len(rejects),
            failed=failed,
            total_images=len(images),
            selected=len(selected),
            move=move,
            output_dir=output_dir,
            from_label=from_label,
            dry_run=True,
            rejects_preview=rejects_preview,
            elapsed=0.0,
        )

    if not selected:
        return SelectResult(
            ok=0,
            skipped=0,
            excluded=len(rejects),
            failed=failed,
            total_images=len(images),
            selected=0,
            move=move,
            output_dir=output_dir,
            from_label=from_label,
            dry_run=False,
            rejects_preview=rejects_preview,
            elapsed=0.0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    ok_count = 0
    skipped_count = 0
    failed_count = failed

    transfer_fn = move_image if move else copy_image

    with logging_redirect_tqdm():
        with tqdm(
            total=len(selected),
            desc="Selecting",
            unit="image",
            disable=not progress,
        ) as pbar:
            for image_path in selected:
                dest = output_dir / image_path.name
                try:
                    if dest.exists():
                        skipped_count += 1
                        logger.debug("Skipped (already exists): %s", image_path.name)
                    else:
                        transfer_fn(image_path, dest)
                        ok_count += 1
                except Exception as e:
                    failed_count += 1
                    logger.error(
                        "Failed to %s %s: %s", transfer_label, image_path.name, e
                    )
                pbar.set_postfix(ok=ok_count, skip=skipped_count, fail=failed_count)
                pbar.update(1)

    return SelectResult(
        ok=ok_count,
        skipped=skipped_count,
        excluded=len(rejects),
        failed=failed_count,
        total_images=len(images),
        selected=len(selected),
        move=move,
        output_dir=output_dir,
        from_label=from_label,
        dry_run=False,
        rejects_preview=rejects_preview,
        elapsed=time.monotonic() - start_time,
    )
