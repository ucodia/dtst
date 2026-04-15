"""Library-layer implementation of ``dtst extract-classes``."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dtst.errors import InputError
from dtst.executor import run_pool
from dtst.files import find_images, resolve_dirs, resolve_workers
from dtst.results import ExtractClassesResult
from dtst.sidecar import read_sidecar, sidecar_path

logger = logging.getLogger(__name__)


def _shift_classes(
    classes: dict, crop_x1: int, crop_y1: int, img_w: int, img_h: int
) -> dict:
    shifted: dict[str, list[dict]] = {}
    for cls, detections in classes.items():
        shifted[cls] = []
        for d in detections:
            x_min, y_min, x_max, y_max = d["box"]
            new_box = [
                max(0, x_min - crop_x1),
                max(0, y_min - crop_y1),
                min(img_w, x_max - crop_x1),
                min(img_h, y_max - crop_y1),
            ]
            shifted[cls].append({"score": d["score"], "box": new_box})
    return shifted


def _process_image(
    args: tuple,
) -> tuple[str, str, int, list[tuple[str, str, dict]], str | None]:
    (
        input_path_s,
        output_dir_s,
        classes_data,
        target_classes,
        margin,
        square,
        min_score,
        skip_partial,
    ) = args
    input_path = Path(input_path_s)
    output_dir = Path(output_dir_s)
    name = input_path.name

    try:
        from PIL import Image

        img = Image.open(input_path)
        img_w, img_h = img.size

        detections: list[tuple[str, dict]] = []
        for cls in target_classes:
            for det in classes_data.get(cls, []):
                if det["score"] >= min_score:
                    detections.append((cls, det))

        if not detections:
            return "no_detections", name, 0, [], None

        class_counts: dict[str, int] = {}
        for cls, _ in detections:
            class_counts[cls] = class_counts.get(cls, 0) + 1

        outputs: list[tuple[str, str, dict]] = []
        class_idx: dict[str, int] = {}
        stem = input_path.stem
        skipped = 0

        for cls, det in detections:
            x_min, y_min, x_max, y_max = det["box"]
            w = x_max - x_min
            h = y_max - y_min

            if square:
                max_side = max(w, h)
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                x_min = cx - max_side / 2
                y_min = cy - max_side / 2
                x_max = cx + max_side / 2
                y_max = cy + max_side / 2
                w = h = max_side

            large_side = max(w, h)
            margin_px = margin * large_side
            x_min -= margin_px
            y_min -= margin_px
            x_max += margin_px
            y_max += margin_px

            if skip_partial and (
                x_min < 0 or y_min < 0 or x_max > img_w or y_max > img_h
            ):
                skipped += 1
                continue

            crop_x1 = max(0, int(x_min))
            crop_y1 = max(0, int(y_min))
            crop_x2 = min(img_w, int(x_max))
            crop_y2 = min(img_h, int(y_max))

            cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            idx = class_idx.get(cls, 0) + 1
            class_idx[cls] = idx

            if class_counts[cls] == 1:
                out_name = f"{stem}_{cls}.jpg"
            else:
                out_name = f"{stem}_{cls}_{idx:02d}.jpg"

            cropped.save(output_dir / out_name, "JPEG", quality=95)

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            shifted = _shift_classes(classes_data, crop_x1, crop_y1, crop_w, crop_h)
            outputs.append((out_name, cls, shifted))

        if not outputs:
            return "no_detections", name, 0, [], None

        return "ok", name, len(outputs), outputs, None

    except Exception as e:
        return "failed", name, 0, [], str(e)


def extract_classes(
    *,
    from_dirs: str,
    to: str,
    classes: str,
    margin: float = 0.0,
    square: bool = False,
    min_score: float = 0.0,
    skip_partial: bool = False,
    workers: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> ExtractClassesResult:
    """Extract image crops from class-detection bounding boxes."""
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not to:
        raise InputError("to is required")
    if not classes:
        raise InputError("classes is required")

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    classes_list = [c.strip() for c in classes.split(",") if c.strip()]
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

    work_items: list[tuple[Path, dict]] = []
    no_sidecar = 0
    for img_path in images:
        sidecar = read_sidecar(img_path)
        classes_data = sidecar.get("classes")
        if not classes_data:
            no_sidecar += 1
            continue
        has_target = any(classes_data.get(c) for c in classes_list)
        if not has_target:
            no_sidecar += 1
            continue
        work_items.append((img_path, classes_data))

    if not work_items:
        raise InputError(
            f"No images with detections for classes [{', '.join(classes_list)}] found"
        )

    num_workers = resolve_workers(workers)
    classes_label = ", ".join(classes_list)
    margin_label = f"{margin:.1%}" if margin else "none"
    logger.info(
        "Extracting classes [%s] from %d images (margin=%s, square=%s, min_score=%.2f, workers=%d)",
        classes_label,
        len(work_items),
        margin_label,
        square,
        min_score,
        num_workers,
    )

    if dry_run:
        total_dets = 0
        for _, classes_data in work_items:
            for cls in classes_list:
                total_dets += sum(
                    1 for d in classes_data.get(cls, []) if d["score"] >= min_score
                )
        return ExtractClassesResult(
            processed=len(work_items),
            crops_extracted=0,
            no_detections=no_sidecar,
            failed=0,
            output_dir=output_dir,
            dry_run=True,
            dry_run_dets=total_dets,
            elapsed=0.0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (
            str(img_path),
            str(output_dir),
            classes_data,
            classes_list,
            margin,
            square,
            min_score,
            skip_partial,
        )
        for img_path, classes_data in work_items
    ]

    start_time = time.monotonic()
    total_crops = 0

    def handle(result, work_item):
        nonlocal total_crops
        status, name, crop_count, outputs, error = result
        if status == "ok":
            total_crops += crop_count
            src_path = Path(work_item[0])
            src_sidecar = read_sidecar(src_path)
            base_data = {
                k: v for k, v in src_sidecar.items() if k not in {"metrics", "classes"}
            }
            for out_name, cls_name, shifted_classes in outputs:
                out_data = {**base_data, "classes": shifted_classes}
                out_path = sidecar_path(output_dir / out_name)
                with open(out_path, "w") as f:
                    json.dump(out_data, f, indent=2)
                    f.write("\n")
            return "ok"
        if status == "no_detections":
            logger.debug("No matching detections in %s", name)
            return "nodet"
        logger.error("Failed to process %s: %s", name, error)
        return "fail"

    counts = run_pool(
        ProcessPoolExecutor,
        _process_image,
        work,
        max_workers=num_workers,
        desc="Extracting classes",
        unit="image",
        on_result=handle,
        postfix_keys=("ok", "nodet", "fail"),
        progress=progress,
    )

    return ExtractClassesResult(
        processed=counts.get("ok", 0),
        crops_extracted=total_crops,
        no_detections=no_sidecar + counts.get("nodet", 0),
        failed=counts.get("fail", 0),
        output_dir=output_dir,
        dry_run=False,
        elapsed=time.monotonic() - start_time,
    )
