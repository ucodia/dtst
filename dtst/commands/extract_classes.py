from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import (
    config_argument,
    dry_run_option,
    from_dirs_option,
    to_dir_option,
    working_dir_option,
    workers_option,
)
from dtst.files import find_images, format_elapsed, resolve_dirs, resolve_workers
from dtst.sidecar import read_sidecar, sidecar_path

logger = logging.getLogger(__name__)


def _shift_classes(
    classes: dict, crop_x1: int, crop_y1: int, img_w: int, img_h: int
) -> dict:
    """Shift bounding-box coordinates by the crop offset and clamp to new image size."""
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
    """Top-level worker function for ProcessPoolExecutor.

    Returns ``(status, filename, crop_count, outputs, error_message)``.
    Each entry in *outputs* is ``(out_name, class_name, shifted_classes)``.
    Status is one of ``"ok"``, ``"no_detections"``, ``"failed"``.
    """
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

        # Count detections per class for naming
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


@click.command("extract-classes")
@config_argument
@working_dir_option(
    help="Working directory containing source folders and where output is written (default: .)."
)
@from_dirs_option()
@to_dir_option()
@click.option(
    "--classes",
    "-c",
    type=str,
    default=None,
    help="Comma-separated class names to extract (must match classes in sidecar data).",
)
@click.option(
    "--margin",
    type=float,
    default=None,
    help="Margin ratio added around the bounding box, based on the larger side (default: 0).",
)
@click.option(
    "--square",
    is_flag=True,
    help="Extend the shorter side of the bounding box to match the larger side.",
)
@click.option(
    "--min-score",
    type=float,
    default=None,
    help="Minimum detection confidence score to include (default: 0).",
)
@click.option(
    "--skip-partial",
    is_flag=True,
    help="Skip detections whose crop extends beyond the image boundary after applying --square and --margin.",
)
@workers_option()
@dry_run_option(help="Preview what would be extracted without writing files.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    classes: str | None,
    margin: float | None,
    square: bool,
    min_score: float | None,
    skip_partial: bool,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Extract image crops from class detection bounding boxes.

    Reads class detections from sidecar JSON files (produced by
    ``dtst detect``) and crops the corresponding regions from each
    image. Supports expanding the bounding box with a margin ratio
    and squaring the crop.

    \b
    Examples:

        dtst extract-classes config.yaml
        dtst extract-classes config.yaml --classes flower --square --margin 0.1
        dtst extract-classes -d ./dahlias --from images --to flowers --classes flower
        dtst extract-classes config.yaml --min-score 0.5 --skip-partial
    """
    if not from_dirs:
        raise click.ClickException(
            "--from is required (or set 'extract_classes.from' in config)"
        )
    if not to:
        raise click.ClickException(
            "--to is required (or set 'extract_classes.to' in config)"
        )
    if not classes:
        raise click.ClickException(
            "--classes is required (or set 'extract_classes.classes' in config)"
        )

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    classes_list = [c.strip() for c in classes.split(",") if c.strip()]
    working = (working_dir or Path(".")).resolve()
    margin = margin if margin is not None else 0.0
    min_score = min_score if min_score is not None else 0.0

    input_dirs = resolve_dirs(working, dirs_list)
    output_dir = working / to

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise click.ClickException(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: {', '.join(missing)}"
        )

    images: list[Path] = []
    for input_dir in input_dirs:
        found = find_images(input_dir)
        logger.info("Found %d images in %s", len(found), input_dir)
        images.extend(found)

    if not images:
        raise click.ClickException(
            f"No images found in: {', '.join(str(d) for d in input_dirs)}"
        )

    # Pre-read sidecars and filter to images that have class detections
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
        raise click.ClickException(
            f"No images with detections for classes [{', '.join(classes_list)}] found"
        )

    classes_label = ", ".join(classes_list)
    margin_label = f"{margin:.1%}" if margin else "none"
    logger.info(
        "Extracting classes [%s] from %d images (margin=%s, square=%s, min_score=%.2f, workers=%d)",
        classes_label,
        len(work_items),
        margin_label,
        square,
        min_score,
        resolve_workers(workers),
    )

    if dry_run:
        total_dets = 0
        for _, classes_data in work_items:
            for cls in classes_list:
                total_dets += sum(
                    1 for d in classes_data.get(cls, []) if d["score"] >= min_score
                )
        click.echo(
            f"Dry run: would extract {total_dets} crops from {len(work_items)} images"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = resolve_workers(workers)

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
    ok_count = 0
    no_det_count = no_sidecar
    failed_count = 0
    total_crops = 0

    with logging_redirect_tqdm():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_image, w): w for w in work}
            with tqdm(
                total=len(futures), desc="Extracting classes", unit="image"
            ) as pbar:
                try:
                    for future in as_completed(futures):
                        status, name, crop_count, outputs, error = future.result()
                        if status == "ok":
                            ok_count += 1
                            total_crops += crop_count
                            src_path = Path(futures[future][0])
                            src_sidecar = read_sidecar(src_path)
                            base_data = {
                                k: v
                                for k, v in src_sidecar.items()
                                if k not in {"metrics", "classes"}
                            }
                            for out_name, cls_name, shifted_classes in outputs:
                                out_data = {**base_data, "classes": shifted_classes}
                                out_path = sidecar_path(output_dir / out_name)
                                with open(out_path, "w") as f:
                                    json.dump(out_data, f, indent=2)
                                    f.write("\n")
                        elif status == "no_detections":
                            no_det_count += 1
                            logger.debug("No matching detections in %s", name)
                        else:
                            failed_count += 1
                            logger.error("Failed to process %s: %s", name, error)
                        pbar.set_postfix(
                            ok=ok_count, nodet=no_det_count, fail=failed_count
                        )
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    elapsed = time.monotonic() - start_time

    click.echo("\nExtract classes complete!")
    click.echo(f"  Processed: {ok_count:,}")
    click.echo(f"  Crops extracted: {total_crops:,}")
    click.echo(f"  No detections: {no_det_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Time: {format_elapsed(elapsed)}")
    click.echo(f"  Output: {output_dir}")
