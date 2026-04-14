from __future__ import annotations

import logging
import time
from pathlib import Path

import click
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import config_argument
from dtst.files import find_images, resolve_dirs
from dtst.sidecar import read_sidecar, sidecar_path, write_sidecar

logger = logging.getLogger(__name__)


@click.command("detect")
@config_argument
@click.option(
    "--from",
    "from_dirs",
    type=str,
    default=None,
    help="Comma-separated source folders (supports globs, e.g. 'images/*').",
)
@click.option(
    "--classes",
    "-c",
    type=str,
    default=None,
    help="Comma-separated object classes to detect (e.g. 'microphone,chair').",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Minimum detection confidence.",
    show_default="0.2",
)
@click.option(
    "--working-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory (default: .).",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=None,
    help="Number of threads for image preloading (default: 4).",
)
@click.option(
    "--max-instances",
    type=int,
    default=None,
    help="Maximum detections per class per image.",
    show_default="1",
)
@click.option(
    "--clear", is_flag=True, help="Remove all detection data from sidecar files."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview what would be detected without writing sidecars.",
)
def cmd(
    from_dirs, classes, threshold, working_dir, workers, max_instances, clear, dry_run
):
    """Detect objects in images using OWL-ViT 2.

    Uses open-vocabulary object detection to find specific objects in images
    and writes the results into per-image sidecar JSON files under a
    "classes" key. Each class gets all detections (score + bounding box)
    sorted by confidence, or null if not found.

    Each run replaces the entire "classes" key in the sidecar.

    \b
    Examples:
        dtst detect -d ./project --from raw --classes "microphone,chair,table"
        dtst detect config.yaml
        dtst detect -d ./project --from raw --classes "microphone" --threshold 0.4
        dtst detect -d ./project --from raw --classes "microphone" --dry-run
        dtst detect -d ./project --from raw --clear
    """
    t0 = time.time()

    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'detect.from' in config)"
        )
    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    working = (working_dir or Path(".")).resolve()
    classes_list = (
        [c.strip() for c in classes.split(",") if c.strip()] if classes else None
    )
    threshold = threshold if threshold is not None else 0.2
    max_instances = max_instances if max_instances is not None else 1

    input_dirs = resolve_dirs(working, dirs_list)

    all_images: list[Path] = []
    for src in input_dirs:
        if not src.is_dir():
            logger.warning("Source directory does not exist, skipping: %s", src)
            continue
        all_images.extend(find_images(src))

    if not all_images:
        raise click.ClickException("No images found in source directories.")

    # --- Clear mode ----------------------------------------------------------

    if clear:
        import json

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

        if dry_run:
            click.echo(
                f"[dry-run] Would clear detection data from {modified:,} sidecar files"
            )
        else:
            elapsed = time.time() - t0
            click.echo(
                f"Cleared detection data from {modified:,} sidecar files ({elapsed:.1f}s)"
            )
        return

    # --- Detect mode ---------------------------------------------------------

    if classes_list is None or not classes_list:
        raise click.ClickException(
            "--classes is required (or set 'detect.classes' in config)"
        )

    needs_work = all_images

    if dry_run:
        click.echo(
            f"[dry-run] Would detect {len(needs_work):,} images for classes: {', '.join(classes_list)}"
        )
        return

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

    if valid:
        click.echo("\nDetection summary:")
        for cls in classes_list:
            click.echo(f"  {cls}: found in {class_counts[cls]}/{valid} images")

    elapsed = time.time() - t0
    click.echo(
        f"\nDone: {written:,} processed, "
        f"{len(needs_work) - valid:,} failed ({elapsed:.1f}s)"
    )
