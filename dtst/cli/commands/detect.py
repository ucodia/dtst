"""Click wrapper for ``dtst detect`` — delegates to :mod:`dtst.core.detect`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    config_argument,
    dry_run_option,
    from_dirs_option,
    working_dir_option,
    workers_option,
)
from dtst.core.detect import detect as core_detect
from dtst.errors import DtstError


@click.command("detect")
@config_argument
@from_dirs_option()
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
@working_dir_option()
@workers_option(help="Number of threads for image preloading (default: 4).")
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
@dry_run_option(help="Preview what would be detected without writing sidecars.")
def cmd(
    from_dirs: str | None,
    classes: str | None,
    threshold: float | None,
    working_dir: Path | None,
    workers: int | None,
    max_instances: int | None,
    clear: bool,
    dry_run: bool,
) -> None:
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
    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'detect.from' in config)"
        )
    if not clear and not classes:
        raise click.ClickException(
            "--classes is required (or set 'detect.classes' in config)"
        )

    try:
        result = core_detect(
            working_dir=working_dir,
            from_dirs=from_dirs,
            classes=classes,
            threshold=threshold if threshold is not None else 0.2,
            workers=workers,
            max_instances=max_instances if max_instances is not None else 1,
            clear=clear,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if clear:
        if result.dry_run:
            click.echo(
                f"[dry-run] Would clear detection data from {result.cleared:,} sidecar files"
            )
        else:
            click.echo(
                f"Cleared detection data from {result.cleared:,} sidecar files ({result.elapsed:.1f}s)"
            )
        return

    if result.dry_run:
        classes_list = [c.strip() for c in (classes or "").split(",") if c.strip()]
        click.echo(
            f"[dry-run] Would detect {result.processed:,} images for classes: {', '.join(classes_list)}"
        )
        return

    if result.valid:
        classes_list = [c.strip() for c in (classes or "").split(",") if c.strip()]
        click.echo("\nDetection summary:")
        for cls in classes_list:
            click.echo(
                f"  {cls}: found in {result.class_counts.get(cls, 0)}/{result.valid} images"
            )

    click.echo(
        f"\nDone: {result.processed:,} processed, "
        f"{result.failed:,} failed ({result.elapsed:.1f}s)"
    )
