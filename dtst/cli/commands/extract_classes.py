"""Click wrapper for ``dtst extract-classes`` — delegates to :mod:`dtst.core.extract_classes`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    apply_working_dir,
    config_argument,
    dry_run_option,
    from_dirs_option,
    to_dir_option,
    working_dir_option,
    workers_option,
)
from dtst.errors import DtstError
from dtst.files import format_elapsed


@click.command("extract-classes")
@config_argument
@working_dir_option()
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

    apply_working_dir(working_dir)
    from dtst.core.extract_classes import extract_classes as core_extract_classes

    try:
        result = core_extract_classes(
            from_dirs=from_dirs,
            to=to,
            classes=classes,
            margin=margin if margin is not None else 0.0,
            square=square,
            min_score=min_score if min_score is not None else 0.0,
            skip_partial=skip_partial,
            workers=workers,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if result.dry_run:
        click.echo(
            f"Dry run: would extract {result.dry_run_dets} crops from {result.processed} images"
        )
        return

    click.echo("\nExtract classes complete!")
    click.echo(f"  Processed: {result.processed:,}")
    click.echo(f"  Crops extracted: {result.crops_extracted:,}")
    click.echo(f"  No detections: {result.no_detections:,}")
    click.echo(f"  Failed: {result.failed:,}")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
