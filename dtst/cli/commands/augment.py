"""Click wrapper for ``dtst augment`` — delegates to :mod:`dtst.core.augment`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    config_argument,
    dry_run_option,
    from_dirs_option,
    to_dir_option,
    working_dir_option,
    workers_option,
)
from dtst.core.augment import augment as core_augment
from dtst.errors import DtstError
from dtst.files import format_elapsed


@click.command("augment")
@config_argument
@working_dir_option(
    help="Working directory containing source folders and where output is written (default: .)."
)
@from_dirs_option()
@to_dir_option()
@click.option("--flipX", "flip_x", is_flag=True, help="Apply horizontal flip.")
@click.option("--flipY", "flip_y", is_flag=True, help="Apply vertical flip.")
@click.option(
    "--flipXY",
    "flip_xy",
    is_flag=True,
    help="Apply both horizontal and vertical flip (180-degree rotation).",
)
@click.option(
    "--no-copy", is_flag=True, help="Do not copy original images to the output folder."
)
@workers_option()
@dry_run_option(help="Preview what would be written without creating files.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    flip_x: bool,
    flip_y: bool,
    flip_xy: bool,
    no_copy: bool,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Augment a dataset by applying image transformations.

    Reads images from one or more source folders and writes transformed
    copies to a destination folder. By default the original images are
    also copied to the output; use --no-copy to write only the
    transformed versions.

    At least one transform flag (--flipX, --flipY, --flipXY) is
    required. Multiple flags can be combined in a single run to
    produce several variants of each image.

    Transformed files are named with a suffix indicating the transform:
    photo.jpg becomes photo_flipX.jpg, photo_flipY.jpg, photo_flipXY.jpg.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst augment -d ./project --from faces --to augmented --flipX
        dtst augment -d ./project --from faces --to augmented --flipX --flipY --flipXY
        dtst augment -d ./project --from faces --to augmented --flipX --no-copy
        dtst augment config.yaml --dry-run
    """
    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'augment.from' in config)"
        )
    if to is None:
        raise click.ClickException("--to is required (or set 'augment.to' in config)")
    if not flip_x and not flip_y and not flip_xy:
        raise click.ClickException(
            "At least one transform flag is required (--flipX, --flipY, --flipXY)"
        )

    try:
        result = core_augment(
            working_dir=working_dir,
            from_dirs=from_dirs,
            to=to,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_xy=flip_xy,
            no_copy=no_copy,
            workers=workers,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if result.dry_run:
        click.echo(
            f"\nDry run -- would augment {result.total_output_estimate // (len(result.transforms) + (1 if result.copy_originals else 0)):,} images"
        )
        click.echo(f"  Transforms: {', '.join(result.transforms)}")
        click.echo(f"  Copy originals: {result.copy_originals}")
        files_per = len(result.transforms) + (1 if result.copy_originals else 0)
        click.echo(f"  Files per image: {files_per}")
        click.echo(f"  Total output files: {result.total_output_estimate:,}")
        click.echo(f"  Output: {result.output_dir}")
        return

    click.echo("\nAugment complete!")
    click.echo(f"  Images processed: {result.ok:,}")
    click.echo(f"  Files written: {result.files_written:,}")
    click.echo(f"  Failed: {result.failed:,}")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
