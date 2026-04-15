"""Click wrapper for ``dtst validate`` — delegates to :mod:`dtst.core.validate`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from dtst.cli.config import (
    apply_working_dir,
    config_argument,
    from_dirs_option,
    working_dir_option,
    workers_option,
)
from dtst.errors import DtstError
from dtst.files import format_elapsed


@click.command("validate")
@config_argument
@from_dirs_option()
@working_dir_option()
@click.option(
    "--square",
    is_flag=True,
    default=False,
    help="Check that all images are square (width == height).",
)
@workers_option()
def cmd(
    from_dirs: str | None,
    working_dir: Path | None,
    square: bool,
    workers: int | None,
) -> None:
    """Validate that all images in a folder are consistent.

    Checks that every image shares the same dimensions and channel mode.
    Optionally checks that images are square. Warns if any PNG files use
    compression (which slows down loading).

    \b
    Examples:
        dtst validate --from faces -d ./my-dataset
        dtst validate --from faces --square -d ./my-dataset
        dtst validate config.yaml
    """
    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'validate.from' in config)"
        )

    apply_working_dir(working_dir)
    from dtst.core.validate import validate as core_validate

    try:
        result = core_validate(
            from_dirs=from_dirs,
            square=square,
            workers=workers,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    click.echo(
        f"\nValidated {result.total:,} images ({format_elapsed(result.elapsed)})"
    )
    click.echo("")

    # Dimensions check
    if len(result.dim_counts) == 1:
        (w, h) = next(iter(result.dim_counts))
        click.echo(f"  Dimensions: PASS (all {w}x{h})")
    else:
        click.echo(f"  Dimensions: FAIL ({len(result.dim_counts)} unique sizes)")
        for (w, h), count in sorted(
            result.dim_counts.items(), key=lambda kv: kv[1], reverse=True
        ):
            click.echo(f"    {w}x{h}: {count:,} images")

    # Channels check
    if len(result.mode_counts) == 1:
        mode = next(iter(result.mode_counts))
        click.echo(f"  Channels:   PASS (all {mode})")
    else:
        click.echo(f"  Channels:   FAIL ({len(result.mode_counts)} unique modes)")
        for mode, count in sorted(
            result.mode_counts.items(), key=lambda kv: kv[1], reverse=True
        ):
            click.echo(f"    {mode}: {count:,} images")

    if result.square_checked:
        if result.non_square == 0:
            click.echo("  Square:     PASS")
        else:
            click.echo(f"  Square:     FAIL ({result.non_square:,} non-square images)")

    if result.total_png > 0:
        if result.compressed_png == 0:
            click.echo(
                f"  PNG comp:   OK (all {result.total_png:,} PNGs at compression level 0)"
            )
        else:
            click.echo(
                f"  PNG comp:   WARN ({result.compressed_png:,}/{result.total_png:,} PNGs are compressed, slower loading)"
            )

    if result.failed > 0:
        click.echo(f"  Errors:     {result.failed:,} images could not be read")

    if not result.passed:
        sys.exit(1)
