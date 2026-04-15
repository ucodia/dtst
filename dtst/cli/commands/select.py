"""Click wrapper for ``dtst select`` — delegates to :mod:`dtst.core.select`."""

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
from dtst.core.select import select as core_select
from dtst.errors import DtstError
from dtst.files import format_elapsed


@click.command("select")
@config_argument
@working_dir_option()
@from_dirs_option()
@to_dir_option()
@click.option(
    "--move", is_flag=True, help="Move images instead of copying (removes originals)."
)
@click.option(
    "--min-side",
    "-s",
    type=int,
    default=None,
    help="Minimum largest side in pixels; images with max(w,h) below this are excluded.",
)
@click.option(
    "--max-side",
    type=int,
    default=None,
    help="Maximum largest side in pixels; images with max(w,h) above this are excluded.",
)
@click.option(
    "--min-width",
    type=int,
    default=None,
    help="Minimum width in pixels; narrower images are excluded.",
)
@click.option(
    "--max-width",
    type=int,
    default=None,
    help="Maximum width in pixels; wider images are excluded.",
)
@click.option(
    "--min-height",
    type=int,
    default=None,
    help="Minimum height in pixels; shorter images are excluded.",
)
@click.option(
    "--max-height",
    type=int,
    default=None,
    help="Maximum height in pixels; taller images are excluded.",
)
@click.option(
    "--min-metric",
    type=(str, float),
    multiple=True,
    default=(),
    help="Minimum metric threshold (e.g. --min-metric blur 5). Can be repeated.",
)
@click.option(
    "--max-metric",
    type=(str, float),
    multiple=True,
    default=(),
    help="Maximum metric threshold (e.g. --max-metric brisque 40). Can be repeated.",
)
@click.option(
    "--max-detect",
    type=(str, float),
    multiple=True,
    default=(),
    help="Exclude images where detection score >= THRESHOLD (e.g. --max-detect microphone 0.5).",
)
@click.option(
    "--min-detect",
    type=(str, float),
    multiple=True,
    default=(),
    help="Exclude images where detection score < THRESHOLD (e.g. --min-detect chair 0.3).",
)
@click.option(
    "--source",
    type=str,
    default=None,
    help="Comma-separated list of sources to include (e.g. 'serper,flickr'); checked against sidecar 'source' field.",
)
@click.option(
    "--license",
    "license_filter",
    type=str,
    default=None,
    help="Comma-separated list of licenses to include (e.g. 'cc-by,none'); checked against sidecar 'license' field.",
)
@workers_option()
@dry_run_option(help="Preview what would be selected without creating files.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    move: bool,
    min_side: int | None,
    max_side: int | None,
    min_width: int | None,
    max_width: int | None,
    min_height: int | None,
    max_height: int | None,
    min_metric: tuple[tuple[str, float], ...],
    max_metric: tuple[tuple[str, float], ...],
    max_detect: tuple[tuple[str, float], ...],
    min_detect: tuple[tuple[str, float], ...],
    source: str | None,
    license_filter: str | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Select images from source folders into a destination folder.

    Copies (or moves with --move) images from one or more source folders
    into a destination folder. When filter criteria are provided, only
    images that pass all criteria are selected. Without criteria, all
    images are selected.

    Files that already exist in the destination (by name) are skipped.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:
        dtst select -d ./project --from raw --to backup
        dtst select -d ./project --from raw,extra --to combined
        dtst select -d ./project --from faces --to curated --min-side 256
        dtst select -d ./project --from faces --to curated --min-metric blur 5
        dtst select -d ./project --from faces --to curated --min-metric blur 5 --min-metric musiq 60
        dtst select -d ./project --from faces --to curated --max-metric brisque 40
        dtst select -d ./project --from raw --to clean --max-detect microphone 0.5
        dtst select -d ./project --from raw --to licensed --source serper,flickr
        dtst select config.yaml --dry-run
    """
    if not from_dirs:
        raise click.ClickException(
            "--from is required (or set 'select.from' in config)"
        )
    if not to:
        raise click.ClickException("--to is required (or set 'select.to' in config)")

    source_list = (
        [s.strip() for s in source.split(",") if s.strip()] if source else None
    )
    license_list = (
        [lf.strip() for lf in license_filter.split(",") if lf.strip()]
        if license_filter
        else None
    )

    apply_working_dir(working_dir)
    try:
        result = core_select(
            from_dirs=from_dirs,
            to=to,
            move=move,
            min_side=min_side,
            max_side=max_side,
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
            min_metric=list(min_metric) if min_metric else None,
            max_metric=list(max_metric) if max_metric else None,
            max_detect=list(max_detect) if max_detect else None,
            min_detect=list(min_detect) if min_detect else None,
            source=source_list,
            license_filter=license_list,
            workers=workers,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    transfer_label = "move" if move else "copy"

    if dry_run:
        click.echo(
            f"\nDry run -- would {transfer_label} {result.selected:,} / {result.total_images:,} images"
        )
        click.echo(f"  From: {result.from_label}")
        click.echo(f"  To: {result.output_dir}")
        if result.excluded:
            click.echo(f"  Excluded: {result.excluded:,}")
            for name, reason in result.rejects_preview:
                click.echo(f"    {name} ({reason})")
            if result.excluded > len(result.rejects_preview):
                click.echo(
                    f"    ... and {result.excluded - len(result.rejects_preview):,} more"
                )
        return

    if result.selected == 0:
        click.echo(
            f"\nNo images passed the filter criteria ({result.excluded:,} excluded)."
        )
        return

    verb = "Moved" if move else "Copied"
    click.echo("\nSelect complete!")
    click.echo(f"  {verb}: {result.ok:,}")
    if result.skipped > 0:
        click.echo(f"  Skipped: {result.skipped:,}")
    if result.excluded > 0:
        click.echo(f"  Excluded: {result.excluded:,}")
    if result.failed > 0:
        click.echo(f"  Failed: {result.failed:,}")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
