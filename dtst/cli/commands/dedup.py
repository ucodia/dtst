"""Click wrapper for ``dtst dedup`` — delegates to :mod:`dtst.core.dedup`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    apply_working_dir,
    config_argument,
    dry_run_option,
    working_dir_option,
    workers_option,
)
from dtst.core.dedup import dedup as core_dedup
from dtst.errors import DtstError


@click.command("dedup")
@config_argument
@working_dir_option()
@click.option(
    "--from",
    "from_dir",
    type=str,
    default=None,
    help="Folder to deduplicate.",
)
@click.option(
    "--to",
    type=str,
    default=None,
    help="Subfolder name for duplicate images.",
    show_default="duplicated",
)
@click.option(
    "--threshold",
    "-t",
    type=int,
    default=None,
    help="Phash hamming distance threshold for near-duplicate detection.",
    show_default="8",
)
@workers_option()
@click.option(
    "--clear",
    is_flag=True,
    help="Restore all deduplicated images back to the source folder.",
)
@dry_run_option(help="Show what would be deduplicated without moving anything.")
@click.option(
    "--prefer-upscaled",
    is_flag=True,
    help="Prefer upscaled images over originals when deduplicating.",
)
def cmd(
    working_dir: Path | None,
    from_dir: str | None,
    to: str | None,
    threshold: int | None,
    workers: int | None,
    clear: bool,
    dry_run: bool,
    prefer_upscaled: bool,
) -> None:
    """Deduplicate images by perceptual hash similarity.

    Groups images by phash hamming distance and keeps the best image
    from each duplicate group. By default, original (non-upscaled)
    images are preferred; use --prefer-upscaled to reverse this. Within
    each preference tier, the winner is chosen by resolution
    (width x height), then file size, then blur sharpness. Losers are
    moved to a duplicated/ subdirectory within the source folder
    (configurable with --to).

    Requires phash sidecar data from ``dtst analyze --metrics phash``. Blur
    scores (from ``dtst analyze --metrics blur``) are used as a tiebreaker
    when available.

    \b
    Examples:
      dtst dedup -d ./project --from faces
      dtst dedup -d ./project --from faces --threshold 4
      dtst dedup -d ./project --from faces --to my-dupes
      dtst dedup config.yaml --dry-run
      dtst dedup -d ./project --from faces --clear
    """
    if from_dir is None:
        raise click.ClickException("--from is required (or set 'dedup.from' in config)")

    apply_working_dir(working_dir)
    try:
        result = core_dedup(
            from_dir=from_dir,
            to=to or "duplicated",
            threshold=threshold if threshold is not None else 8,
            workers=workers,
            clear=clear,
            dry_run=dry_run,
            prefer_upscaled=prefer_upscaled,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if result.mode == "restore":
        if result.message:
            click.echo(result.message)
            return
        if result.dry_run:
            click.echo(
                f"\nDry run -- would restore {result.total_losers:,} images to {result.source_dir}"
            )
            return
        click.echo("\nRestore complete!")
        click.echo(f"  Restored: {result.restored:,}")
        click.echo(f"  Source: {result.source_dir}")
        return

    if result.mode == "noop":
        if result.message:
            click.echo(
                f"{result.message} ({result.elapsed:.1f}s)"
                if "No duplicates found" in (result.message or "")
                else result.message
            )
        return

    if result.dry_run:
        click.echo(
            f"\nDry run -- would keep {result.kept:,}, move {result.total_losers:,} duplicates from {result.groups:,} groups"
        )
        for name, reason in result.losers_preview:
            click.echo(f"  {name} ({reason})")
        if result.total_losers > len(result.losers_preview):
            click.echo(
                f"  ... and {result.total_losers - len(result.losers_preview):,} more"
            )
        return

    click.echo("\nDedup complete!")
    click.echo(f"  Groups: {result.groups:,}")
    click.echo(f"  Kept: {result.kept:,}")
    click.echo(f"  Moved: {result.moved:,}")
    if result.errors > 0:
        click.echo(f"  Errors: {result.errors:,}")
    click.echo(f"  Source: {result.source_dir}")
    click.echo(f"  Duplicates: {result.duplicated_dir}")
    click.echo(f"  Time: {result.elapsed:.1f}s")
