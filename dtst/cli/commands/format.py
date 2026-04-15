"""Click wrapper for ``dtst format`` — delegates to :mod:`dtst.core.format`."""

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


@click.command("format")
@config_argument
@working_dir_option()
@from_dirs_option()
@to_dir_option()
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["jpg", "png", "webp"]),
    default=None,
    help="Output image format. When omitted the source format is preserved.",
)
@click.option(
    "--quality",
    "-q",
    type=int,
    default=None,
    help="JPEG/WebP output quality, 1-100 (default: 95). Ignored for PNG.",
)
@click.option(
    "--compress-level",
    type=int,
    default=None,
    help="PNG compression level, 0 (none) to 9 (max). Default: 0. Ignored for JPEG/WebP.",
)
@click.option(
    "--strip-metadata",
    is_flag=True,
    default=False,
    help="Remove EXIF data and embedded ICC profiles from output images.",
)
@click.option(
    "--channels",
    "-c",
    type=click.Choice(["rgb", "grayscale"]),
    default=None,
    help="Enforce channel mode. 'rgb' converts to 3-channel RGB (drops alpha). 'grayscale' converts to single-channel.",
)
@click.option(
    "--background",
    type=str,
    default=None,
    help="Background color for alpha compositing (default: white). Accepts named colors or hex codes.",
)
@workers_option()
@dry_run_option(help="Preview what would be written without creating files.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    fmt: str | None,
    quality: int | None,
    compress_level: int | None,
    strip_metadata: bool,
    channels: str | None,
    background: str | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Convert and normalize image formats, channels, and metadata.

    Reads images from source folders and writes converted copies to a
    destination folder.  Can change format (jpg/png/webp), enforce
    channel mode (rgb/grayscale), and strip EXIF metadata.

    When --format is omitted the source format is preserved, but other
    transformations (--channels, --strip-metadata) still apply.

    \b
    Examples:
        dtst format -d ./project --from raw --to formatted -f jpg -q 90
        dtst format -d ./project --from raw --to clean --strip-metadata --channels rgb
        dtst format -d ./project --from raw --to gray --channels grayscale
        dtst format config.yaml --dry-run
    """
    if not from_dirs:
        raise click.ClickException("--from is required (or set 'from' in config)")
    if not to:
        raise click.ClickException("--to is required (or set 'to' in config)")

    apply_working_dir(working_dir)
    from dtst.core.format import format as core_format

    try:
        result = core_format(
            from_dirs=from_dirs,
            to=to,
            fmt=fmt,
            quality=quality if quality is not None else 95,
            compress_level=compress_level if compress_level is not None else 0,
            strip_metadata=strip_metadata,
            channels=channels,
            background=background or "white",
            workers=workers,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if result.dry_run:
        click.echo(f"\nDry run -- would format {result.total_images:,} images")
        if result.fmt:
            click.echo(f"  Format: {result.fmt}")
        if result.channels:
            click.echo(f"  Channels: {result.channels}")
        if result.strip_metadata:
            click.echo("  Strip metadata: yes")
        if result.fmt in ("jpg", "webp"):
            click.echo(f"  Quality: {result.quality}")
        if result.fmt == "png" or result.fmt is None:
            click.echo(f"  Compress level: {result.compress_level}")
        click.echo(f"  Output: {result.output_dir}")
        return

    click.echo("\nFormat complete!")
    click.echo(f"  Converted: {result.converted:,}")
    click.echo(f"  Failed: {result.failed:,}")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
