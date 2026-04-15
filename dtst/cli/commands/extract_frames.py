"""Click wrapper for ``dtst extract-frames`` — delegates to :mod:`dtst.core.extract_frames`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    VALID_FRAME_FORMATS,
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


@click.command("extract-frames")
@config_argument
@working_dir_option()
@from_dirs_option()
@to_dir_option()
@click.option(
    "--keyframes",
    "-k",
    type=float,
    default=None,
    help="Minimum interval in seconds between extracted keyframes. Only I-frames are considered; frames closer together than this value are skipped (default: 10).",
)
@click.option(
    "--format",
    "-F",
    "fmt",
    type=click.Choice(sorted(VALID_FRAME_FORMATS), case_sensitive=False),
    default=None,
    help="Output image format (default: jpg).",
)
@workers_option()
@dry_run_option(help="Preview what would be done without extracting frames.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    keyframes: float | None,
    fmt: str | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Extract keyframes from video files using ffmpeg.

    Reads video files from one or more source folders and extracts
    keyframes (I-frames) to a destination folder. Each video produces
    a set of numbered images named as
    ``{video_stem}_{frame_number}.{format}``.

    Only I-frames are decoded, which avoids interpolated or blurry
    frames and produces the sharpest possible output. The --keyframes
    option sets the minimum interval between extracted frames: with
    the default of 10, at most one keyframe every 10 seconds is kept.
    Lower values produce more frames, higher values produce fewer.

    Supported video formats: .mp4, .mkv, .avi, .mov, .webm, .flv,
    .wmv, .m4v.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst extract-frames -d ./project --from videos --to frames
        dtst extract-frames -d ./project --from videos --to frames --keyframes 5
        dtst extract-frames -d ./project --from videos --to frames --keyframes 30 --format png
        dtst extract-frames config.yaml
        dtst extract-frames config.yaml --keyframes 20 --dry-run
    """
    if keyframes is not None and keyframes <= 0:
        raise click.ClickException("--keyframes must be a positive number")
    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'extract_frames.from' in config)"
        )
    if to is None:
        raise click.ClickException(
            "--to is required (or set 'extract_frames.to' in config)"
        )

    apply_working_dir(working_dir)
    from dtst.core.extract_frames import extract_frames as core_extract_frames

    try:
        result = core_extract_frames(
            from_dirs=from_dirs,
            to=to,
            keyframes=keyframes if keyframes is not None else 10.0,
            fmt=fmt or "jpg",
            workers=workers,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if result.dry_run:
        click.echo(
            f"\nDry run -- would extract keyframes from {result.total_videos:,} videos"
        )
        click.echo(f"  Min interval: {result.keyframes}s")
        click.echo(f"  Format: {result.fmt}")
        click.echo(f"  Output: {result.output_dir}")
        return

    click.echo("\nExtract-frames complete!")
    click.echo(f"  Processed: {result.processed:,} videos")
    click.echo(f"  Frames extracted: {result.frames_extracted:,}")
    click.echo(f"  Skipped (existing): {result.skipped:,}")
    click.echo(f"  Failed: {result.failed:,}")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
