"""Click wrapper for ``dtst extract-faces`` — delegates to :mod:`dtst.core.extract_faces`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    config_argument,
    from_dirs_option,
    to_dir_option,
    working_dir_option,
    workers_option,
)
from dtst.core.extract_faces import extract_faces as core_extract_faces
from dtst.errors import DtstError
from dtst.files import format_elapsed


@click.command("extract-faces")
@config_argument
@working_dir_option(
    help="Working directory containing source folders and where output is written (default: .)."
)
@from_dirs_option()
@to_dir_option()
@click.option(
    "--max-size",
    "-M",
    type=int,
    default=None,
    help="Maximum side length in pixels; faces smaller than this are kept at natural size (default: no limit).",
)
@click.option(
    "--engine",
    "-e",
    type=click.Choice(["mediapipe", "dlib"], case_sensitive=False),
    default=None,
    help="Face detection engine (default: mediapipe).",
)
@click.option(
    "--max-faces",
    "-m",
    type=int,
    default=None,
    help="Max faces to extract per image (default: 1).",
)
@workers_option()
@click.option(
    "--padding/--no-padding",
    default=None,
    help="Enable/disable reflective padding on crops (default: enabled).",
)
@click.option(
    "--skip-partial",
    is_flag=True,
    help="Skip faces whose crop extends beyond the image boundary instead of padding them.",
)
@click.option(
    "--refine-landmarks",
    is_flag=True,
    help="Enable MediaPipe refined landmarks (478 vs 468).",
)
@click.option("--debug", is_flag=True, help="Overlay landmark points on output images.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    max_size: int | None,
    engine: str | None,
    max_faces: int | None,
    workers: int | None,
    padding: bool | None,
    skip_partial: bool,
    refine_landmarks: bool,
    debug: bool,
) -> None:
    """Extract aligned face crops from images.

    Detects faces in each image using MediaPipe (default) or dlib,
    then produces an aligned and cropped face image for each detection.
    The alignment normalises eye and mouth positions for consistent
    face crops.

    Reads images from one or more source folders within the working
    directory and writes face crops to a destination folder. Multiple
    source folders can be specified as a comma-separated list with
    --from.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst extract-faces config.yaml
        dtst extract-faces config.yaml --engine dlib --max-size 512
        dtst extract-faces -d ./crowd --from raw --to faces
        dtst extract-faces -d ./crowd --from raw,extra --to faces
        dtst extract-faces config.yaml --max-faces 3 --no-padding
    """
    if not from_dirs:
        raise click.ClickException(
            "--from is required (or set 'extract_faces.from' in config)"
        )
    if not to:
        raise click.ClickException(
            "--to is required (or set 'extract_faces.to' in config)"
        )

    try:
        result = core_extract_faces(
            working_dir=working_dir,
            from_dirs=from_dirs,
            to=to,
            max_size=max_size,
            engine=engine or "mediapipe",
            max_faces=max_faces if max_faces is not None else 1,
            workers=workers,
            padding=True if padding is None else padding,
            skip_partial=skip_partial,
            refine_landmarks=refine_landmarks,
            debug=debug,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    click.echo("\nExtract faces complete!")
    click.echo(f"  Processed: {result.processed:,}")
    click.echo(f"  Faces extracted: {result.faces_extracted:,}")
    click.echo(f"  No faces detected: {result.no_faces:,}")
    click.echo(f"  Failed: {result.failed:,}")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
