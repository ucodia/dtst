"""Click wrapper for ``dtst frame`` — delegates to :mod:`dtst.core.frame`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    FRAME_FILLS,
    FRAME_GRAVITIES,
    FRAME_MODES,
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


@click.command("frame")
@config_argument
@working_dir_option()
@from_dirs_option()
@to_dir_option()
@click.option(
    "--width",
    "-W",
    type=int,
    default=None,
    help="Target width in pixels. If --height is omitted, aspect ratio is preserved.",
)
@click.option(
    "--height",
    "-H",
    type=int,
    default=None,
    help="Target height in pixels. If --width is omitted, aspect ratio is preserved.",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(FRAME_MODES, case_sensitive=False),
    default=None,
    help="Resize mode when both width and height are given (default: crop).",
)
@click.option(
    "--gravity",
    "-g",
    type=click.Choice(FRAME_GRAVITIES, case_sensitive=False),
    default=None,
    help="Anchor position for crop (part to keep) or pad (where to place image). Default: center.",
)
@click.option(
    "--fill",
    "-f",
    type=click.Choice(FRAME_FILLS, case_sensitive=False),
    default=None,
    help="Fill strategy for pad mode: color, edge, reflect, or blur (default: color).",
)
@click.option(
    "--fill-color",
    type=str,
    default=None,
    help="Hex color for pad fill when --fill=color (default: #000000).",
)
@click.option(
    "--trim",
    is_flag=True,
    default=False,
    help=(
        "Crop each image to its content bounding box before framing. Assumes a "
        "uniform background reachable from the image corners."
    ),
)
@click.option(
    "--trim-tolerance",
    type=click.IntRange(0, 255),
    default=None,
    help=(
        "How far a pixel may stray from the detected background and still count "
        "as background; on images with alpha it is the opacity threshold above "
        "which a pixel counts as content (default: 8)."
    ),
)
@click.option(
    "--margin",
    type=str,
    default=None,
    help=(
        "Margin around content inside the target canvas: pixels ('48') or a "
        "percentage of the shorter target side ('5%'). Requires --mode pad and "
        "both --width and --height. The margin band is filled according to "
        "--fill, so only --fill color leaves it empty."
    ),
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
@workers_option()
@dry_run_option(help="Preview what would be written without creating files.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    width: int | None,
    height: int | None,
    mode: str | None,
    gravity: str | None,
    fill: str | None,
    fill_color: str | None,
    trim: bool,
    trim_tolerance: int | None,
    margin: str | None,
    quality: int | None,
    compress_level: int | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Resize images to a target width and/or height.

    Reads images from one or more source folders and writes resized
    copies to a destination folder. Uses Lanczos resampling for
    high-quality downscaling.

    When both --width and --height are given, the --mode option controls
    how aspect ratio differences are handled:

    \b
      stretch  Resize to exact dimensions, distorting if needed.
      crop     Scale to cover the target area, then trim excess (default).
      pad      Scale to fit within the target area, then fill the rest.

    When only one dimension is given, the other is computed proportionally
    and --mode is ignored.

    `--trim` crops each image to its content bounding box first, using the
    alpha channel when there is one and otherwise a background colour taken
    from the corner pixels. Combined with `--margin`, this gives every image
    in a dataset the same margin around its subject regardless of how the
    source was framed — which is what StyleGAN-style training wants.

    \b
    Examples:

        dtst frame -d ./project --from faces --to resized -W 512 -H 512
        dtst frame -d ./project --from faces --to resized -W 512 -H 512 --mode pad --fill blur
        dtst frame -d ./project --from faces --to resized -W 512 -H 512 --mode crop --gravity top
        dtst frame -d ./project --from faces --to resized --width 512
        dtst frame -d ./project --from raw --to out -W 1024 -H 1024 --mode pad --trim --margin 5%
        dtst frame config.yaml --dry-run
    """
    if not from_dirs:
        raise click.ClickException("--from is required (or set 'frame.from' in config)")
    if not to:
        raise click.ClickException("--to is required (or set 'frame.to' in config)")
    if width is None and height is None:
        raise click.ClickException("At least one of --width or --height is required")
    if margin is not None:
        if width is None or height is None:
            raise click.ClickException("--margin requires both --width and --height")
        if (mode or "crop") != "pad":
            raise click.ClickException("--margin requires --mode pad")

    apply_working_dir(working_dir)
    from dtst.core.frame import frame as core_frame

    try:
        result = core_frame(
            from_dirs=from_dirs,
            to=to,
            width=width,
            height=height,
            mode=mode or "crop",
            gravity=gravity or "center",
            fill=fill or "color",
            fill_color=fill_color or "#000000",
            trim=trim,
            trim_tolerance=trim_tolerance if trim_tolerance is not None else 8,
            margin=margin,
            quality=quality if quality is not None else 95,
            compress_level=compress_level if compress_level is not None else 0,
            workers=workers,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    width_label = str(result.width) if result.width is not None else "auto"
    height_label = str(result.height) if result.height is not None else "auto"
    both_dims = result.width is not None and result.height is not None

    if result.dry_run:
        click.echo(f"\nDry run -- would resize {result.total_images:,} images")
        click.echo(f"  Target: {width_label} x {height_label}")
        if both_dims:
            click.echo(f"  Mode: {result.mode}")
            if result.mode in ("crop", "pad"):
                click.echo(f"  Gravity: {result.gravity}")
            if result.mode == "pad":
                fill_label = result.fill
                if result.fill == "color":
                    fill_label += f" ({result.fill_color})"
                click.echo(f"  Fill: {fill_label}")
        if result.trim:
            click.echo("  Trim: content bbox")
        if margin is not None:
            click.echo(f"  Margin: {result.margin}px")
        click.echo(f"  Output: {result.output_dir}")
        return

    click.echo("\nFrame complete!")
    click.echo(f"  Resized: {result.resized:,}")
    click.echo(f"  Failed: {result.failed:,}")
    click.echo(f"  Target: {width_label} x {height_label}")
    if both_dims:
        click.echo(f"  Mode: {result.mode}")
    if result.trim:
        click.echo("  Trim: content bbox")
    if margin is not None:
        click.echo(f"  Margin: {result.margin}px")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
