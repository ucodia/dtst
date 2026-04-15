"""Click wrapper for ``dtst upscale`` — delegates to :mod:`dtst.core.upscale`."""

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


@click.command("upscale")
@config_argument
@working_dir_option()
@from_dirs_option()
@to_dir_option()
@click.option(
    "--scale",
    "-s",
    type=click.Choice(["2", "4"]),
    default=None,
    help="Upscale factor. Ignored when --model is provided (default: 4).",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Model preset name or path to a .pth file. Overrides --scale.",
)
@click.option(
    "--tile-size",
    "-t",
    type=int,
    default=None,
    help="Tile size in pixels for processing; 0 disables tiling (default: 512).",
)
@click.option(
    "--tile-pad",
    type=int,
    default=None,
    help="Overlap padding between tiles in pixels (default: 32).",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["jpg", "png", "webp"]),
    default=None,
    help="Output image format. Default preserves the source format.",
)
@click.option(
    "--quality",
    "-q",
    type=int,
    default=None,
    help="JPEG/WebP output quality, 1-100 (default: 95).",
)
@click.option(
    "--denoise",
    "-n",
    type=float,
    default=None,
    help="Denoise strength 0.0-1.0. Lower preserves more texture. Only available at 4x.",
)
@workers_option(help="Number of threads for image preloading (default: 4).")
@dry_run_option(help="Preview what would be written without processing.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    scale: str | None,
    model: str | None,
    tile_size: int | None,
    tile_pad: int | None,
    fmt: str | None,
    quality: int | None,
    denoise: float | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Upscale images using AI super-resolution models.

    Reads images from one or more source folders and writes upscaled
    copies to a destination folder. Uses spandrel to load PyTorch
    super-resolution models (Real-ESRGAN, SwinIR, HAT, etc.).

    By default uses a 4x Real-ESRGAN model. Use --scale to choose
    between 2x and 4x upscaling, or --model to provide a custom
    .pth weights file (scale is auto-detected from the model).

    Use --denoise to control how much denoising is applied (4x only).
    0.0 preserves the most texture, 1.0 applies full denoising.
    This activates a lighter general-purpose model with controllable
    denoise strength via weight interpolation.

    Large images are processed in tiles to avoid GPU memory issues.
    Adjust --tile-size to control memory usage (smaller = less VRAM).

    \b
    Examples:
        dtst upscale -d ./project --from faces --to upscaled
        dtst upscale -d ./project --from faces --to upscaled --scale 2
        dtst upscale -d ./project --from faces --to upscaled --denoise 0.5
        dtst upscale -d ./project --from faces --to upscaled --model ./custom.pth
        dtst upscale config.yaml --dry-run
    """
    if not from_dirs:
        raise click.ClickException(
            "--from is required (or set 'upscale.from' in config)"
        )
    if not to:
        raise click.ClickException("--to is required (or set 'upscale.to' in config)")

    apply_working_dir(working_dir)
    from dtst.core.upscale import upscale as core_upscale

    try:
        result = core_upscale(
            from_dirs=from_dirs,
            to=to,
            scale=int(scale) if scale is not None else 4,
            model=model,
            tile_size=tile_size if tile_size is not None else 512,
            tile_pad=tile_pad if tile_pad is not None else 32,
            fmt=fmt,
            quality=quality if quality is not None else 95,
            denoise=denoise,
            workers=workers if workers is not None else 4,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if dry_run:
        click.echo(f"\nDry run -- would upscale {result.total_images:,} images")
        click.echo(f"  Model: {result.model_label}")
        click.echo(f"  Source: {result.from_label}")
        click.echo(f"  Output: {result.output_dir}")
        return

    click.echo("\nUpscale complete!")
    click.echo(f"  Upscaled: {result.ok:,}")
    click.echo(f"  Failed: {result.failed:,}")
    click.echo(f"  Scale: {result.scale}x ({result.model_label})")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
