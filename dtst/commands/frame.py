from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import (
    FRAME_FILLS,
    FRAME_GRAVITIES,
    FRAME_MODES,
    config_argument,
    dry_run_option,
    from_dirs_option,
    to_dir_option,
    working_dir_option,
    workers_option,
)
from dtst.files import (
    build_save_kwargs,
    find_images,
    format_elapsed,
    resolve_dirs,
    resolve_workers,
)
from dtst.sidecar import EXCLUDE_METRICS_AND_CLASSES, copy_sidecar

from PIL import Image

logger = logging.getLogger(__name__)


def _parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    """Parse a hex color string like '#FF00AA' or 'FF00AA' into an (R, G, B) tuple."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_str!r}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _gravity_offset(
    gravity: str, canvas_w: int, canvas_h: int, img_w: int, img_h: int
) -> tuple[int, int]:
    """Return (x, y) paste offset for placing an image on a canvas according to gravity."""
    gx = {"left": 0, "center": 0.5, "right": 1}
    gy = {"top": 0, "center": 0.5, "bottom": 1}

    hx = gravity if gravity in gx else "center"
    vy = gravity if gravity in gy else "center"

    x = round((canvas_w - img_w) * gx[hx])
    y = round((canvas_h - img_h) * gy[vy])
    return x, y


def _np_pad_fill(
    resized, canvas_size: tuple[int, int], paste_x: int, paste_y: int, mode: str
) -> Image.Image:
    """Pad a resized image using numpy's pad with the given mode ('edge' or 'reflect')."""
    import numpy as np

    arr = np.array(resized)
    ih, iw = arr.shape[:2]
    cw, ch = canvas_size
    pad_top = paste_y
    pad_bottom = ch - paste_y - ih
    pad_left = paste_x
    pad_right = cw - paste_x - iw
    padded = np.pad(
        arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=mode
    )
    return Image.fromarray(padded)


def _resize_image(args: tuple) -> tuple[str, str, str | None]:
    """Top-level worker function for ProcessPoolExecutor.

    Returns ``(status, filename, error_message)``.
    Status is one of ``"ok"`` or ``"failed"``.
    """
    (
        input_path_s,
        output_dir_s,
        target_width,
        target_height,
        mode,
        gravity,
        fill,
        fill_color,
        quality,
        compress_level,
    ) = args
    input_path = Path(input_path_s)
    output_dir = Path(output_dir_s)
    name = input_path.name

    try:
        img = Image.open(input_path)
        orig_w, orig_h = img.size
        save_kw = build_save_kwargs(
            input_path, quality=quality, compress_level=compress_level
        )

        # Single-dimension: proportional resize (mode irrelevant)
        if target_width is None or target_height is None:
            if target_width is not None:
                ratio = target_width / orig_w
                new_w, new_h = target_width, round(orig_h * ratio)
            else:
                ratio = target_height / orig_h
                new_w, new_h = round(orig_w * ratio), target_height

            if new_w == orig_w and new_h == orig_h:
                img.save(output_dir / name, **save_kw)
            else:
                resized = img.resize((new_w, new_h), Image.LANCZOS)
                resized.save(output_dir / name, **save_kw)
                resized.close()
            img.close()
            return "ok", name, None

        # Both dimensions given: apply mode
        tw, th = target_width, target_height

        if orig_w == tw and orig_h == th:
            img.save(output_dir / name, **save_kw)
            img.close()
            return "ok", name, None

        if mode == "stretch":
            result = img.resize((tw, th), Image.LANCZOS)

        elif mode == "crop":
            scale = max(tw / orig_w, th / orig_h)
            scaled_w = round(orig_w * scale)
            scaled_h = round(orig_h * scale)
            scaled = img.resize((scaled_w, scaled_h), Image.LANCZOS)

            ox, oy = _gravity_offset(gravity, scaled_w, scaled_h, tw, th)
            result = scaled.crop((ox, oy, ox + tw, oy + th))
            scaled.close()

        elif mode == "pad":
            scale = min(tw / orig_w, th / orig_h)
            scaled_w = round(orig_w * scale)
            scaled_h = round(orig_h * scale)
            resized = img.resize((scaled_w, scaled_h), Image.LANCZOS)

            paste_x, paste_y = _gravity_offset(gravity, tw, th, scaled_w, scaled_h)

            if fill == "blur":
                from PIL import ImageFilter

                bg = img.resize((tw, th), Image.LANCZOS)
                bg = bg.filter(ImageFilter.GaussianBlur(radius=30))
                bg.paste(resized, (paste_x, paste_y))
                result = bg
            elif fill in ("edge", "reflect"):
                result = _np_pad_fill(resized, (tw, th), paste_x, paste_y, fill)
            else:
                parsed_color = _parse_hex_color(fill_color)
                canvas = Image.new("RGB", (tw, th), parsed_color)
                canvas.paste(resized, (paste_x, paste_y))
                result = canvas
            resized.close()

        result.save(output_dir / name, **save_kw)
        result.close()
        img.close()
        return "ok", name, None

    except Exception as e:
        return "failed", name, str(e)


@click.command("frame")
@config_argument
@working_dir_option(
    help="Working directory containing source folders and where output is written (default: .)."
)
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

    \b
    Examples:

        dtst frame -d ./project --from faces --to resized -W 512 -H 512
        dtst frame -d ./project --from faces --to resized -W 512 -H 512 --mode pad --fill blur
        dtst frame -d ./project --from faces --to resized -W 512 -H 512 --mode crop --gravity top
        dtst frame -d ./project --from faces --to resized --width 512
        dtst frame config.yaml --dry-run
    """
    if not from_dirs:
        raise click.ClickException("--from is required (or set 'frame.from' in config)")
    if not to:
        raise click.ClickException("--to is required (or set 'frame.to' in config)")

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    working = (working_dir or Path(".")).resolve()
    mode = mode or "crop"
    gravity = gravity or "center"
    fill = fill or "color"
    fill_color = fill_color or "#000000"
    quality = quality if quality is not None else 95
    compress_level = compress_level if compress_level is not None else 0

    if width is None and height is None:
        raise click.ClickException("At least one of --width or --height is required")

    input_dirs = resolve_dirs(working, dirs_list)
    output_dir = working / to

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise click.ClickException(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: {', '.join(missing)}"
        )

    images: list[Path] = []
    for input_dir in input_dirs:
        found = find_images(input_dir)
        logger.info("Found %d images in %s", len(found), input_dir)
        images.extend(found)

    if not images:
        raise click.ClickException(
            f"No images found in: {', '.join(str(d) for d in input_dirs)}"
        )

    width_label = str(width) if width is not None else "auto"
    height_label = str(height) if height is not None else "auto"
    both_dims = width is not None and height is not None
    from_label = ", ".join(str(d) for d in input_dirs)
    num_workers = resolve_workers(workers)

    logger.info(
        "Resizing %d images from [%s] to %sx%s mode=%s (workers=%d)",
        len(images),
        from_label,
        width_label,
        height_label,
        mode if both_dims else "proportional",
        num_workers,
    )

    if dry_run:
        click.echo(f"\nDry run -- would resize {len(images):,} images")
        click.echo(f"  Target: {width_label} x {height_label}")
        if both_dims:
            click.echo(f"  Mode: {mode}")
            if mode in ("crop", "pad"):
                click.echo(f"  Gravity: {gravity}")
            if mode == "pad":
                fill_label = fill
                if fill == "color":
                    fill_label += f" ({fill_color})"
                click.echo(f"  Fill: {fill_label}")
        click.echo(f"  Output: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (
            str(img_path),
            str(output_dir),
            width,
            height,
            mode,
            gravity,
            fill,
            fill_color,
            quality,
            compress_level,
        )
        for img_path in images
    ]

    start_time = time.monotonic()
    ok_count = 0
    failed_count = 0

    with logging_redirect_tqdm():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_resize_image, w): w for w in work}
            with tqdm(total=len(futures), desc="Resizing", unit="image") as pbar:
                try:
                    for future in as_completed(futures):
                        status, name, error = future.result()
                        if status == "ok":
                            ok_count += 1
                            src_path = Path(futures[future][0])
                            copy_sidecar(
                                src_path,
                                output_dir / name,
                                exclude=EXCLUDE_METRICS_AND_CLASSES,
                            )
                        else:
                            failed_count += 1
                            logger.error("Failed to resize %s: %s", name, error)
                        pbar.set_postfix(ok=ok_count, fail=failed_count)
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    elapsed = time.monotonic() - start_time

    click.echo("\nFrame complete!")
    click.echo(f"  Resized: {ok_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Target: {width_label} x {height_label}")
    if both_dims:
        click.echo(f"  Mode: {mode}")
    click.echo(f"  Time: {format_elapsed(elapsed)}")
    click.echo(f"  Output: {output_dir}")
