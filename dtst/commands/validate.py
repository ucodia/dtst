from __future__ import annotations

import logging
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

from dtst.config import (
    config_argument,
    from_dirs_option,
    working_dir_option,
    workers_option,
)
from dtst.executor import run_pool
from dtst.files import format_elapsed, gather_images, resolve_workers

logger = logging.getLogger(__name__)


def _png_flevel(path_str: str) -> int | None:
    """Return zlib FLEVEL (0-3) from the first IDAT chunk of a PNG file."""
    try:
        with open(path_str, "rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            while True:
                header = f.read(8)
                if len(header) < 8:
                    return None
                length = struct.unpack(">I", header[:4])[0]
                chunk_type = header[4:8]
                if chunk_type == b"IDAT":
                    zlib_header = f.read(2)
                    if len(zlib_header) < 2:
                        return None
                    return (zlib_header[1] >> 6) & 3
                f.seek(length + 4, 1)  # skip data + CRC
    except Exception:
        return None


def _check_image(
    args: tuple,
) -> tuple[str, int, int, str, bool, int | None, str | None]:
    """Return (filename, width, height, mode, is_png, png_flevel, error)."""
    (path_str,) = args
    try:
        from PIL import Image

        with Image.open(path_str) as img:
            w, h = img.size
            mode = img.mode

        is_png = Path(path_str).suffix.lower() == ".png"
        flevel = _png_flevel(path_str) if is_png else None
        return (Path(path_str).name, w, h, mode, is_png, flevel, None)
    except Exception as e:
        return (Path(path_str).name, 0, 0, "", False, None, str(e))


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
def cmd(from_dirs, working_dir, square, workers):
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
    workers = resolve_workers(workers)
    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'validate.from' in config)"
        )

    _working, _input_dirs, all_images = gather_images(working_dir, from_dirs)
    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]

    logger.info("Validating %d images from %s", len(all_images), ", ".join(dirs_list))

    t0 = time.monotonic()
    dim_counts: Counter[tuple[int, int]] = Counter()
    mode_counts: Counter[str] = Counter()
    non_square = 0
    compressed_png = 0
    total_png = 0

    work = [(str(img),) for img in all_images]

    def handle(result, _work_item):
        nonlocal non_square, total_png, compressed_png
        name, w, h, mode, is_png, flevel, error = result
        if error is not None:
            logger.error("Failed to read %s: %s", name, error)
            return "fail"
        dim_counts[(w, h)] += 1
        mode_counts[mode] += 1
        if square and w != h:
            non_square += 1
        if is_png:
            total_png += 1
            if flevel is not None and flevel > 0:
                compressed_png += 1
        return "ok"

    counts = run_pool(
        ProcessPoolExecutor,
        _check_image,
        work,
        max_workers=workers,
        desc="Validating",
        unit="image",
        on_result=handle,
    )
    failed = counts.get("fail", 0)

    elapsed = time.monotonic() - t0
    total = len(all_images)
    checks_passed = True

    click.echo(f"\nValidated {total:,} images ({format_elapsed(elapsed)})")
    click.echo("")

    # Dimensions check
    if len(dim_counts) == 1:
        (w, h), _ = dim_counts.most_common(1)[0]
        click.echo(f"  Dimensions: PASS (all {w}x{h})")
    else:
        checks_passed = False
        click.echo(f"  Dimensions: FAIL ({len(dim_counts)} unique sizes)")
        for (w, h), count in dim_counts.most_common():
            click.echo(f"    {w}x{h}: {count:,} images")

    # Channels check
    if len(mode_counts) == 1:
        mode, _ = mode_counts.most_common(1)[0]
        click.echo(f"  Channels:   PASS (all {mode})")
    else:
        checks_passed = False
        click.echo(f"  Channels:   FAIL ({len(mode_counts)} unique modes)")
        for mode, count in mode_counts.most_common():
            click.echo(f"    {mode}: {count:,} images")

    # Square check
    if square:
        if non_square == 0:
            click.echo("  Square:     PASS")
        else:
            checks_passed = False
            click.echo(f"  Square:     FAIL ({non_square:,} non-square images)")

    # PNG compression warning
    if total_png > 0:
        if compressed_png == 0:
            click.echo(
                f"  PNG comp:   OK (all {total_png:,} PNGs at compression level 0)"
            )
        else:
            click.echo(
                f"  PNG comp:   WARN ({compressed_png:,}/{total_png:,} PNGs are compressed, slower loading)"
            )

    # Errors
    if failed > 0:
        checks_passed = False
        click.echo(f"  Errors:     {failed:,} images could not be read")

    if not checks_passed:
        sys.exit(1)
