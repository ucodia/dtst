"""Library-layer implementation of ``dtst validate``."""

from __future__ import annotations

import logging
import struct
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dtst.errors import InputError
from dtst.executor import run_pool
from dtst.files import gather_images, resolve_workers
from dtst.results import ValidateResult

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


def validate(
    *,
    from_dirs: str,
    square: bool = False,
    workers: int | None = None,
    progress: bool = True,
) -> ValidateResult:
    """Check that images in a folder share dimensions, mode, and optionally squareness.

    Returns a :class:`ValidateResult` — inspect ``.passed`` for overall
    pass/fail or the individual counters for detail.  Does not raise on
    failed checks; only raises :class:`InputError` when inputs are
    missing or unreadable.
    """
    if not from_dirs:
        raise InputError("from_dirs is required")

    num_workers = resolve_workers(workers)
    _input_dirs, all_images = gather_images(from_dirs)
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
        max_workers=num_workers,
        desc="Validating",
        unit="image",
        on_result=handle,
        progress=progress,
    )

    return ValidateResult(
        total=len(all_images),
        dim_counts=dict(dim_counts),
        mode_counts=dict(mode_counts),
        non_square=non_square,
        total_png=total_png,
        compressed_png=compressed_png,
        failed=counts.get("fail", 0),
        square_checked=square,
        elapsed=time.monotonic() - t0,
    )
