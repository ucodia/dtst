"""Library-layer implementation of ``dtst frame``."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from dtst.errors import InputError
from dtst.executor import run_pool
from dtst.files import (
    build_save_kwargs,
    find_images,
    resolve_dirs,
    resolve_workers,
)
from dtst.results import FrameResult
from dtst.sidecar import EXCLUDE_METRICS_AND_CLASSES, copy_sidecar

logger = logging.getLogger(__name__)


def _parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_str!r}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _gravity_offset(
    gravity: str, canvas_w: int, canvas_h: int, img_w: int, img_h: int
) -> tuple[int, int]:
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


def frame(
    *,
    from_dirs: str,
    to: str,
    width: int | None = None,
    height: int | None = None,
    mode: str = "crop",
    gravity: str = "center",
    fill: str = "color",
    fill_color: str = "#000000",
    quality: int = 95,
    compress_level: int = 0,
    workers: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> FrameResult:
    """Resize images to a target width and/or height."""
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not to:
        raise InputError("to is required")
    if width is None and height is None:
        raise InputError("At least one of width or height is required")

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    input_dirs = resolve_dirs(dirs_list)
    output_dir = Path(to).expanduser().resolve()

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise InputError(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: {', '.join(missing)}"
        )

    images: list[Path] = []
    for input_dir in input_dirs:
        found = find_images(input_dir)
        logger.info("Found %d images in %s", len(found), input_dir)
        images.extend(found)

    if not images:
        raise InputError(f"No images found in: {', '.join(str(d) for d in input_dirs)}")

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
        return FrameResult(
            resized=0,
            failed=0,
            output_dir=output_dir,
            dry_run=True,
            width=width,
            height=height,
            mode=mode,
            gravity=gravity,
            fill=fill,
            fill_color=fill_color,
            total_images=len(images),
            elapsed=0.0,
        )

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

    def handle(result, work_item):
        status, name, error = result
        if status == "ok":
            copy_sidecar(
                Path(work_item[0]),
                output_dir / name,
                exclude=EXCLUDE_METRICS_AND_CLASSES,
            )
            return "ok"
        logger.error("Failed to resize %s: %s", name, error)
        return "fail"

    counts = run_pool(
        ProcessPoolExecutor,
        _resize_image,
        work,
        max_workers=num_workers,
        desc="Resizing",
        unit="image",
        on_result=handle,
        postfix_keys=("ok", "fail"),
        progress=progress,
    )

    return FrameResult(
        resized=counts.get("ok", 0),
        failed=counts.get("fail", 0),
        output_dir=output_dir,
        dry_run=False,
        width=width,
        height=height,
        mode=mode,
        gravity=gravity,
        fill=fill,
        fill_color=fill_color,
        total_images=len(images),
        elapsed=time.monotonic() - start_time,
    )
