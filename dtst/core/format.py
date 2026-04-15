"""Library-layer implementation of ``dtst format``."""

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
from dtst.results import FormatResult
from dtst.sidecar import EXCLUDE_METRICS, copy_sidecar

logger = logging.getLogger(__name__)


def _format_image(args: tuple) -> tuple[str, str, str | None]:
    (
        input_path_s,
        output_dir_s,
        fmt,
        quality,
        compress_level,
        strip_metadata,
        channels,
        background,
    ) = args
    input_path = Path(input_path_s)
    output_dir = Path(output_dir_s)

    try:
        img = Image.open(input_path)

        if fmt is not None:
            out_name = input_path.stem + "." + fmt
        else:
            out_name = input_path.name
        out_suffix = Path(out_name).suffix.lower()

        if channels == "rgb":
            if img.mode in ("RGBA", "LA", "PA"):
                bg = Image.new("RGBA", img.size, background)
                bg.paste(img, mask=img.split()[-1])
                img = bg.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
        elif channels == "grayscale":
            if img.mode in ("RGBA", "LA", "PA"):
                bg = Image.new("RGBA", img.size, background)
                bg.paste(img, mask=img.split()[-1])
                img = bg.convert("L")
            else:
                img = img.convert("L")

        if out_suffix in (".jpg", ".jpeg") and img.mode in ("RGBA", "LA", "PA"):
            bg = Image.new("RGBA", img.size, background)
            bg.paste(img, mask=img.split()[-1])
            img = bg.convert("RGB")
        elif out_suffix in (".jpg", ".jpeg") and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        save_kwargs: dict = {}

        if not strip_metadata:
            exif = img.info.get("exif")
            if exif:
                save_kwargs["exif"] = exif
            icc = img.info.get("icc_profile")
            if icc:
                save_kwargs["icc_profile"] = icc

        save_kwargs.update(
            build_save_kwargs(
                Path(out_name), quality=quality, compress_level=compress_level
            )
        )

        if fmt is not None:
            pil_format = "JPEG" if fmt == "jpg" else fmt.upper()
            save_kwargs["format"] = pil_format

        img.save(output_dir / out_name, **save_kwargs)
        img.close()
        return "ok", out_name, None

    except Exception as e:
        return "failed", input_path.name, str(e)


def format(
    *,
    from_dirs: str,
    to: str,
    fmt: str | None = None,
    quality: int = 95,
    compress_level: int = 0,
    strip_metadata: bool = False,
    channels: str | None = None,
    background: str = "white",
    workers: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> FormatResult:
    """Convert and normalize image formats, channels, and metadata."""
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not to:
        raise InputError("to is required")

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

    from_label = ", ".join(str(d) for d in input_dirs)
    num_workers = resolve_workers(workers)

    ops: list[str] = []
    if fmt:
        ops.append(f"format={fmt}")
    if channels:
        ops.append(f"channels={channels}")
    if strip_metadata:
        ops.append("strip-metadata")

    logger.info(
        "Formatting %d images from [%s] → %s (%s, workers=%d)",
        len(images),
        from_label,
        output_dir,
        ", ".join(ops) if ops else "copy",
        num_workers,
    )

    if dry_run:
        return FormatResult(
            converted=0,
            failed=0,
            output_dir=output_dir,
            dry_run=True,
            fmt=fmt,
            channels=channels,
            strip_metadata=strip_metadata,
            quality=quality,
            compress_level=compress_level,
            total_images=len(images),
            elapsed=0.0,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (
            str(img_path),
            str(output_dir),
            fmt,
            quality,
            compress_level,
            strip_metadata,
            channels,
            background,
        )
        for img_path in images
    ]

    start_time = time.monotonic()

    def handle(result, work_item):
        status, name, error = result
        if status == "ok":
            copy_sidecar(Path(work_item[0]), output_dir / name, exclude=EXCLUDE_METRICS)
            return "ok"
        logger.error("Failed to format %s: %s", name, error)
        return "fail"

    counts = run_pool(
        ProcessPoolExecutor,
        _format_image,
        work,
        max_workers=num_workers,
        desc="Formatting",
        unit="image",
        on_result=handle,
        postfix_keys=("ok", "fail"),
        progress=progress,
    )

    return FormatResult(
        converted=counts.get("ok", 0),
        failed=counts.get("fail", 0),
        output_dir=output_dir,
        dry_run=False,
        fmt=fmt,
        channels=channels,
        strip_metadata=strip_metadata,
        quality=quality,
        compress_level=compress_level,
        total_images=len(images),
        elapsed=time.monotonic() - start_time,
    )
