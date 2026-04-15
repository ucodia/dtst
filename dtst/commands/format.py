import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import (
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
from dtst.sidecar import EXCLUDE_METRICS, copy_sidecar

logger = logging.getLogger(__name__)


def _format_image(args: tuple) -> tuple[str, str, str | None]:
    """Top-level worker for ProcessPoolExecutor.

    Returns (status, output_filename, error_message).
    """
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

        # Determine output name and suffix
        if fmt is not None:
            out_name = input_path.stem + "." + fmt
        else:
            out_name = input_path.name
        out_suffix = Path(out_name).suffix.lower()

        # Convert channels
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

        # JPEG requires RGB (or L) — drop alpha if still present
        if out_suffix in (".jpg", ".jpeg") and img.mode in ("RGBA", "LA", "PA"):
            bg = Image.new("RGBA", img.size, background)
            bg.paste(img, mask=img.split()[-1])
            img = bg.convert("RGB")
        elif out_suffix in (".jpg", ".jpeg") and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Build save kwargs
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


@click.command("format")
@config_argument
@working_dir_option(
    help="Working directory containing source folders and where output is written (default: .)."
)
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

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    working = (working_dir or Path(".")).resolve()
    quality = quality if quality is not None else 95
    compress_level = compress_level if compress_level is not None else 0
    background = background or "white"

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
        click.echo(f"\nDry run -- would format {len(images):,} images")
        if fmt:
            click.echo(f"  Format: {fmt}")
        if channels:
            click.echo(f"  Channels: {channels}")
        if strip_metadata:
            click.echo("  Strip metadata: yes")
        if fmt in ("jpg", "webp"):
            click.echo(f"  Quality: {quality}")
        if fmt == "png" or fmt is None:
            click.echo(f"  Compress level: {compress_level}")
        click.echo(f"  Output: {output_dir}")
        return

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
    ok_count = 0
    failed_count = 0

    with logging_redirect_tqdm():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_format_image, w): w for w in work}
            with tqdm(total=len(futures), desc="Formatting", unit="image") as pbar:
                try:
                    for future in as_completed(futures):
                        status, name, error = future.result()
                        if status == "ok":
                            ok_count += 1
                            src_path = Path(futures[future][0])
                            copy_sidecar(
                                src_path, output_dir / name, exclude=EXCLUDE_METRICS
                            )
                        else:
                            failed_count += 1
                            logger.error("Failed to format %s: %s", name, error)
                        pbar.set_postfix(ok=ok_count, fail=failed_count)
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    elapsed = time.monotonic() - start_time

    click.echo("\nFormat complete!")
    click.echo(f"  Converted: {ok_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Time: {format_elapsed(elapsed)}")
    click.echo(f"  Output: {output_dir}")
