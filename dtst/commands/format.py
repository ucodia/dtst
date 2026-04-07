import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import click
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import FormatConfig, load_format_config
from dtst.files import find_images, resolve_dirs
from dtst.sidecar import copy_sidecar

logger = logging.getLogger(__name__)


def _format_image(args: tuple) -> tuple[str, str, str | None]:
    """Top-level worker for ProcessPoolExecutor.

    Returns (status, output_filename, error_message).
    """
    (input_path_s, output_dir_s, fmt, quality,
     strip_metadata, channels, background) = args
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

        if out_suffix in (".jpg", ".jpeg"):
            save_kwargs["quality"] = quality
        elif out_suffix == ".webp":
            save_kwargs["quality"] = quality
        elif out_suffix == ".png":
            save_kwargs["compress_level"] = 6

        if fmt is not None:
            pil_format = "JPEG" if fmt == "jpg" else fmt.upper()
            save_kwargs["format"] = pil_format

        img.save(output_dir / out_name, **save_kwargs)
        img.close()
        return "ok", out_name, None

    except Exception as e:
        return "failed", input_path.name, str(e)


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: list[str] | None,
    to: str | None,
    fmt: str | None,
    quality: int | None,
    strip_metadata: bool,
    channels: str | None,
    background: str | None,
) -> FormatConfig:
    if config is not None:
        cfg = load_format_config(config)
    else:
        cfg = FormatConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dirs is not None:
        cfg.from_dirs = from_dirs
    if to is not None:
        cfg.to = to
    if fmt is not None:
        cfg.format = fmt
    if quality is not None:
        cfg.quality = quality
    if strip_metadata:
        cfg.strip_metadata = True
    if channels is not None:
        cfg.channels = channels
    if background is not None:
        cfg.background = background

    if cfg.from_dirs is None:
        raise click.ClickException("--from is required (or set 'from' in config)")
    if cfg.to is None:
        raise click.ClickException("--to is required (or set 'to' in config)")

    return cfg


@click.command("format")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory containing source folders and where output is written (default: .).")
@click.option("--from", "from_dirs", type=str, default=None, help="Comma-separated source folders within the working directory (supports globs).")
@click.option("--to", type=str, default=None, help="Destination folder name within the working directory.")
@click.option("--format", "-f", "fmt", type=click.Choice(["jpg", "png", "webp"]), default=None, help="Output image format. When omitted the source format is preserved.")
@click.option("--quality", "-q", type=int, default=None, help="JPEG/WebP output quality, 1-100 (default: 95). Ignored for PNG.")
@click.option("--strip-metadata", is_flag=True, default=False, help="Remove EXIF data and embedded ICC profiles from output images.")
@click.option("--channels", "-c", type=click.Choice(["rgb", "grayscale"]), default=None, help="Enforce channel mode. 'rgb' converts to 3-channel RGB (drops alpha). 'grayscale' converts to single-channel.")
@click.option("--background", type=str, default=None, help="Background color for alpha compositing (default: white). Accepts named colors or hex codes.")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count).")
@click.option("--dry-run", is_flag=True, help="Preview what would be written without creating files.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    fmt: str | None,
    quality: int | None,
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
    parsed_from_dirs: list[str] | None = None
    if from_dirs is not None:
        parsed_from_dirs = [d.strip() for d in from_dirs.split(",") if d.strip()]
        if not parsed_from_dirs:
            raise click.ClickException("--from must contain at least one folder name")

    cfg = _resolve_config(config, working_dir, parsed_from_dirs, to, fmt, quality, strip_metadata, channels, background)

    input_dirs = resolve_dirs(cfg.working_dir, cfg.from_dirs)
    output_dir = cfg.working_dir / cfg.to

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
    num_workers = workers if workers is not None else cpu_count() or 4

    ops: list[str] = []
    if cfg.format:
        ops.append(f"format={cfg.format}")
    if cfg.channels:
        ops.append(f"channels={cfg.channels}")
    if cfg.strip_metadata:
        ops.append("strip-metadata")

    logger.info(
        "Formatting %d images from [%s] → %s (%s, workers=%d)",
        len(images), from_label, output_dir,
        ", ".join(ops) if ops else "copy", num_workers,
    )

    if dry_run:
        click.echo(f"\nDry run -- would format {len(images):,} images")
        if cfg.format:
            click.echo(f"  Format: {cfg.format}")
        if cfg.channels:
            click.echo(f"  Channels: {cfg.channels}")
        if cfg.strip_metadata:
            click.echo(f"  Strip metadata: yes")
        if cfg.format in ("jpg", "webp"):
            click.echo(f"  Quality: {cfg.quality}")
        click.echo(f"  Output: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (str(img_path), str(output_dir), cfg.format, cfg.quality,
         cfg.strip_metadata, cfg.channels, cfg.background)
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
                            copy_sidecar(src_path, output_dir / name, exclude={"metrics"})
                        else:
                            failed_count += 1
                            logger.error("Failed to format %s: %s", name, error)
                        pbar.set_postfix(ok=ok_count, fail=failed_count)
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nFormat complete!")
    click.echo(f"  Converted: {ok_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {output_dir}")
