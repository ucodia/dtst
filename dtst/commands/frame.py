from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import FrameConfig, load_frame_config
from dtst.files import find_images, resolve_dirs

logger = logging.getLogger(__name__)


def _resize_image(args: tuple) -> tuple[str, str, str | None]:
    """Top-level worker function for ProcessPoolExecutor.

    Returns ``(status, filename, error_message)``.
    Status is one of ``"ok"`` or ``"failed"``.
    """
    input_path_s, output_dir_s, target_width, target_height = args
    input_path = Path(input_path_s)
    output_dir = Path(output_dir_s)
    name = input_path.name

    try:
        from PIL import Image

        img = Image.open(input_path)
        orig_w, orig_h = img.size

        if target_width is not None and target_height is not None:
            new_w, new_h = target_width, target_height
        elif target_width is not None:
            ratio = target_width / orig_w
            new_w = target_width
            new_h = round(orig_h * ratio)
        else:
            ratio = target_height / orig_h
            new_w = round(orig_w * ratio)
            new_h = target_height

        if new_w == orig_w and new_h == orig_h:
            img.save(output_dir / name, quality=95)
            img.close()
            return "ok", name, None

        resized = img.resize((new_w, new_h), Image.LANCZOS)
        resized.save(output_dir / name, quality=95)
        resized.close()
        img.close()
        return "ok", name, None

    except Exception as e:
        return "failed", name, str(e)


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: list[str] | None,
    to: str | None,
    width: int | None,
    height: int | None,
) -> FrameConfig:
    if config is not None:
        cfg = load_frame_config(config)
    else:
        cfg = FrameConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dirs is not None:
        cfg.from_dirs = from_dirs
    if to is not None:
        cfg.to = to
    if width is not None:
        cfg.width = width
    if height is not None:
        cfg.height = height

    if cfg.from_dirs is None:
        raise click.ClickException("--from is required (or set 'frame.from' in config)")
    if cfg.to is None:
        raise click.ClickException("--to is required (or set 'frame.to' in config)")
    if cfg.width is None and cfg.height is None:
        raise click.ClickException(
            "At least one of --width or --height is required"
        )

    return cfg


@click.command("frame")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory containing source folders and where output is written (default: .).")
@click.option("--from", "from_dirs", type=str, default=None, help="Comma-separated source folders within the working directory (supports globs, e.g. 'images/*').")
@click.option("--to", type=str, default=None, help="Destination folder name within the working directory.")
@click.option("--width", "-W", type=int, default=None, help="Target width in pixels. If --height is omitted, aspect ratio is preserved.")
@click.option("--height", "-H", type=int, default=None, help="Target height in pixels. If --width is omitted, aspect ratio is preserved.")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count).")
@click.option("--dry-run", is_flag=True, help="Preview what would be written without creating files.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    width: int | None,
    height: int | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Resize images to a target width and/or height.

    Reads images from one or more source folders and writes resized
    copies to a destination folder. Uses Lanczos resampling for
    high-quality downscaling.

    When both --width and --height are given, images are resized to
    exactly those dimensions (aspect ratio is not preserved). When
    only one dimension is given, the other is computed proportionally
    to preserve the original aspect ratio.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst frame -d ./project --from faces --to resized --width 512 --height 512
        dtst frame -d ./project --from faces --to resized --width 512
        dtst frame -d ./project --from raw --to small --height 256
        dtst frame config.yaml --dry-run
    """
    parsed_from_dirs: list[str] | None = None
    if from_dirs is not None:
        parsed_from_dirs = [d.strip() for d in from_dirs.split(",") if d.strip()]
        if not parsed_from_dirs:
            raise click.ClickException("--from must contain at least one folder name")

    cfg = _resolve_config(config, working_dir, parsed_from_dirs, to, width, height)

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

    width_label = str(cfg.width) if cfg.width is not None else "auto"
    height_label = str(cfg.height) if cfg.height is not None else "auto"
    from_label = ", ".join(str(d) for d in input_dirs)
    num_workers = workers if workers is not None else cpu_count() or 4

    logger.info(
        "Resizing %d images from [%s] to %sx%s (workers=%d)",
        len(images), from_label, width_label, height_label, num_workers,
    )

    if dry_run:
        click.echo(f"\nDry run -- would resize {len(images):,} images")
        click.echo(f"  Target: {width_label} x {height_label}")
        click.echo(f"  Output: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (str(img_path), str(output_dir), cfg.width, cfg.height)
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
                        else:
                            failed_count += 1
                            logger.error("Failed to resize %s: %s", name, error)
                        pbar.set_postfix(ok=ok_count, fail=failed_count)
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nFrame complete!")
    click.echo(f"  Resized: {ok_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Target: {width_label} x {height_label}")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {output_dir}")
