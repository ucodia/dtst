from __future__ import annotations

import logging
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import FilterConfig, load_filter_config
from dtst.images import find_images

logger = logging.getLogger(__name__)

FILTERED_DIR_NAME = "filtered"


def _check_image_size(args: tuple) -> tuple[str, str, int, int, str | None]:
    """Check image dimensions. Returns (status, filename, width, height, error).

    Must be a top-level function for ProcessPoolExecutor.
    """
    input_path_s, min_size = args
    input_path = Path(input_path_s)
    name = input_path.name

    try:
        from PIL import Image

        with Image.open(input_path) as img:
            w, h = img.size

        if max(w, h) < min_size:
            return "reject", name, w, h, None
        return "keep", name, w, h, None

    except Exception as e:
        return "failed", name, 0, 0, str(e)


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dir: str | None,
    min_size: int | None,
) -> FilterConfig:
    if config is not None:
        cfg = load_filter_config(config)
    else:
        cfg = FilterConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dir is not None:
        cfg.from_dir = from_dir
    if min_size is not None:
        cfg.min_size = min_size

    return cfg


@click.command("filter")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory (default: .).")
@click.option("--from", "from_dir", type=str, default=None, help="Folder name to filter within the working directory (default: faces).")
@click.option("--min-size", "-s", type=int, default=None, help="Minimum image dimension in pixels; images smaller are filtered out.")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count).")
@click.option("--clear", is_flag=True, help="Restore all filtered images back to the source folder.")
@click.option("--dry-run", is_flag=True, help="Show what would be filtered without moving anything.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dir: str | None,
    min_size: int | None,
    workers: int | None,
    clear: bool,
    dry_run: bool,
) -> None:
    """Filter images by moving rejects to a filtered/ subfolder.

    Evaluates images in a source folder against filter criteria and
    moves those that fail into a filtered/ subdirectory within the
    source folder. Filtered images can be restored with --clear.

    This is a non-destructive operation: no images are deleted, only
    moved. The file explorer serves as the UI for reviewing what was
    filtered. To undo individual decisions, move files back manually.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:
        dtst filter -d ./project --from faces --min-size 256
        dtst filter config.yaml --min-size 128
        dtst filter -d ./project --from faces --clear
        dtst filter -d ./project --from faces --min-size 256 --dry-run
    """
    if clear and (min_size is not None):
        raise click.ClickException("--clear cannot be combined with filter criteria")

    cfg = _resolve_config(config, working_dir, from_dir, min_size)
    source_dir = cfg.working_dir / cfg.from_dir
    filtered_dir = source_dir / FILTERED_DIR_NAME

    if not source_dir.is_dir():
        raise click.ClickException(f"Source directory not found: {source_dir}")

    # --- Clear mode ----------------------------------------------------------

    if clear:
        if not filtered_dir.is_dir():
            click.echo("Nothing to restore (no filtered/ directory found).")
            return

        filtered_images = find_images(filtered_dir)
        if not filtered_images:
            click.echo("Nothing to restore (filtered/ is empty).")
            return

        if dry_run:
            click.echo(f"\nDry run -- would restore {len(filtered_images):,} images to {source_dir}")
            return

        restored = 0
        with logging_redirect_tqdm():
            with tqdm(total=len(filtered_images), desc="Restoring", unit="image") as pbar:
                for path in filtered_images:
                    dest = source_dir / path.name
                    if dest.exists():
                        # Avoid overwriting; skip with warning
                        logger.warning("Skipping %s (already exists in source)", path.name)
                    else:
                        path.rename(dest)
                        restored += 1
                    pbar.update(1)

        # Remove filtered dir if empty
        try:
            filtered_dir.rmdir()
        except OSError:
            pass

        click.echo(f"\nRestore complete!")
        click.echo(f"  Restored: {restored:,}")
        click.echo(f"  Source: {source_dir}")
        return

    # --- Filter mode ---------------------------------------------------------

    if cfg.min_size is None:
        raise click.ClickException("No filter criteria specified (use --min-size or check config)")

    images = find_images(source_dir)
    if not images:
        raise click.ClickException(f"No images found in: {source_dir}")

    num_workers = workers if workers is not None else cpu_count() or 4
    logger.info("Filtering %d images in %s (min_size=%d)", len(images), source_dir, cfg.min_size)

    # Check dimensions in parallel (CPU-bound: opening image headers)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    work = [(str(p), cfg.min_size) for p in images]
    rejects: list[tuple[Path, int, int]] = []
    kept = 0
    failed = 0

    with logging_redirect_tqdm():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_check_image_size, w): w for w in work}
            with tqdm(total=len(futures), desc="Checking", unit="image") as pbar:
                try:
                    for future in as_completed(futures):
                        status, name, w, h, error = future.result()
                        if status == "reject":
                            # Find the original path from the work item
                            original_path = Path(futures[future][0])
                            rejects.append((original_path, w, h))
                        elif status == "keep":
                            kept += 1
                        else:
                            failed += 1
                            logger.error("Failed to check %s: %s", name, error)
                        pbar.set_postfix(keep=kept, reject=len(rejects), fail=failed)
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    if not rejects:
        click.echo(f"\nAll {kept:,} images pass the filter criteria.")
        return

    if dry_run:
        click.echo(f"\nDry run -- would filter {len(rejects):,} / {len(images):,} images")
        for path, w, h in rejects[:10]:
            click.echo(f"  {path.name} ({w}x{h})")
        if len(rejects) > 10:
            click.echo(f"  ... and {len(rejects) - 10:,} more")
        return

    # Move rejects to filtered/
    filtered_dir.mkdir(parents=True, exist_ok=True)
    moved = 0

    with logging_redirect_tqdm():
        with tqdm(total=len(rejects), desc="Filtering", unit="image") as pbar:
            for path, w, h in rejects:
                try:
                    dest = filtered_dir / path.name
                    path.rename(dest)
                    moved += 1
                    logger.debug("Filtered %s (%dx%d)", path.name, w, h)
                except OSError as e:
                    logger.error("Failed to move %s: %s", path.name, e)
                pbar.update(1)

    click.echo(f"\nFilter complete!")
    click.echo(f"  Kept: {kept:,}")
    click.echo(f"  Filtered: {moved:,}")
    if failed > 0:
        click.echo(f"  Failed: {failed:,}")
    click.echo(f"  Source: {source_dir}")
    click.echo(f"  Filtered to: {filtered_dir}")
