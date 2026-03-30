from __future__ import annotations

import logging
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import FilterConfig, load_filter_config
from dtst.files import find_images
from dtst.sidecar import read_sidecar, sidecar_path

logger = logging.getLogger(__name__)


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
    to: str | None,
    min_size: int | None,
    min_blur: float | None,
) -> FilterConfig:
    if config is not None:
        cfg = load_filter_config(config)
    else:
        cfg = FilterConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dir is not None:
        cfg.from_dir = from_dir
    if to is not None:
        cfg.to = to
    if min_size is not None:
        cfg.min_size = min_size
    if min_blur is not None:
        cfg.min_blur = min_blur

    return cfg


@click.command("filter")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory (default: .).")
@click.option("--from", "from_dir", type=str, default=None, help="Folder name to filter within the working directory.")
@click.option("--to", type=str, default=None, help="Subfolder name for rejected images.", show_default="filtered")
@click.option("--min-size", "-s", type=int, default=None, help="Minimum image dimension in pixels; images smaller are filtered out.")
@click.option("--min-blur", type=float, default=None, help="Minimum blur score (Laplacian variance) to keep; lower-scoring images are filtered as too blurry.")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count).")
@click.option("--clear", is_flag=True, help="Restore all filtered images back to the source folder.")
@click.option("--dry-run", is_flag=True, help="Show what would be filtered without moving anything.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dir: str | None,
    to: str | None,
    min_size: int | None,
    min_blur: float | None,
    workers: int | None,
    clear: bool,
    dry_run: bool,
) -> None:
    """Filter images by moving rejects to a subfolder.

    Evaluates images in a source folder against filter criteria and
    moves those that fail into a subdirectory within the source
    folder (default: filtered/). Filtered images can be restored
    with --clear.

    This is a non-destructive operation: no images are deleted, only
    moved. The file explorer serves as the UI for reviewing what was
    filtered. To undo individual decisions, move files back manually.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:
        dtst filter -d ./project --from faces --min-size 256
        dtst filter -d ./project --from faces --min-blur 50
        dtst filter -d ./project --from faces --min-size 256 --min-blur 50
        dtst filter -d ./project --from faces --to rejects --min-size 256
        dtst filter config.yaml --min-size 128
        dtst filter -d ./project --from faces --clear
        dtst filter -d ./project --from faces --min-size 256 --dry-run
    """
    has_criteria = min_size is not None or min_blur is not None
    if clear and has_criteria:
        raise click.ClickException("--clear cannot be combined with filter criteria")

    cfg = _resolve_config(config, working_dir, from_dir, to, min_size, min_blur)

    if cfg.from_dir is None:
        raise click.ClickException("--from is required (or set 'filter.from' in config)")

    source_dir = cfg.working_dir / cfg.from_dir
    filtered_dir = source_dir / cfg.to

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
                        logger.warning("Skipping %s (already exists in source)", path.name)
                    else:
                        path.rename(dest)
                        sc = sidecar_path(path)
                        if sc.exists():
                            sc.rename(sidecar_path(dest))
                        restored += 1
                    pbar.update(1)

        try:
            filtered_dir.rmdir()
        except OSError:
            pass

        click.echo(f"\nRestore complete!")
        click.echo(f"  Restored: {restored:,}")
        click.echo(f"  Source: {source_dir}")
        return

    # --- Filter mode ---------------------------------------------------------

    if cfg.min_size is None and cfg.min_blur is None:
        raise click.ClickException("No filter criteria specified (use --min-size, --min-blur, or check config)")

    images = find_images(source_dir)
    if not images:
        raise click.ClickException(f"No images found in: {source_dir}")

    num_workers = workers if workers is not None else cpu_count() or 4

    criteria_parts = []
    if cfg.min_size is not None:
        criteria_parts.append(f"min_size={cfg.min_size}")
    if cfg.min_blur is not None:
        criteria_parts.append(f"min_blur={cfg.min_blur}")
    logger.info("Filtering %d images in %s (%s)", len(images), source_dir, ", ".join(criteria_parts))

    rejects: dict[Path, str] = {}
    kept_set: set[Path] = set(images)
    failed = 0

    # --- Size check (parallel, CPU-bound) ------------------------------------

    if cfg.min_size is not None:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        work = [(str(p), cfg.min_size) for p in images]
        with logging_redirect_tqdm():
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_check_image_size, w): w for w in work}
                with tqdm(total=len(futures), desc="Checking size", unit="image") as pbar:
                    try:
                        for future in as_completed(futures):
                            status, name, w, h, error = future.result()
                            if status == "reject":
                                original_path = Path(futures[future][0])
                                rejects[original_path] = f"too small ({w}x{h})"
                                kept_set.discard(original_path)
                            elif status == "failed":
                                failed += 1
                                logger.error("Failed to check %s: %s", name, error)
                            pbar.update(1)
                    except KeyboardInterrupt:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

    # --- Blur check (sidecar lookup, no parallelism needed) ------------------

    if cfg.min_blur is not None:
        remaining = sorted(kept_set)
        for img_path in remaining:
            sidecar = read_sidecar(img_path)
            blur_data = sidecar.get("blur")
            if blur_data is None:
                logger.warning("No blur data in sidecar for %s, skipping blur check", img_path.name)
                continue
            score = blur_data.get("score")
            if score is None:
                logger.warning("No blur score in sidecar for %s, skipping blur check", img_path.name)
                continue
            if score < cfg.min_blur:
                rejects[img_path] = f"too blurry (score={score:.2f})"
                kept_set.discard(img_path)

    # --- Results -------------------------------------------------------------

    kept = len(kept_set)

    if not rejects:
        click.echo(f"\nAll {kept:,} images pass the filter criteria.")
        return

    if dry_run:
        click.echo(f"\nDry run -- would filter {len(rejects):,} / {len(images):,} images")
        for path, reason in list(rejects.items())[:10]:
            click.echo(f"  {path.name} ({reason})")
        if len(rejects) > 10:
            click.echo(f"  ... and {len(rejects) - 10:,} more")
        return

    filtered_dir.mkdir(parents=True, exist_ok=True)
    moved = 0

    with logging_redirect_tqdm():
        with tqdm(total=len(rejects), desc="Filtering", unit="image") as pbar:
            for path, reason in rejects.items():
                try:
                    dest = filtered_dir / path.name
                    path.rename(dest)
                    sc = sidecar_path(path)
                    if sc.exists():
                        sc.rename(sidecar_path(dest))
                    moved += 1
                    logger.debug("Filtered %s (%s)", path.name, reason)
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
