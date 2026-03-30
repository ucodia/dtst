from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import AnalyzeConfig, load_analyze_config
from dtst.files import find_images, resolve_dirs
from dtst.sidecar import read_sidecar, sidecar_path, write_sidecar

logger = logging.getLogger(__name__)


def _compute_phash(args: tuple) -> tuple[str, str | None, str | None]:
    (image_path_str,) = args
    try:
        import imagehash
        from PIL import Image

        img = Image.open(image_path_str)
        h = imagehash.phash(img)
        return (image_path_str, str(h), None)
    except Exception as exc:
        return (image_path_str, None, str(exc))


def _compute_blur(args: tuple) -> tuple[str, float | None, str | None]:
    (image_path_str,) = args
    try:
        import cv2

        img = cv2.imread(image_path_str)
        if img is None:
            return (image_path_str, None, "could not read image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return (image_path_str, float(score), None)
    except Exception as exc:
        return (image_path_str, None, str(exc))


@click.command("analyze")
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
)
@click.option(
    "--from",
    "from_dirs",
    type=str,
    default=None,
    help="Comma-separated source folders (supports globs, e.g. 'images/*').",
)
@click.option("--phash", is_flag=True, default=False, help="Compute perceptual hash for each image.")
@click.option("--blur", is_flag=True, default=False, help="Compute blur score (Laplacian variance) for each image.")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Recompute all analyzers even if sidecar data already exists.",
)
@click.option(
    "--working-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory (default: .).",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Number of parallel workers (default: CPU count).",
)
@click.option("--clear", is_flag=True, help="Remove all sidecar files from source folders.")
@click.option("--dry-run", is_flag=True, help="Preview what would be computed without writing sidecars.")
def cmd(config, from_dirs, phash, blur, force, working_dir, workers, clear, dry_run):
    """Compute image metadata and write JSON sidecars.

    Analyzes images in the source folders and writes per-image sidecar
    JSON files containing the requested metadata (perceptual hash,
    blur score, or both). Sidecars are merged incrementally — running
    with --phash then --blur accumulates both.

    At least one analyzer flag (--phash, --blur) is required unless
    using --clear.

    \b
    Examples:
    
      dtst analyze --from raw --phash --blur -d ./my-dataset
      dtst analyze config.yaml --phash
      dtst analyze --from raw,extra --blur --force
      dtst analyze --from raw --phash --dry-run -d ./my-dataset
      dtst analyze --from raw --clear -d ./my-dataset
    """
    t0 = time.time()

    cfg = AnalyzeConfig()
    if config is not None:
        cfg = load_analyze_config(config)

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dirs is not None:
        cfg.from_dirs = [d.strip() for d in from_dirs.split(",") if d.strip()]
    if phash:
        cfg.phash = True
    if blur:
        cfg.blur = True

    if cfg.from_dirs is None:
        raise click.ClickException("--from is required (or set 'analyze.from' in config)")

    working = cfg.working_dir.resolve()

    input_dirs = resolve_dirs(working, cfg.from_dirs)

    if clear:
        all_images: list[Path] = []
        for src in input_dirs:
            if not src.is_dir():
                logger.warning("Source directory does not exist, skipping: %s", src)
                continue
            all_images.extend(find_images(src))

        if not all_images:
            raise click.ClickException("No images found in source directories.")

        sidecars = [sidecar_path(img) for img in all_images if sidecar_path(img).exists()]

        if not sidecars:
            click.echo("No sidecar files found. Nothing to clear.")
            return

        if dry_run:
            click.echo(f"[dry-run] Would remove {len(sidecars):,} sidecar files")
            return

        removed = 0
        for sc in sidecars:
            try:
                sc.unlink()
                removed += 1
            except OSError as e:
                logger.error("Failed to remove %s: %s", sc.name, e)

        elapsed = time.time() - t0
        click.echo(f"Removed {removed:,} sidecar files ({elapsed:.1f}s)")
        return

    if not cfg.phash and not cfg.blur:
        raise click.ClickException("At least one analyzer flag is required (--phash, --blur).")

    if workers is None:
        workers = cpu_count()

    analyzers = []
    if cfg.phash:
        analyzers.append("phash")
    if cfg.blur:
        analyzers.append("blur")

    all_images: list[Path] = []
    for src in input_dirs:
        if not src.is_dir():
            logger.warning("Source directory does not exist, skipping: %s", src)
            continue
        all_images.extend(find_images(src))

    if not all_images:
        raise click.ClickException("No images found in source directories.")

    logger.info(
        "Found %d images in %s, analyzers: %s",
        len(all_images),
        ", ".join(cfg.from_dirs),
        ", ".join(analyzers),
    )

    needs_work: dict[Path, list[str]] = {}
    skipped = 0
    for img in all_images:
        existing = read_sidecar(img) if not force else {}
        missing = [a for a in analyzers if a not in existing]
        if missing:
            needs_work[img] = missing
        else:
            skipped += 1

    if dry_run:
        click.echo(f"[dry-run] Would analyze {len(needs_work)} images, skip {skipped} (already computed)")
        for img, pending in needs_work.items():
            click.echo(f"  {img.name}: {', '.join(pending)}")
        return

    if not needs_work:
        click.echo(f"All {len(all_images)} images already have requested metadata. Nothing to do.")
        return

    logger.info("Analyzing %d images (%d skipped)", len(needs_work), skipped)

    results: dict[Path, dict] = {img: {} for img in needs_work}
    errors = 0

    with logging_redirect_tqdm():
        if "phash" in analyzers:
            phash_images = [img for img, pending in needs_work.items() if "phash" in pending]
            if phash_images:
                work = [(str(img),) for img in phash_images]
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_compute_phash, w): w for w in work}
                    with tqdm(total=len(work), desc="Computing phash", unit="image") as pbar:
                        for future in as_completed(futures):
                            path_str, hash_val, err = future.result()
                            img_path = Path(path_str)
                            if err:
                                logger.error("phash failed for %s: %s", img_path.name, err)
                                errors += 1
                            else:
                                results[img_path]["phash"] = {"hash": hash_val}
                            pbar.update(1)

        if "blur" in analyzers:
            blur_images = [img for img, pending in needs_work.items() if "blur" in pending]
            if blur_images:
                work = [(str(img),) for img in blur_images]
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_compute_blur, w): w for w in work}
                    with tqdm(total=len(work), desc="Computing blur", unit="image") as pbar:
                        for future in as_completed(futures):
                            path_str, score, err = future.result()
                            img_path = Path(path_str)
                            if err:
                                logger.error("blur failed for %s: %s", img_path.name, err)
                                errors += 1
                            else:
                                results[img_path]["blur"] = {"score": round(score, 2)}
                            pbar.update(1)

    written = 0
    for img, data in results.items():
        if data:
            write_sidecar(img, data)
            written += 1

    elapsed = time.time() - t0
    click.echo(
        f"Done: {written} analyzed, {skipped} skipped, {errors} failed ({elapsed:.1f}s)"
    )
