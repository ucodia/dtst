from __future__ import annotations

import logging
import time
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import SelectConfig, load_select_config
from dtst.files import copy_image, find_images, move_image, resolve_dirs
from dtst.sidecar import read_sidecar

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
    from_dirs: list[str] | None,
    to: str | None,
    move: bool,
    min_size: int | None,
    min_blur: float | None,
    max_tag: tuple[tuple[str, float], ...] | None = None,
    min_tag: tuple[tuple[str, float], ...] | None = None,
    max_detect: tuple[tuple[str, float], ...] | None = None,
    min_detect: tuple[tuple[str, float], ...] | None = None,
) -> SelectConfig:
    if config is not None:
        cfg = load_select_config(config)
    else:
        cfg = SelectConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dirs is not None:
        cfg.from_dirs = from_dirs
    if to is not None:
        cfg.to = to
    if move:
        cfg.move = True
    if min_size is not None:
        cfg.min_size = min_size
    if min_blur is not None:
        cfg.min_blur = min_blur
    if max_tag:
        cfg.max_tag = list(max_tag)
    if min_tag:
        cfg.min_tag = list(min_tag)
    if max_detect:
        cfg.max_detect = list(max_detect)
    if min_detect:
        cfg.min_detect = list(min_detect)

    if cfg.from_dirs is None:
        raise click.ClickException("--from is required (or set 'select.from' in config)")
    if cfg.to is None:
        raise click.ClickException("--to is required (or set 'select.to' in config)")

    return cfg


@click.command("select")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory containing source folders and where output is written (default: .).")
@click.option("--from", "from_dirs", type=str, default=None, help="Comma-separated source folders within the working directory (supports globs, e.g. 'images/*').")
@click.option("--to", type=str, default=None, help="Destination folder name within the working directory.")
@click.option("--move", is_flag=True, help="Move images instead of copying (removes originals).")
@click.option("--min-size", "-s", type=int, default=None, help="Minimum image dimension in pixels; smaller images are excluded.")
@click.option("--min-blur", type=float, default=None, help="Minimum blur score (Laplacian variance); lower-scoring images are excluded as too blurry.")
@click.option("--max-tag", type=(str, float), multiple=True, default=(), help="Exclude images where TAG score >= THRESHOLD (e.g. --max-tag microphone 0.25).")
@click.option("--min-tag", type=(str, float), multiple=True, default=(), help="Exclude images where TAG score < THRESHOLD (e.g. --min-tag photograph 0.2).")
@click.option("--max-detect", type=(str, float), multiple=True, default=(), help="Exclude images where detection score >= THRESHOLD (e.g. --max-detect microphone 0.5).")
@click.option("--min-detect", type=(str, float), multiple=True, default=(), help="Exclude images where detection score < THRESHOLD (e.g. --min-detect chair 0.3).")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count).")
@click.option("--dry-run", is_flag=True, help="Preview what would be selected without creating files.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    move: bool,
    min_size: int | None,
    min_blur: float | None,
    max_tag: tuple[tuple[str, float], ...],
    min_tag: tuple[tuple[str, float], ...],
    max_detect: tuple[tuple[str, float], ...],
    min_detect: tuple[tuple[str, float], ...],
    workers: int | None,
    dry_run: bool,
) -> None:
    """Select images from source folders into a destination folder.

    Copies (or moves with --move) images from one or more source folders
    into a destination folder. When filter criteria are provided, only
    images that pass all criteria are selected. Without criteria, all
    images are selected.

    Files that already exist in the destination (by name) are skipped.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:
        dtst select -d ./project --from raw --to backup
        dtst select -d ./project --from raw,extra --to combined
        dtst select -d ./project --from faces --to curated --min-size 256
        dtst select -d ./project --from faces --to curated --move --min-blur 50
        dtst select -d ./project --from raw --to clean --max-tag microphone 0.25
        dtst select -d ./project --from raw --to clean --max-detect microphone 0.5
        dtst select config.yaml --dry-run
    """
    parsed_from_dirs: list[str] | None = None
    if from_dirs is not None:
        parsed_from_dirs = [d.strip() for d in from_dirs.split(",") if d.strip()]
        if not parsed_from_dirs:
            raise click.ClickException("--from must contain at least one folder name")

    cfg = _resolve_config(config, working_dir, parsed_from_dirs, to, move, min_size, min_blur, max_tag, min_tag, max_detect, min_detect)

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

    num_workers = workers if workers is not None else cpu_count() or 4
    has_criteria = cfg.min_size is not None or cfg.min_blur is not None or cfg.max_tag or cfg.min_tag or cfg.max_detect or cfg.min_detect
    transfer_label = "move" if cfg.move else "copy"

    # --- Filter images -------------------------------------------------------

    rejects: dict[Path, str] = {}
    kept_set: set[Path] = set(images)
    failed = 0

    if has_criteria:
        criteria_parts = []
        if cfg.min_size is not None:
            criteria_parts.append(f"min_size={cfg.min_size}")
        if cfg.min_blur is not None:
            criteria_parts.append(f"min_blur={cfg.min_blur}")
        if cfg.max_tag:
            for label, threshold in cfg.max_tag:
                criteria_parts.append(f"max_tag({label})={threshold}")
        if cfg.min_tag:
            for label, threshold in cfg.min_tag:
                criteria_parts.append(f"min_tag({label})={threshold}")
        if cfg.max_detect:
            for cls, threshold in cfg.max_detect:
                criteria_parts.append(f"max_detect({cls})={threshold}")
        if cfg.min_detect:
            for cls, threshold in cfg.min_detect:
                criteria_parts.append(f"min_detect({cls})={threshold}")
        logger.info("Filtering %d images (%s)", len(images), ", ".join(criteria_parts))

        # Size check (parallel, CPU-bound)
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

        # Blur check (sidecar lookup)
        if cfg.min_blur is not None:
            remaining = sorted(kept_set)
            for img_path in remaining:
                sidecar = read_sidecar(img_path)
                blur_data = sidecar.get("blur")
                if blur_data is None or blur_data.get("score") is None:
                    rejects[img_path] = "missing blur data"
                    kept_set.discard(img_path)
                    continue
                score = blur_data["score"]
                if score < cfg.min_blur:
                    rejects[img_path] = f"too blurry (score={score:.2f})"
                    kept_set.discard(img_path)

        # Tag check (sidecar lookup)
        if cfg.max_tag or cfg.min_tag:
            remaining = sorted(kept_set)
            for img_path in remaining:
                sidecar = read_sidecar(img_path)
                tags_data = sidecar.get("tags")
                if tags_data is None:
                    rejects[img_path] = "missing tag data"
                    kept_set.discard(img_path)
                    continue
                scores = tags_data.get("scores", {})

                rejected = False
                if cfg.max_tag:
                    for label, threshold in cfg.max_tag:
                        score = scores.get(label)
                        if score is None:
                            rejects[img_path] = f"missing score for tag '{label}'"
                            kept_set.discard(img_path)
                            rejected = True
                            break
                        if score >= threshold:
                            rejects[img_path] = f"tag '{label}' score {score:.3f} >= {threshold}"
                            kept_set.discard(img_path)
                            rejected = True
                            break

                if not rejected and cfg.min_tag:
                    for label, threshold in cfg.min_tag:
                        score = scores.get(label)
                        if score is None:
                            rejects[img_path] = f"missing score for tag '{label}'"
                            kept_set.discard(img_path)
                            break
                        if score < threshold:
                            rejects[img_path] = f"tag '{label}' score {score:.3f} < {threshold}"
                            kept_set.discard(img_path)
                            break

        # Detection check (sidecar lookup)
        if cfg.max_detect or cfg.min_detect:
            remaining = sorted(kept_set)
            for img_path in remaining:
                sidecar = read_sidecar(img_path)
                detect_data = sidecar.get("classes")
                if detect_data is None:
                    rejects[img_path] = "missing detection data"
                    kept_set.discard(img_path)
                    continue

                rejected = False
                if cfg.max_detect:
                    for cls, threshold in cfg.max_detect:
                        entry = detect_data.get(cls, [])
                        if not entry:
                            continue
                        score = entry[0]["score"]
                        if score >= threshold:
                            rejects[img_path] = f"detection '{cls}' score {score:.3f} >= {threshold}"
                            kept_set.discard(img_path)
                            rejected = True
                            break

                if not rejected and cfg.min_detect:
                    for cls, threshold in cfg.min_detect:
                        entry = detect_data.get(cls, [])
                        if not entry:
                            rejects[img_path] = f"'{cls}' not detected"
                            kept_set.discard(img_path)
                            break
                        score = entry[0]["score"]
                        if score < threshold:
                            rejects[img_path] = f"detection '{cls}' score {score:.3f} < {threshold}"
                            kept_set.discard(img_path)
                            break

    selected = sorted(kept_set)

    from_label = ", ".join(str(d) for d in input_dirs)
    logger.info(
        "Selecting %d / %d images from [%s] to %s (%s)",
        len(selected), len(images), from_label, output_dir, transfer_label,
    )

    # --- Dry run -------------------------------------------------------------

    if dry_run:
        click.echo(f"\nDry run -- would {transfer_label} {len(selected):,} / {len(images):,} images")
        click.echo(f"  From: {from_label}")
        click.echo(f"  To: {output_dir}")
        if rejects:
            click.echo(f"  Excluded: {len(rejects):,}")
            for path, reason in list(rejects.items())[:10]:
                click.echo(f"    {path.name} ({reason})")
            if len(rejects) > 10:
                click.echo(f"    ... and {len(rejects) - 10:,} more")
        return

    if not selected:
        click.echo(f"\nNo images passed the filter criteria ({len(rejects):,} excluded).")
        return

    # --- Transfer images -----------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    ok_count = 0
    skipped_count = 0
    failed_count = 0

    transfer_fn = move_image if cfg.move else copy_image

    with logging_redirect_tqdm():
        with tqdm(total=len(selected), desc="Selecting", unit="image") as pbar:
            for image_path in selected:
                dest = output_dir / image_path.name
                try:
                    if dest.exists():
                        skipped_count += 1
                        logger.debug("Skipped (already exists): %s", image_path.name)
                    else:
                        transfer_fn(image_path, dest)
                        ok_count += 1
                except Exception as e:
                    failed_count += 1
                    logger.error("Failed to %s %s: %s", transfer_label, image_path.name, e)
                pbar.set_postfix(ok=ok_count, skip=skipped_count, fail=failed_count)
                pbar.update(1)

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    verb = "Moved" if cfg.move else "Copied"
    click.echo(f"\nSelect complete!")
    click.echo(f"  {verb}: {ok_count:,}")
    if skipped_count > 0:
        click.echo(f"  Skipped: {skipped_count:,}")
    if len(rejects) > 0:
        click.echo(f"  Excluded: {len(rejects):,}")
    if failed_count > 0:
        click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {output_dir}")
