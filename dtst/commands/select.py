from __future__ import annotations

import logging
import time
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import config_argument
from dtst.files import copy_image, find_images, move_image, resolve_dirs
from dtst.sidecar import read_sidecar

logger = logging.getLogger(__name__)


def _check_image_dimensions(args: tuple) -> tuple[str, str, int, int, str | None]:
    """Check image dimensions against all criteria.

    Returns (status, filename, width, height, reject_reason_or_error).
    Must be a top-level function for ProcessPoolExecutor.
    """
    input_path_s, min_side, max_side, min_width, max_width, min_height, max_height = args
    input_path = Path(input_path_s)
    name = input_path.name

    try:
        from PIL import Image

        with Image.open(input_path) as img:
            w, h = img.size

        largest = max(w, h)
        if min_side is not None and largest < min_side:
            return "reject", name, w, h, f"side too small ({w}x{h}, min_side={min_side})"
        if max_side is not None and largest > max_side:
            return "reject", name, w, h, f"side too large ({w}x{h}, max_side={max_side})"
        if min_width is not None and w < min_width:
            return "reject", name, w, h, f"width too small ({w}, min_width={min_width})"
        if max_width is not None and w > max_width:
            return "reject", name, w, h, f"width too large ({w}, max_width={max_width})"
        if min_height is not None and h < min_height:
            return "reject", name, w, h, f"height too small ({h}, min_height={min_height})"
        if max_height is not None and h > max_height:
            return "reject", name, w, h, f"height too large ({h}, max_height={max_height})"
        return "keep", name, w, h, None

    except Exception as e:
        return "failed", name, 0, 0, str(e)


@click.command("select")
@config_argument
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory containing source folders and where output is written (default: .).")
@click.option("--from", "from_dirs", type=str, default=None, help="Comma-separated source folders within the working directory (supports globs, e.g. 'images/*').")
@click.option("--to", type=str, default=None, help="Destination folder name within the working directory.")
@click.option("--move", is_flag=True, help="Move images instead of copying (removes originals).")
@click.option("--min-side", "-s", type=int, default=None, help="Minimum largest side in pixels; images with max(w,h) below this are excluded.")
@click.option("--max-side", type=int, default=None, help="Maximum largest side in pixels; images with max(w,h) above this are excluded.")
@click.option("--min-width", type=int, default=None, help="Minimum width in pixels; narrower images are excluded.")
@click.option("--max-width", type=int, default=None, help="Maximum width in pixels; wider images are excluded.")
@click.option("--min-height", type=int, default=None, help="Minimum height in pixels; shorter images are excluded.")
@click.option("--max-height", type=int, default=None, help="Maximum height in pixels; taller images are excluded.")
@click.option("--min-metric", type=(str, float), multiple=True, default=(), help="Minimum metric threshold (e.g. --min-metric blur 5). Can be repeated.")
@click.option("--max-metric", type=(str, float), multiple=True, default=(), help="Maximum metric threshold (e.g. --max-metric brisque 40). Can be repeated.")
@click.option("--max-detect", type=(str, float), multiple=True, default=(), help="Exclude images where detection score >= THRESHOLD (e.g. --max-detect microphone 0.5).")
@click.option("--min-detect", type=(str, float), multiple=True, default=(), help="Exclude images where detection score < THRESHOLD (e.g. --min-detect chair 0.3).")
@click.option("--source", type=str, default=None, help="Comma-separated list of sources to include (e.g. 'serper,flickr'); checked against sidecar 'source' field.")
@click.option("--license", "license_filter", type=str, default=None, help="Comma-separated list of licenses to include (e.g. 'cc-by,none'); checked against sidecar 'license' field.")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count).")
@click.option("--dry-run", is_flag=True, help="Preview what would be selected without creating files.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    move: bool,
    min_side: int | None,
    max_side: int | None,
    min_width: int | None,
    max_width: int | None,
    min_height: int | None,
    max_height: int | None,
    min_metric: tuple[tuple[str, float], ...],
    max_metric: tuple[tuple[str, float], ...],
    max_detect: tuple[tuple[str, float], ...],
    min_detect: tuple[tuple[str, float], ...],
    source: str | None,
    license_filter: str | None,
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
        dtst select -d ./project --from faces --to curated --min-side 256
        dtst select -d ./project --from faces --to curated --min-metric blur 5
        dtst select -d ./project --from faces --to curated --min-metric blur 5 --min-metric musiq 60
        dtst select -d ./project --from faces --to curated --max-metric brisque 40
        dtst select -d ./project --from raw --to clean --max-detect microphone 0.5
        dtst select -d ./project --from raw --to licensed --source serper,flickr
        dtst select config.yaml --dry-run
    """
    if not from_dirs:
        raise click.ClickException("--from is required (or set 'select.from' in config)")
    if not to:
        raise click.ClickException("--to is required (or set 'select.to' in config)")

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    working = (working_dir or Path(".")).resolve()
    source_list = [s.strip().lower() for s in source.split(",") if s.strip()] if source else None
    license_list = [lf.strip().lower() for lf in license_filter.split(",") if lf.strip()] if license_filter else None
    min_metric_list = list(min_metric) if min_metric else None
    max_metric_list = list(max_metric) if max_metric else None
    min_detect_list = list(min_detect) if min_detect else None
    max_detect_list = list(max_detect) if max_detect else None

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

    num_workers = workers if workers is not None else cpu_count() or 4
    has_dimension_criteria = any(v is not None for v in (
        min_side, max_side, min_width, max_width, min_height, max_height,
    ))
    has_metric_criteria = min_metric_list is not None or max_metric_list is not None
    has_sidecar_criteria = source_list is not None or license_list is not None
    has_criteria = has_dimension_criteria or has_metric_criteria or max_detect_list or min_detect_list or has_sidecar_criteria
    transfer_label = "move" if move else "copy"

    # --- Filter images -------------------------------------------------------

    rejects: dict[Path, str] = {}
    kept_set: set[Path] = set(images)
    failed = 0

    if has_criteria:
        criteria_parts = []
        for name, val in [
            ("min_side", min_side), ("max_side", max_side),
            ("min_width", min_width), ("max_width", max_width),
            ("min_height", min_height), ("max_height", max_height),
        ]:
            if val is not None:
                criteria_parts.append(f"{name}={val}")
        if min_metric_list:
            for metric_name, threshold in min_metric_list:
                criteria_parts.append(f"min_metric({metric_name})={threshold}")
        if max_metric_list:
            for metric_name, threshold in max_metric_list:
                criteria_parts.append(f"max_metric({metric_name})={threshold}")
        if max_detect_list:
            for cls, threshold in max_detect_list:
                criteria_parts.append(f"max_detect({cls})={threshold}")
        if min_detect_list:
            for cls, threshold in min_detect_list:
                criteria_parts.append(f"min_detect({cls})={threshold}")
        if source_list:
            criteria_parts.append(f"source={','.join(source_list)}")
        if license_list:
            criteria_parts.append(f"license={','.join(license_list)}")
        logger.info("Filtering %d images (%s)", len(images), ", ".join(criteria_parts))

        # Dimension check (parallel, CPU-bound)
        if has_dimension_criteria:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            work = [
                (str(p), min_side, max_side,
                 min_width, max_width, min_height, max_height)
                for p in images
            ]
            with logging_redirect_tqdm():
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(_check_image_dimensions, w): w for w in work}
                    with tqdm(total=len(futures), desc="Checking dimensions", unit="image") as pbar:
                        try:
                            for future in as_completed(futures):
                                status, name, w, h, reason = future.result()
                                if status == "reject":
                                    original_path = Path(futures[future][0])
                                    rejects[original_path] = reason
                                    kept_set.discard(original_path)
                                elif status == "failed":
                                    failed += 1
                                    logger.error("Failed to check %s: %s", name, reason)
                                pbar.update(1)
                        except KeyboardInterrupt:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise

        # Metric check (sidecar lookup)
        if has_metric_criteria:
            remaining = sorted(kept_set)
            for img_path in remaining:
                sidecar = read_sidecar(img_path)
                metrics = sidecar.get("metrics", {})

                rejected = False
                if min_metric_list:
                    for metric_name, threshold in min_metric_list:
                        score = metrics.get(metric_name)
                        if score is None:
                            rejects[img_path] = f"missing '{metric_name}' metric data"
                            kept_set.discard(img_path)
                            rejected = True
                            break
                        if score < threshold:
                            rejects[img_path] = f"{metric_name} too low ({score} < {threshold})"
                            kept_set.discard(img_path)
                            rejected = True
                            break

                if not rejected and max_metric_list:
                    for metric_name, threshold in max_metric_list:
                        score = metrics.get(metric_name)
                        if score is None:
                            rejects[img_path] = f"missing '{metric_name}' metric data"
                            kept_set.discard(img_path)
                            break
                        if score > threshold:
                            rejects[img_path] = f"{metric_name} too high ({score} > {threshold})"
                            kept_set.discard(img_path)
                            break

        # Detection check (sidecar lookup)
        if max_detect_list or min_detect_list:
            remaining = sorted(kept_set)
            for img_path in remaining:
                sidecar = read_sidecar(img_path)
                detect_data = sidecar.get("classes")
                if detect_data is None:
                    rejects[img_path] = "missing detection data"
                    kept_set.discard(img_path)
                    continue

                rejected = False
                if max_detect_list:
                    for cls, threshold in max_detect_list:
                        entry = detect_data.get(cls, [])
                        if not entry:
                            continue
                        score = entry[0]["score"]
                        if score >= threshold:
                            rejects[img_path] = f"detection '{cls}' score {score:.3f} >= {threshold}"
                            kept_set.discard(img_path)
                            rejected = True
                            break

                if not rejected and min_detect_list:
                    for cls, threshold in min_detect_list:
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

        # Source / license check (sidecar lookup)
        if has_sidecar_criteria:
            remaining = sorted(kept_set)
            for img_path in remaining:
                sidecar = read_sidecar(img_path)

                if source_list is not None:
                    img_source = sidecar.get("source")
                    if img_source is None:
                        rejects[img_path] = "missing source data"
                        kept_set.discard(img_path)
                        continue
                    if str(img_source).lower() not in source_list:
                        rejects[img_path] = f"source '{img_source}' not in {source_list}"
                        kept_set.discard(img_path)
                        continue

                if license_list is not None:
                    img_license = sidecar.get("license")
                    if img_license is None:
                        rejects[img_path] = "missing license data"
                        kept_set.discard(img_path)
                        continue
                    if str(img_license).lower() not in license_list:
                        rejects[img_path] = f"license '{img_license}' not in {license_list}"
                        kept_set.discard(img_path)
                        continue

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

    transfer_fn = move_image if move else copy_image

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

    verb = "Moved" if move else "Copied"
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
