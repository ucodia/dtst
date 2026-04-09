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

CPU_METRICS = {"phash", "blur"}


def _get_iqa_metrics() -> set[str]:
    try:
        import pyiqa

        return set(pyiqa.list_models())
    except ImportError:
        return set()


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


_CPU_COMPUTE_FNS = {
    "phash": _compute_phash,
    "blur": _compute_blur,
}


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
@click.option(
    "--metrics",
    "-m",
    type=str,
    default=None,
    help="Comma-separated metric names (e.g. 'phash,blur,musiq,clipiqa').",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Recompute all metrics even if sidecar data already exists.",
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
    help="Number of parallel workers for CPU metrics (default: CPU count).",
)
@click.option("--clear", is_flag=True, help="Remove all sidecar files from source folders.")
@click.option("--dry-run", is_flag=True, help="Preview what would be computed without writing sidecars.")
def cmd(config, from_dirs, metrics, force, working_dir, workers, clear, dry_run):
    """Compute image metrics and write JSON sidecars.

    Analyzes images in the source folders and writes per-image sidecar
    JSON files containing the requested metrics. Sidecars are merged
    incrementally — running with different metrics accumulates results.

    CPU metrics: phash, blur.
    IQA metrics (GPU-accelerated): any metric from IQA-PyTorch (e.g.
    musiq, clipiqa, topiq_nr, dbcnn, hyperiqa, niqe, brisque).

    \b
    Examples:
      dtst analyze --from raw --metrics phash,blur -d ./my-dataset
      dtst analyze config.yaml --metrics phash
      dtst analyze --from raw --metrics musiq,clipiqa -d ./my-dataset
      dtst analyze --from raw --metrics phash,blur,musiq --force
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
    if metrics is not None:
        cli_metrics = [m.strip() for m in metrics.split(",") if m.strip()]
        cfg.metrics = cli_metrics

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

    if not cfg.metrics:
        raise click.ClickException(
            "At least one metric is required via --metrics (e.g. --metrics phash,blur,musiq)."
        )

    iqa_metrics = _get_iqa_metrics()
    known_metrics = CPU_METRICS | iqa_metrics
    unknown = [m for m in cfg.metrics if m not in known_metrics]
    if unknown:
        raise click.ClickException(
            f"Unknown metric(s): {', '.join(unknown)}. "
            f"CPU metrics: {', '.join(sorted(CPU_METRICS))}. "
            f"IQA metrics: use any name from pyiqa.list_models()."
        )

    requested_cpu = [m for m in cfg.metrics if m in CPU_METRICS]
    requested_iqa = [m for m in cfg.metrics if m in iqa_metrics]

    if requested_iqa:
        from dtst.metrics.iqa import validate_iqa_metrics

        fr_metrics = validate_iqa_metrics(requested_iqa)
        if fr_metrics:
            raise click.ClickException(
                f"Full-reference (FR) metrics are not supported: {', '.join(fr_metrics)}. "
                f"The analyze command only supports no-reference (NR) metrics."
            )

    if workers is None:
        workers = cpu_count()

    all_images: list[Path] = []
    for src in input_dirs:
        if not src.is_dir():
            logger.warning("Source directory does not exist, skipping: %s", src)
            continue
        all_images.extend(find_images(src))

    if not all_images:
        raise click.ClickException("No images found in source directories.")

    logger.info(
        "Found %d images in %s, metrics: %s",
        len(all_images),
        ", ".join(cfg.from_dirs),
        ", ".join(cfg.metrics),
    )

    needs_work: dict[Path, list[str]] = {}
    skipped = 0
    for img in all_images:
        existing = read_sidecar(img) if not force else {}
        existing_metrics = existing.get("metrics", {})
        missing = [m for m in cfg.metrics if m not in existing_metrics]
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
        click.echo(f"All {len(all_images)} images already have requested metrics. Nothing to do.")
        return

    logger.info("Analyzing %d images (%d skipped)", len(needs_work), skipped)

    results: dict[Path, dict] = {img: {} for img in needs_work}
    errors = 0

    # --- CPU metrics via ProcessPoolExecutor ---
    with logging_redirect_tqdm():
        for metric_name in requested_cpu:
            compute_fn = _CPU_COMPUTE_FNS[metric_name]
            metric_images = [img for img, pending in needs_work.items() if metric_name in pending]
            if not metric_images:
                continue

            work = [(str(img),) for img in metric_images]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(compute_fn, w): w for w in work}
                with tqdm(total=len(work), desc=f"Computing {metric_name}", unit="image") as pbar:
                    for future in as_completed(futures):
                        path_str, value, err = future.result()
                        img_path = Path(path_str)
                        if err:
                            logger.error("%s failed for %s: %s", metric_name, img_path.name, err)
                            errors += 1
                        else:
                            if isinstance(value, float):
                                value = round(value, 2)
                            results[img_path][metric_name] = value
                        pbar.update(1)

    # --- IQA metrics via batched GPU inference ---
    if requested_iqa:
        iqa_images_by_metric: dict[str, list[Path]] = {}
        for metric_name in requested_iqa:
            imgs = [img for img, pending in needs_work.items() if metric_name in pending]
            if imgs:
                iqa_images_by_metric[metric_name] = imgs

        if iqa_images_by_metric:
            from dtst.metrics.iqa import compute_iqa_metrics

            all_iqa_images = sorted(set().union(*iqa_images_by_metric.values()))
            iqa_metric_names = list(iqa_images_by_metric.keys())

            with logging_redirect_tqdm():
                iqa_results = compute_iqa_metrics(iqa_metric_names, all_iqa_images)

            for img, scores in iqa_results.items():
                if img in results:
                    results[img].update(scores)

    written = 0
    for img, data in results.items():
        if data:
            existing = read_sidecar(img)
            merged_metrics = existing.get("metrics", {})
            merged_metrics.update(data)
            write_sidecar(img, {"metrics": merged_metrics})
            written += 1

    elapsed = time.time() - t0
    click.echo(
        f"Done: {written} analyzed, {skipped} skipped, {errors} failed ({elapsed:.1f}s)"
    )
