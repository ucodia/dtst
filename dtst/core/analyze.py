"""Library-layer implementation of ``dtst analyze``."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.errors import InputError
from dtst.executor import run_pool
from dtst.files import gather_images, resolve_workers
from dtst.results import AnalyzeResult
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


def analyze(
    *,
    from_dirs: str,
    metrics: str | None = None,
    force: bool = False,
    workers: int | None = None,
    clear: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> AnalyzeResult:
    """Compute image metrics and write per-image sidecar JSON files."""
    t0 = time.time()

    if not from_dirs:
        raise InputError("from_dirs is required")
    metrics_list = (
        [m.strip() for m in metrics.split(",") if m.strip()] if metrics else []
    )
    _input_dirs, all_images = gather_images(from_dirs)
    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]

    if clear:
        sidecars = [
            sidecar_path(img) for img in all_images if sidecar_path(img).exists()
        ]

        if not sidecars:
            return AnalyzeResult(
                analyzed=0,
                skipped=0,
                failed=0,
                cleared=0,
                dry_run=dry_run,
                elapsed=time.time() - t0,
            )

        if dry_run:
            return AnalyzeResult(
                analyzed=0,
                skipped=0,
                failed=0,
                cleared=len(sidecars),
                dry_run=True,
                elapsed=time.time() - t0,
            )

        removed = 0
        for sc in sidecars:
            try:
                sc.unlink()
                removed += 1
            except OSError as e:
                logger.error("Failed to remove %s: %s", sc.name, e)

        return AnalyzeResult(
            analyzed=0,
            skipped=0,
            failed=0,
            cleared=removed,
            dry_run=False,
            elapsed=time.time() - t0,
        )

    if not metrics_list:
        raise InputError(
            "At least one metric is required via metrics (e.g. phash,blur,musiq)."
        )

    iqa_metrics = _get_iqa_metrics()
    known_metrics = CPU_METRICS | iqa_metrics
    unknown = [m for m in metrics_list if m not in known_metrics]
    if unknown:
        raise InputError(
            f"Unknown metric(s): {', '.join(unknown)}. "
            f"CPU metrics: {', '.join(sorted(CPU_METRICS))}. "
            f"IQA metrics: use any name from pyiqa.list_models()."
        )

    requested_cpu = [m for m in metrics_list if m in CPU_METRICS]
    requested_iqa = [m for m in metrics_list if m in iqa_metrics]

    if requested_iqa:
        from dtst.metrics.iqa import validate_iqa_metrics

        fr_metrics = validate_iqa_metrics(requested_iqa)
        if fr_metrics:
            raise InputError(
                f"Full-reference (FR) metrics are not supported: {', '.join(fr_metrics)}. "
                f"The analyze command only supports no-reference (NR) metrics."
            )

    num_workers = resolve_workers(workers)

    logger.info(
        "Found %d images in %s, metrics: %s",
        len(all_images),
        ", ".join(dirs_list),
        ", ".join(metrics_list),
    )

    needs_work: dict[Path, list[str]] = {}
    skipped = 0
    for img in all_images:
        existing = read_sidecar(img) if not force else {}
        existing_metrics = existing.get("metrics", {})
        missing = [m for m in metrics_list if m not in existing_metrics]
        if missing:
            needs_work[img] = missing
        else:
            skipped += 1

    if dry_run:
        for img, pending in needs_work.items():
            logger.debug("%s: %s", img.name, ", ".join(pending))
        return AnalyzeResult(
            analyzed=len(needs_work),
            skipped=skipped,
            failed=0,
            cleared=0,
            dry_run=True,
            elapsed=time.time() - t0,
        )

    if not needs_work:
        return AnalyzeResult(
            analyzed=0,
            skipped=skipped,
            failed=0,
            cleared=0,
            dry_run=False,
            elapsed=time.time() - t0,
        )

    logger.info("Analyzing %d images (%d skipped)", len(needs_work), skipped)

    results: dict[Path, dict] = {img: {} for img in needs_work}
    errors = 0

    for metric_name in requested_cpu:
        compute_fn = _CPU_COMPUTE_FNS[metric_name]
        metric_images = [
            img for img, pending in needs_work.items() if metric_name in pending
        ]
        if not metric_images:
            continue

        work = [(str(img),) for img in metric_images]

        def handle(result, _work_item, metric_name=metric_name):
            path_str, value, err = result
            img_path = Path(path_str)
            if err:
                logger.error("%s failed for %s: %s", metric_name, img_path.name, err)
                return "fail"
            if isinstance(value, float):
                value = round(value, 2)
            results[img_path][metric_name] = value
            return "ok"

        counts = run_pool(
            ProcessPoolExecutor,
            compute_fn,
            work,
            max_workers=num_workers,
            desc=f"Computing {metric_name}",
            unit="image",
            on_result=handle,
            postfix_keys=("ok", "fail"),
            progress=progress,
        )
        errors += counts.get("fail", 0)

    if requested_iqa:
        iqa_images_by_metric: dict[str, list[Path]] = {}
        for metric_name in requested_iqa:
            imgs = [
                img for img, pending in needs_work.items() if metric_name in pending
            ]
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

    return AnalyzeResult(
        analyzed=written,
        skipped=skipped,
        failed=errors,
        cleared=0,
        dry_run=False,
        elapsed=time.time() - t0,
    )
