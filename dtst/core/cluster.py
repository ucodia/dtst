"""Library-layer implementation of ``dtst cluster``."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.cache import load_embeddings, save_embeddings
from dtst.embeddings import detect_device, get_backend
from dtst.errors import InputError, PipelineError
from dtst.files import copy_image, find_images, resolve_dirs, resolve_workers
from dtst.results import ClusterInfo, ClusterResult

logger = logging.getLogger(__name__)


def cluster(
    *,
    from_dirs: str,
    to: str,
    model: str = "arcface",
    top: int | None = None,
    min_cluster_size: int = 5,
    min_samples: int = 2,
    batch_size: int = 32,
    workers: int | None = None,
    no_cache: bool = False,
    clean: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> ClusterResult:
    """Cluster images by visual similarity using HDBSCAN.

    Returns a :class:`ClusterResult`.  Raises :class:`InputError` for
    missing inputs or :class:`PipelineError` if no clusters emerge.

    Set ``progress=False`` to silence tqdm bars.
    """
    if not from_dirs:
        raise InputError("from_dirs is required")
    if not to:
        raise InputError("to is required")

    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    input_dirs = resolve_dirs(dirs_list)
    output_dir = Path(to).expanduser().resolve()

    missing = [str(d) for d in input_dirs if not d.is_dir()]
    if missing:
        raise InputError(
            f"Source director{'y' if len(missing) == 1 else 'ies'} not found: "
            f"{', '.join(missing)}"
        )

    images: list[Path] = []
    for input_dir in input_dirs:
        found = find_images(input_dir)
        logger.info("Found %d images in %s", len(found), input_dir)
        images.extend(found)

    if not images:
        raise InputError(f"No images found in: {', '.join(str(d) for d in input_dirs)}")

    num_workers = resolve_workers(workers)
    from_label = ", ".join(str(d) for d in input_dirs)
    top_label = str(top) if top is not None else "all"

    logger.info(
        "Clustering %d images from [%s] (model=%s, min_cluster_size=%d, "
        "min_samples=%d, top=%s, batch_size=%d)",
        len(images),
        from_label,
        model,
        min_cluster_size,
        min_samples,
        top_label,
        batch_size,
    )

    if dry_run:
        return ClusterResult(
            model=model,
            total_images=len(images),
            embedded_images=0,
            clusters=[],
            noise_images=0,
            output_dir=output_dir,
            elapsed=0.0,
        )

    start_time = time.monotonic()
    cached = load_embeddings(model, images) if not no_cache else None

    if cached is not None:
        embeddings, valid_paths = cached
    else:
        device = detect_device()
        backend = get_backend(model)
        backend.load(device)

        with logging_redirect_tqdm():
            embeddings, valid_paths = backend.embed(
                images,
                batch_size=batch_size,
                num_workers=num_workers,
            )

        if len(valid_paths) == 0:
            raise PipelineError("No images produced valid embeddings")

        if not no_cache:
            save_embeddings(model, images, embeddings, valid_paths)

    embed_time = time.monotonic() - start_time
    logger.info(
        "Embedded %d / %d images in %.1fs",
        len(valid_paths),
        len(images),
        embed_time,
    )

    import hdbscan

    cluster_start = time.monotonic()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=0.0,
    )
    labels = clusterer.fit_predict(embeddings)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_info = [
        (int(label), int(count))
        for label, count in zip(unique_labels, counts)
        if label != -1
    ]
    cluster_info.sort(key=lambda x: x[1], reverse=True)

    noise_count = int(np.sum(labels == -1))
    cluster_time = time.monotonic() - cluster_start

    logger.info(
        "Found %d clusters + %d noise images in %.1fs",
        len(cluster_info),
        noise_count,
        cluster_time,
    )

    if not cluster_info:
        raise PipelineError(
            "No clusters found -- all images classified as noise. "
            "Try reducing min_cluster_size."
        )

    if top is not None:
        cluster_info = cluster_info[:top]

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("Cleaned output directory: %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    selected_labels = {label for label, _ in cluster_info}
    label_to_rank = {label: rank for rank, (label, _) in enumerate(cluster_info)}

    with logging_redirect_tqdm():
        with tqdm(
            total=len(valid_paths),
            desc="Writing clusters",
            unit="image",
            disable=not progress,
        ) as pbar:
            for path, label in zip(valid_paths, labels):
                label_int = int(label)
                if label_int == -1 or label_int not in selected_labels:
                    dest_dir = output_dir / "noise"
                else:
                    rank = label_to_rank[label_int]
                    dest_dir = output_dir / f"{rank:03d}"
                dest_dir.mkdir(parents=True, exist_ok=True)
                copy_image(path, dest_dir / path.name)
                pbar.update(1)

    metadata: dict = {
        "model": model,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "total_images": len(images),
        "embedded_images": len(valid_paths),
        "num_clusters": len(cluster_info),
        "noise_images": noise_count,
        "clusters": [
            {"rank": rank, "label": label, "size": count}
            for rank, (label, count) in enumerate(cluster_info)
        ],
    }
    with open(output_dir / "clusters.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return ClusterResult(
        model=model,
        total_images=len(images),
        embedded_images=len(valid_paths),
        clusters=[
            ClusterInfo(rank=rank, label=label, size=count)
            for rank, (label, count) in enumerate(cluster_info)
        ],
        noise_images=noise_count,
        output_dir=output_dir,
        elapsed=time.monotonic() - start_time,
    )
