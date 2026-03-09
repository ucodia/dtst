from __future__ import annotations

import json
import logging
import shutil
import time
from multiprocessing import cpu_count
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import ClusterConfig, load_cluster_config
from dtst.embeddings import VALID_MODELS, detect_device, get_backend
from dtst.images import find_images

logger = logging.getLogger(__name__)


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: list[str] | None,
    to: str | None,
    model: str | None,
    top: int | None,
    min_cluster_size: int | None,
    batch_size: int | None,
) -> ClusterConfig:
    if config is not None:
        cfg = load_cluster_config(config)
    else:
        cfg = ClusterConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dirs is not None:
        cfg.from_dirs = from_dirs
    if to is not None:
        cfg.to = to
    if model is not None:
        cfg.model = model
    if top is not None:
        cfg.top = top
    if min_cluster_size is not None:
        cfg.min_cluster_size = min_cluster_size
    if batch_size is not None:
        cfg.batch_size = batch_size

    return cfg


@click.command("cluster")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory containing source folders and where output is written (default: .).")
@click.option("--from", "from_dirs", type=str, default=None, help="Comma-separated source folder names within the working directory (default: faces).")
@click.option("--to", "-t", type=str, default=None, help="Destination folder name within the working directory (default: clusters).")
@click.option("--model", "-m", type=click.Choice(sorted(VALID_MODELS), case_sensitive=False), default=None, help="Embedding model for similarity (default: arcface).")
@click.option("--top", "-n", type=int, default=None, help="Maximum number of clusters to output; omit for all clusters.")
@click.option("--min-cluster-size", type=int, default=None, help="Minimum images to form a cluster (default: 5).")
@click.option("--batch-size", "-b", type=int, default=None, help="Images per inference batch (default: 32).")
@click.option("--workers", "-w", type=int, default=None, help="Number of workers for image preloading (default: CPU count).")
@click.option("--dry-run", is_flag=True, help="Show image count and configuration without clustering.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    model: str | None,
    top: int | None,
    min_cluster_size: int | None,
    batch_size: int | None,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Cluster images by visual similarity.

    Groups images into clusters based on embedding similarity using
    HDBSCAN. Each cluster is written to a numbered subdirectory
    (000 = largest, 001 = second largest, etc.) within the output
    folder. Images that do not belong to any cluster are placed in
    a noise/ subdirectory.

    Supports two embedding models: arcface for face identity
    clustering (requires face images) and clip for general visual
    similarity clustering (works with any images).

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:
        dtst cluster config.yaml
        dtst cluster -d ./project --from faces --to clusters
        dtst cluster -d ./project --model clip --from raw --to clusters
        dtst cluster -d ./project --top 3 --min-cluster-size 10
        dtst cluster config.yaml --model arcface --dry-run
    """
    parsed_from_dirs: list[str] | None = None
    if from_dirs is not None:
        parsed_from_dirs = [d.strip() for d in from_dirs.split(",") if d.strip()]
        if not parsed_from_dirs:
            raise click.ClickException("--from must contain at least one folder name")

    cfg = _resolve_config(
        config, working_dir, parsed_from_dirs, to, model, top, min_cluster_size, batch_size,
    )

    input_dirs = [cfg.working_dir / d for d in cfg.from_dirs]
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
    from_label = ", ".join(str(d) for d in input_dirs)
    top_label = str(cfg.top) if cfg.top is not None else "all"

    logger.info(
        "Clustering %d images from [%s] (model=%s, min_cluster_size=%d, top=%s, batch_size=%d)",
        len(images), from_label, cfg.model, cfg.min_cluster_size, top_label, cfg.batch_size,
    )

    if dry_run:
        click.echo(f"\nDry run -- would cluster {len(images):,} images")
        click.echo(f"  Model: {cfg.model}")
        click.echo(f"  Min cluster size: {cfg.min_cluster_size}")
        click.echo(f"  Top clusters: {top_label}")
        click.echo(f"  Output: {output_dir}")
        return

    # --- Embedding -----------------------------------------------------------

    start_time = time.monotonic()
    device = detect_device()
    backend = get_backend(cfg.model)
    backend.load(device)

    with logging_redirect_tqdm():
        embeddings, valid_paths = backend.embed(
            images,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
        )

    if len(valid_paths) == 0:
        raise click.ClickException("No images produced valid embeddings")

    embed_time = time.monotonic() - start_time
    logger.info(
        "Embedded %d / %d images in %.1fs",
        len(valid_paths), len(images), embed_time,
    )

    # --- Clustering ----------------------------------------------------------

    import hdbscan

    cluster_start = time.monotonic()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        metric="euclidean",
        cluster_selection_epsilon=0.0,
    )
    labels = clusterer.fit_predict(embeddings)

    # Gather clusters sorted by size (descending)
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
        len(cluster_info), noise_count, cluster_time,
    )

    if not cluster_info:
        raise click.ClickException(
            "No clusters found -- all images classified as noise. "
            "Try reducing --min-cluster-size."
        )

    # Apply --top limit
    if cfg.top is not None:
        cluster_info = cluster_info[: cfg.top]

    # --- Write output --------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    # Build label-to-rank mapping for selected clusters
    selected_labels = {label for label, _ in cluster_info}
    label_to_rank = {label: rank for rank, (label, _) in enumerate(cluster_info)}

    with logging_redirect_tqdm():
        with tqdm(total=len(valid_paths), desc="Writing clusters", unit="image") as pbar:
            for i, (path, label) in enumerate(zip(valid_paths, labels)):
                label_int = int(label)

                if label_int == -1:
                    dest_dir = output_dir / "noise"
                elif label_int in selected_labels:
                    rank = label_to_rank[label_int]
                    dest_dir = output_dir / f"{rank:03d}"
                else:
                    # Cluster exists but not in --top selection; treat as noise
                    dest_dir = output_dir / "noise"

                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest_dir / path.name)
                copied += 1
                pbar.update(1)

    # --- Write metadata ------------------------------------------------------

    metadata = {
        "model": cfg.model,
        "min_cluster_size": cfg.min_cluster_size,
        "total_images": len(images),
        "embedded_images": len(valid_paths),
        "num_clusters": len(cluster_info),
        "noise_images": noise_count,
        "clusters": [
            {"rank": rank, "label": label, "size": count}
            for rank, (label, count) in enumerate(cluster_info)
        ],
    }

    meta_path = output_dir / "clusters.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Summary -------------------------------------------------------------

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nCluster complete!")
    click.echo(f"  Images processed: {len(valid_paths):,} / {len(images):,}")
    click.echo(f"  Clusters: {len(cluster_info):,}")
    for rank, (label, count) in enumerate(cluster_info):
        click.echo(f"    {rank:03d}/: {count:,} images")
    click.echo(f"  Noise: {noise_count:,} images")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {output_dir}")
