"""Click wrapper for ``dtst cluster`` — delegates to :mod:`dtst.core.cluster`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    apply_working_dir,
    config_argument,
    dry_run_option,
    from_dirs_option,
    working_dir_option,
    workers_option,
)
from dtst.core.cluster import cluster as core_cluster
from dtst.embeddings import VALID_MODELS
from dtst.errors import DtstError
from dtst.files import format_elapsed


@click.command("cluster")
@config_argument
@working_dir_option()
@from_dirs_option()
@click.option(
    "--to",
    "-t",
    type=str,
    default=None,
    help="Destination folder.",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(sorted(VALID_MODELS), case_sensitive=False),
    default=None,
    help="Embedding model for similarity (default: arcface).",
)
@click.option(
    "--top",
    "-n",
    type=int,
    default=None,
    help="Maximum number of clusters to output; omit for all clusters.",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=None,
    help="Minimum images to form a cluster (default: 5).",
)
@click.option(
    "--min-samples",
    type=int,
    default=None,
    help="How many close neighbors a point needs to join a cluster; lower values include more borderline images (default: 2).",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=None,
    help="Images per inference batch (default: 32).",
)
@workers_option(help="Number of workers for image preloading (default: CPU count).")
@click.option(
    "--no-cache",
    is_flag=True,
    help="Skip the embedding cache and recompute from scratch.",
)
@click.option(
    "--clean",
    is_flag=True,
    help="Remove the output directory before writing new clusters.",
)
@dry_run_option(help="Show image count and configuration without clustering.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    model: str | None,
    top: int | None,
    min_cluster_size: int | None,
    min_samples: int | None,
    batch_size: int | None,
    workers: int | None,
    no_cache: bool,
    clean: bool,
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

    --min-cluster-size sets the smallest group HDBSCAN will consider
    a real cluster (default: 5). Raise it to suppress small or
    spurious clusters; lower it to capture smaller groups.

    --min-samples controls how conservative the density estimate is
    (default: 2). It decides how many close neighbors a point needs
    before it can join a cluster. Lower values (1-2) let borderline
    images in; higher values push more images into the noise folder.
    Keeping this low while adjusting --min-cluster-size is usually
    the best starting point.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst cluster config.yaml
        dtst cluster -d ./project --from faces --to clusters
        dtst cluster -d ./project --model clip --from raw --to clusters
        dtst cluster -d ./project --top 3 --min-cluster-size 10
        dtst cluster -d ./project --min-samples 1 --min-cluster-size 8
        dtst cluster config.yaml --model arcface --dry-run
    """
    if not from_dirs:
        raise click.ClickException(
            "--from is required (or set 'cluster.from' in config)"
        )
    if not to:
        raise click.ClickException("--to is required (or set 'cluster.to' in config)")

    apply_working_dir(working_dir)
    try:
        result = core_cluster(
            from_dirs=from_dirs,
            to=to,
            model=model or "arcface",
            top=top,
            min_cluster_size=min_cluster_size if min_cluster_size is not None else 5,
            min_samples=min_samples if min_samples is not None else 2,
            batch_size=batch_size if batch_size is not None else 32,
            workers=workers,
            no_cache=no_cache,
            clean=clean,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if dry_run:
        click.echo(f"\nDry run -- would cluster {result.total_images:,} images")
        click.echo(f"  Model: {result.model}")
        click.echo(f"  Output: {result.output_dir}")
        return

    click.echo("\nCluster complete!")
    click.echo(
        f"  Images processed: {result.embedded_images:,} / {result.total_images:,}"
    )
    click.echo(f"  Clusters: {len(result.clusters):,}")
    for info in result.clusters:
        click.echo(f"    {info.rank:03d}/: {info.size:,} images")
    click.echo(f"  Noise: {result.noise_images:,} images")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
