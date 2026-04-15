"""Click wrapper for ``dtst analyze`` — delegates to :mod:`dtst.core.analyze`."""

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
from dtst.errors import DtstError


@click.command("analyze")
@config_argument
@from_dirs_option()
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
@working_dir_option()
@workers_option(help="Number of parallel workers for CPU metrics (default: CPU count).")
@click.option(
    "--clear", is_flag=True, help="Remove all sidecar files from source folders."
)
@dry_run_option(help="Preview what would be computed without writing sidecars.")
def cmd(
    from_dirs: str | None,
    metrics: str | None,
    force: bool,
    working_dir: Path | None,
    workers: int | None,
    clear: bool,
    dry_run: bool,
) -> None:
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
    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'analyze.from' in config)"
        )
    if not clear and not metrics:
        raise click.ClickException(
            "At least one metric is required via --metrics (e.g. --metrics phash,blur,musiq)."
        )

    apply_working_dir(working_dir)
    from dtst.core.analyze import analyze as core_analyze

    try:
        result = core_analyze(
            from_dirs=from_dirs,
            metrics=metrics,
            force=force,
            workers=workers,
            clear=clear,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if clear:
        if result.cleared == 0 and not result.dry_run:
            click.echo("No sidecar files found. Nothing to clear.")
            return
        if result.dry_run:
            click.echo(f"[dry-run] Would remove {result.cleared:,} sidecar files")
            return
        click.echo(f"Removed {result.cleared:,} sidecar files ({result.elapsed:.1f}s)")
        return

    if result.dry_run:
        click.echo(
            f"[dry-run] Would analyze {result.analyzed} images, skip {result.skipped} (already computed)"
        )
        return

    if result.analyzed == 0 and result.failed == 0:
        total = result.analyzed + result.skipped
        click.echo(f"All {total} images already have requested metrics. Nothing to do.")
        return

    click.echo(
        f"Done: {result.analyzed} analyzed, {result.skipped} skipped, {result.failed} failed ({result.elapsed:.1f}s)"
    )
