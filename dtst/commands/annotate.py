from __future__ import annotations

import logging
import time
from pathlib import Path

import click

from dtst.config import config_argument
from dtst.files import find_images, resolve_dirs
from dtst.sidecar import read_sidecar, write_sidecar

logger = logging.getLogger(__name__)


@click.command("annotate")
@config_argument
@click.option(
    "--from",
    "from_dirs",
    type=str,
    default=None,
    help="Comma-separated source folders (supports globs, e.g. 'images/*').",
)
@click.option(
    "--source",
    "-s",
    type=str,
    default=None,
    help="Source name to write (e.g. 'unsplash', 'personal').",
)
@click.option(
    "--license",
    "-l",
    type=str,
    default=None,
    help="License string to write (e.g. 'cc-by', 'cc0', 'all-rights-reserved').",
)
@click.option(
    "--origin",
    "-o",
    type=str,
    default=None,
    help="Origin URL to write (applied to all images).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing source/license/origin values in sidecars.",
)
@click.option(
    "--working-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory (default: .).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview what would be annotated without writing sidecars.",
)
def cmd(from_dirs, source, license, origin, overwrite, working_dir, dry_run):
    """Write source and license metadata into image sidecars.

    Annotates all images in the given folders with provenance metadata
    (source, license, origin). Useful for manually imported images that
    were not fetched through the pipeline. Sidecars are merged
    incrementally — existing fields are preserved unless --overwrite
    is used.

    At least one of --source, --license, or --origin is required.

    \b
    Examples:
        dtst annotate --from extra --source "unsplash" --license "cc0" -d ./my-dataset
        dtst annotate config.yaml
        dtst annotate --from raw,extra --source "personal" --license "all-rights-reserved"
        dtst annotate --from extra --source "flickr" --overwrite -d ./my-dataset
        dtst annotate --from extra --source "personal" --dry-run
    """
    t0 = time.time()

    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'annotate.from' in config)"
        )
    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    working = (working_dir or Path(".")).resolve()

    if not source and not license and not origin:
        raise click.ClickException(
            "At least one of --source, --license, or --origin is required."
        )

    input_dirs = resolve_dirs(working, dirs_list)

    all_images: list[Path] = []
    for src in input_dirs:
        if not src.is_dir():
            logger.warning("Source directory does not exist, skipping: %s", src)
            continue
        all_images.extend(find_images(src))

    if not all_images:
        raise click.ClickException("No images found in source directories.")

    annotation: dict[str, str] = {}
    if source:
        annotation["source"] = source
    if license:
        annotation["license"] = license
    if origin:
        annotation["origin"] = origin

    logger.info(
        "Found %d images in %s, annotating: %s",
        len(all_images),
        ", ".join(dirs_list),
        ", ".join(f"{k}={v}" for k, v in annotation.items()),
    )

    annotated = 0
    skipped = 0

    for img in all_images:
        existing = read_sidecar(img)
        if not overwrite:
            new_data = {k: v for k, v in annotation.items() if k not in existing}
        else:
            new_data = dict(annotation)

        if not new_data:
            skipped += 1
            continue

        if dry_run:
            fields = ", ".join(f"{k}={v}" for k, v in new_data.items())
            logger.debug("%s: %s", img.name, fields)
            annotated += 1
            continue

        write_sidecar(img, new_data)
        annotated += 1

    elapsed = time.time() - t0

    if dry_run:
        click.echo(
            f"[dry-run] Would annotate {annotated} images, skip {skipped} (already set)"
        )
    else:
        click.echo(f"Done: {annotated} annotated, {skipped} skipped ({elapsed:.1f}s)")
