"""Click wrapper for ``dtst annotate`` — delegates to :mod:`dtst.core.annotate`."""

from __future__ import annotations


import click

from dtst.cli.config import (
    apply_working_dir,
    config_argument,
    dry_run_option,
    from_dirs_option,
    working_dir_option,
)
from dtst.errors import DtstError


@click.command("annotate")
@config_argument
@from_dirs_option()
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
@working_dir_option()
@dry_run_option(help="Preview what would be annotated without writing sidecars.")
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
    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'annotate.from' in config)"
        )
    if not source and not license and not origin:
        raise click.ClickException(
            "At least one of --source, --license, or --origin is required."
        )

    apply_working_dir(working_dir)
    from dtst.core.annotate import annotate as core_annotate

    try:
        result = core_annotate(
            from_dirs=from_dirs,
            source=source,
            license=license,
            origin=origin,
            overwrite=overwrite,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if result.dry_run:
        click.echo(
            f"[dry-run] Would annotate {result.annotated} images, skip {result.skipped} (already set)"
        )
    else:
        click.echo(
            f"Done: {result.annotated} annotated, {result.skipped} skipped ({result.elapsed:.1f}s)"
        )
