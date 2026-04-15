"""Click wrapper for ``dtst rename`` — delegates to :mod:`dtst.core.rename`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    config_argument,
    dry_run_option,
    from_dirs_option,
    working_dir_option,
)
from dtst.core.rename import rename as core_rename
from dtst.errors import DtstError


@click.command("rename")
@config_argument
@from_dirs_option()
@click.option(
    "--prefix",
    "-p",
    type=str,
    default=None,
    help="Filename prefix for renamed files (default: '').",
)
@click.option(
    "--digits",
    "-n",
    type=int,
    default=None,
    help="Number of zero-padded digits (default: auto based on total count).",
)
@working_dir_option()
@dry_run_option(help="Preview renames without executing.")
def cmd(
    from_dirs: str | None,
    prefix: str | None,
    digits: int | None,
    working_dir: Path | None,
    dry_run: bool,
) -> None:
    """Sequentially rename images in-place with a prefix and zero-padded number.

    Renames all images in the given folders to {prefix}{number}.{ext},
    where the number is zero-padded to the specified number of digits.
    Sidecar JSON files are moved along with their images. Operates
    in-place — there is no --to option.

    \b
    Examples:
        dtst rename --from raw --prefix "img_" -d ./my-dataset
        dtst rename --from raw --prefix "photo_" --digits 5 -d ./my-dataset
        dtst rename config.yaml --dry-run
        dtst rename --from faces --prefix "face_" -n 4
    """
    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'rename.from' in config)"
        )

    try:
        result = core_rename(
            working_dir=working_dir,
            from_dirs=from_dirs,
            prefix=prefix or "",
            digits=digits,
            dry_run=dry_run,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if result.dry_run:
        click.echo(f"[dry-run] Would rename {result.renamed} images")
    else:
        click.echo(f"Done: {result.renamed} images renamed ({result.elapsed:.1f}s)")
