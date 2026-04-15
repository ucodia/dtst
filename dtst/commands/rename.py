from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import click

from dtst.config import (
    config_argument,
    dry_run_option,
    from_dirs_option,
    working_dir_option,
)
from dtst.files import gather_images, move_image

logger = logging.getLogger(__name__)


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
def cmd(from_dirs, prefix, digits, working_dir, dry_run):
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
    t0 = time.time()

    if from_dirs is None:
        raise click.ClickException(
            "--from is required (or set 'rename.from' in config)"
        )

    prefix = prefix or ""
    _working, _input_dirs, all_images = gather_images(working_dir, from_dirs)
    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]

    pad = digits if digits is not None else len(str(len(all_images)))

    logger.info(
        "Found %d images in %s, renaming with prefix='%s', digits=%d",
        len(all_images),
        ", ".join(dirs_list),
        prefix,
        pad,
    )

    if dry_run:
        for i, img in enumerate(all_images, start=1):
            new_name = f"{prefix}{i:0{pad}d}{img.suffix}"
            logger.debug("%s -> %s", img.name, new_name)
        click.echo(f"[dry-run] Would rename {len(all_images)} images")
        return

    # Phase 1: move to temporary names to avoid collisions
    tag = uuid.uuid4().hex[:8]
    temp_pairs: list[tuple[Path, Path, str]] = []
    for i, img in enumerate(all_images, start=1):
        tmp_name = f"__rename_tmp_{tag}_{i}{img.suffix}"
        tmp_path = img.parent / tmp_name
        move_image(img, tmp_path)
        final_name = f"{prefix}{i:0{pad}d}{img.suffix}"
        temp_pairs.append((tmp_path, img.parent / final_name, img.suffix))

    # Phase 2: move from temporary to final names
    renamed = 0
    for tmp_path, final_path, _ in temp_pairs:
        move_image(tmp_path, final_path)
        renamed += 1

    elapsed = time.time() - t0
    click.echo(f"Done: {renamed} images renamed ({elapsed:.1f}s)")
