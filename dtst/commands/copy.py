from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import CopyConfig, load_copy_config
from dtst.files import find_images, resolve_dirs

logger = logging.getLogger(__name__)


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: list[str] | None,
    to: str | None,
) -> CopyConfig:
    if config is not None:
        cfg = load_copy_config(config)
    else:
        cfg = CopyConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dirs is not None:
        cfg.from_dirs = from_dirs
    if to is not None:
        cfg.to = to

    if cfg.from_dirs is None:
        raise click.ClickException("--from is required (or set 'copy.from' in config)")
    if cfg.to is None:
        raise click.ClickException("--to is required (or set 'copy.to' in config)")

    return cfg


@click.command("copy")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory containing source folders and where output is written (default: .).")
@click.option("--from", "from_dirs", type=str, default=None, help="Comma-separated source folders within the working directory (supports globs, e.g. 'images/*').")
@click.option("--to", type=str, default=None, help="Destination folder name within the working directory.")
@click.option("--dry-run", is_flag=True, help="Preview what would be copied without creating files.")
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    dry_run: bool,
) -> None:
    """Copy images from one or more folders to a destination folder.

    Duplicates the contents of the source folders into the destination
    without any transformation. Files that already exist in the
    destination (by name) are skipped.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:
        dtst copy -d ./project --from raw --to backup
        dtst copy -d ./project --from raw,extra --to combined
        dtst copy config.yaml --dry-run
    """
    parsed_from_dirs: list[str] | None = None
    if from_dirs is not None:
        parsed_from_dirs = [d.strip() for d in from_dirs.split(",") if d.strip()]
        if not parsed_from_dirs:
            raise click.ClickException("--from must contain at least one folder name")

    cfg = _resolve_config(config, working_dir, parsed_from_dirs, to)

    input_dirs = resolve_dirs(cfg.working_dir, cfg.from_dirs)
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

    from_label = ", ".join(str(d) for d in input_dirs)
    logger.info("Copying %d images from [%s] to %s", len(images), from_label, output_dir)

    if dry_run:
        click.echo(f"\nDry run -- would copy {len(images):,} images")
        click.echo(f"  From: {from_label}")
        click.echo(f"  To: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    ok_count = 0
    skipped_count = 0
    failed_count = 0

    with logging_redirect_tqdm():
        with tqdm(total=len(images), desc="Copying", unit="image") as pbar:
            for image_path in images:
                dest = output_dir / image_path.name
                try:
                    if dest.exists():
                        skipped_count += 1
                        logger.debug("Skipped (already exists): %s", image_path.name)
                    else:
                        shutil.copy2(image_path, dest)
                        ok_count += 1
                except Exception as e:
                    failed_count += 1
                    logger.error("Failed to copy %s: %s", image_path.name, e)
                pbar.set_postfix(ok=ok_count, skip=skipped_count, fail=failed_count)
                pbar.update(1)

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nCopy complete!")
    click.echo(f"  Copied: {ok_count:,}")
    click.echo(f"  Skipped: {skipped_count:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {output_dir}")
