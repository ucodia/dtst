from __future__ import annotations

import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import config_argument
from dtst.files import build_save_kwargs, find_images, resolve_dirs
from dtst.sidecar import copy_sidecar

logger = logging.getLogger(__name__)


def _transform_image(args: tuple) -> tuple[str, str, list[str], str | None]:
    """Top-level worker function for ProcessPoolExecutor.

    Returns ``(status, filename, output_files, error_message)``.
    Status is one of ``"ok"`` or ``"failed"``.
    """
    (
        input_path_s,
        output_dir_s,
        flip_x,
        flip_y,
        flip_xy,
        copy_original,
    ) = args
    input_path = Path(input_path_s)
    output_dir = Path(output_dir_s)
    name = input_path.name
    stem = input_path.stem
    ext = input_path.suffix

    created: list[str] = []

    try:
        from PIL import Image

        img = Image.open(input_path)
        save_kw = build_save_kwargs(input_path)

        if copy_original:
            dest = output_dir / name
            shutil.copy2(input_path, dest)
            created.append(name)

        if flip_x:
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            out_name = f"{stem}_flipX{ext}"
            flipped.save(output_dir / out_name, **save_kw)
            created.append(out_name)

        if flip_y:
            flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
            out_name = f"{stem}_flipY{ext}"
            flipped.save(output_dir / out_name, **save_kw)
            created.append(out_name)

        if flip_xy:
            flipped = img.transpose(Image.ROTATE_180)
            out_name = f"{stem}_flipXY{ext}"
            flipped.save(output_dir / out_name, **save_kw)
            created.append(out_name)

        img.close()
        return "ok", name, created, None

    except Exception as e:
        return "failed", name, created, str(e)


@click.command("augment")
@config_argument
@click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None, help="Working directory containing source folders and where output is written (default: .).")
@click.option("--from", "from_dirs", type=str, default=None, help="Comma-separated source folders within the working directory (supports globs, e.g. 'images/*').")
@click.option("--to", type=str, default=None, help="Destination folder name within the working directory.")
@click.option("--flipX", "flip_x", is_flag=True, help="Apply horizontal flip.")
@click.option("--flipY", "flip_y", is_flag=True, help="Apply vertical flip.")
@click.option("--flipXY", "flip_xy", is_flag=True, help="Apply both horizontal and vertical flip (180-degree rotation).")
@click.option("--no-copy", is_flag=True, help="Do not copy original images to the output folder.")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count).")
@click.option("--dry-run", is_flag=True, help="Preview what would be written without creating files.")
def cmd(
    working_dir: Path | None,
    from_dirs: str | None,
    to: str | None,
    flip_x: bool,
    flip_y: bool,
    flip_xy: bool,
    no_copy: bool,
    workers: int | None,
    dry_run: bool,
) -> None:
    """Augment a dataset by applying image transformations.

    Reads images from one or more source folders and writes transformed
    copies to a destination folder. By default the original images are
    also copied to the output; use --no-copy to write only the
    transformed versions.

    At least one transform flag (--flipX, --flipY, --flipXY) is
    required. Multiple flags can be combined in a single run to
    produce several variants of each image.

    Transformed files are named with a suffix indicating the transform:
    photo.jpg becomes photo_flipX.jpg, photo_flipY.jpg, photo_flipXY.jpg.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst augment -d ./project --from faces --to augmented --flipX
        dtst augment -d ./project --from faces --to augmented --flipX --flipY --flipXY
        dtst augment -d ./project --from faces --to augmented --flipX --no-copy
        dtst augment config.yaml --dry-run
    """
    if from_dirs is None:
        raise click.ClickException("--from is required (or set 'augment.from' in config)")
    if to is None:
        raise click.ClickException("--to is required (or set 'augment.to' in config)")
    dirs_list = [d.strip() for d in from_dirs.split(",") if d.strip()]
    working = (working_dir or Path(".")).resolve()
    if not flip_x and not flip_y and not flip_xy:
        raise click.ClickException(
            "At least one transform flag is required (--flipX, --flipY, --flipXY)"
        )

    input_dirs = resolve_dirs(working, dirs_list)
    output_dir = working / to

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

    transforms = []
    if flip_x:
        transforms.append("flipX")
    if flip_y:
        transforms.append("flipY")
    if flip_xy:
        transforms.append("flipXY")

    copies_per_image = len(transforms) + (0 if no_copy else 1)
    total_output = len(images) * copies_per_image

    from_label = ", ".join(str(d) for d in input_dirs)
    logger.info(
        "Augmenting %d images from [%s] with transforms [%s] (copy_original=%s, workers=%d expected output=%d)",
        len(images), from_label, ", ".join(transforms),
        not no_copy, workers if workers is not None else cpu_count() or 4,
        total_output,
    )

    if dry_run:
        click.echo(f"\nDry run -- would augment {len(images):,} images")
        click.echo(f"  Transforms: {', '.join(transforms)}")
        click.echo(f"  Copy originals: {not no_copy}")
        click.echo(f"  Files per image: {copies_per_image}")
        click.echo(f"  Total output files: {total_output:,}")
        click.echo(f"  Output: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    num_workers = workers if workers is not None else cpu_count() or 4
    copy_original = not no_copy

    work = [
        (
            str(img_path),
            str(output_dir),
            flip_x,
            flip_y,
            flip_xy,
            copy_original,
        )
        for img_path in images
    ]

    start_time = time.monotonic()
    ok_count = 0
    failed_count = 0
    total_files = 0

    with logging_redirect_tqdm():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_transform_image, w): w for w in work}
            with tqdm(total=len(futures), desc="Augmenting", unit="image") as pbar:
                try:
                    for future in as_completed(futures):
                        status, name, created, error = future.result()
                        if status == "ok":
                            ok_count += 1
                            total_files += len(created)
                            src_path = Path(futures[future][0])
                            for out_name in created:
                                copy_sidecar(src_path, output_dir / out_name, exclude={"metrics", "classes"})
                        else:
                            failed_count += 1
                            total_files += len(created)
                            logger.error("Failed to process %s: %s", name, error)
                        pbar.set_postfix(ok=ok_count, fail=failed_count)
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"\nAugment complete!")
    click.echo(f"  Images processed: {ok_count:,}")
    click.echo(f"  Files written: {total_files:,}")
    click.echo(f"  Failed: {failed_count:,}")
    click.echo(f"  Time: {minutes}m {seconds}s")
    click.echo(f"  Output: {output_dir}")
