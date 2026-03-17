from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import DedupConfig, load_dedup_config
from dtst.images import find_images
from dtst.sidecar import read_all_sidecars, sidecar_path

logger = logging.getLogger(__name__)


def _read_image_info(args: tuple) -> tuple[str, int, int, int, str | None]:
    (image_path_str,) = args
    try:
        from PIL import Image

        path = Path(image_path_str)
        file_size = path.stat().st_size
        with Image.open(path) as img:
            w, h = img.size
        return (image_path_str, w, h, file_size, None)
    except Exception as exc:
        return (image_path_str, 0, 0, 0, str(exc))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self._parent)):
            result[self.find(i)].append(i)
        return result


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dir: str | None,
    to: str | None,
    threshold: int | None,
) -> DedupConfig:
    if config is not None:
        cfg = load_dedup_config(config)
    else:
        cfg = DedupConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dir is not None:
        cfg.from_dir = from_dir
    if to is not None:
        cfg.to = to
    if threshold is not None:
        cfg.threshold = threshold

    if cfg.from_dir is None:
        raise click.ClickException("--from is required (or set 'dedup.from' in config)")

    return cfg


@click.command("dedup")
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
)
@click.option(
    "--working-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory (default: .).",
)
@click.option(
    "--from",
    "from_dir",
    type=str,
    default=None,
    help="Folder name to deduplicate within the working directory.",
)
@click.option(
    "--to",
    type=str,
    default=None,
    help="Subfolder name for duplicate images.",
    show_default="duplicated",
)
@click.option(
    "--threshold",
    "-t",
    type=int,
    default=None,
    help="Phash hamming distance threshold for near-duplicate detection.",
    show_default="8",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count).",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Restore all deduplicated images back to the source folder.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deduplicated without moving anything.",
)
def cmd(
    config: Path | None,
    working_dir: Path | None,
    from_dir: str | None,
    to: str | None,
    threshold: int | None,
    workers: int | None,
    clear: bool,
    dry_run: bool,
) -> None:
    """Deduplicate images by perceptual hash similarity.

    Groups images by phash hamming distance and keeps the best image
    from each duplicate group. The winner is chosen by resolution
    (width x height), then file size, then blur sharpness. Losers are
    moved to a duplicated/ subdirectory within the source folder
    (configurable with --to).

    Requires phash sidecar data from ``dtst analyze --phash``. Blur
    scores (from ``dtst analyze --blur``) are used as a tiebreaker
    when available.

    \b
    Examples:
      dtst dedup -d ./project --from faces
      dtst dedup -d ./project --from faces --threshold 4
      dtst dedup -d ./project --from faces --to my-dupes
      dtst dedup config.yaml --dry-run
      dtst dedup -d ./project --from faces --clear
    """
    t0 = time.time()

    cfg = _resolve_config(config, working_dir, from_dir, to, threshold)
    source_dir = cfg.working_dir.resolve() / cfg.from_dir
    duplicated_dir = source_dir / cfg.to

    if not source_dir.is_dir():
        raise click.ClickException(f"Source directory not found: {source_dir}")

    if workers is None:
        workers = cpu_count()

    # --- Clear mode ----------------------------------------------------------

    if clear:
        if not duplicated_dir.is_dir():
            click.echo("Nothing to restore (no duplicated/ directory found).")
            return

        dup_images = find_images(duplicated_dir)
        if not dup_images:
            click.echo("Nothing to restore (duplicated/ is empty).")
            return

        if dry_run:
            click.echo(f"\nDry run -- would restore {len(dup_images):,} images to {source_dir}")
            return

        restored = 0
        with logging_redirect_tqdm():
            with tqdm(total=len(dup_images), desc="Restoring", unit="image") as pbar:
                for path in dup_images:
                    dest = source_dir / path.name
                    if dest.exists():
                        logger.warning("Skipping %s (already exists in source)", path.name)
                    else:
                        path.rename(dest)
                        sc = sidecar_path(path)
                        if sc.exists():
                            sc.rename(sidecar_path(dest))
                        restored += 1
                    pbar.update(1)

        try:
            duplicated_dir.rmdir()
        except OSError:
            pass

        click.echo(f"\nRestore complete!")
        click.echo(f"  Restored: {restored:,}")
        click.echo(f"  Source: {source_dir}")
        return

    # --- Dedup mode ----------------------------------------------------------

    images = find_images(source_dir)
    if not images:
        raise click.ClickException(f"No images found in: {source_dir}")

    logger.info("Found %d images in %s", len(images), source_dir)

    sidecars = read_all_sidecars(images)

    has_phash: list[Path] = []
    for img in images:
        sc = sidecars.get(img, {})
        phash_data = sc.get("phash")
        if phash_data and phash_data.get("hash"):
            has_phash.append(img)
        else:
            logger.warning("No phash data for %s, skipping", img.name)

    if len(has_phash) < 2:
        click.echo("Fewer than 2 images with phash data. Nothing to deduplicate.")
        return

    logger.info("Reading image metadata for %d images", len(has_phash))
    image_info: dict[Path, tuple[int, int, int]] = {}
    errors = 0

    with logging_redirect_tqdm():
        work = [(str(p),) for p in has_phash]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_read_image_info, w): w for w in work}
            with tqdm(total=len(work), desc="Reading metadata", unit="image") as pbar:
                for future in as_completed(futures):
                    path_str, w, h, file_size, err = future.result()
                    if err:
                        logger.error("Failed to read %s: %s", Path(path_str).name, err)
                        errors += 1
                    else:
                        image_info[Path(path_str)] = (w, h, file_size)
                    pbar.update(1)

    valid_images = [p for p in has_phash if p in image_info]
    if len(valid_images) < 2:
        click.echo("Fewer than 2 readable images with phash data. Nothing to deduplicate.")
        return

    import imagehash

    hashes: list[imagehash.ImageHash] = []
    for img in valid_images:
        h = imagehash.hex_to_hash(sidecars[img]["phash"]["hash"])
        hashes.append(h)

    logger.info("Computing pairwise hamming distances (threshold=%d)", cfg.threshold)
    n = len(valid_images)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= cfg.threshold:
                uf.union(i, j)

    groups = uf.groups()
    dup_groups = {root: members for root, members in groups.items() if len(members) > 1}

    if not dup_groups:
        elapsed = time.time() - t0
        click.echo(f"No duplicates found among {n:,} images ({elapsed:.1f}s)")
        return

    total_duplicates = sum(len(m) - 1 for m in dup_groups.values())
    logger.info("Found %d duplicate groups (%d images to move)", len(dup_groups), total_duplicates)

    losers: list[tuple[Path, str]] = []
    for members in dup_groups.values():
        group_paths = [valid_images[i] for i in members]

        def sort_key(p: Path) -> tuple[int, int, float]:
            w, h, fsize = image_info[p]
            resolution = w * h
            sc = sidecars.get(p, {})
            blur_score = sc.get("blur", {}).get("score", 0.0)
            return (resolution, fsize, blur_score)

        group_paths.sort(key=sort_key, reverse=True)
        winner = group_paths[0]
        w_w, w_h, w_fsize = image_info[winner]
        logger.debug("Winner: %s (%dx%d, %d bytes)", winner.name, w_w, w_h, w_fsize)
        for loser in group_paths[1:]:
            l_w, l_h, l_fsize = image_info[loser]
            reason = f"dup of {winner.name} ({l_w}x{l_h}, {l_fsize} bytes)"
            losers.append((loser, reason))

    if dry_run:
        click.echo(f"\nDry run -- would move {len(losers):,} duplicates from {len(dup_groups):,} groups")
        for path, reason in losers[:10]:
            click.echo(f"  {path.name} ({reason})")
        if len(losers) > 10:
            click.echo(f"  ... and {len(losers) - 10:,} more")
        return

    duplicated_dir.mkdir(parents=True, exist_ok=True)
    moved = 0

    with logging_redirect_tqdm():
        with tqdm(total=len(losers), desc="Moving duplicates", unit="image") as pbar:
            for path, reason in losers:
                try:
                    dest = duplicated_dir / path.name
                    path.rename(dest)
                    sc = sidecar_path(path)
                    if sc.exists():
                        sc.rename(sidecar_path(dest))
                    moved += 1
                    logger.debug("Moved %s (%s)", path.name, reason)
                except OSError as e:
                    logger.error("Failed to move %s: %s", path.name, e)
                pbar.update(1)

    elapsed = time.time() - t0
    click.echo(f"\nDedup complete!")
    click.echo(f"  Groups: {len(dup_groups):,}")
    click.echo(f"  Kept: {len(valid_images) - moved:,}")
    click.echo(f"  Moved: {moved:,}")
    if errors > 0:
        click.echo(f"  Errors: {errors:,}")
    click.echo(f"  Source: {source_dir}")
    click.echo(f"  Duplicates: {duplicated_dir}")
    click.echo(f"  Time: {elapsed:.1f}s")
