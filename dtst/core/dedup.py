"""Library-layer implementation of ``dtst dedup``."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.errors import InputError
from dtst.executor import run_pool
from dtst.files import find_images, resolve_workers
from dtst.results import DedupResult
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


def dedup(
    *,
    from_dir: str,
    to: str = "duplicated",
    threshold: int = 8,
    workers: int | None = None,
    clear: bool = False,
    dry_run: bool = False,
    prefer_upscaled: bool = False,
    progress: bool = True,
) -> DedupResult:
    """Deduplicate images by perceptual hash similarity.

    ``to`` is a subfolder *inside* ``from_dir`` where duplicates are
    moved.
    """
    t0 = time.time()

    if not from_dir:
        raise InputError("from_dir is required")

    source_dir = Path(from_dir).expanduser().resolve()
    duplicated_dir = source_dir / to

    if not source_dir.is_dir():
        raise InputError(f"Source directory not found: {source_dir}")

    num_workers = resolve_workers(workers)

    if clear:
        if not duplicated_dir.is_dir():
            return DedupResult(
                mode="restore",
                message="Nothing to restore (no duplicated/ directory found).",
                source_dir=source_dir,
                duplicated_dir=duplicated_dir,
                elapsed=time.time() - t0,
            )

        dup_images = find_images(duplicated_dir)
        if not dup_images:
            return DedupResult(
                mode="restore",
                message="Nothing to restore (duplicated/ is empty).",
                source_dir=source_dir,
                duplicated_dir=duplicated_dir,
                elapsed=time.time() - t0,
            )

        if dry_run:
            return DedupResult(
                mode="restore",
                total_losers=len(dup_images),
                dry_run=True,
                source_dir=source_dir,
                duplicated_dir=duplicated_dir,
                elapsed=time.time() - t0,
            )

        restored = 0
        with logging_redirect_tqdm():
            with tqdm(
                total=len(dup_images),
                desc="Restoring",
                unit="image",
                disable=not progress,
            ) as pbar:
                for path in dup_images:
                    dest = source_dir / path.name
                    if dest.exists():
                        logger.warning(
                            "Skipping %s (already exists in source)", path.name
                        )
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

        return DedupResult(
            mode="restore",
            restored=restored,
            source_dir=source_dir,
            duplicated_dir=duplicated_dir,
            elapsed=time.time() - t0,
        )

    images = find_images(source_dir)
    if not images:
        raise InputError(f"No images found in: {source_dir}")

    logger.info("Found %d images in %s", len(images), source_dir)

    sidecars = read_all_sidecars(images)

    has_phash: list[Path] = []
    for img in images:
        sc = sidecars.get(img, {})
        metrics = sc.get("metrics", {})
        if metrics.get("phash"):
            has_phash.append(img)
        else:
            logger.warning("No phash data for %s, skipping", img.name)

    if len(has_phash) < 2:
        return DedupResult(
            mode="noop",
            message="Fewer than 2 images with phash data. Nothing to deduplicate.",
            source_dir=source_dir,
            duplicated_dir=duplicated_dir,
            elapsed=time.time() - t0,
        )

    logger.info("Reading image metadata for %d images", len(has_phash))
    image_info: dict[Path, tuple[int, int, int]] = {}

    work = [(str(p),) for p in has_phash]

    def handle(result, _work_item):
        path_str, w, h, file_size, err = result
        if err:
            logger.error("Failed to read %s: %s", Path(path_str).name, err)
            return "fail"
        image_info[Path(path_str)] = (w, h, file_size)
        return "ok"

    counts = run_pool(
        ProcessPoolExecutor,
        _read_image_info,
        work,
        max_workers=num_workers,
        desc="Reading metadata",
        unit="image",
        on_result=handle,
        progress=progress,
    )
    errors = counts.get("fail", 0)

    valid_images = [p for p in has_phash if p in image_info]
    if len(valid_images) < 2:
        return DedupResult(
            mode="noop",
            message="Fewer than 2 readable images with phash data. Nothing to deduplicate.",
            errors=errors,
            source_dir=source_dir,
            duplicated_dir=duplicated_dir,
            elapsed=time.time() - t0,
        )

    import imagehash

    hashes: list[imagehash.ImageHash] = []
    for img in valid_images:
        h = imagehash.hex_to_hash(sidecars[img]["metrics"]["phash"])
        hashes.append(h)

    logger.info("Computing pairwise hamming distances (threshold=%d)", threshold)
    n = len(valid_images)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= threshold:
                uf.union(i, j)

    groups = uf.groups()
    dup_groups = {root: members for root, members in groups.items() if len(members) > 1}

    if not dup_groups:
        return DedupResult(
            mode="noop",
            message=f"No duplicates found among {n:,} images",
            kept=n,
            source_dir=source_dir,
            duplicated_dir=duplicated_dir,
            elapsed=time.time() - t0,
        )

    total_duplicates = sum(len(m) - 1 for m in dup_groups.values())
    logger.info(
        "Found %d duplicate groups (%d images to move)",
        len(dup_groups),
        total_duplicates,
    )

    losers: list[tuple[Path, str]] = []
    for members in dup_groups.values():
        group_paths = [valid_images[i] for i in members]

        def sort_key(p: Path) -> tuple[int, int, int, float]:
            w, h, fsize = image_info[p]
            resolution = w * h
            sc = sidecars.get(p, {})
            blur_score = sc.get("metrics", {}).get("blur", 0.0)
            is_upscaled = 1 if sc.get("upscale") else 0
            preference = is_upscaled if prefer_upscaled else (1 - is_upscaled)
            return (preference, resolution, fsize, blur_score)

        group_paths.sort(key=sort_key, reverse=True)
        winner = group_paths[0]
        w_w, w_h, w_fsize = image_info[winner]
        logger.debug("Winner: %s (%dx%d, %d bytes)", winner.name, w_w, w_h, w_fsize)
        for loser in group_paths[1:]:
            l_w, l_h, l_fsize = image_info[loser]
            reason = f"dup of {winner.name} ({l_w}x{l_h}, {l_fsize} bytes)"
            losers.append((loser, reason))

    if dry_run:
        kept = len(valid_images) - len(losers)
        preview = [(p.name, r) for p, r in losers[:10]]
        return DedupResult(
            mode="dedup",
            groups=len(dup_groups),
            kept=kept,
            total_losers=len(losers),
            losers_preview=preview,
            dry_run=True,
            errors=errors,
            source_dir=source_dir,
            duplicated_dir=duplicated_dir,
            elapsed=time.time() - t0,
        )

    duplicated_dir.mkdir(parents=True, exist_ok=True)
    moved = 0

    with logging_redirect_tqdm():
        with tqdm(
            total=len(losers),
            desc="Moving duplicates",
            unit="image",
            disable=not progress,
        ) as pbar:
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

    return DedupResult(
        mode="dedup",
        groups=len(dup_groups),
        kept=len(valid_images) - moved,
        moved=moved,
        errors=errors,
        total_losers=len(losers),
        source_dir=source_dir,
        duplicated_dir=duplicated_dir,
        elapsed=time.time() - t0,
    )
