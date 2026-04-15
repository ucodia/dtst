import logging
import shutil
from multiprocessing import cpu_count
from pathlib import Path

import click

from dtst.sidecar import sidecar_path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
)


def resolve_dirs(working_dir: Path, names: list[str]) -> list[Path]:
    """Expand folder names relative to *working_dir*, supporting globs.

    Literal names (e.g. ``"raw"``) resolve to a single path.  Patterns
    containing ``*`` or ``?`` (e.g. ``"images/*"``) are expanded via
    :meth:`Path.glob` and only existing directories are kept.  Results
    are returned in sorted order with duplicates removed.
    """
    dirs: dict[Path, None] = {}
    for name in names:
        if "*" in name or "?" in name:
            for p in sorted(working_dir.glob(name)):
                if p.is_dir():
                    dirs[p] = None
        else:
            dirs[(working_dir / name).resolve()] = None
    return list(dirs)


def find_images(directory: Path, recursive: bool = False) -> list[Path]:
    """Return sorted list of image files in *directory*.

    Parameters
    ----------
    directory:
        Root directory to scan.
    recursive:
        If ``True``, search subdirectories as well.
    """
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_videos(directory: Path, recursive: bool = False) -> list[Path]:
    """Return sorted list of video files in *directory*.

    Parameters
    ----------
    directory:
        Root directory to scan.
    recursive:
        If ``True``, search subdirectories as well.
    """
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_save_kwargs(
    path: Path, *, quality: int = 95, compress_level: int = 6
) -> dict:
    """Build format-appropriate save kwargs for :meth:`PIL.Image.save`.

    Returns ``quality`` for JPEG/WebP and ``compress_level`` for PNG.
    """
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return {"quality": quality}
    elif suffix == ".webp":
        return {"quality": quality}
    elif suffix == ".png":
        return {"compress_level": compress_level}
    return {}


def move_image(src: Path, dest: Path) -> None:
    """Move an image and its sidecar file to a new location."""
    src.rename(dest)
    sc = sidecar_path(src)
    if sc.exists():
        sc.rename(sidecar_path(dest))


def copy_image(src: Path, dest: Path) -> None:
    """Copy an image and its sidecar file to a new location."""
    shutil.copy2(src, dest)
    sc = sidecar_path(src)
    if sc.exists():
        shutil.copy2(sc, sidecar_path(dest))


def resolve_workers(workers: int | None, fallback: int = 4) -> int:
    """Resolve a ``--workers`` CLI value, falling back to CPU count."""
    if workers is not None:
        return workers
    return cpu_count() or fallback


def format_elapsed(seconds: float) -> str:
    """Format an elapsed duration as ``"2m 34s"``."""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _gather_files(
    working_dir: Path | None,
    from_dirs_csv: str,
    finder,
    *,
    kind: str,
    recursive: bool = False,
) -> tuple[Path, list[Path], list[Path]]:
    working = (working_dir or Path(".")).resolve()
    dirs_list = _split_csv(from_dirs_csv)
    input_dirs = resolve_dirs(working, dirs_list)

    items: list[Path] = []
    for src in input_dirs:
        if not src.is_dir():
            logger.warning("Source directory does not exist, skipping: %s", src)
            continue
        items.extend(finder(src, recursive=recursive))

    if not items:
        raise click.ClickException(f"No {kind} found in source directories.")
    return working, input_dirs, items


def gather_images(
    working_dir: Path | None,
    from_dirs_csv: str,
    *,
    recursive: bool = False,
) -> tuple[Path, list[Path], list[Path]]:
    """Resolve a comma-separated ``--from`` value into images.

    Returns ``(working_dir, input_dirs, images)``.  Non-existent source
    directories are logged and skipped.  Raises ``click.ClickException``
    if no images are found across all inputs.
    """
    return _gather_files(
        working_dir, from_dirs_csv, find_images, kind="images", recursive=recursive
    )


def gather_videos(
    working_dir: Path | None,
    from_dirs_csv: str,
    *,
    recursive: bool = False,
) -> tuple[Path, list[Path], list[Path]]:
    """Resolve a comma-separated ``--from`` value into videos.

    See :func:`gather_images` for semantics.
    """
    return _gather_files(
        working_dir, from_dirs_csv, find_videos, kind="videos", recursive=recursive
    )
