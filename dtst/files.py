import shutil
from pathlib import Path

from dtst.sidecar import sidecar_path

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
