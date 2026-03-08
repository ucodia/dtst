from pathlib import Path

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


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
