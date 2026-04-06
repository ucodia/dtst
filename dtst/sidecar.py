import json
from pathlib import Path


def sidecar_path(image_path: Path) -> Path:
    return image_path.parent / (image_path.name + ".json")


def read_sidecar(image_path: Path) -> dict:
    path = sidecar_path(image_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def write_sidecar(image_path: Path, data: dict) -> None:
    path = sidecar_path(image_path)
    existing = read_sidecar(image_path)
    existing.update(data)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")


def copy_sidecar(src: Path, dest: Path, exclude: set[str] | None = None) -> None:
    """Copy sidecar data from *src* image to *dest* image.

    When *exclude* is given, those top-level keys are omitted from the
    copy.  This is useful when a transformation invalidates computed
    fields (e.g. ``metrics``, ``classes``) but provenance should be kept.
    """
    data = read_sidecar(src)
    if not data:
        return
    if exclude:
        data = {k: v for k, v in data.items() if k not in exclude}
    if not data:
        return
    path = sidecar_path(dest)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def scale_classes(classes: dict, factor: float) -> dict:
    """Scale bounding-box coordinates by *factor*, preserving scores."""
    scaled = {}
    for cls, detections in classes.items():
        scaled[cls] = [
            {"score": d["score"], "box": [int(c * factor) for c in d["box"]]}
            for d in detections
        ]
    return scaled


def read_all_sidecars(image_paths: list[Path]) -> dict[Path, dict]:
    return {p: read_sidecar(p) for p in image_paths}
