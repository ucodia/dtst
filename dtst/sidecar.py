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


def read_all_sidecars(image_paths: list[Path]) -> dict[Path, dict]:
    return {p: read_sidecar(p) for p in image_paths}
