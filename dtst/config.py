from dataclasses import dataclass, field
from pathlib import Path

import click
import yaml

VALID_SEARCH_ENGINES = frozenset({"brave", "flickr", "serper", "wikimedia"})
VALID_FACE_ENGINES = frozenset({"mediapipe", "dlib"})
VALID_EMBEDDING_MODELS = frozenset({"arcface", "clip"})


def load_yaml(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    with open(config_path) as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        raise click.ClickException("Config must be a non-empty YAML object")
    return data, config_path.parent


@dataclass
class SearchConfig:
    terms: list[str] = field(default_factory=list)
    suffixes: list[str] = field(default_factory=list)
    engines: list[str] = field(default_factory=list)
    working_dir: Path = field(default_factory=lambda: Path("."))
    min_size: int = 512
    output: str = "results.jsonl"

    def query_matrix(self, suffix_only: bool = False) -> list[str]:
        queries: list[str] = []
        if not suffix_only:
            queries.extend(self.terms)
        queries.extend(
            f"{term} {suffix}".strip()
            for term in self.terms
            for suffix in self.suffixes
            if suffix
        )
        return queries


def _resolve_working_dir(data: dict, config_dir: Path) -> Path:
    working_dir = data.get("working_dir")
    if working_dir is None:
        return Path(".")
    if not isinstance(working_dir, str) or not working_dir.strip():
        raise click.ClickException("'working_dir' must be a non-empty string")
    return config_dir / working_dir.strip()


def load_search_config(path: str | Path) -> SearchConfig:
    data, config_dir = load_yaml(path)
    section = data.get("search")
    if not section or not isinstance(section, dict):
        raise click.ClickException("Config must have a 'search' section")

    terms = section.get("terms")
    if terms is not None and not isinstance(terms, list):
        raise click.ClickException("'search.terms' must be a list of strings")
    terms = [str(t) for t in terms] if terms else []

    suffixes = section.get("suffixes")
    if suffixes is not None and not isinstance(suffixes, list):
        raise click.ClickException("'search.suffixes' must be a list of strings")
    suffixes = [str(s) for s in suffixes] if suffixes else []

    engines = section.get("engines")
    if engines is not None and not isinstance(engines, list):
        raise click.ClickException("'search.engines' must be a list of strings")
    if engines:
        engines = [str(e).strip().lower() for e in engines]
        invalid = set(engines) - VALID_SEARCH_ENGINES
        if invalid:
            raise click.ClickException(
                f"Invalid engine(s): {invalid}; valid: {sorted(VALID_SEARCH_ENGINES)}"
            )
    else:
        engines = []

    min_size = section.get("min_size", 512)
    if not isinstance(min_size, int) or min_size < 0:
        raise click.ClickException("'search.min_size' must be a non-negative integer")

    output = section.get("output", "results.jsonl")
    if not isinstance(output, str) or not output.strip():
        raise click.ClickException("'search.output' must be a non-empty string")

    return SearchConfig(
        terms=terms,
        suffixes=suffixes,
        engines=engines,
        working_dir=_resolve_working_dir(data, config_dir),
        min_size=min_size,
        output=output.strip(),
    )


@dataclass
class FetchConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    to: str | None = None
    input: str | None = None
    min_size: int = 512
    license: str | None = None


def load_fetch_config(path: str | Path) -> FetchConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("fetch")
    if not section or not isinstance(section, dict):
        return FetchConfig(working_dir=resolved_working_dir)

    min_size = section.get("min_size", 512)
    if not isinstance(min_size, int) or min_size < 0:
        raise click.ClickException("'fetch.min_size' must be a non-negative integer")

    to = section.get("to")
    if to is not None and (not isinstance(to, str) or not to.strip()):
        raise click.ClickException("'fetch.to' must be a non-empty string")

    input_file = section.get("input")
    if input_file is not None:
        if not isinstance(input_file, str) or not input_file.strip():
            raise click.ClickException("'fetch.input' must be a non-empty string")
        input_file = input_file.strip()

    license_filter = section.get("license")
    if license_filter is not None:
        if not isinstance(license_filter, str) or not license_filter.strip():
            raise click.ClickException("'fetch.license' must be a non-empty string")
        license_filter = license_filter.strip()

    return FetchConfig(
        working_dir=resolved_working_dir,
        to=to.strip() if to else None,
        input=input_file,
        min_size=min_size,
        license=license_filter,
    )


@dataclass
class ExtractFacesConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    to: str | None = None
    max_size: int | None = None
    engine: str = "mediapipe"
    max_faces: int = 3
    padding: bool = True
    skip_partial: bool = False
    refine_landmarks: bool = False
    debug: bool = False


def load_extract_faces_config(path: str | Path) -> ExtractFacesConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("extract_faces")
    if not section or not isinstance(section, dict):
        return ExtractFacesConfig(working_dir=resolved_working_dir)

    max_size = section.get("max_size")
    if max_size is not None and (not isinstance(max_size, int) or max_size < 1):
        raise click.ClickException("'extract_faces.max_size' must be a positive integer")

    engine = str(section.get("engine", "mediapipe")).strip().lower()
    if engine not in VALID_FACE_ENGINES:
        raise click.ClickException(
            f"Invalid face engine: {engine!r}; valid: {sorted(VALID_FACE_ENGINES)}"
        )

    max_faces = section.get("max_faces", 1)
    if not isinstance(max_faces, int) or max_faces < 1:
        raise click.ClickException("'extract_faces.max_faces' must be a positive integer")

    padding = section.get("padding", True)
    if not isinstance(padding, bool):
        raise click.ClickException("'extract_faces.padding' must be a boolean")

    skip_partial = section.get("skip_partial", False)
    if not isinstance(skip_partial, bool):
        raise click.ClickException("'extract_faces.skip_partial' must be a boolean")

    refine_landmarks = section.get("refine_landmarks", False)
    if not isinstance(refine_landmarks, bool):
        raise click.ClickException("'extract_faces.refine_landmarks' must be a boolean")

    debug = section.get("debug", False)
    if not isinstance(debug, bool):
        raise click.ClickException("'extract_faces.debug' must be a boolean")

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'extract_faces.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'extract_faces.from' must contain at least one directory name")
    else:
        from_dirs = None

    to = section.get("to")
    if to is not None and (not isinstance(to, str) or not to.strip()):
        raise click.ClickException("'extract_faces.to' must be a non-empty string")

    return ExtractFacesConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        to=to.strip() if to else None,
        max_size=max_size,
        engine=engine,
        max_faces=max_faces,
        padding=padding,
        skip_partial=skip_partial,
        refine_landmarks=refine_landmarks,
        debug=debug,
    )


@dataclass
class ClusterConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    to: str | None = None
    model: str = "arcface"
    top: int | None = None
    min_cluster_size: int = 5
    min_samples: int = 2
    batch_size: int = 32
    no_cache: bool = False
    clean: bool = False


def load_cluster_config(path: str | Path) -> ClusterConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("cluster")
    if not section or not isinstance(section, dict):
        return ClusterConfig(working_dir=resolved_working_dir)

    model = str(section.get("model", "arcface")).strip().lower()
    if model not in VALID_EMBEDDING_MODELS:
        raise click.ClickException(
            f"Invalid embedding model: {model!r}; valid: {sorted(VALID_EMBEDDING_MODELS)}"
        )

    top = section.get("top")
    if top is not None:
        if not isinstance(top, int) or top < 1:
            raise click.ClickException("'cluster.top' must be a positive integer")

    min_cluster_size = section.get("min_cluster_size", 5)
    if not isinstance(min_cluster_size, int) or min_cluster_size < 2:
        raise click.ClickException("'cluster.min_cluster_size' must be an integer >= 2")

    min_samples = section.get("min_samples", 2)
    if not isinstance(min_samples, int) or min_samples < 1:
        raise click.ClickException("'cluster.min_samples' must be a positive integer")

    batch_size = section.get("batch_size", 32)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise click.ClickException("'cluster.batch_size' must be a positive integer")

    clean = section.get("clean", False)
    if not isinstance(clean, bool):
        raise click.ClickException("'cluster.clean' must be a boolean")

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'cluster.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'cluster.from' must contain at least one directory name")
    else:
        from_dirs = None

    to = section.get("to")
    if to is not None and (not isinstance(to, str) or not to.strip()):
        raise click.ClickException("'cluster.to' must be a non-empty string")

    return ClusterConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        to=to.strip() if to else None,
        model=model,
        top=top,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        batch_size=batch_size,
        clean=clean,
    )


@dataclass
class SelectConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    to: str | None = None
    move: bool = False
    min_size: int | None = None
    min_blur: float | None = None
    max_tag: list[tuple[str, float]] | None = None
    min_tag: list[tuple[str, float]] | None = None
    max_detect: list[tuple[str, float]] | None = None
    min_detect: list[tuple[str, float]] | None = None


def _parse_tag_thresholds(
    section: dict, key: str, section_name: str = "select"
) -> list[tuple[str, float]] | None:
    raw = section.get(key)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise click.ClickException(f"'{section_name}.{key}' must be a mapping of label to threshold")
    result = []
    for label, threshold in raw.items():
        if not isinstance(label, str) or not label.strip():
            raise click.ClickException(f"'{section_name}.{key}' keys must be non-empty strings")
        if not isinstance(threshold, (int, float)):
            raise click.ClickException(f"'{section_name}.{key}.{label}' must be a number")
        result.append((label.strip(), float(threshold)))
    return result if result else None


def load_select_config(path: str | Path) -> SelectConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("select")
    if not section or not isinstance(section, dict):
        return SelectConfig(working_dir=resolved_working_dir)

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'select.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'select.from' must contain at least one directory name")
    else:
        from_dirs = None

    to = section.get("to")
    if to is not None and (not isinstance(to, str) or not to.strip()):
        raise click.ClickException("'select.to' must be a non-empty string")

    move = section.get("move", False)
    if not isinstance(move, bool):
        raise click.ClickException("'select.move' must be a boolean")

    min_size = section.get("min_size")
    if min_size is not None:
        if not isinstance(min_size, int) or min_size < 1:
            raise click.ClickException("'select.min_size' must be a positive integer")

    min_blur = section.get("min_blur")
    if min_blur is not None:
        if not isinstance(min_blur, (int, float)) or min_blur < 0:
            raise click.ClickException("'select.min_blur' must be a non-negative number")
        min_blur = float(min_blur)

    max_tag = _parse_tag_thresholds(section, "max_tag")
    min_tag = _parse_tag_thresholds(section, "min_tag")
    max_detect = _parse_tag_thresholds(section, "max_detect")
    min_detect = _parse_tag_thresholds(section, "min_detect")

    return SelectConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        to=to.strip() if to else None,
        move=move,
        min_size=min_size,
        min_blur=min_blur,
        max_tag=max_tag,
        min_tag=min_tag,
        max_detect=max_detect,
        min_detect=min_detect,
    )


@dataclass
class TagConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    labels: list[str] | None = None
    batch_size: int = 32


def load_tag_config(path: str | Path) -> TagConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("tag")
    if not section or not isinstance(section, dict):
        return TagConfig(working_dir=resolved_working_dir)

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'tag.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'tag.from' must contain at least one directory name")
    else:
        from_dirs = None

    labels_raw = section.get("labels")
    if labels_raw is not None:
        if isinstance(labels_raw, list):
            labels = [str(l).strip() for l in labels_raw if str(l).strip()]
        elif isinstance(labels_raw, str):
            labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
        else:
            raise click.ClickException("'tag.labels' must be a string or list of strings")
        if not labels:
            raise click.ClickException("'tag.labels' must contain at least one label")
    else:
        labels = None

    batch_size = section.get("batch_size", 32)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise click.ClickException("'tag.batch_size' must be a positive integer")

    return TagConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        labels=labels,
        batch_size=batch_size,
    )


@dataclass
class DetectConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    classes: list[str] | None = None
    threshold: float = 0.2
    max_instances: int = 1


def load_detect_config(path: str | Path) -> DetectConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("detect")
    if not section or not isinstance(section, dict):
        return DetectConfig(working_dir=resolved_working_dir)

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'detect.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'detect.from' must contain at least one directory name")
    else:
        from_dirs = None

    classes_raw = section.get("classes")
    if classes_raw is not None:
        if isinstance(classes_raw, list):
            classes = [str(c).strip() for c in classes_raw if str(c).strip()]
        elif isinstance(classes_raw, str):
            classes = [c.strip() for c in classes_raw.split(",") if c.strip()]
        else:
            raise click.ClickException("'detect.classes' must be a string or list of strings")
        if not classes:
            raise click.ClickException("'detect.classes' must contain at least one class")
    else:
        classes = None

    threshold = section.get("threshold", 0.2)
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise click.ClickException("'detect.threshold' must be a non-negative number")

    max_instances = section.get("max_instances", 1)
    if not isinstance(max_instances, int) or max_instances < 1:
        raise click.ClickException("'detect.max_instances' must be a positive integer")

    return DetectConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        classes=classes,
        threshold=float(threshold),
        max_instances=max_instances,
    )


@dataclass
class DedupConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dir: str | None = None
    to: str = "duplicated"
    threshold: int = 8


def load_dedup_config(path: str | Path) -> DedupConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("dedup")
    if not section or not isinstance(section, dict):
        return DedupConfig(working_dir=resolved_working_dir)

    from_dir = section.get("from")
    if from_dir is not None and (not isinstance(from_dir, str) or not from_dir.strip()):
        raise click.ClickException("'dedup.from' must be a non-empty string")

    to = section.get("to", "duplicated")
    if not isinstance(to, str) or not to.strip():
        raise click.ClickException("'dedup.to' must be a non-empty string")

    threshold = section.get("threshold", 8)
    if not isinstance(threshold, int) or threshold < 0:
        raise click.ClickException("'dedup.threshold' must be a non-negative integer")

    return DedupConfig(
        working_dir=resolved_working_dir,
        from_dir=from_dir.strip() if from_dir else None,
        to=to.strip(),
        threshold=threshold,
    )


@dataclass
class AnalyzeConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    phash: bool = False
    blur: bool = False


def load_analyze_config(path: str | Path) -> AnalyzeConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("analyze")
    if not section or not isinstance(section, dict):
        return AnalyzeConfig(working_dir=resolved_working_dir)

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'analyze.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'analyze.from' must contain at least one directory name")
    else:
        from_dirs = None

    phash = section.get("phash", False)
    if not isinstance(phash, bool):
        raise click.ClickException("'analyze.phash' must be a boolean")

    blur = section.get("blur", False)
    if not isinstance(blur, bool):
        raise click.ClickException("'analyze.blur' must be a boolean")

    return AnalyzeConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs if from_dirs else None,
        phash=phash,
        blur=blur,
    )


@dataclass
class AugmentConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    to: str | None = None
    flip_x: bool = False
    flip_y: bool = False
    flip_xy: bool = False
    no_copy: bool = False


def load_augment_config(path: str | Path) -> AugmentConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("augment")
    if not section or not isinstance(section, dict):
        return AugmentConfig(working_dir=resolved_working_dir)

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'augment.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'augment.from' must contain at least one directory name")
    else:
        from_dirs = None

    to = section.get("to")
    if to is not None and (not isinstance(to, str) or not to.strip()):
        raise click.ClickException("'augment.to' must be a non-empty string")

    flip_x = section.get("flip_x", False)
    if not isinstance(flip_x, bool):
        raise click.ClickException("'augment.flip_x' must be a boolean")

    flip_y = section.get("flip_y", False)
    if not isinstance(flip_y, bool):
        raise click.ClickException("'augment.flip_y' must be a boolean")

    flip_xy = section.get("flip_xy", False)
    if not isinstance(flip_xy, bool):
        raise click.ClickException("'augment.flip_xy' must be a boolean")

    no_copy = section.get("no_copy", False)
    if not isinstance(no_copy, bool):
        raise click.ClickException("'augment.no_copy' must be a boolean")

    return AugmentConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        to=to.strip() if to else None,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_xy=flip_xy,
        no_copy=no_copy,
    )


VALID_UPSCALE_FORMATS = frozenset({"jpg", "png", "webp"})
VALID_UPSCALE_SCALES = frozenset({2, 4})


@dataclass
class UpscaleConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    to: str | None = None
    scale: int = 4
    model: str | None = None
    tile_size: int = 512
    tile_pad: int = 32
    format: str | None = None
    quality: int = 95
    denoise: float | None = None


def load_upscale_config(path: str | Path) -> UpscaleConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("upscale")
    if not section or not isinstance(section, dict):
        return UpscaleConfig(working_dir=resolved_working_dir)

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'upscale.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'upscale.from' must contain at least one directory name")
    else:
        from_dirs = None

    to = section.get("to")
    if to is not None and (not isinstance(to, str) or not to.strip()):
        raise click.ClickException("'upscale.to' must be a non-empty string")

    scale = section.get("scale", 4)
    if not isinstance(scale, int) or scale not in VALID_UPSCALE_SCALES:
        raise click.ClickException(
            f"'upscale.scale' must be one of {sorted(VALID_UPSCALE_SCALES)}"
        )

    model = section.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise click.ClickException("'upscale.model' must be a non-empty string")

    tile_size = section.get("tile_size", 512)
    if not isinstance(tile_size, int) or tile_size < 0:
        raise click.ClickException("'upscale.tile_size' must be a non-negative integer")

    tile_pad = section.get("tile_pad", 32)
    if not isinstance(tile_pad, int) or tile_pad < 0:
        raise click.ClickException("'upscale.tile_pad' must be a non-negative integer")

    fmt = section.get("format")
    if fmt is not None:
        fmt = str(fmt).strip().lower()
        if fmt not in VALID_UPSCALE_FORMATS:
            raise click.ClickException(
                f"Invalid upscale format: {fmt!r}; valid: {sorted(VALID_UPSCALE_FORMATS)}"
            )

    quality = section.get("quality", 95)
    if not isinstance(quality, int) or quality < 1 or quality > 100:
        raise click.ClickException("'upscale.quality' must be an integer between 1 and 100")

    denoise = section.get("denoise")
    if denoise is not None:
        if not isinstance(denoise, (int, float)):
            raise click.ClickException("'upscale.denoise' must be a number between 0.0 and 1.0")
        denoise = float(denoise)
        if denoise < 0.0 or denoise > 1.0:
            raise click.ClickException("'upscale.denoise' must be between 0.0 and 1.0")

    return UpscaleConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        to=to.strip() if to else None,
        scale=scale,
        model=model.strip() if model else None,
        tile_size=tile_size,
        tile_pad=tile_pad,
        format=fmt,
        quality=quality,
        denoise=denoise,
    )


VALID_FRAME_FORMATS = frozenset({"jpg", "png"})


@dataclass
class ExtractFramesConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    to: str | None = None
    keyframes: float = 10.0
    format: str = "jpg"


def load_extract_frames_config(path: str | Path) -> ExtractFramesConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("extract_frames")
    if not section or not isinstance(section, dict):
        return ExtractFramesConfig(working_dir=resolved_working_dir)

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'extract_frames.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'extract_frames.from' must contain at least one directory name")
    else:
        from_dirs = None

    to = section.get("to")
    if to is not None and (not isinstance(to, str) or not to.strip()):
        raise click.ClickException("'extract_frames.to' must be a non-empty string")

    keyframes = section.get("keyframes", 10.0)
    if not isinstance(keyframes, (int, float)) or keyframes <= 0:
        raise click.ClickException("'extract_frames.keyframes' must be a positive number")
    keyframes = float(keyframes)

    fmt = str(section.get("format", "jpg")).strip().lower()
    if fmt not in VALID_FRAME_FORMATS:
        raise click.ClickException(
            f"Invalid frame format: {fmt!r}; valid: {sorted(VALID_FRAME_FORMATS)}"
        )

    return ExtractFramesConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        to=to.strip() if to else None,
        keyframes=keyframes,
        format=fmt,
    )


FRAME_MODES = ("stretch", "crop", "pad")
FRAME_GRAVITIES = ("center", "top", "bottom", "left", "right")
FRAME_FILLS = ("color", "edge", "reflect", "blur")


@dataclass
class FrameConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    to: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str = "crop"
    gravity: str = "center"
    fill: str = "color"
    fill_color: str = "#000000"


def load_frame_config(path: str | Path) -> FrameConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("frame")
    if not section or not isinstance(section, dict):
        return FrameConfig(working_dir=resolved_working_dir)

    from_raw = section.get("from")
    if from_raw is not None:
        if isinstance(from_raw, list):
            from_dirs = [str(d).strip() for d in from_raw if str(d).strip()]
        elif isinstance(from_raw, str):
            from_dirs = [d.strip() for d in from_raw.split(",") if d.strip()]
        else:
            raise click.ClickException("'frame.from' must be a string or list of strings")
        if not from_dirs:
            raise click.ClickException("'frame.from' must contain at least one directory name")
    else:
        from_dirs = None

    to = section.get("to")
    if to is not None and (not isinstance(to, str) or not to.strip()):
        raise click.ClickException("'frame.to' must be a non-empty string")

    width = section.get("width")
    if width is not None:
        if not isinstance(width, int) or width < 1:
            raise click.ClickException("'frame.width' must be a positive integer")

    height = section.get("height")
    if height is not None:
        if not isinstance(height, int) or height < 1:
            raise click.ClickException("'frame.height' must be a positive integer")

    mode = section.get("mode", "crop")
    if mode not in FRAME_MODES:
        raise click.ClickException(
            f"'frame.mode' must be one of {', '.join(FRAME_MODES)}"
        )

    gravity = section.get("gravity", "center")
    if gravity not in FRAME_GRAVITIES:
        raise click.ClickException(
            f"'frame.gravity' must be one of {', '.join(FRAME_GRAVITIES)}"
        )

    fill = section.get("fill", "color")
    if fill not in FRAME_FILLS:
        raise click.ClickException(
            f"'frame.fill' must be one of {', '.join(FRAME_FILLS)}"
        )

    fill_color = section.get("fill_color", "#000000")
    if not isinstance(fill_color, str) or not fill_color.strip():
        raise click.ClickException("'frame.fill_color' must be a non-empty string")

    return FrameConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        to=to.strip() if to else None,
        width=width,
        height=height,
        mode=mode,
        gravity=gravity,
        fill=fill,
        fill_color=fill_color.strip(),
    )


@dataclass
class ReviewConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dir: str | None = None
    to: str = "rejected"
    port: int = 8888


def load_review_config(path: str | Path) -> ReviewConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("review")
    if not section or not isinstance(section, dict):
        return ReviewConfig(working_dir=resolved_working_dir)

    from_dir = section.get("from")
    if from_dir is not None and (not isinstance(from_dir, str) or not from_dir.strip()):
        raise click.ClickException("'review.from' must be a non-empty string")

    to = section.get("to", "rejected")
    if not isinstance(to, str) or not to.strip():
        raise click.ClickException("'review.to' must be a non-empty string")

    port = section.get("port", 8888)
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise click.ClickException("'review.port' must be an integer between 1 and 65535")

    return ReviewConfig(
        working_dir=resolved_working_dir,
        from_dir=from_dir.strip() if from_dir else None,
        to=to.strip(),
        port=port,
    )


@dataclass
class WorkflowStep:
    command: str | None = None
    exec: str | None = None
    inherit: bool = True
    overrides: dict = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    steps: list[WorkflowStep] = field(default_factory=list)


def load_workflow_config(path: str | Path, workflow_name: str) -> WorkflowConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    workflows = data.get("workflows")
    if not workflows or not isinstance(workflows, dict):
        raise click.ClickException("Config must have a 'workflows' section")

    workflow = workflows.get(workflow_name)
    if workflow is None:
        available = ", ".join(sorted(workflows.keys()))
        raise click.ClickException(
            f"Workflow '{workflow_name}' not found; available: {available}"
        )
    if not isinstance(workflow, list):
        raise click.ClickException(
            f"Workflow '{workflow_name}' must be a list of steps"
        )

    steps: list[WorkflowStep] = []
    for i, raw_step in enumerate(workflow, 1):
        if isinstance(raw_step, str):
            steps.append(WorkflowStep(command=raw_step))
        elif isinstance(raw_step, dict):
            if "exec" in raw_step:
                exec_cmd = raw_step["exec"]
                if not isinstance(exec_cmd, str) or not exec_cmd.strip():
                    raise click.ClickException(
                        f"Step {i}: 'exec' must be a non-empty string"
                    )
                steps.append(WorkflowStep(exec=exec_cmd.strip()))
            else:
                keys = list(raw_step.keys())
                if len(keys) != 1:
                    raise click.ClickException(
                        f"Step {i}: expected a single command key, got {keys}"
                    )
                cmd_name = keys[0]
                raw_overrides = raw_step[cmd_name]
                if raw_overrides is None:
                    overrides = {}
                elif isinstance(raw_overrides, dict):
                    overrides = dict(raw_overrides)
                else:
                    raise click.ClickException(
                        f"Step {i}: overrides for '{cmd_name}' must be a mapping"
                    )
                inherit = overrides.pop("inherit", True)
                if not isinstance(inherit, bool):
                    raise click.ClickException(
                        f"Step {i}: 'inherit' must be a boolean"
                    )
                steps.append(
                    WorkflowStep(
                        command=cmd_name, inherit=inherit, overrides=overrides
                    )
                )
        else:
            raise click.ClickException(
                f"Step {i}: must be a command name or mapping"
            )

    return WorkflowConfig(working_dir=resolved_working_dir, steps=steps)
