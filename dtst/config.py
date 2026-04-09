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


def _parse_str_list(
    section: dict, key: str, section_name: str, *, lower: bool = False, required: bool = True
) -> list[str] | None:
    raw = section.get(key)
    if raw is None:
        return None
    if isinstance(raw, list):
        items = [str(d).strip() for d in raw if str(d).strip()]
    elif isinstance(raw, str):
        items = [d.strip() for d in raw.split(",") if d.strip()]
    else:
        raise click.ClickException(f"'{section_name}.{key}' must be a string or list of strings")
    if not items:
        if required:
            raise click.ClickException(f"'{section_name}.{key}' must contain at least one entry")
        return None
    if lower:
        items = [i.lower() for i in items]
    return items


def _parse_optional_str(section: dict, key: str, section_name: str) -> str | None:
    val = section.get(key)
    if val is None:
        return None
    if not isinstance(val, str) or not val.strip():
        raise click.ClickException(f"'{section_name}.{key}' must be a non-empty string")
    return val.strip()


def _parse_bool(section: dict, key: str, default: bool, section_name: str) -> bool:
    val = section.get(key, default)
    if not isinstance(val, bool):
        raise click.ClickException(f"'{section_name}.{key}' must be a boolean")
    return val


def _parse_int(
    section: dict,
    key: str,
    section_name: str,
    *,
    default: int | None = None,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int | None:
    val = section.get(key, default)
    if val is None:
        return None
    if not isinstance(val, int) or isinstance(val, bool):
        raise click.ClickException(f"'{section_name}.{key}' must be an integer")
    if min_val is not None and val < min_val:
        if max_val is not None:
            raise click.ClickException(
                f"'{section_name}.{key}' must be an integer between {min_val} and {max_val}"
            )
        raise click.ClickException(f"'{section_name}.{key}' must be an integer >= {min_val}")
    if max_val is not None and val > max_val:
        raise click.ClickException(
            f"'{section_name}.{key}' must be an integer between {min_val} and {max_val}"
        )
    return val


def _parse_float(
    section: dict,
    key: str,
    section_name: str,
    *,
    default: float | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float | None:
    val = section.get(key, default)
    if val is None:
        return None
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        raise click.ClickException(f"'{section_name}.{key}' must be a number")
    val = float(val)
    if min_val is not None and val < min_val:
        raise click.ClickException(
            f"'{section_name}.{key}' must be >= {min_val}"
        )
    if max_val is not None and val > max_val:
        raise click.ClickException(
            f"'{section_name}.{key}' must be <= {max_val}"
        )
    return val


def load_search_config(path: str | Path) -> SearchConfig:
    data, config_dir = load_yaml(path)
    section = data.get("search")
    if not section or not isinstance(section, dict):
        raise click.ClickException("Config must have a 'search' section")

    s = "search"
    terms = section.get("terms")
    if terms is not None and not isinstance(terms, list):
        raise click.ClickException(f"'{s}.terms' must be a list of strings")
    terms = [str(t) for t in terms] if terms else []

    suffixes = section.get("suffixes")
    if suffixes is not None and not isinstance(suffixes, list):
        raise click.ClickException(f"'{s}.suffixes' must be a list of strings")
    suffixes = [str(sf) for sf in suffixes] if suffixes else []

    engines = section.get("engines")
    if engines is not None and not isinstance(engines, list):
        raise click.ClickException(f"'{s}.engines' must be a list of strings")
    if engines:
        engines = [str(e).strip().lower() for e in engines]
        invalid = set(engines) - VALID_SEARCH_ENGINES
        if invalid:
            raise click.ClickException(
                f"Invalid engine(s): {invalid}; valid: {sorted(VALID_SEARCH_ENGINES)}"
            )
    else:
        engines = []

    output = section.get("output", "results.jsonl")
    if not isinstance(output, str) or not output.strip():
        raise click.ClickException(f"'{s}.output' must be a non-empty string")

    return SearchConfig(
        terms=terms,
        suffixes=suffixes,
        engines=engines,
        working_dir=_resolve_working_dir(data, config_dir),
        min_size=_parse_int(section, "min_size", s, default=512, min_val=0),
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

    s = "fetch"
    return FetchConfig(
        working_dir=resolved_working_dir,
        to=_parse_optional_str(section, "to", s),
        input=_parse_optional_str(section, "input", s),
        min_size=_parse_int(section, "min_size", s, default=512, min_val=0),
        license=_parse_optional_str(section, "license", s),
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

    s = "extract_faces"
    engine = str(section.get("engine", "mediapipe")).strip().lower()
    if engine not in VALID_FACE_ENGINES:
        raise click.ClickException(
            f"Invalid face engine: {engine!r}; valid: {sorted(VALID_FACE_ENGINES)}"
        )

    return ExtractFacesConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        to=_parse_optional_str(section, "to", s),
        max_size=_parse_int(section, "max_size", s, min_val=1),
        engine=engine,
        max_faces=_parse_int(section, "max_faces", s, default=1, min_val=1),
        padding=_parse_bool(section, "padding", True, s),
        skip_partial=_parse_bool(section, "skip_partial", False, s),
        refine_landmarks=_parse_bool(section, "refine_landmarks", False, s),
        debug=_parse_bool(section, "debug", False, s),
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

    s = "cluster"
    model = str(section.get("model", "arcface")).strip().lower()
    if model not in VALID_EMBEDDING_MODELS:
        raise click.ClickException(
            f"Invalid embedding model: {model!r}; valid: {sorted(VALID_EMBEDDING_MODELS)}"
        )

    return ClusterConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        to=_parse_optional_str(section, "to", s),
        model=model,
        top=_parse_int(section, "top", s, min_val=1),
        min_cluster_size=_parse_int(section, "min_cluster_size", s, default=5, min_val=2),
        min_samples=_parse_int(section, "min_samples", s, default=2, min_val=1),
        batch_size=_parse_int(section, "batch_size", s, default=32, min_val=1),
        clean=_parse_bool(section, "clean", False, s),
    )


@dataclass
class SelectConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    to: str | None = None
    move: bool = False
    min_side: int | None = None
    max_side: int | None = None
    min_width: int | None = None
    max_width: int | None = None
    min_height: int | None = None
    max_height: int | None = None
    min_metric: list[tuple[str, float]] | None = None
    max_metric: list[tuple[str, float]] | None = None
    max_detect: list[tuple[str, float]] | None = None
    min_detect: list[tuple[str, float]] | None = None
    source: list[str] | None = None
    license: list[str] | None = None


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

    s = "select"
    dim_fields = {}
    for field_name in ("min_side", "max_side", "min_width", "max_width", "min_height", "max_height"):
        value = section.get(field_name)
        if value is None and field_name == "min_side":
            value = section.get("min_size")
        if value is not None:
            if not isinstance(value, int) or value < 1:
                raise click.ClickException(f"'{s}.{field_name}' must be a positive integer")
        dim_fields[field_name] = value

    return SelectConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        to=_parse_optional_str(section, "to", s),
        move=_parse_bool(section, "move", False, s),
        min_side=dim_fields["min_side"],
        max_side=dim_fields["max_side"],
        min_width=dim_fields["min_width"],
        max_width=dim_fields["max_width"],
        min_height=dim_fields["min_height"],
        max_height=dim_fields["max_height"],
        min_metric=_parse_tag_thresholds(section, "min_metric"),
        max_metric=_parse_tag_thresholds(section, "max_metric"),
        max_detect=_parse_tag_thresholds(section, "max_detect"),
        min_detect=_parse_tag_thresholds(section, "min_detect"),
        source=_parse_str_list(section, "source", s, lower=True, required=False),
        license=_parse_str_list(section, "license", s, lower=True, required=False),
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

    s = "detect"
    return DetectConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        classes=_parse_str_list(section, "classes", s),
        threshold=_parse_float(section, "threshold", s, default=0.2, min_val=0.0),
        max_instances=_parse_int(section, "max_instances", s, default=1, min_val=1),
    )


@dataclass
class DedupConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dir: str | None = None
    to: str = "duplicated"
    threshold: int = 8
    prefer_upscaled: bool = False


def load_dedup_config(path: str | Path) -> DedupConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("dedup")
    if not section or not isinstance(section, dict):
        return DedupConfig(working_dir=resolved_working_dir)

    s = "dedup"
    to = section.get("to", "duplicated")
    if not isinstance(to, str) or not to.strip():
        raise click.ClickException(f"'{s}.to' must be a non-empty string")

    return DedupConfig(
        working_dir=resolved_working_dir,
        from_dir=_parse_optional_str(section, "from", s),
        to=to.strip(),
        threshold=_parse_int(section, "threshold", s, default=8, min_val=0),
        prefer_upscaled=_parse_bool(section, "prefer_upscaled", False, s),
    )


@dataclass
class AnnotateConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    source: str | None = None
    license: str | None = None
    origin: str | None = None


def load_annotate_config(path: str | Path) -> AnnotateConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("annotate")
    if not section or not isinstance(section, dict):
        return AnnotateConfig(working_dir=resolved_working_dir)

    s = "annotate"
    return AnnotateConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        source=_parse_optional_str(section, "source", s),
        license=_parse_optional_str(section, "license", s),
        origin=_parse_optional_str(section, "origin", s),
    )


@dataclass
class AnalyzeConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    metrics: list[str] = field(default_factory=list)


def load_analyze_config(path: str | Path) -> AnalyzeConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("analyze")
    if not section or not isinstance(section, dict):
        return AnalyzeConfig(working_dir=resolved_working_dir)

    s = "analyze"
    metrics_raw = section.get("metrics", [])
    if isinstance(metrics_raw, list):
        metrics = [str(m).strip() for m in metrics_raw if str(m).strip()]
    elif isinstance(metrics_raw, str):
        metrics = [m.strip() for m in metrics_raw.split(",") if m.strip()]
    else:
        raise click.ClickException(f"'{s}.metrics' must be a list or comma-separated string")

    return AnalyzeConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        metrics=metrics,
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

    s = "augment"
    return AugmentConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        to=_parse_optional_str(section, "to", s),
        flip_x=_parse_bool(section, "flip_x", False, s),
        flip_y=_parse_bool(section, "flip_y", False, s),
        flip_xy=_parse_bool(section, "flip_xy", False, s),
        no_copy=_parse_bool(section, "no_copy", False, s),
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

    s = "upscale"
    scale = section.get("scale", 4)
    if not isinstance(scale, int) or scale not in VALID_UPSCALE_SCALES:
        raise click.ClickException(
            f"'{s}.scale' must be one of {sorted(VALID_UPSCALE_SCALES)}"
        )

    fmt = section.get("format")
    if fmt is not None:
        fmt = str(fmt).strip().lower()
        if fmt not in VALID_UPSCALE_FORMATS:
            raise click.ClickException(
                f"Invalid upscale format: {fmt!r}; valid: {sorted(VALID_UPSCALE_FORMATS)}"
            )

    return UpscaleConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        to=_parse_optional_str(section, "to", s),
        scale=scale,
        model=_parse_optional_str(section, "model", s),
        tile_size=_parse_int(section, "tile_size", s, default=512, min_val=0),
        tile_pad=_parse_int(section, "tile_pad", s, default=32, min_val=0),
        format=fmt,
        quality=_parse_int(section, "quality", s, default=95, min_val=1, max_val=100),
        denoise=_parse_float(section, "denoise", s, min_val=0.0, max_val=1.0),
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

    s = "extract_frames"
    fmt = str(section.get("format", "jpg")).strip().lower()
    if fmt not in VALID_FRAME_FORMATS:
        raise click.ClickException(
            f"Invalid frame format: {fmt!r}; valid: {sorted(VALID_FRAME_FORMATS)}"
        )

    return ExtractFramesConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        to=_parse_optional_str(section, "to", s),
        keyframes=_parse_float(section, "keyframes", s, default=10.0, min_val=0.0),
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
    quality: int = 95
    compress_level: int = 0


def load_frame_config(path: str | Path) -> FrameConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("frame")
    if not section or not isinstance(section, dict):
        return FrameConfig(working_dir=resolved_working_dir)

    s = "frame"
    mode = section.get("mode", "crop")
    if mode not in FRAME_MODES:
        raise click.ClickException(
            f"'{s}.mode' must be one of {', '.join(FRAME_MODES)}"
        )

    gravity = section.get("gravity", "center")
    if gravity not in FRAME_GRAVITIES:
        raise click.ClickException(
            f"'{s}.gravity' must be one of {', '.join(FRAME_GRAVITIES)}"
        )

    fill = section.get("fill", "color")
    if fill not in FRAME_FILLS:
        raise click.ClickException(
            f"'{s}.fill' must be one of {', '.join(FRAME_FILLS)}"
        )

    fill_color = section.get("fill_color", "#000000")
    if not isinstance(fill_color, str) or not fill_color.strip():
        raise click.ClickException(f"'{s}.fill_color' must be a non-empty string")

    return FrameConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        to=_parse_optional_str(section, "to", s),
        width=_parse_int(section, "width", s, min_val=1),
        height=_parse_int(section, "height", s, min_val=1),
        mode=mode,
        gravity=gravity,
        fill=fill,
        fill_color=fill_color.strip(),
        quality=_parse_int(section, "quality", s, default=95, min_val=1, max_val=100),
        compress_level=_parse_int(section, "compress_level", s, default=0, min_val=0, max_val=9),
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

    s = "review"
    to = section.get("to", "rejected")
    if not isinstance(to, str) or not to.strip():
        raise click.ClickException(f"'{s}.to' must be a non-empty string")

    return ReviewConfig(
        working_dir=resolved_working_dir,
        from_dir=_parse_optional_str(section, "from", s),
        to=to.strip(),
        port=_parse_int(section, "port", s, default=8888, min_val=1, max_val=65535),
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


@dataclass
class RenameConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    prefix: str = ""
    digits: int | None = None


def load_rename_config(path: str | Path) -> RenameConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("rename")
    if not section or not isinstance(section, dict):
        return RenameConfig(working_dir=resolved_working_dir)

    s = "rename"
    prefix = section.get("prefix", "")
    if not isinstance(prefix, str):
        raise click.ClickException(f"'{s}.prefix' must be a string")

    return RenameConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        prefix=prefix,
        digits=_parse_int(section, "digits", s, min_val=1),
    )


@dataclass
class ValidateConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    square: bool = False


def load_validate_config(path: str | Path) -> ValidateConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("validate")
    if not section or not isinstance(section, dict):
        return ValidateConfig(working_dir=resolved_working_dir)

    s = "validate"
    return ValidateConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        square=_parse_bool(section, "square", False, s),
    )


VALID_FORMAT_CHANNELS = frozenset({"rgb", "grayscale"})


@dataclass
class FormatConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] | None = None
    to: str | None = None
    format: str | None = None
    quality: int = 95
    compress_level: int = 0
    strip_metadata: bool = False
    channels: str | None = None
    background: str = "white"


def load_format_config(path: str | Path) -> FormatConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("format")
    if not section or not isinstance(section, dict):
        return FormatConfig(working_dir=resolved_working_dir)

    s = "format"
    fmt = section.get("format")
    if fmt is not None:
        if fmt not in ("jpg", "png", "webp"):
            raise click.ClickException(f"'{s}.format' must be one of: jpg, png, webp")

    channels = section.get("channels")
    if channels is not None:
        if channels not in VALID_FORMAT_CHANNELS:
            raise click.ClickException(f"'{s}.channels' must be one of: {', '.join(sorted(VALID_FORMAT_CHANNELS))}")

    background = section.get("background", "white")
    if not isinstance(background, str) or not background.strip():
        raise click.ClickException(f"'{s}.background' must be a non-empty string")

    return FormatConfig(
        working_dir=resolved_working_dir,
        from_dirs=_parse_str_list(section, "from", s),
        to=_parse_optional_str(section, "to", s),
        format=fmt,
        quality=_parse_int(section, "quality", s, default=95, min_val=1, max_val=100),
        compress_level=_parse_int(section, "compress_level", s, default=0, min_val=0, max_val=9),
        strip_metadata=_parse_bool(section, "strip_metadata", False, s),
        channels=channels,
        background=background.strip(),
    )
