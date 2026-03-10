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

    return SearchConfig(
        terms=terms,
        suffixes=suffixes,
        engines=engines,
        working_dir=_resolve_working_dir(data, config_dir),
        min_size=min_size,
    )


@dataclass
class FetchConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    to: str = "raw"
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

    to = section.get("to", "raw")
    if not isinstance(to, str) or not to.strip():
        raise click.ClickException("'fetch.to' must be a non-empty string")

    license_filter = section.get("license")
    if license_filter is not None:
        if not isinstance(license_filter, str) or not license_filter.strip():
            raise click.ClickException("'fetch.license' must be a non-empty string")
        license_filter = license_filter.strip()

    return FetchConfig(
        working_dir=resolved_working_dir,
        to=to.strip(),
        min_size=min_size,
        license=license_filter,
    )


@dataclass
class ExtractFacesConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] = field(default_factory=lambda: ["raw"])
    to: str = "faces"
    max_size: int | None = None
    engine: str = "mediapipe"
    max_faces: int = 3
    padding: bool = True
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
        from_dirs = ["raw"]

    to = section.get("to", "faces")
    if not isinstance(to, str) or not to.strip():
        raise click.ClickException("'extract_faces.to' must be a non-empty string")

    return ExtractFacesConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        to=to.strip(),
        max_size=max_size,
        engine=engine,
        max_faces=max_faces,
        padding=padding,
        refine_landmarks=refine_landmarks,
        debug=debug,
    )


@dataclass
class ClusterConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dirs: list[str] = field(default_factory=lambda: ["faces"])
    to: str = "clusters"
    model: str = "arcface"
    top: int | None = None
    min_cluster_size: int = 5
    min_samples: int = 2
    batch_size: int = 32
    no_cache: bool = False


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
        from_dirs = ["faces"]

    to = section.get("to", "clusters")
    if not isinstance(to, str) or not to.strip():
        raise click.ClickException("'cluster.to' must be a non-empty string")

    return ClusterConfig(
        working_dir=resolved_working_dir,
        from_dirs=from_dirs,
        to=to.strip(),
        model=model,
        top=top,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        batch_size=batch_size,
    )


@dataclass
class FilterConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    from_dir: str = "faces"
    min_size: int | None = None


def load_filter_config(path: str | Path) -> FilterConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    section = data.get("filter")
    if not section or not isinstance(section, dict):
        return FilterConfig(working_dir=resolved_working_dir)

    from_dir = section.get("from", "faces")
    if not isinstance(from_dir, str) or not from_dir.strip():
        raise click.ClickException("'filter.from' must be a non-empty string")

    min_size = section.get("min_size")
    if min_size is not None:
        if not isinstance(min_size, int) or min_size < 1:
            raise click.ClickException("'filter.min_size' must be a positive integer")

    return FilterConfig(
        working_dir=resolved_working_dir,
        from_dir=from_dir.strip(),
        min_size=min_size,
    )
