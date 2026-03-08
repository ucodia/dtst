from dataclasses import dataclass, field
from pathlib import Path

import click
import yaml

VALID_SEARCH_ENGINES = frozenset({"brave", "flickr", "serper", "wikimedia"})
VALID_FACE_ENGINES = frozenset({"mediapipe", "dlib"})


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
    output_dir: Path = Path(".")
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


def _resolve_output_dir(data: dict, config_dir: Path) -> Path:
    output_dir = data.get("output_dir")
    if output_dir is None:
        return Path(".")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise click.ClickException("'output_dir' must be a non-empty string")
    return config_dir / output_dir.strip()


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
        output_dir=_resolve_output_dir(data, config_dir),
        min_size=min_size,
    )


@dataclass
class FetchConfig:
    output_dir: Path = Path(".")
    min_size: int = 512


def load_fetch_config(path: str | Path) -> FetchConfig:
    data, config_dir = load_yaml(path)
    resolved_output_dir = _resolve_output_dir(data, config_dir)

    section = data.get("fetch")
    if not section or not isinstance(section, dict):
        return FetchConfig(output_dir=resolved_output_dir)

    min_size = section.get("min_size", 512)
    if not isinstance(min_size, int) or min_size < 0:
        raise click.ClickException("'fetch.min_size' must be a non-negative integer")

    return FetchConfig(
        output_dir=resolved_output_dir,
        min_size=min_size,
    )


@dataclass
class ExtractFacesConfig:
    input_dir: Path | None = None
    output_dir: Path | None = None
    max_size: int | None = None
    engine: str = "mediapipe"
    max_faces: int = 3
    padding: bool = True
    refine_landmarks: bool = False
    no_stretch: bool = False
    debug: bool = False


def load_extract_faces_config(path: str | Path) -> ExtractFacesConfig:
    data, config_dir = load_yaml(path)
    resolved_output_dir = _resolve_output_dir(data, config_dir)

    section = data.get("extract_faces")
    if not section or not isinstance(section, dict):
        return ExtractFacesConfig(
            input_dir=resolved_output_dir / "raw",
            output_dir=resolved_output_dir / "faces",
        )

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

    no_stretch = section.get("no_stretch", False)
    if not isinstance(no_stretch, bool):
        raise click.ClickException("'extract_faces.no_stretch' must be a boolean")

    debug = section.get("debug", False)
    if not isinstance(debug, bool):
        raise click.ClickException("'extract_faces.debug' must be a boolean")

    input_dir_raw = section.get("input_dir")
    if input_dir_raw is not None:
        input_dir = config_dir / str(input_dir_raw).strip()
    else:
        input_dir = resolved_output_dir / "raw"

    output_dir_raw = section.get("output_dir")
    if output_dir_raw is not None:
        output_dir = config_dir / str(output_dir_raw).strip()
    else:
        output_dir = resolved_output_dir / "faces"

    return ExtractFacesConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        max_size=max_size,
        engine=engine,
        max_faces=max_faces,
        padding=padding,
        refine_landmarks=refine_landmarks,
        no_stretch=no_stretch,
        debug=debug,
    )
