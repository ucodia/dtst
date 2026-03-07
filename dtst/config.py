from dataclasses import dataclass, field
from pathlib import Path

import click
import yaml

VALID_ENGINES = frozenset({"brave", "flickr", "serper", "wikimedia"})


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
        invalid = set(engines) - VALID_ENGINES
        if invalid:
            raise click.ClickException(
                f"Invalid engine(s): {invalid}; valid: {sorted(VALID_ENGINES)}"
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
