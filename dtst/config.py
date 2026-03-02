from dataclasses import dataclass
from pathlib import Path

import click
import yaml

VALID_ENGINES = frozenset({"flickr", "serper", "bing", "wikimedia"})


@dataclass
class SubjectConfig:
    name: str
    aliases: list[str]
    query_contexts: list[str]
    engines: list[str]
    output_dir: Path

    def subject_terms(self) -> list[str]:
        return [self.name] + (self.aliases or [])

    def query_matrix(self) -> list[str]:
        terms = self.subject_terms()
        return [f"{term} {ctx}".strip() for term in terms for ctx in (self.query_contexts or [])]


def load_config(path: str | Path) -> SubjectConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        raise click.ClickException("Config must be a non-empty YAML object")
    name = data.get("name")
    if not name or not isinstance(name, str):
        raise click.ClickException("Config must have a non-empty string 'name'")
    aliases = data.get("aliases")
    if aliases is not None and not isinstance(aliases, list):
        raise click.ClickException("'aliases' must be a list of strings")
    if aliases is not None:
        aliases = [a for a in aliases if isinstance(a, str)]
    else:
        aliases = []
    query_contexts = data.get("query_contexts")
    if not query_contexts or not isinstance(query_contexts, list):
        raise click.ClickException("Config must have a non-empty list 'query_contexts'")
    query_contexts = [str(c) for c in query_contexts]
    engines = data.get("engines")
    if not engines or not isinstance(engines, list):
        raise click.ClickException("Config must have a non-empty list 'engines'")
    engines = [str(e).strip().lower() for e in engines]
    invalid = set(engines) - VALID_ENGINES
    if invalid:
        raise click.ClickException(f"Invalid engine(s): {invalid}; valid: {sorted(VALID_ENGINES)}")
    output_dir = data.get("output_dir")
    if not output_dir or not isinstance(output_dir, str):
        raise click.ClickException("Config must have a non-empty string 'output_dir'")
    return SubjectConfig(
        name=name.strip(),
        aliases=aliases,
        query_contexts=query_contexts,
        engines=engines,
        output_dir=Path(output_dir.strip()),
    )
