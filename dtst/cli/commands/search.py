"""Click wrapper for ``dtst search`` — delegates to :mod:`dtst.core.search`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    config_argument,
    working_dir_option,
    workers_option,
)
from dtst.core.search import search as core_search
from dtst.errors import DtstError
from dtst.files import format_elapsed


@click.command("search")
@config_argument
@click.option(
    "--terms",
    type=str,
    default=None,
    help="Comma-separated search terms (override config).",
)
@click.option(
    "--suffixes",
    type=str,
    default=None,
    help="Comma-separated query suffixes (override config).",
)
@working_dir_option(help="Working directory where results are written (default: .).")
@click.option(
    "--output",
    "-o",
    type=str,
    default=None,
    help="Output filename within the working directory (default: results.jsonl).",
)
@click.option(
    "--max-pages",
    "-m",
    type=int,
    default=None,
    help="Limit pages per engine per query.",
)
@click.option(
    "--engines",
    "-e",
    type=str,
    default=None,
    help="Comma-separated engine list (override config).",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Print query matrix and exit without searching.",
)
@workers_option(help="Parallel workers (default: CPU count).")
@click.option(
    "--min-size",
    "-s",
    type=int,
    default=None,
    help="Minimum image dimension in pixels (default: 512).",
)
@click.option(
    "--retries",
    "-r",
    type=int,
    default=3,
    show_default=True,
    help="Number of retries per request (with exponential backoff).",
)
@click.option(
    "--timeout",
    "-t",
    type=float,
    default=30,
    show_default=True,
    help="Request timeout in seconds.",
)
@click.option(
    "--suffix-only",
    is_flag=True,
    help="Run only queries that include a suffix (e.g. 'term suffix'). Skip bare term queries.",
)
@click.option(
    "--taxon-ids",
    type=str,
    default=None,
    help="Comma-separated iNaturalist taxon IDs (implies --engines inaturalist).",
)
def cmd(
    terms: str | None,
    suffixes: str | None,
    working_dir: Path | None,
    output: str | None,
    max_pages: int | None,
    engines: str | None,
    dry_run: bool,
    workers: int | None,
    min_size: int | None,
    retries: int,
    timeout: int | float,
    suffix_only: bool,
    taxon_ids: str | None,
) -> None:
    """Search for images across multiple engines.

    Reads an optional YAML config file and generates image URLs from
    Flickr, Serper (Google Images), Brave, Wikimedia Commons, and
    iNaturalist. Text-based engines use an expanded query matrix of
    search terms and suffixes. iNaturalist uses taxon IDs instead.
    Results are deduplicated and written to a JSONL file in the working
    directory (default: results.jsonl) so multiple runs accumulate new
    results.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    Query matrix: By default, the command runs two kinds of queries for
    each term: (1) the term alone, e.g. "chanterelle"; (2) the term
    with each suffix, e.g. "chanterelle mushroom", "chanterelle forest".
    Use --suffix-only to run only the second kind.

    \b
    Examples:

        dtst search config.yaml
        dtst search config.yaml --dry-run
        dtst search config.yaml --max-pages 3 --engines flickr,wikimedia
        dtst search --terms "chanterelle" --suffixes "mushroom,forest" --engines brave -d ./chanterelle
        dtst search --taxon-ids 47169,54743 -d ./fungi
    """
    terms_list = [t.strip() for t in terms.split(",") if t.strip()] if terms else []
    suffixes_list = (
        [s.strip() for s in suffixes.split(",") if s.strip()] if suffixes else []
    )
    engine_list = (
        [e.strip().lower() for e in engines.split(",") if e.strip()] if engines else []
    )

    taxon_ids_list: list[int] = []
    if taxon_ids:
        for t in taxon_ids.split(","):
            t = t.strip()
            if t:
                try:
                    taxon_ids_list.append(int(t))
                except ValueError:
                    raise click.ClickException(
                        f"Invalid taxon ID: {t!r}; must be an integer."
                    )

    try:
        result = core_search(
            terms=terms_list,
            suffixes=suffixes_list,
            working_dir=working_dir,
            output=output or "results.jsonl",
            max_pages=max_pages,
            engines=engine_list,
            dry_run=dry_run,
            workers=workers,
            min_size=min_size if min_size is not None else 512,
            retries=retries,
            timeout=timeout,
            suffix_only=suffix_only,
            taxon_ids=taxon_ids_list,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    if dry_run:
        if result.queries_preview:
            click.echo("Query matrix:")
            for q in result.queries_preview:
                click.echo(f"  {q}")
        if "inaturalist" in result.engines:
            click.echo(
                "iNaturalist taxon IDs: " + ", ".join(str(t) for t in result.taxon_ids)
            )
        click.echo("Engines: " + ", ".join(result.engines))
        click.echo(f"Min size: {result.min_size}px")
        return

    click.echo(
        f"Queries run: {result.queries_run} across {len(result.engines)} engines"
    )
    for en in result.engines:
        click.echo(f"  {en}: {result.engine_counts.get(en, 0)} results")
    click.echo(f"Total unique results (after dedup): {result.total_unique}")
    click.echo(f"New URLs added: {result.new_urls}")
    click.echo(f"Errors: {result.errors}")
    click.echo(f"Time: {format_elapsed(result.elapsed)}")
