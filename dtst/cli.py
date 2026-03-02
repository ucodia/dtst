import logging

import click

from dtst.commands import fetch, search


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """dtst - dataset toolkit for datasets creation and curation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


cli.add_command(fetch.cmd, "fetch")
cli.add_command(search.cmd, "search")


def main() -> None:
    cli()
