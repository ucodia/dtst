from __future__ import annotations

import webbrowser
from pathlib import Path

import click

from dtst.config import config_argument, require_extra, working_dir_option


@click.command("review")
@config_argument
@click.option(
    "--from",
    "from_dir",
    type=str,
    default=None,
    help="Source folder name within working directory.",
)
@click.option(
    "--to",
    type=str,
    default="rejected",
    help="Subfolder name for filtered images.",
    show_default=True,
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8888,
    help="Port for the web server.",
    show_default=True,
)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Do not open the browser automatically.",
)
@working_dir_option()
def cmd(from_dir, to, port, no_open, working_dir):
    """Launch a web UI for manual image review.

    Opens a local web server with an image grid. Click images to
    select or deselect them, then apply to move filtered images
    into a subfolder. Use the view toggle to switch between source
    and filtered images to restore previously filtered images.

    Press Ctrl+C to stop the server.

    \b
    Examples:
        dtst review config.yaml
        dtst review -d ./project --from faces
        dtst review -d ./project --from faces --to rejected --port 9000
        dtst review config.yaml --no-open
    """
    working = (working_dir or Path(".")).resolve()

    if from_dir is not None:
        source = working / from_dir
        if not source.is_dir():
            raise click.ClickException(f"Source directory does not exist: {source}")
        filtered = source / to
    else:
        source = None
        filtered = None

    require_extra("fastapi", extra="server")
    import uvicorn

    from dtst.review.server import create_app

    app = create_app(working, source, filtered)

    url = f"http://localhost:{port}"
    click.echo(f"Starting review server at {url}")
    if source is not None:
        click.echo(f"  Source: {source}")
        click.echo(f"  Filtered: {filtered}")
    else:
        click.echo(f"  Working dir: {working}")
        click.echo("  Select buckets in the browser to begin.")
    click.echo("  Press Ctrl+C to stop.\n")

    if not no_open:
        webbrowser.open(url)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
