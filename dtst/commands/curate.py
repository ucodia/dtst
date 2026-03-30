from __future__ import annotations

import webbrowser
from pathlib import Path

import click

from dtst.config import CurateConfig, load_curate_config


def _resolve_config(
    config: Path | None,
    working_dir: Path | None,
    from_dir: str | None,
    to: str | None,
    port: int | None,
) -> CurateConfig:
    if config is not None:
        cfg = load_curate_config(config)
    else:
        cfg = CurateConfig()

    if working_dir is not None:
        cfg.working_dir = working_dir
    if from_dir is not None:
        cfg.from_dir = from_dir
    if to is not None:
        cfg.to = to
    if port is not None:
        cfg.port = port

    return cfg


@click.command("curate")
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
)
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
    default=None,
    help="Subfolder name for filtered images.",
    show_default="filtered_manual",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=None,
    help="Port for the web server.",
    show_default="8888",
)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Do not open the browser automatically.",
)
@click.option(
    "--working-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory (default: .).",
)
def cmd(config, from_dir, to, port, no_open, working_dir):
    """Launch a web UI for manual image curation.

    Opens a local web server with an image grid. Click images to
    select or deselect them, then apply to move filtered images
    into a subfolder. Use the view toggle to switch between source
    and filtered images to restore previously filtered images.

    Press Ctrl+C to stop the server.

    \b
    Examples:
        dtst curate config.yaml
        dtst curate -d ./project --from faces
        dtst curate -d ./project --from faces --to rejected --port 9000
        dtst curate config.yaml --no-open
    """
    cfg = _resolve_config(config, working_dir, from_dir, to, port)

    if cfg.from_dir is None:
        raise click.ClickException("--from is required (or set 'curate.from' in config)")

    working = cfg.working_dir.resolve()
    source_dir = working / cfg.from_dir
    filtered_dir = source_dir / cfg.to

    if not source_dir.is_dir():
        raise click.ClickException(f"Source directory does not exist: {source_dir}")

    import uvicorn

    from dtst.curate.server import create_app

    app = create_app(source_dir, filtered_dir)

    url = f"http://localhost:{cfg.port}"
    click.echo(f"Starting curate server at {url}")
    click.echo(f"  Source: {source_dir}")
    click.echo(f"  Filtered: {filtered_dir}")
    click.echo("  Press Ctrl+C to stop.\n")

    if not no_open:
        webbrowser.open(url)

    uvicorn.run(app, host="127.0.0.1", port=cfg.port, log_level="warning")
