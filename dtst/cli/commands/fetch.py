"""Click wrapper for ``dtst fetch`` — delegates to :mod:`dtst.core.fetch`."""

from __future__ import annotations

from pathlib import Path

import click

from dtst.cli.config import (
    apply_working_dir,
    config_argument,
    to_dir_option,
    working_dir_option,
    workers_option,
)
from dtst.errors import DtstError
from dtst.files import format_elapsed


@click.command("fetch")
@config_argument
@working_dir_option()
@to_dir_option()
@click.option(
    "--input",
    "-i",
    "input_file",
    type=str,
    default=None,
    help="Input file path (.jsonl or .txt).",
)
@click.option(
    "--min-size",
    "-s",
    type=int,
    default=None,
    help="Minimum image dimension in pixels; only applies to .jsonl input (default: 512).",
)
@workers_option(
    help="Number of parallel download threads (default: CPU count for images, 2 for video)."
)
@click.option(
    "--timeout",
    "-t",
    type=int,
    default=30,
    show_default=True,
    help="Per-request timeout in seconds.",
)
@click.option(
    "--force", "-f", is_flag=True, help="Re-download files even if they already exist."
)
@click.option(
    "--max-wait",
    "-W",
    type=int,
    default=None,
    help="Max seconds to honor a Retry-After header (default: unlimited).",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="Never wait for Retry-After headers; use fast exponential backoff instead.",
)
@click.option(
    "--license",
    "-l",
    "license_filter",
    type=str,
    default=None,
    help="Only download images whose license starts with this prefix (e.g. 'cc'); only applies to .jsonl input.",
)
def cmd(
    working_dir: Path | None,
    to: str | None,
    input_file: str | None,
    min_size: int | None,
    workers: int | None,
    timeout: int,
    force: bool,
    max_wait: int | None,
    no_wait: bool,
    license_filter: str | None,
) -> None:
    """Download images and videos from a URL list.

    Reads a URL list from the working directory specified by --input.
    Two formats are supported:

    \b
      .jsonl  JSON Lines with a "url" field per line (search output).
              Supports --min-size and --license filtering.
      .txt    Plain text with one URL per line. Lines starting with
              # are treated as comments.

    URLs are routed automatically: known video hosting domains
    (YouTube, Vimeo, etc.) are downloaded with yt-dlp, all other
    URLs are downloaded directly with HTTP requests.

    Image files are named by the MD5 hash of the URL. Video files
    are named by yt-dlp using the video ID and original extension.
    Existing files are skipped unless --force is set.

    Can be invoked with just a config file, just CLI options, or both.
    When both are provided, CLI options override config file values.

    \b
    Examples:

        dtst fetch config.yaml
        dtst fetch -d ./chanterelle --to raw --input results.jsonl
        dtst fetch -d ./project --to videos --input urls.txt
        dtst fetch config.yaml --workers 16 --timeout 60
        dtst fetch config.yaml --force
        dtst fetch -d ./chanterelle --to raw --input results.jsonl --no-wait --license cc
    """
    if no_wait and max_wait is not None:
        raise click.ClickException("--no-wait and --max-wait are mutually exclusive")
    if to is None:
        raise click.ClickException("--to is required (or set 'fetch.to' in config)")
    if input_file is None:
        raise click.ClickException(
            "--input is required (or set 'fetch.input' in config)"
        )

    apply_working_dir(working_dir)
    from dtst.core.fetch import fetch as core_fetch

    try:
        result = core_fetch(
            to=to,
            input_file=input_file,
            min_size=min_size if min_size is not None else 512,
            workers=workers,
            timeout=timeout,
            force=force,
            max_wait=max_wait,
            no_wait=no_wait,
            license_filter=license_filter,
        )
    except DtstError as e:
        raise click.ClickException(str(e)) from e

    click.echo("\nFetch complete!")
    click.echo(f"  Downloaded: {result.downloaded:,}")
    click.echo(f"  Skipped (existing): {result.skipped_existing:,}")
    if result.skipped_unsupported > 0:
        click.echo(f"  Skipped (unsupported format): {result.skipped_unsupported:,}")
    if result.rate_limited > 0:
        domains_str = ", ".join(result.rate_limited_domains)
        click.echo(f"  Rate limited: {result.rate_limited:,} ({domains_str})")
    click.echo(f"  Failed: {result.failed:,}")
    click.echo(f"  Time: {format_elapsed(result.elapsed)}")
    click.echo(f"  Output: {result.output_dir}")
