import logging

import click
from dotenv import load_dotenv

from dtst.commands import analyze, annotate, augment, cluster, dedup, detect, extract_faces, extract_frames, fetch, format, frame, rename, review, run, search, select, upscale


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """dtst - dataset toolkit for datasets creation and curation."""
    load_dotenv()
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


cli.add_command(analyze.cmd, "analyze")
cli.add_command(annotate.cmd, "annotate")
cli.add_command(augment.cmd, "augment")
cli.add_command(cluster.cmd, "cluster")
cli.add_command(review.cmd, "review")
cli.add_command(dedup.cmd, "dedup")
cli.add_command(detect.cmd, "detect")
cli.add_command(extract_faces.cmd, "extract-faces")
cli.add_command(extract_frames.cmd, "extract-frames")
cli.add_command(fetch.cmd, "fetch")
cli.add_command(format.cmd, "format")
cli.add_command(frame.cmd, "frame")
cli.add_command(rename.cmd, "rename")
cli.add_command(run.cmd, "run")
cli.add_command(search.cmd, "search")
cli.add_command(select.cmd, "select")
cli.add_command(upscale.cmd, "upscale")


def main() -> None:
    cli()
