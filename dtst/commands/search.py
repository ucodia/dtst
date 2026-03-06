import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import load_config
from dtst.engines import ENGINE_REGISTRY

logger = logging.getLogger(__name__)

DEFAULT_MAX_PAGES = {
    "brave": 10,
    "flickr": 40,
    "serper": 10,
    "wikimedia": 20,
}


def _run_task(args: tuple[str, str, int]) -> tuple[str, list[str], str | None]:
    query, engine_name, page = args
    try:
        engine_cls = ENGINE_REGISTRY.get(engine_name)
        if not engine_cls:
            return engine_name, [], None
        engine = engine_cls()
        urls = engine.search(query, page)
        return engine_name, urls, None
    except Exception as e:
        logger.error(
            "Task failed %s %s page %s: %s", query[:40], engine_name, page, e
        )
        return engine_name, [], str(e)


@click.command("search")
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option("--max-pages", "-m", type=int, default=None, help="Limit pages per engine per query.")
@click.option("--engines", "-e", type=str, default=None, help="Comma-separated engine list (override config).")
@click.option("--dry-run", "-n", is_flag=True, help="Print query matrix and exit without searching.")
@click.option("--workers", "-w", type=int, default=None, show_default=True, help="Parallel workers (default: CPU count).")
@click.option(
    "--context-only",
    is_flag=True,
    help="Run only queries that include a context suffix (e.g. 'name face'). Skip queries that are just the name or alias alone.",
)
def cmd(
    config: Path,
    max_pages: int | None,
    engines: str | None,
    dry_run: bool,
    workers: int | None,
    context_only: bool,
) -> None:
    """Search for images across multiple engines.

    Reads a subject YAML config file and generates image URLs from
    Flickr, Serper (Google Images), and Wikimedia Commons using
    an expanded query matrix of name variations and contextual terms.
    Results are deduplicated and appended to urls.txt in the output
    directory so multiple runs accumulate new URLs.

    Query matrix: By default, the command runs two kinds of queries for
    each subject term (name and aliases): (1) the term alone, e.g.
    "chanterelle"; (2) the term with each context from query_contexts,
    e.g. "chanterelle mushroom", "chanterelle forest". Use --context-only
    to run only the second kind: every query will be "term + context", and
    no query will be just the bare name or alias.

    \b
    Examples:

        dtst search subjects/chanterelle.yaml
        dtst search subjects/chanterelle.yaml --dry-run
        dtst search subjects/chanterelle.yaml --max-pages 3 --engines flickr,wikimedia
        dtst search subjects/chanterelle.yaml --context-only
    """
    from dotenv import load_dotenv

    load_dotenv()

    cfg = load_config(config)
    engine_list = [e.strip().lower() for e in engines.split(",")] if engines else cfg.engines
    queries = cfg.query_matrix(context_only=context_only)

    if dry_run:
        click.echo("Query matrix:")
        for q in queries:
            click.echo(f"  {q}")
        click.echo("Engines: " + ", ".join(engine_list))
        return

    num_workers = workers if workers is not None else cpu_count() or 4

    tasks: list[tuple[str, str, int]] = []
    for query in queries:
        for en in engine_list:
            if en not in ENGINE_REGISTRY:
                continue
            limit = max_pages if max_pages is not None else DEFAULT_MAX_PAGES.get(en, 10)
            for page in range(1, limit + 1):
                tasks.append((query, en, page))

    logger.info(
        'Searching for "%s" across %d engines (%d queries, %d pages, %d workers)',
        cfg.name, len(engine_list), len(queries), len(tasks), num_workers,
    )

    start_time = time.monotonic()
    engine_counts: dict[str, int] = {en: 0 for en in engine_list}
    all_urls: list[str] = []
    error_count = 0
    total_found = 0

    with logging_redirect_tqdm():
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run_task, t): t for t in tasks}
            with tqdm(total=len(futures), desc="Searching", unit="page", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]") as pbar:
                for fut in as_completed(futures):
                    engine_name, urls, error = fut.result()
                    if error:
                        error_count += 1
                    engine_counts[engine_name] = engine_counts.get(engine_name, 0) + len(urls)
                    all_urls.extend(urls)
                    total_found += len(urls)
                    pbar.set_postfix(urls=total_found, errors=error_count)
                    pbar.update(1)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    urls_file = out_dir / "urls.txt"

    with open(urls_file, "a") as f:
        for u in all_urls:
            f.write(u.strip() + "\n")

    with open(urls_file) as f:
        seen = {ln.strip() for ln in f if ln.strip()}

    with open(urls_file, "w") as f:
        for u in sorted(seen):
            f.write(u + "\n")

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"Queries run: {len(queries)} x {len(engine_list)} engines")
    for en in engine_list:
        click.echo(f"  {en}: {engine_counts.get(en, 0)} URLs")
    click.echo(f"Total unique URLs (after dedup): {len(seen)}")
    click.echo(f"Errors: {error_count}")
    click.echo(f"Time: {minutes}m {seconds}s")
