import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import click
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.config import SearchConfig, load_search_config
from dtst.engines import ENGINE_REGISTRY

logger = logging.getLogger(__name__)

DEFAULT_MAX_PAGES = {
    "brave": 10,
    "flickr": 40,
    "serper": 10,
    "wikimedia": 20,
}


def _run_task(
    args: tuple[str, str, int, int, int, int | float],
) -> tuple[str, list[dict], str | None]:
    query, engine_name, page, min_size, retries, timeout = args
    try:
        engine_cls = ENGINE_REGISTRY.get(engine_name)
        if not engine_cls:
            return engine_name, [], None
        engine = engine_cls(
            min_size=min_size,
            retries=retries,
            timeout=timeout,
        )
        results = engine.search(query, page)
        return engine_name, results, None
    except Exception as e:
        logger.error(
            "Task failed %s %s page %s: %s", query[:40], engine_name, page, e
        )
        return engine_name, [], str(e)


def _dedup_results(results: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in results:
        url = r.get("url")
        if not url:
            continue
        existing = seen.get(url)
        if existing is None:
            seen[url] = r
        else:
            existing_non_null = sum(1 for v in existing.values() if v is not None)
            new_non_null = sum(1 for v in r.values() if v is not None)
            if new_non_null > existing_non_null:
                seen[url] = r
    return list(seen.values())


def _resolve_config(
    config: Path | None,
    terms: str | None,
    suffixes: str | None,
    engines: str | None,
    output_dir: Path | None,
    min_size: int | None,
) -> SearchConfig:
    if config is not None:
        cfg = load_search_config(config)
    else:
        cfg = SearchConfig()

    if terms is not None:
        cfg.terms = [t.strip() for t in terms.split(",") if t.strip()]
    if suffixes is not None:
        cfg.suffixes = [s.strip() for s in suffixes.split(",") if s.strip()]
    if engines is not None:
        cfg.engines = [e.strip().lower() for e in engines.split(",") if e.strip()]
    if output_dir is not None:
        cfg.output_dir = output_dir
    if min_size is not None:
        cfg.min_size = min_size

    if not cfg.terms:
        raise click.ClickException("Search terms must be provided via config or --terms.")
    if not cfg.suffixes:
        raise click.ClickException("Suffixes must be provided via config or --suffixes.")
    if not cfg.engines:
        raise click.ClickException("At least one engine must be specified via config or --engines.")

    return cfg


@click.command("search")
@click.argument("config", type=click.Path(exists=True, path_type=Path), required=False, default=None)
@click.option("--terms", type=str, default=None, help="Comma-separated search terms (override config).")
@click.option("--suffixes", type=str, default=None, help="Comma-separated query suffixes (override config).")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None, help="Output directory (default: .).")
@click.option("--max-pages", "-m", type=int, default=None, help="Limit pages per engine per query.")
@click.option("--engines", "-e", type=str, default=None, help="Comma-separated engine list (override config).")
@click.option("--dry-run", "-n", is_flag=True, help="Print query matrix and exit without searching.")
@click.option("--workers", "-w", type=int, default=None, show_default=True, help="Parallel workers (default: CPU count).")
@click.option("--min-size", "-s", type=int, default=None, help="Minimum image dimension in pixels (default: 512).")
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
def cmd(
    config: Path | None,
    terms: str | None,
    suffixes: str | None,
    output_dir: Path | None,
    max_pages: int | None,
    engines: str | None,
    dry_run: bool,
    workers: int | None,
    min_size: int | None,
    retries: int,
    timeout: int | float,
    suffix_only: bool,
) -> None:
    """Search for images across multiple engines.

    Reads an optional YAML config file and generates image URLs from
    Flickr, Serper (Google Images), Brave and Wikimedia Commons using
    an expanded query matrix of search terms and suffixes.
    Results are deduplicated and written to results.jsonl in the output
    directory so multiple runs accumulate new results.

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
        dtst search --terms "chanterelle,mushroom" --suffixes "face,portrait" --engines brave -o ./out
    """
    cfg = _resolve_config(config, terms, suffixes, engines, output_dir, min_size)

    engine_list = cfg.engines
    invalid = [e for e in engine_list if e not in ENGINE_REGISTRY]
    if invalid:
        raise click.ClickException(
            f"Invalid engine(s): {set(invalid)}; valid: {sorted(ENGINE_REGISTRY)}"
        )
    queries = cfg.query_matrix(suffix_only=suffix_only)

    if dry_run:
        click.echo("Query matrix:")
        for q in queries:
            click.echo(f"  {q}")
        click.echo("Engines: " + ", ".join(engine_list))
        click.echo(f"Min size: {cfg.min_size}px")
        return

    num_workers = workers if workers is not None else cpu_count() or 4

    tasks: list[tuple[str, str, int, int, int, int | float]] = []
    for query in queries:
        for en in engine_list:
            if en not in ENGINE_REGISTRY:
                continue
            limit = max_pages if max_pages is not None else DEFAULT_MAX_PAGES.get(en, 10)
            for page in range(1, limit + 1):
                tasks.append((query, en, page, cfg.min_size, retries, timeout))

    logger.info(
        'Searching for "%s" across %d engines (%d queries, %d pages, %d workers)',
        cfg.terms[0], len(engine_list), len(queries), len(tasks), num_workers,
    )

    start_time = time.monotonic()
    engine_counts: dict[str, int] = {en: 0 for en in engine_list}
    all_results: list[dict] = []
    error_count = 0
    total_found = 0

    with logging_redirect_tqdm():
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run_task, t): t for t in tasks}
            with tqdm(total=len(futures), desc="Searching", unit="page", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]") as pbar:
                try:
                    for fut in as_completed(futures):
                        engine_name, results, error = fut.result()
                        if error:
                            error_count += 1
                        engine_counts[engine_name] = engine_counts.get(engine_name, 0) + len(results)
                        all_results.extend(results)
                        total_found += len(results)
                        pbar.set_postfix(results=total_found, errors=error_count)
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.jsonl"

    existing_results: list[dict] = []
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    combined = existing_results + all_results
    deduped = _dedup_results(combined)

    with open(results_file, "w") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.monotonic() - start_time
    minutes, seconds = divmod(int(elapsed), 60)

    click.echo(f"Queries run: {len(queries)} x {len(engine_list)} engines")
    for en in engine_list:
        click.echo(f"  {en}: {engine_counts.get(en, 0)} results")
    click.echo(f"Total unique results (after dedup): {len(deduped)}")
    click.echo(f"New URLs added: {len(deduped) - len(existing_results)}")
    click.echo(f"Errors: {error_count}")
    click.echo(f"Time: {minutes}m {seconds}s")
