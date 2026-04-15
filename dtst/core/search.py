"""Library-layer implementation of ``dtst search``."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dtst.engines import ENGINE_REGISTRY
from dtst.errors import InputError
from dtst.files import resolve_workers
from dtst.results import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_MAX_PAGES = {
    "brave": 1,
    "flickr": 40,
    "inaturalist": 50,
    "serper": 1,
    "wikimedia": 20,
}

TEXT_ENGINES = {"brave", "flickr", "serper", "wikimedia"}


def _run_task(
    args: tuple[str, str, int, int, int, int | float],
) -> tuple[str, list[dict], str | None]:
    query, engine_name, page, min_size, retries, timeout = args
    try:
        engine_cls = ENGINE_REGISTRY.get(engine_name)
        if not engine_cls:
            return engine_name, [], None
        engine = engine_cls(min_size=min_size, retries=retries, timeout=timeout)
        results = engine.search(query, page)
        return engine_name, results, None
    except Exception as e:
        logger.error("Task failed %s %s page %s: %s", query[:40], engine_name, page, e)
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


def search(
    *,
    terms: list[str] | None = None,
    suffixes: list[str] | None = None,
    output: str = "results.jsonl",
    max_pages: int | None = None,
    engines: list[str] | None = None,
    dry_run: bool = False,
    workers: int | None = None,
    min_size: int = 512,
    retries: int = 3,
    timeout: float = 30,
    suffix_only: bool = False,
    taxon_ids: list[int] | None = None,
    progress: bool = True,
) -> SearchResult:
    """Search for images across multiple engines and append to a JSONL file."""
    terms_list = list(terms or [])
    suffixes_list = list(suffixes or [])
    engine_list = [e.strip().lower() for e in (engines or []) if e.strip()]
    taxon_ids_list = list(taxon_ids or [])

    if taxon_ids_list and "inaturalist" not in engine_list:
        engine_list.append("inaturalist")

    if not engine_list:
        raise InputError("At least one engine must be specified.")

    text_engines = [e for e in engine_list if e in TEXT_ENGINES]
    if text_engines:
        if not terms_list:
            raise InputError("Search terms must be provided.")
        if not suffixes_list:
            raise InputError("Suffixes must be provided.")

    if "inaturalist" in engine_list and not taxon_ids_list:
        raise InputError("iNaturalist requires taxon IDs.")

    invalid = [e for e in engine_list if e not in ENGINE_REGISTRY]
    if invalid:
        raise InputError(
            f"Invalid engine(s): {set(invalid)}; valid: {sorted(ENGINE_REGISTRY)}"
        )

    def query_matrix(suffix_only: bool = False) -> list[str]:
        queries: list[str] = []
        if not suffix_only:
            queries.extend(terms_list)
        queries.extend(
            f"{term} {suffix}".strip()
            for term in terms_list
            for suffix in suffixes_list
            if suffix
        )
        return queries

    queries = query_matrix(suffix_only=suffix_only) if text_engines else []
    results_file = Path(output).expanduser().resolve()

    if dry_run:
        return SearchResult(
            queries_run=0,
            engines=engine_list,
            engine_counts={en: 0 for en in engine_list},
            total_unique=0,
            new_urls=0,
            errors=0,
            output_file=results_file,
            dry_run=True,
            queries_preview=queries,
            taxon_ids=taxon_ids_list,
            min_size=min_size,
        )

    num_workers = resolve_workers(workers)

    tasks: list[tuple[str, str, int, int, int, int | float]] = []
    for query in queries:
        for en in text_engines:
            if en not in ENGINE_REGISTRY:
                continue
            limit = (
                max_pages if max_pages is not None else DEFAULT_MAX_PAGES.get(en, 10)
            )
            for page in range(1, limit + 1):
                tasks.append((query, en, page, min_size, retries, timeout))

    if "inaturalist" in engine_list:
        limit = (
            max_pages
            if max_pages is not None
            else DEFAULT_MAX_PAGES.get("inaturalist", 50)
        )
        for tid in taxon_ids_list:
            for page in range(1, limit + 1):
                tasks.append(
                    (str(tid), "inaturalist", page, min_size, retries, timeout)
                )

    query_label = (
        terms_list[0]
        if terms_list
        else str(taxon_ids_list[0])
        if taxon_ids_list
        else "?"
    )
    logger.info(
        'Searching for "%s" across %d engines (%d tasks, %d workers)',
        query_label,
        len(engine_list),
        len(tasks),
        num_workers,
    )

    start_time = time.monotonic()
    engine_counts: dict[str, int] = {en: 0 for en in engine_list}
    all_results: list[dict] = []
    error_count = 0
    total_found = 0

    with logging_redirect_tqdm():
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_run_task, t): t for t in tasks}
            with tqdm(
                total=len(futures),
                desc="Searching",
                unit="page",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]",
                disable=not progress,
            ) as pbar:
                try:
                    for fut in as_completed(futures):
                        engine_name, results, error = fut.result()
                        if error:
                            error_count += 1
                        engine_counts[engine_name] = engine_counts.get(
                            engine_name, 0
                        ) + len(results)
                        all_results.extend(results)
                        total_found += len(results)
                        pbar.set_postfix(results=total_found, errors=error_count)
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    results_file.parent.mkdir(parents=True, exist_ok=True)

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

    query_count = len(queries)
    if "inaturalist" in engine_list:
        query_count += len(taxon_ids_list)

    return SearchResult(
        queries_run=query_count,
        engines=engine_list,
        engine_counts=engine_counts,
        total_unique=len(deduped),
        new_urls=len(deduped) - len(existing_results),
        errors=error_count,
        output_file=results_file,
        dry_run=False,
        queries_preview=queries,
        taxon_ids=taxon_ids_list,
        min_size=min_size,
        elapsed=time.monotonic() - start_time,
    )
