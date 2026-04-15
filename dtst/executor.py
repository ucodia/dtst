"""Shared scaffolding for ``concurrent.futures`` pools with tqdm progress.

Commands wrap a ``ProcessPoolExecutor`` or ``ThreadPoolExecutor`` in
``logging_redirect_tqdm`` with identical boilerplate for progress bars,
interrupt handling, and postfix counters.  :func:`run_pool` consolidates
that scaffolding; callers provide the worker function, the work items,
and a per-result handler that performs any side effects and returns a
status string for counting.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Executor, as_completed
from typing import Callable, Iterable, Sequence, TypeVar

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

W = TypeVar("W")
R = TypeVar("R")


def run_pool(
    executor_cls: type[Executor],
    worker_fn: Callable[[W], R],
    work_items: Sequence[W],
    *,
    max_workers: int,
    desc: str,
    unit: str,
    on_result: Callable[[R, W], str | None],
    postfix_keys: Iterable[str] | None = None,
    bar_format: str | None = None,
    progress: bool = True,
) -> dict[str, int]:
    """Run ``worker_fn`` over ``work_items`` with a tqdm-tracked pool.

    ``on_result(result, work_item)`` is called for each completed future
    in completion order.  Its return value — a short status string such
    as ``"ok"`` or ``"failed"`` — is tallied into a counter dict and
    surfaced via ``pbar.set_postfix``.  Return ``None`` to skip counting.

    ``postfix_keys`` fixes which counters appear in the live postfix
    and in what order; if omitted, every status seen so far is shown.

    ``KeyboardInterrupt`` shuts the pool down with ``cancel_futures=True``
    before re-raising.  The whole run is wrapped in
    ``logging_redirect_tqdm()`` so log lines do not collide with the bar.

    Set ``progress=False`` to silence the tqdm bar — useful for library
    callers that want no terminal output.

    Returns the final counter dict so callers can build their summary.
    """
    counts: Counter[str] = Counter()
    keys = tuple(postfix_keys) if postfix_keys is not None else None
    tqdm_kwargs: dict = {}
    if bar_format is not None:
        tqdm_kwargs["bar_format"] = bar_format

    with logging_redirect_tqdm():
        with executor_cls(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_fn, w): w for w in work_items}
            with tqdm(
                total=len(futures),
                desc=desc,
                unit=unit,
                disable=not progress,
                **tqdm_kwargs,
            ) as pbar:
                try:
                    for future in as_completed(futures):
                        work_item = futures[future]
                        result = future.result()
                        status = on_result(result, work_item)
                        if status is not None:
                            counts[status] += 1
                        if keys is None:
                            pbar.set_postfix(dict(counts))
                        else:
                            pbar.set_postfix({k: counts[k] for k in keys})
                        pbar.update(1)
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    return dict(counts)
