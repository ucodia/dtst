"""Async wrappers for :mod:`dtst.core`.

Each function here is a thin ``asyncio.to_thread`` shim around the
corresponding sync core function.  Signatures, docstrings, and return
types are preserved via :func:`functools.wraps`, so::

    from dtst.aio import fetch
    result = await fetch(working_dir=wd, to="raw", input_file="urls.txt")

behaves exactly like the sync version but does not block the event
loop.  There is no native async I/O underneath — dtst's workloads are
CPU/GPU-bound or already use thread pools, so ``to_thread`` is the
right tool.
"""

from __future__ import annotations

import asyncio
import functools

from dtst import core


def _wrap(fn):
    @functools.wraps(fn)
    async def awrap(**kwargs):
        return await asyncio.to_thread(fn, **kwargs)

    return awrap


for _name in core.__all__:
    globals()[_name] = _wrap(getattr(core, _name))

__all__ = list(core.__all__)
