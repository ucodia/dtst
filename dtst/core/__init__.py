"""Library-layer core functions for each dtst command.

Each module exposes a single function (``rename``, ``validate``,
``cluster``, …) that can be called directly from Python without any
Click dependency.  Results are returned as dataclasses from
:mod:`dtst.results`; errors are raised as subclasses of
:class:`dtst.errors.DtstError`.

Submodules are loaded lazily (PEP 562) so that accessing one command
does not import the others or their dependencies.
"""

from __future__ import annotations

_COMMANDS = frozenset(
    {
        "analyze",
        "annotate",
        "augment",
        "cluster",
        "dedup",
        "detect",
        "extract_classes",
        "extract_faces",
        "extract_frames",
        "fetch",
        "format",
        "frame",
        "rename",
        "search",
        "select",
        "upscale",
        "validate",
    }
)


def __getattr__(name: str):
    if name in _COMMANDS:
        import importlib

        mod = importlib.import_module(f"dtst.core.{name}")
        return getattr(mod, name)
    raise AttributeError(f"module 'dtst.core' has no attribute {name!r}")


__all__ = sorted(_COMMANDS)
