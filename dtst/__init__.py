"""dtst — a Python toolkit for image dataset creation and curation.

Two entry points:

* **CLI** (``dtst`` command): see :mod:`dtst.cli`.
* **Library**: import command functions from :mod:`dtst.core` or from
  the top-level package.  Each returns a result dataclass from
  :mod:`dtst.results` and raises :class:`dtst.errors.DtstError`
  subclasses on failure.

Example::

    from dtst import validate

    result = validate(from_dirs="./my-dataset/faces", progress=False)
    if not result.passed:
        ...
"""

from dtst.errors import ConfigError, DtstError, InputError, PipelineError
from dtst.results import (
    AnalyzeResult,
    AnnotateResult,
    AugmentResult,
    ClusterInfo,
    ClusterResult,
    DedupResult,
    DetectResult,
    ExtractClassesResult,
    ExtractFacesResult,
    ExtractFramesResult,
    FetchResult,
    FormatResult,
    FrameResult,
    RenameResult,
    SearchResult,
    SelectResult,
    UpscaleResult,
    ValidateResult,
)

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
    raise AttributeError(f"module 'dtst' has no attribute {name!r}")


__all__ = [
    # commands
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
    # errors
    "DtstError",
    "InputError",
    "ConfigError",
    "PipelineError",
    # results
    "AnalyzeResult",
    "AnnotateResult",
    "AugmentResult",
    "ClusterInfo",
    "ClusterResult",
    "DedupResult",
    "DetectResult",
    "ExtractClassesResult",
    "ExtractFacesResult",
    "ExtractFramesResult",
    "FetchResult",
    "FormatResult",
    "FrameResult",
    "RenameResult",
    "SearchResult",
    "SelectResult",
    "UpscaleResult",
    "ValidateResult",
]
