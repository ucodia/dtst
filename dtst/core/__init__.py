"""Library-layer core functions for each dtst command.

Each module exposes a single function (``rename``, ``validate``,
``cluster``, …) that can be called directly from Python without any
Click dependency.  Results are returned as dataclasses from
:mod:`dtst.results`; errors are raised as subclasses of
:class:`dtst.errors.DtstError`.
"""

from dtst.core.analyze import analyze
from dtst.core.annotate import annotate
from dtst.core.augment import augment
from dtst.core.cluster import cluster
from dtst.core.dedup import dedup
from dtst.core.detect import detect
from dtst.core.extract_classes import extract_classes
from dtst.core.extract_faces import extract_faces
from dtst.core.extract_frames import extract_frames
from dtst.core.fetch import fetch
from dtst.core.format import format
from dtst.core.frame import frame
from dtst.core.rename import rename
from dtst.core.search import search
from dtst.core.select import select
from dtst.core.upscale import upscale
from dtst.core.validate import validate

__all__ = [
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
]
