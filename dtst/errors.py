"""Library-layer exceptions.

These are raised by :mod:`dtst.core` functions so library callers never
have to depend on Click.  The CLI layer in :mod:`dtst.cli` catches them
and re-raises as ``click.ClickException`` for user-friendly output.
"""

from __future__ import annotations


class DtstError(Exception):
    """Base class for all library-layer errors."""


class InputError(DtstError):
    """Invalid or missing user input (missing --from, bad directory, etc.)."""


class ConfigError(DtstError):
    """Invalid configuration file or values."""


class PipelineError(DtstError):
    """Failure during pipeline execution (e.g. no clusters found)."""
