"""Compatibility shim — re-exports from :mod:`dtst.cli.config`.

Historical import path kept so unmigrated commands in
``dtst.commands`` continue to work.  New code should import from
:mod:`dtst.cli.config` directly.
"""

from dtst.cli.config import *  # noqa: F401, F403
from dtst.cli.config import (  # noqa: F401
    _YAML_TO_CLICK,
    _coerce_for_click,
    _find_param,
    _resolve_working_dir,
    apply_working_dir,
)
