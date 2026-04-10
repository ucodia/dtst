from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import yaml

# Constants used by command Click options (click.Choice)
VALID_FRAME_FORMATS = frozenset({"jpg", "png"})
FRAME_MODES = ("stretch", "crop", "pad")
FRAME_GRAVITIES = ("center", "top", "bottom", "left", "right")
FRAME_FILLS = ("color", "edge", "reflect", "blur")


def load_yaml(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    with open(config_path) as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        raise click.ClickException("Config must be a non-empty YAML object")
    return data, config_path.parent


def _resolve_working_dir(data: dict, config_dir: Path) -> Path:
    working_dir = data.get("working_dir")
    if working_dir is None:
        return Path(".")
    if not isinstance(working_dir, str) or not working_dir.strip():
        raise click.ClickException("'working_dir' must be a non-empty string")
    return config_dir / working_dir.strip()


# Mapping from YAML config keys to Click parameter names where they differ.
_YAML_TO_CLICK = {
    "from": "from_dirs",
    "format": "fmt",
    "input": "input_file",
    "license": "license_filter",
}


def _find_param(command: click.BaseCommand, name: str) -> click.Parameter | None:
    for p in command.params:
        if p.name == name:
            return p
    return None


def _coerce_for_click(param: click.Parameter, value: Any) -> Any:
    """Convert a YAML value to what Click expects for a given parameter."""
    is_multi = getattr(param, "multiple", False)
    is_tuple = isinstance(param.type, click.Tuple)

    if isinstance(value, bool):
        return value

    # dict → tuple of tuples (for type=(str, float), multiple=True)
    if isinstance(value, dict) and is_tuple and is_multi:
        return tuple((str(k), float(v)) for k, v in value.items())

    # list → comma-joined string (for type=str, multiple=False like --from)
    if isinstance(value, list) and not is_multi:
        return ",".join(str(v) for v in value)

    # list → tuple (for multiple=True options)
    if isinstance(value, list) and is_multi:
        return tuple(str(v) for v in value)

    # scalar for a multiple option → single-element tuple
    if is_multi and not isinstance(value, (list, tuple)):
        return (value,)

    return value


def apply_config_defaults(
    ctx: click.Context, _param: click.Parameter, value: Path | None
) -> Path | None:
    """Eager callback: parse YAML config and set ctx.default_map."""
    if value is None:
        return None
    data, config_dir = load_yaml(value)
    section_key = ctx.info_name.replace("-", "_")
    section = data.get(section_key, {})
    if not isinstance(section, dict):
        section = {}

    defaults: dict[str, Any] = {"working_dir": _resolve_working_dir(data, config_dir)}
    for yaml_key, yaml_val in section.items():
        click_key = _YAML_TO_CLICK.get(yaml_key, yaml_key)
        param = _find_param(ctx.command, click_key)
        if param:
            defaults[click_key] = _coerce_for_click(param, yaml_val)
    ctx.default_map = defaults
    return value


def config_argument(f):
    """Decorator: add an optional YAML config positional argument to a command."""
    return click.argument(
        "config",
        type=click.Path(exists=True, path_type=Path),
        required=False,
        default=None,
        is_eager=True,
        callback=apply_config_defaults,
        expose_value=False,
    )(f)


# ---------------------------------------------------------------------------
# Workflow support (used by the `run` command)
# ---------------------------------------------------------------------------


@dataclass
class WorkflowStep:
    command: str | None = None
    exec: str | None = None
    inherit: bool = True
    overrides: dict = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    working_dir: Path = field(default_factory=lambda: Path("."))
    steps: list[WorkflowStep] = field(default_factory=list)


def load_workflow_config(path: str | Path, workflow_name: str) -> WorkflowConfig:
    data, config_dir = load_yaml(path)
    resolved_working_dir = _resolve_working_dir(data, config_dir)

    workflows = data.get("workflows")
    if not workflows or not isinstance(workflows, dict):
        raise click.ClickException("Config must have a 'workflows' section")

    workflow = workflows.get(workflow_name)
    if workflow is None:
        available = ", ".join(sorted(workflows.keys()))
        raise click.ClickException(
            f"Workflow '{workflow_name}' not found; available: {available}"
        )
    if not isinstance(workflow, list):
        raise click.ClickException(
            f"Workflow '{workflow_name}' must be a list of steps"
        )

    steps: list[WorkflowStep] = []
    for i, raw_step in enumerate(workflow, 1):
        if isinstance(raw_step, str):
            steps.append(WorkflowStep(command=raw_step))
        elif isinstance(raw_step, dict):
            if "exec" in raw_step:
                exec_cmd = raw_step["exec"]
                if not isinstance(exec_cmd, str) or not exec_cmd.strip():
                    raise click.ClickException(
                        f"Step {i}: 'exec' must be a non-empty string"
                    )
                steps.append(WorkflowStep(exec=exec_cmd.strip()))
            else:
                keys = list(raw_step.keys())
                if len(keys) != 1:
                    raise click.ClickException(
                        f"Step {i}: expected a single command key, got {keys}"
                    )
                cmd_name = keys[0]
                raw_overrides = raw_step[cmd_name]
                if raw_overrides is None:
                    overrides = {}
                elif isinstance(raw_overrides, dict):
                    overrides = dict(raw_overrides)
                else:
                    raise click.ClickException(
                        f"Step {i}: overrides for '{cmd_name}' must be a mapping"
                    )
                inherit = overrides.pop("inherit", True)
                if not isinstance(inherit, bool):
                    raise click.ClickException(f"Step {i}: 'inherit' must be a boolean")
                steps.append(
                    WorkflowStep(command=cmd_name, inherit=inherit, overrides=overrides)
                )
        else:
            raise click.ClickException(f"Step {i}: must be a command name or mapping")

    return WorkflowConfig(working_dir=resolved_working_dir, steps=steps)
