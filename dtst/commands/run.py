import logging
import subprocess
from pathlib import Path

import click

from dtst.config import load_workflow_config

logger = logging.getLogger(__name__)


def _resolve_param_name(click_cmd, key):
    """Map a YAML override key to the Click parameter name for a command."""
    param_names = {p.name for p in click_cmd.params}
    if key in param_names:
        return key
    if key == "from":
        for candidate in ("from_dirs", "from_dir"):
            if candidate in param_names:
                return candidate
    return None


def _prepare_kwargs(click_cmd, overrides):
    """Convert YAML step overrides to ctx.invoke keyword arguments."""
    kwargs = {}
    for key, value in overrides.items():
        param_name = _resolve_param_name(click_cmd, key)
        if param_name is None:
            raise click.ClickException(
                f"Unknown parameter '{key}' for command '{click_cmd.name}'"
            )
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        if param_name == "working_dir" and value is not None:
            value = Path(value)
        kwargs[param_name] = value
    return kwargs


@click.command("run")
@click.argument("workflow", type=str)
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--working-dir",
    "-d",
    type=click.Path(path_type=Path),
    default=None,
    help="Override working directory.",
)
@click.option("--dry-run", is_flag=True, help="Print steps without executing.")
@click.pass_context
def cmd(ctx, workflow, config, working_dir, dry_run):
    """Run a named workflow defined in a config file.

    Executes a sequence of dtst commands and shell commands as defined
    in the workflows section of the config file. Each command step
    inherits its defaults from the corresponding config section unless
    inherit: false is set.

    \b
    Examples:
        dtst run pipeline config.yaml
        dtst run pipeline config.yaml --dry-run
        dtst run pipeline config.yaml -d ./my_dataset
    """
    config_path = config.resolve()
    workflow_cfg = load_workflow_config(config_path, workflow)

    parent_group = ctx.parent.command
    registered = parent_group.commands

    if workflow in registered:
        raise click.ClickException(
            f"Workflow name '{workflow}' conflicts with a registered command"
        )

    for i, step in enumerate(workflow_cfg.steps, 1):
        if step.command and step.command not in registered:
            raise click.ClickException(
                f"Step {i}: unknown command '{step.command}'"
            )

    total = len(workflow_cfg.steps)

    for i, step in enumerate(workflow_cfg.steps, 1):
        if step.exec:
            logger.info('[%d/%d] Running: exec "%s"', i, total, step.exec)
            if dry_run:
                click.echo(f"  [dry-run] exec: {step.exec}")
                continue
            cwd = str(working_dir if working_dir else workflow_cfg.working_dir)
            result = subprocess.run(step.exec, shell=True, cwd=cwd)
            if result.returncode != 0:
                raise click.ClickException(
                    f'Step {i} failed: exec "{step.exec}" '
                    f"(exit code {result.returncode})"
                )
            continue

        cmd_name = step.command
        click_cmd = registered[cmd_name]

        logger.info("[%d/%d] Running: %s", i, total, cmd_name)
        if dry_run:
            if step.overrides:
                click.echo(f"  [dry-run] {cmd_name}: {step.overrides}")
            else:
                click.echo(f"  [dry-run] {cmd_name}")
            continue

        kwargs = _prepare_kwargs(click_cmd, step.overrides)

        if step.inherit:
            kwargs.setdefault("config", config_path)
        else:
            kwargs.setdefault("config", None)
            kwargs.setdefault(
                "working_dir", working_dir or workflow_cfg.working_dir
            )

        if working_dir is not None:
            kwargs["working_dir"] = working_dir

        ctx.invoke(click_cmd, **kwargs)

    click.echo(f"\nWorkflow '{workflow}' completed ({total} steps)")
