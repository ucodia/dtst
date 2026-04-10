import logging
import subprocess
from pathlib import Path

import click

from dtst.config import (
    _YAML_TO_CLICK,
    _coerce_for_click,
    _resolve_working_dir,
    load_workflow_config,
    load_yaml,
)

logger = logging.getLogger(__name__)


def _build_ctx_params(click_cmd, step, config_path, working_dir, workflow_working_dir):
    """Build a ctx.params dict for a Click command from workflow step overrides.

    Loads the command's config section from the YAML file (when inherit is
    True), applies step overrides on top, and coerces all values to the
    types Click expects.
    """
    param_info = {p.name: p for p in click_cmd.params}

    # Start with defaults for all params (skip expose_value=False arguments)
    params = {}
    for p in click_cmd.params:
        if isinstance(p, click.Argument):
            if not p.expose_value:
                continue
            params[p.name] = p.default
        elif isinstance(p, click.Option):
            if p.multiple:
                params[p.name] = p.default or ()
            else:
                params[p.name] = p.default

    # Load config section defaults (replaces the old config= pass-through)
    if step.inherit:
        data, config_dir = load_yaml(config_path)
        section_key = click_cmd.name.replace("-", "_")
        section = data.get(section_key, {})
        if isinstance(section, dict):
            for yaml_key, value in section.items():
                param_name = _YAML_TO_CLICK.get(yaml_key, yaml_key)
                p = param_info.get(param_name)
                if p is not None:
                    params[param_name] = _coerce_for_click(p, value)
        # Inject resolved working_dir
        if "working_dir" in param_info:
            params["working_dir"] = _resolve_working_dir(data, config_dir)

    # CLI working_dir override takes precedence
    wd = working_dir or (workflow_working_dir if not step.inherit else None)
    if wd and "working_dir" in param_info:
        params["working_dir"] = wd

    # Apply step overrides with type coercion
    for key, value in step.overrides.items():
        param_name = key.replace("-", "_")
        param_name = _YAML_TO_CLICK.get(param_name, param_name)

        p = param_info.get(param_name)
        if p is None:
            raise click.ClickException(
                f"Unknown parameter '{key}' for command '{click_cmd.name}'"
            )

        params[param_name] = _coerce_for_click(p, value)

    return params


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
            raise click.ClickException(f"Step {i}: unknown command '{step.command}'")

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

        params = _build_ctx_params(
            click_cmd, step, config_path, working_dir, workflow_cfg.working_dir
        )
        sub_ctx = click.Context(click_cmd, info_name=cmd_name, parent=ctx)
        sub_ctx.params = params
        with sub_ctx:
            click_cmd.invoke(sub_ctx)

    click.echo(f"\nWorkflow '{workflow}' completed ({total} steps)")
