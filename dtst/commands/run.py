import logging
import subprocess
from pathlib import Path

import click

from dtst.config import load_workflow_config

logger = logging.getLogger(__name__)


def _build_cli_args(step, config_path, working_dir, workflow_working_dir):
    """Convert a workflow step into CLI argument strings for Click to parse."""
    args = []

    if step.inherit:
        args.append(str(config_path))

    wd = working_dir or (workflow_working_dir if not step.inherit else None)
    if wd:
        args.extend(["--working-dir", str(wd)])

    for key, value in step.overrides.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            if all(isinstance(v, str) for v in value):
                args.extend([flag, ",".join(value)])
            else:
                args.append(flag)
                args.extend(str(v) for v in value)
        else:
            args.extend([flag, str(value)])

    return args


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

        args = _build_cli_args(
            step, config_path, working_dir, workflow_cfg.working_dir
        )
        sub_ctx = click_cmd.make_context(cmd_name, args, parent=ctx)
        with sub_ctx:
            click_cmd.invoke(sub_ctx)

    click.echo(f"\nWorkflow '{workflow}' completed ({total} steps)")
