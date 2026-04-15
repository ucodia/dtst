"""CLI smoke tests for the ``dtst`` command group.

These are thin integration checks that ensure the Click CLI is wired
together correctly — not that the commands do real work.  We verify:

* ``dtst --help`` mentions every registered subcommand
* every subcommand's ``--help`` renders without errors
* the ``@config_argument`` + ``default_map`` override mechanism works
  end-to-end using ``fetch`` with a monkeypatched core function
* required-field validation for ``--from`` emits the expected error
* unknown subcommands fail with a helpful error
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from dtst.cli import cli

ALL_COMMANDS = [
    "analyze",
    "annotate",
    "augment",
    "cluster",
    "dedup",
    "detect",
    "extract-classes",
    "extract-faces",
    "extract-frames",
    "fetch",
    "format",
    "frame",
    "rename",
    "review",
    "run",
    "search",
    "select",
    "upscale",
    "validate",
]


# ---------------------------------------------------------------------------
# 1. Top-level help
# ---------------------------------------------------------------------------


def test_top_level_help_mentions_all_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    for name in ALL_COMMANDS:
        assert name in result.output, f"{name} missing from top-level --help"


def test_top_level_no_args_exits_nonzero() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, [])
    # Click prints help and exits with code 2 when no subcommand is given
    # on a group.  Some Click versions exit 0; accept any nonzero.
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# 2. Per-command --help
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_COMMANDS)
def test_subcommand_help(name: str) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, [name, "--help"])
    assert result.exit_code == 0, (
        f"{name} --help exited {result.exit_code}\n{result.output}"
    )
    assert "Usage:" in result.output, f"{name} --help missing 'Usage:'"
    # Ensure there is some descriptive text beyond the Usage/Options sections.
    lines = [line.strip() for line in result.output.splitlines() if line.strip()]
    assert len(lines) > 3, f"{name} --help output looks empty: {result.output!r}"
    assert "Traceback" not in result.output, (
        f"{name} --help produced a traceback:\n{result.output}"
    )


# ---------------------------------------------------------------------------
# 3. Config-file + CLI-override integration test
# ---------------------------------------------------------------------------


def test_fetch_config_defaults_and_cli_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify YAML defaults land in the core call and CLI flags override them."""
    calls: list[dict] = []

    def spy(**kwargs):
        calls.append(kwargs)

        class _Result:
            downloaded = 0
            skipped_existing = 0
            skipped_unsupported = 0
            rate_limited = 0
            rate_limited_domains: list[str] = []
            failed = 0
            elapsed = 0.0
            output_dir = "raw"

        return _Result()

    # Patch the core function as imported into the CLI module.
    monkeypatch.setattr("dtst.cli.commands.fetch.core_fetch", spy)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        workdir = Path(td)
        # Provide an input file so the YAML path is plausible.
        (workdir / "results.jsonl").write_text("")
        config_path = workdir / "conf.yaml"
        config_path.write_text(
            "fetch:\n"
            "  to: raw\n"
            "  input: results.jsonl\n"
            "  timeout: 77\n"
            "  min_size: 256\n"
        )

        # First invocation: pure config, no CLI overrides.
        result = runner.invoke(cli, ["fetch", str(config_path)])
        assert result.exit_code == 0, result.output
        assert len(calls) == 1, f"spy not called once: {calls}"
        kwargs = calls[0]
        assert kwargs["to"] == "raw"
        assert kwargs["input_file"] == "results.jsonl"
        assert kwargs["timeout"] == 77
        assert kwargs["min_size"] == 256

        # Second invocation: CLI flag must override YAML.
        result = runner.invoke(
            cli, ["fetch", str(config_path), "--timeout", "3", "--to", "other"]
        )
        assert result.exit_code == 0, result.output
        assert len(calls) == 2
        kwargs2 = calls[1]
        assert kwargs2["timeout"] == 3, "CLI --timeout should win over YAML"
        assert kwargs2["to"] == "other", "CLI --to should win over YAML"
        # Non-overridden YAML values still flow through.
        assert kwargs2["input_file"] == "results.jsonl"
        assert kwargs2["min_size"] == 256


# ---------------------------------------------------------------------------
# 4. Required-field error path
# ---------------------------------------------------------------------------


def test_dedup_without_from_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["dedup"])
    assert result.exit_code != 0
    assert "--from is required" in result.output


# ---------------------------------------------------------------------------
# 5. Unknown subcommand
# ---------------------------------------------------------------------------


def test_unknown_subcommand() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["nonexistent-command-xyz"])
    assert result.exit_code != 0
    # Click prints "No such command" for unknown subcommands on a group.
    assert "No such command" in result.output or "Usage" in result.output
