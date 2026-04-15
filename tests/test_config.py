"""Tests for dtst.cli.config — YAML loading, coercion, working-dir, workflows."""

from __future__ import annotations

import os
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from dtst.cli.config import (
    WorkflowConfig,
    WorkflowStep,
    _coerce_for_click,
    _find_param,
    _resolve_working_dir,
    apply_config_defaults,
    apply_working_dir,
    config_argument,
    load_workflow_config,
    load_yaml,
)


# ---------------------------------------------------------------------------
# load_yaml
# ---------------------------------------------------------------------------


def test_load_yaml_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(click.ClickException) as exc:
        load_yaml(p)
    assert "non-empty YAML object" in exc.value.message


def test_load_yaml_top_level_list_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(click.ClickException):
        load_yaml(p)


def test_load_yaml_valid_dict_returns_data_and_parent_dir(tmp_path: Path) -> None:
    p = tmp_path / "good.yaml"
    p.write_text("foo: 1\nbar: two\n")
    data, config_dir = load_yaml(p)
    assert data == {"foo": 1, "bar": "two"}
    assert config_dir == tmp_path.resolve()


def test_load_yaml_returns_absolute_parent_for_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "cfg.yaml"
    p.write_text("k: v\n")
    _, config_dir = load_yaml("cfg.yaml")
    assert config_dir.is_absolute()
    assert config_dir == tmp_path.resolve()


# ---------------------------------------------------------------------------
# _resolve_working_dir
# ---------------------------------------------------------------------------


def test_resolve_working_dir_missing_returns_dot(tmp_path: Path) -> None:
    result = _resolve_working_dir({}, tmp_path)
    assert result == Path(".")


def test_resolve_working_dir_non_string_raises(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException):
        _resolve_working_dir({"working_dir": 42}, tmp_path)
    with pytest.raises(click.ClickException):
        _resolve_working_dir({"working_dir": ["x"]}, tmp_path)


def test_resolve_working_dir_empty_string_raises(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException):
        _resolve_working_dir({"working_dir": ""}, tmp_path)
    with pytest.raises(click.ClickException):
        _resolve_working_dir({"working_dir": "   "}, tmp_path)


def test_resolve_working_dir_relative_joins_with_config_dir(tmp_path: Path) -> None:
    result = _resolve_working_dir({"working_dir": "  ./scratch  "}, tmp_path)
    assert result == tmp_path / "./scratch"


def test_resolve_working_dir_absolute_overrides_config_dir(tmp_path: Path) -> None:
    # Path("/abs") / "/other" yields Path("/other") — absolute wins.
    abs_target = (tmp_path / "elsewhere").resolve()
    result = _resolve_working_dir({"working_dir": str(abs_target)}, tmp_path)
    assert result == abs_target


# ---------------------------------------------------------------------------
# apply_working_dir
# ---------------------------------------------------------------------------


def test_apply_working_dir_none_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = Path.cwd()
    apply_working_dir(None)
    assert Path.cwd() == before


def test_apply_working_dir_creates_and_chdirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "new_subdir"
    assert not target.exists()
    apply_working_dir(target)
    assert target.is_dir()
    assert Path(os.getcwd()).resolve() == target.resolve()


def test_apply_working_dir_expands_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    apply_working_dir(Path("~/dtst-test-subdir"))
    expected = (tmp_path / "dtst-test-subdir").resolve()
    assert expected.is_dir()
    assert Path(os.getcwd()).resolve() == expected


# ---------------------------------------------------------------------------
# _coerce_for_click + _find_param
# ---------------------------------------------------------------------------


@click.command()
@click.option("--from", "from_dirs", type=str, default=None)
@click.option("--include", multiple=True)
@click.option("--tag", type=(str, float), multiple=True)
@click.option("--flag", is_flag=True)
@click.option("--quality", type=int, default=None)
def _fake_cmd(from_dirs, include, tag, flag, quality):  # pragma: no cover
    pass


def test_find_param_found() -> None:
    p = _find_param(_fake_cmd, "from_dirs")
    assert p is not None
    assert p.name == "from_dirs"


def test_find_param_not_found() -> None:
    assert _find_param(_fake_cmd, "does_not_exist") is None


def test_coerce_bool_passthrough_even_for_multi() -> None:
    # bool short-circuits first
    param = _find_param(_fake_cmd, "include")
    assert _coerce_for_click(param, True) is True
    assert _coerce_for_click(param, False) is False


def test_coerce_list_to_comma_for_non_multiple() -> None:
    param = _find_param(_fake_cmd, "from_dirs")
    assert _coerce_for_click(param, ["a", "b", "c"]) == "a,b,c"


def test_coerce_list_of_ints_stringified_when_joined() -> None:
    param = _find_param(_fake_cmd, "from_dirs")
    assert _coerce_for_click(param, [1, 2, 3]) == "1,2,3"


def test_coerce_list_to_tuple_for_multiple() -> None:
    param = _find_param(_fake_cmd, "include")
    assert _coerce_for_click(param, ["a", "b"]) == ("a", "b")


def test_coerce_list_of_ints_for_multiple_stringified() -> None:
    param = _find_param(_fake_cmd, "include")
    assert _coerce_for_click(param, [1, 2]) == ("1", "2")


def test_coerce_dict_for_tuple_multiple() -> None:
    param = _find_param(_fake_cmd, "tag")
    result = _coerce_for_click(param, {"a": 1, "b": 2.5})
    assert result == (("a", 1.0), ("b", 2.5))


def test_coerce_scalar_for_multiple_wraps_in_tuple() -> None:
    param = _find_param(_fake_cmd, "include")
    assert _coerce_for_click(param, "x") == ("x",)


def test_coerce_scalar_for_non_multiple_unchanged() -> None:
    param = _find_param(_fake_cmd, "quality")
    assert _coerce_for_click(param, 42) == 42


# ---------------------------------------------------------------------------
# apply_config_defaults — integration via CliRunner
# ---------------------------------------------------------------------------


def _build_search_cmd():
    @click.command("search")
    @config_argument
    @click.option("--from", "from_dirs", type=str, default=None)
    @click.option("--quality", type=int, default=None)
    @click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None)
    def search(from_dirs, quality, working_dir):
        click.echo(f"from={from_dirs}|quality={quality}|wd={working_dir}")

    return search


def _build_hyphen_cmd():
    @click.command("extract-faces")
    @config_argument
    @click.option("--to", type=str, default=None)
    @click.option("--working-dir", "-d", type=click.Path(path_type=Path), default=None)
    def extract_faces(to, working_dir):
        click.echo(f"to={to}|wd={working_dir}")

    return extract_faces


def test_apply_config_sets_defaults_from_yaml() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("cfg.yaml").write_text(
            "working_dir: ./sub\nsearch:\n  from: [a, b]\n  quality: 80\n"
        )
        result = runner.invoke(_build_search_cmd(), ["cfg.yaml"])
        assert result.exit_code == 0, result.output
        assert "from=a,b" in result.output
        assert "quality=80" in result.output
        # wd is a Path object pointing to the resolved ./sub
        assert "wd=" in result.output
        assert "sub" in result.output


def test_cli_overrides_yaml_value() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("cfg.yaml").write_text("search:\n  quality: 80\n")
        result = runner.invoke(_build_search_cmd(), ["cfg.yaml", "--quality", "99"])
        assert result.exit_code == 0, result.output
        assert "quality=99" in result.output


def test_yaml_from_key_maps_to_from_dirs_click_param() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("cfg.yaml").write_text("search:\n  from: raw\n")
        result = runner.invoke(_build_search_cmd(), ["cfg.yaml"])
        assert result.exit_code == 0, result.output
        assert "from=raw" in result.output


def test_hyphenated_command_name_maps_to_underscore_section() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("cfg.yaml").write_text("extract_faces:\n  to: faces\n")
        result = runner.invoke(_build_hyphen_cmd(), ["cfg.yaml"])
        assert result.exit_code == 0, result.output
        assert "to=faces" in result.output


def test_missing_section_only_sets_working_dir() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("cfg.yaml").write_text("other: 1\n")
        result = runner.invoke(_build_search_cmd(), ["cfg.yaml"])
        assert result.exit_code == 0, result.output
        assert "from=None" in result.output
        assert "quality=None" in result.output


def test_non_dict_section_silently_ignored() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("cfg.yaml").write_text("search: 42\n")
        result = runner.invoke(_build_search_cmd(), ["cfg.yaml"])
        assert result.exit_code == 0, result.output
        assert "from=None" in result.output
        assert "quality=None" in result.output


def test_apply_config_defaults_returns_none_when_value_none() -> None:
    # Unit-level: callback returns None when value is None (no config path passed)
    assert apply_config_defaults(click.Context(_build_search_cmd()), None, None) is None


# ---------------------------------------------------------------------------
# load_workflow_config
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "wf.yaml"
    p.write_text(body)
    return p


def test_workflow_missing_section_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "other: 1\n")
    with pytest.raises(click.ClickException) as exc:
        load_workflow_config(p, "default")
    assert "workflows" in exc.value.message


def test_workflow_unknown_name_lists_available(tmp_path: Path) -> None:
    p = _write(tmp_path, "workflows:\n  alpha: []\n  beta: []\n")
    with pytest.raises(click.ClickException) as exc:
        load_workflow_config(p, "gamma")
    assert "gamma" in exc.value.message
    assert "alpha" in exc.value.message
    assert "beta" in exc.value.message


def test_workflow_not_a_list_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "workflows:\n  default:\n    key: value\n")
    with pytest.raises(click.ClickException) as exc:
        load_workflow_config(p, "default")
    assert "list of steps" in exc.value.message


def test_workflow_step_plain_string(tmp_path: Path) -> None:
    p = _write(tmp_path, "workflows:\n  default:\n    - fetch\n")
    cfg = load_workflow_config(p, "default")
    assert isinstance(cfg, WorkflowConfig)
    assert len(cfg.steps) == 1
    s = cfg.steps[0]
    assert s.command == "fetch"
    assert s.exec is None
    assert s.inherit is True
    assert s.overrides == {}


def test_workflow_step_exec(tmp_path: Path) -> None:
    p = _write(tmp_path, 'workflows:\n  default:\n    - exec: "echo hi"\n')
    cfg = load_workflow_config(p, "default")
    s = cfg.steps[0]
    assert s.exec == "echo hi"
    assert s.command is None


def test_workflow_step_exec_empty_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, 'workflows:\n  default:\n    - exec: ""\n')
    with pytest.raises(click.ClickException):
        load_workflow_config(p, "default")


def test_workflow_step_exec_whitespace_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, 'workflows:\n  default:\n    - exec: "   "\n')
    with pytest.raises(click.ClickException):
        load_workflow_config(p, "default")


def test_workflow_step_exec_non_string_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "workflows:\n  default:\n    - exec: 42\n")
    with pytest.raises(click.ClickException):
        load_workflow_config(p, "default")


def test_workflow_step_mapping_with_overrides(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "workflows:\n  default:\n    - fetch:\n        to: raw\n        min_size: 512\n",
    )
    cfg = load_workflow_config(p, "default")
    s = cfg.steps[0]
    assert s.command == "fetch"
    assert s.overrides == {"to": "raw", "min_size": 512}
    assert s.inherit is True


def test_workflow_step_mapping_with_inherit_false(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "workflows:\n  default:\n    - fetch:\n        inherit: false\n        to: raw\n",
    )
    cfg = load_workflow_config(p, "default")
    s = cfg.steps[0]
    assert s.command == "fetch"
    assert s.inherit is False
    assert s.overrides == {"to": "raw"}


def test_workflow_step_mapping_with_null_overrides(tmp_path: Path) -> None:
    p = _write(tmp_path, "workflows:\n  default:\n    - fetch: ~\n")
    cfg = load_workflow_config(p, "default")
    s = cfg.steps[0]
    assert s.command == "fetch"
    assert s.overrides == {}
    assert s.inherit is True


def test_workflow_step_inherit_non_bool_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        'workflows:\n  default:\n    - fetch:\n        inherit: "yes"\n',
    )
    with pytest.raises(click.ClickException) as exc:
        load_workflow_config(p, "default")
    assert "inherit" in exc.value.message


def test_workflow_step_multi_key_mapping_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "workflows:\n  default:\n    - a: {}\n      b: {}\n",
    )
    with pytest.raises(click.ClickException) as exc:
        load_workflow_config(p, "default")
    assert "single command key" in exc.value.message


def test_workflow_step_overrides_not_mapping_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "workflows:\n  default:\n    - fetch: [1, 2]\n")
    with pytest.raises(click.ClickException) as exc:
        load_workflow_config(p, "default")
    assert "must be a mapping" in exc.value.message


def test_workflow_step_bare_integer_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "workflows:\n  default:\n    - 42\n")
    with pytest.raises(click.ClickException) as exc:
        load_workflow_config(p, "default")
    assert "command name or mapping" in exc.value.message


def test_workflow_working_dir_resolved_relative_to_yaml(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "working_dir: ./scratch\nworkflows:\n  default:\n    - fetch\n",
    )
    cfg = load_workflow_config(p, "default")
    assert cfg.working_dir == tmp_path.resolve() / "./scratch"


def test_workflow_default_working_dir_when_absent(tmp_path: Path) -> None:
    p = _write(tmp_path, "workflows:\n  default:\n    - fetch\n")
    cfg = load_workflow_config(p, "default")
    assert cfg.working_dir == Path(".")


def test_workflow_step_dataclass_defaults() -> None:
    s = WorkflowStep()
    assert s.command is None
    assert s.exec is None
    assert s.inherit is True
    assert s.overrides == {}
