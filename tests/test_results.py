"""Tests for dtst.results dataclasses and dtst.errors hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest

from dtst.errors import ConfigError, DtstError, InputError, PipelineError
from dtst.results import FetchResult, RenameResult, ValidateResult


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_dtst_error_is_exception(self) -> None:
        assert issubclass(DtstError, Exception)

    @pytest.mark.parametrize("cls", [InputError, ConfigError, PipelineError])
    def test_subclass_of_dtst_error(self, cls: type) -> None:
        assert issubclass(cls, DtstError)
        assert issubclass(cls, Exception)

    def test_can_raise_and_catch_as_dtst_error(self) -> None:
        with pytest.raises(DtstError):
            raise InputError("boom")
        with pytest.raises(DtstError):
            raise ConfigError("boom")
        with pytest.raises(DtstError):
            raise PipelineError("boom")

    def test_message_preserved(self) -> None:
        err = InputError("missing --from")
        assert str(err) == "missing --from"


# ---------------------------------------------------------------------------
# ValidateResult.passed
# ---------------------------------------------------------------------------


def make_validate(
    *,
    dim_counts: dict | None = None,
    mode_counts: dict | None = None,
    non_square: int = 0,
    failed: int = 0,
    square_checked: bool = False,
) -> ValidateResult:
    return ValidateResult(
        total=0,
        dim_counts=dim_counts if dim_counts is not None else {},
        mode_counts=mode_counts if mode_counts is not None else {},
        non_square=non_square,
        total_png=0,
        compressed_png=0,
        failed=failed,
        square_checked=square_checked,
        elapsed=0.0,
    )


class TestValidateResultPassed:
    def test_all_empty_no_square_check_passes(self) -> None:
        assert make_validate().passed is True

    def test_single_dim_single_mode_square_ok_passes(self) -> None:
        r = make_validate(
            dim_counts={(512, 512): 10},
            mode_counts={"RGB": 10},
            square_checked=True,
            non_square=0,
        )
        assert r.passed is True

    def test_multiple_dims_fails(self) -> None:
        r = make_validate(dim_counts={(512, 512): 1, (256, 256): 1})
        assert r.passed is False

    def test_multiple_modes_fails(self) -> None:
        r = make_validate(mode_counts={"RGB": 1, "RGBA": 1})
        assert r.passed is False

    def test_square_checked_non_square_fails(self) -> None:
        r = make_validate(square_checked=True, non_square=5)
        assert r.passed is False

    def test_failed_nonzero_fails(self) -> None:
        r = make_validate(failed=1)
        assert r.passed is False

    def test_non_square_irrelevant_when_not_checked(self) -> None:
        r = make_validate(square_checked=False, non_square=999)
        assert r.passed is True


# ---------------------------------------------------------------------------
# Sample dataclass construction
# ---------------------------------------------------------------------------


class TestDataclassConstruction:
    def test_rename_result(self) -> None:
        r = RenameResult(renamed=3, dry_run=False, elapsed=1.25)
        assert r.renamed == 3
        assert r.dry_run is False
        assert r.elapsed == 1.25

    def test_fetch_result(self) -> None:
        r = FetchResult(
            downloaded=10,
            skipped_existing=2,
            skipped_unsupported=1,
            failed=0,
            rate_limited=0,
            rate_limited_domains=[],
            output_dir=Path("/tmp/out"),
            elapsed=5.0,
        )
        assert r.downloaded == 10
        assert r.output_dir == Path("/tmp/out")
        assert r.rate_limited_domains == []
