"""Tests for the filter chain in dtst.core.select.

Drives ``select()`` end-to-end against ``tmp_path`` with synthetic image files
and sidecar JSON. The dimension-check filter is skipped intentionally — it
spawns a ProcessPoolExecutor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dtst.core.select import select
from dtst.sidecar import write_sidecar


def _call_select(
    src: Path,
    dst: Path,
    **kwargs,
):
    """Invoke ``select`` in dry-run mode with sensible defaults."""
    return select(
        from_dirs=str(src),
        to=str(dst),
        dry_run=True,
        progress=False,
        **kwargs,
    )


def _reject_reasons(result) -> dict[str, str]:
    return {name: reason for name, reason in result.rejects_preview}


# ---------------------------------------------------------------------------
# Metric threshold filters
# ---------------------------------------------------------------------------


class TestMetricFilters:
    def test_min_metric_rejects_below_threshold(
        self, tmp_path: Path, make_image
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        low = make_image(name="src/low.jpg")
        high = make_image(name="src/high.jpg")
        write_sidecar(low, {"metrics": {"blur": 50.0}})
        write_sidecar(high, {"metrics": {"blur": 150.0}})

        result = _call_select(
            src,
            tmp_path / "out",
            min_metric=[("blur", 100.0)],
        )

        assert result.selected == 1
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert "low.jpg" in reasons
        assert "blur too low" in reasons["low.jpg"]

    def test_max_metric_rejects_above_threshold(
        self, tmp_path: Path, make_image
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        low = make_image(name="src/low.jpg")
        high = make_image(name="src/high.jpg")
        write_sidecar(low, {"metrics": {"blur": 50.0}})
        write_sidecar(high, {"metrics": {"blur": 250.0}})

        result = _call_select(
            src,
            tmp_path / "out",
            max_metric=[("blur", 200.0)],
        )

        assert result.selected == 1
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert "high.jpg" in reasons
        assert "blur too high" in reasons["high.jpg"]

    def test_missing_metric_rejected(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        img = make_image(name="src/a.jpg")
        write_sidecar(img, {"metrics": {"sharpness": 0.9}})

        result = _call_select(
            src,
            tmp_path / "out",
            min_metric=[("blur", 100.0)],
        )

        assert result.selected == 0
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert reasons["a.jpg"] == "missing 'blur' metric data"

    def test_missing_metric_rejected_for_max(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        img = make_image(name="src/a.jpg")
        write_sidecar(img, {"metrics": {"sharpness": 0.9}})

        result = _call_select(
            src,
            tmp_path / "out",
            max_metric=[("blur", 200.0)],
        )

        assert result.selected == 0
        reasons = _reject_reasons(result)
        assert reasons["a.jpg"] == "missing 'blur' metric data"

    def test_min_and_max_combined(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        too_low = make_image(name="src/too_low.jpg")
        in_range = make_image(name="src/in_range.jpg")
        too_high = make_image(name="src/too_high.jpg")
        write_sidecar(too_low, {"metrics": {"blur": 50.0}})
        write_sidecar(in_range, {"metrics": {"blur": 150.0}})
        write_sidecar(too_high, {"metrics": {"blur": 300.0}})

        result = _call_select(
            src,
            tmp_path / "out",
            min_metric=[("blur", 100.0)],
            max_metric=[("blur", 200.0)],
        )

        assert result.selected == 1
        assert result.excluded == 2


# ---------------------------------------------------------------------------
# Detection class threshold filters
# ---------------------------------------------------------------------------


class TestDetectionFilters:
    def test_min_detect_requires_score_above_threshold(
        self, tmp_path: Path, make_image
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        strong = make_image(name="src/strong.jpg")
        weak = make_image(name="src/weak.jpg")
        write_sidecar(
            strong, {"classes": {"face": [{"score": 0.9, "box": [0, 0, 1, 1]}]}}
        )
        write_sidecar(
            weak, {"classes": {"face": [{"score": 0.2, "box": [0, 0, 1, 1]}]}}
        )

        result = _call_select(
            src,
            tmp_path / "out",
            min_detect=[("face", 0.5)],
        )

        assert result.selected == 1
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert "weak.jpg" in reasons
        assert "face" in reasons["weak.jpg"]

    def test_min_detect_rejects_missing_class(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        img = make_image(name="src/a.jpg")
        write_sidecar(img, {"classes": {"cat": [{"score": 0.9, "box": [0, 0, 1, 1]}]}})

        result = _call_select(
            src,
            tmp_path / "out",
            min_detect=[("face", 0.5)],
        )

        assert result.selected == 0
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert reasons["a.jpg"] == "'face' not detected"

    def test_max_detect_rejects_when_score_at_or_above_threshold(
        self, tmp_path: Path, make_image
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        strong = make_image(name="src/strong.jpg")
        weak = make_image(name="src/weak.jpg")
        write_sidecar(
            strong, {"classes": {"face": [{"score": 0.95, "box": [0, 0, 1, 1]}]}}
        )
        write_sidecar(
            weak, {"classes": {"face": [{"score": 0.1, "box": [0, 0, 1, 1]}]}}
        )

        result = _call_select(
            src,
            tmp_path / "out",
            max_detect=[("face", 0.9)],
        )

        assert result.selected == 1
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert "strong.jpg" in reasons
        assert ">= 0.9" in reasons["strong.jpg"]

    def test_max_detect_ignores_missing_class(self, tmp_path: Path, make_image) -> None:
        """When the class isn't present, max_detect has nothing to reject on."""
        src = tmp_path / "src"
        src.mkdir()
        img = make_image(name="src/a.jpg")
        write_sidecar(img, {"classes": {"cat": [{"score": 0.9, "box": [0, 0, 1, 1]}]}})

        result = _call_select(
            src,
            tmp_path / "out",
            max_detect=[("face", 0.9)],
        )

        assert result.selected == 1
        assert result.excluded == 0

    def test_missing_classes_rejected(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        img = make_image(name="src/a.jpg")
        write_sidecar(img, {"metrics": {"blur": 150.0}})

        result = _call_select(
            src,
            tmp_path / "out",
            min_detect=[("face", 0.5)],
        )

        assert result.selected == 0
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert reasons["a.jpg"] == "missing detection data"


# ---------------------------------------------------------------------------
# Source / license filters
# ---------------------------------------------------------------------------


class TestSourceLicenseFilters:
    def test_source_allow_list(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        flickr = make_image(name="src/flickr.jpg")
        brave = make_image(name="src/brave.jpg")
        wiki = make_image(name="src/wiki.jpg")
        write_sidecar(flickr, {"source": "Flickr"})
        write_sidecar(brave, {"source": "brave"})
        write_sidecar(wiki, {"source": "Wikimedia"})

        result = _call_select(
            src,
            tmp_path / "out",
            source=["flickr", "brave"],
        )

        assert result.selected == 2
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert "wiki.jpg" in reasons
        assert "source 'Wikimedia' not in" in reasons["wiki.jpg"]

    def test_source_case_insensitive_match(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        img = make_image(name="src/a.jpg")
        write_sidecar(img, {"source": "FLICKR"})

        result = _call_select(
            src,
            tmp_path / "out",
            source=["flickr"],
        )

        assert result.selected == 1
        assert result.excluded == 0

    def test_missing_source_rejected(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        img = make_image(name="src/a.jpg")
        write_sidecar(img, {"metrics": {"blur": 150.0}})

        result = _call_select(
            src,
            tmp_path / "out",
            source=["flickr"],
        )

        assert result.selected == 0
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert reasons["a.jpg"] == "missing source data"

    def test_license_allow_list(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        cc0 = make_image(name="src/cc0.jpg")
        by = make_image(name="src/by.jpg")
        nd = make_image(name="src/nd.jpg")
        write_sidecar(cc0, {"license": "CC0"})
        write_sidecar(by, {"license": "cc-by"})
        write_sidecar(nd, {"license": "cc-nd"})

        result = _call_select(
            src,
            tmp_path / "out",
            license_filter=["cc0", "cc-by"],
        )

        assert result.selected == 2
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert "nd.jpg" in reasons
        assert "license 'cc-nd' not in" in reasons["nd.jpg"]

    def test_missing_license_rejected(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        img = make_image(name="src/a.jpg")
        write_sidecar(img, {"source": "flickr"})

        result = _call_select(
            src,
            tmp_path / "out",
            license_filter=["cc0"],
        )

        assert result.selected == 0
        assert result.excluded == 1
        reasons = _reject_reasons(result)
        assert reasons["a.jpg"] == "missing license data"

    def test_source_and_license_combined(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        ok = make_image(name="src/ok.jpg")
        bad_license = make_image(name="src/bad_license.jpg")
        bad_source = make_image(name="src/bad_source.jpg")
        write_sidecar(ok, {"source": "flickr", "license": "cc0"})
        write_sidecar(bad_license, {"source": "flickr", "license": "cc-nd"})
        write_sidecar(bad_source, {"source": "brave", "license": "cc0"})

        result = _call_select(
            src,
            tmp_path / "out",
            source=["flickr"],
            license_filter=["cc0"],
        )

        assert result.selected == 1
        assert result.excluded == 2


# ---------------------------------------------------------------------------
# No criteria: all images pass
# ---------------------------------------------------------------------------


class TestNoFilters:
    def test_no_criteria_keeps_everything(self, tmp_path: Path, make_image) -> None:
        src = tmp_path / "src"
        src.mkdir()
        make_image(name="src/a.jpg")
        make_image(name="src/b.jpg")

        result = _call_select(src, tmp_path / "out")

        assert result.selected == 2
        assert result.excluded == 0
        assert result.total_images == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_from_raises(self, tmp_path: Path) -> None:
        from dtst.errors import InputError

        with pytest.raises(InputError):
            select(from_dirs="", to=str(tmp_path / "out"), dry_run=True)

    def test_empty_to_raises(self, tmp_path: Path, make_image) -> None:
        from dtst.errors import InputError

        src = tmp_path / "src"
        src.mkdir()
        make_image(name="src/a.jpg")

        with pytest.raises(InputError):
            select(from_dirs=str(src), to="", dry_run=True)

    def test_missing_source_dir_raises(self, tmp_path: Path) -> None:
        from dtst.errors import InputError

        with pytest.raises(InputError):
            select(
                from_dirs=str(tmp_path / "nope"),
                to=str(tmp_path / "out"),
                dry_run=True,
            )

    def test_empty_source_dir_raises(self, tmp_path: Path) -> None:
        from dtst.errors import InputError

        src = tmp_path / "src"
        src.mkdir()

        with pytest.raises(InputError):
            select(from_dirs=str(src), to=str(tmp_path / "out"), dry_run=True)
