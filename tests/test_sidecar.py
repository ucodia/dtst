import json

import pytest

from dtst.sidecar import (
    EXCLUDE_METRICS,
    EXCLUDE_METRICS_AND_CLASSES,
    copy_sidecar,
    read_all_sidecars,
    read_sidecar,
    scale_classes,
    sidecar_path,
    write_sidecar,
)


class TestSidecarPath:
    def test_simple_name(self, tmp_path):
        img = tmp_path / "foo.jpg"
        assert sidecar_path(img) == tmp_path / "foo.jpg.json"

    def test_multi_dot_name(self, tmp_path):
        img = tmp_path / "image.2024.png"
        assert sidecar_path(img) == tmp_path / "image.2024.png.json"

    def test_nested_path_preserves_parent(self, tmp_path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        img = nested / "pic.jpg"
        result = sidecar_path(img)
        assert result == nested / "pic.jpg.json"
        assert result.parent == nested


class TestReadSidecar:
    def test_missing_returns_empty(self, tmp_path):
        img = tmp_path / "missing.jpg"
        assert read_sidecar(img) == {}

    def test_existing_returns_parsed(self, tmp_path):
        img = tmp_path / "foo.jpg"
        data = {"score": 0.9, "tags": ["a", "b"]}
        (tmp_path / "foo.jpg.json").write_text(json.dumps(data))
        assert read_sidecar(img) == data

    def test_invalid_json_raises(self, tmp_path):
        img = tmp_path / "foo.jpg"
        (tmp_path / "foo.jpg.json").write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            read_sidecar(img)


class TestWriteSidecar:
    def test_writes_new(self, tmp_path):
        img = tmp_path / "foo.jpg"
        write_sidecar(img, {"a": 1})
        sp = tmp_path / "foo.jpg.json"
        assert sp.exists()
        assert json.loads(sp.read_text()) == {"a": 1}

    def test_merges_with_existing(self, tmp_path):
        img = tmp_path / "foo.jpg"
        write_sidecar(img, {"a": 1, "b": 2})
        write_sidecar(img, {"b": 99, "c": 3})
        assert json.loads((tmp_path / "foo.jpg.json").read_text()) == {
            "a": 1,
            "b": 99,
            "c": 3,
        }

    def test_ends_with_newline_and_valid_json(self, tmp_path):
        img = tmp_path / "foo.jpg"
        write_sidecar(img, {"x": 1})
        text = (tmp_path / "foo.jpg.json").read_text()
        assert text.endswith("\n")
        assert json.loads(text) == {"x": 1}


class TestCopySidecar:
    def test_noop_when_source_has_no_sidecar(self, tmp_path):
        src = tmp_path / "src.jpg"
        dest = tmp_path / "dest.jpg"
        copy_sidecar(src, dest)
        assert not (tmp_path / "dest.jpg.json").exists()

    def test_full_copy_without_exclude(self, tmp_path):
        src = tmp_path / "src.jpg"
        dest = tmp_path / "dest.jpg"
        data = {"a": 1, "metrics": {"blur": 0.1}, "classes": {"face": []}}
        write_sidecar(src, data)
        copy_sidecar(src, dest)
        assert json.loads((tmp_path / "dest.jpg.json").read_text()) == data

    def test_exclude_metrics(self, tmp_path):
        src = tmp_path / "src.jpg"
        dest = tmp_path / "dest.jpg"
        write_sidecar(src, {"a": 1, "metrics": {"blur": 0.1}, "classes": {}})
        copy_sidecar(src, dest, exclude=EXCLUDE_METRICS)
        result = json.loads((tmp_path / "dest.jpg.json").read_text())
        assert result == {"a": 1, "classes": {}}

    def test_exclude_metrics_and_classes_constant(self, tmp_path):
        assert EXCLUDE_METRICS_AND_CLASSES == frozenset({"metrics", "classes"})
        src = tmp_path / "src.jpg"
        dest = tmp_path / "dest.jpg"
        write_sidecar(src, {"a": 1, "metrics": {}, "classes": {}})
        copy_sidecar(src, dest, exclude=EXCLUDE_METRICS_AND_CLASSES)
        result = json.loads((tmp_path / "dest.jpg.json").read_text())
        assert result == {"a": 1}

    def test_noop_when_all_keys_excluded(self, tmp_path):
        src = tmp_path / "src.jpg"
        dest = tmp_path / "dest.jpg"
        write_sidecar(src, {"metrics": {}, "classes": {}})
        copy_sidecar(src, dest, exclude=EXCLUDE_METRICS_AND_CLASSES)
        assert not (tmp_path / "dest.jpg.json").exists()


class TestScaleClasses:
    def test_empty(self):
        assert scale_classes({}, 2.0) == {}

    def test_factor_two_doubles_boxes(self):
        classes = {"face": [{"score": 0.9, "box": [10, 20, 30, 40]}]}
        result = scale_classes(classes, 2.0)
        assert result == {"face": [{"score": 0.9, "box": [20, 40, 60, 80]}]}

    def test_factor_half_truncates(self):
        classes = {"face": [{"score": 0.5, "box": [101, 201, 0, 0]}]}
        result = scale_classes(classes, 0.5)
        assert result == {"face": [{"score": 0.5, "box": [50, 100, 0, 0]}]}

    def test_multiple_classes_and_detections(self):
        classes = {
            "face": [
                {"score": 0.9, "box": [10, 20, 30, 40]},
                {"score": 0.8, "box": [1, 2, 3, 4]},
            ],
            "cat": [{"score": 0.7, "box": [100, 100, 200, 200]}],
        }
        result = scale_classes(classes, 2.0)
        assert result == {
            "face": [
                {"score": 0.9, "box": [20, 40, 60, 80]},
                {"score": 0.8, "box": [2, 4, 6, 8]},
            ],
            "cat": [{"score": 0.7, "box": [200, 200, 400, 400]}],
        }


class TestReadAllSidecars:
    def test_mix_of_missing_and_present(self, tmp_path):
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        c = tmp_path / "c.jpg"
        write_sidecar(a, {"x": 1})
        write_sidecar(c, {"z": 3})
        result = read_all_sidecars([a, b, c])
        assert result == {a: {"x": 1}, b: {}, c: {"z": 3}}
        assert set(result.keys()) == {a, b, c}
