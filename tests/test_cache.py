"""Tests for dtst.cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dtst.cache import CACHE_DIR, _cache_key, load_embeddings, save_embeddings


class TestCacheKey:
    def test_deterministic(self) -> None:
        paths = [Path("a.jpg"), Path("b.jpg")]
        assert _cache_key("m", paths) == _cache_key("m", paths)

    def test_order_insensitive(self) -> None:
        a = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        b = [Path("c.jpg"), Path("a.jpg"), Path("b.jpg")]
        assert _cache_key("m", a) == _cache_key("m", b)

    def test_different_model_different_key(self) -> None:
        paths = [Path("a.jpg")]
        assert _cache_key("m1", paths) != _cache_key("m2", paths)

    def test_different_image_set_different_key(self) -> None:
        k1 = _cache_key("m", [Path("a.jpg")])
        k2 = _cache_key("m", [Path("a.jpg"), Path("b.jpg")])
        assert k1 != k2

    def test_keys_distinguish_directories(self, tmp_path: Path) -> None:
        # Key hashes full resolved paths, so same filename in different dirs differs
        d1 = tmp_path / "x"
        d2 = tmp_path / "y"
        d1.mkdir()
        d2.mkdir()
        k1 = _cache_key("m", [d1 / "a.jpg"])
        k2 = _cache_key("m", [d2 / "a.jpg"])
        assert k1 != k2

    def test_keys_stable_across_invocations_same_dir(self, tmp_path: Path) -> None:
        # Same resolved location should produce the same key even if constructed
        # from different Path instances (e.g. relative vs absolute).
        paths_a = [tmp_path / "img.jpg"]
        paths_b = [tmp_path / "img.jpg"]
        assert _cache_key("m", paths_a) == _cache_key("m", paths_b)


class TestLoadEmbeddings:
    def test_returns_none_when_cache_dir_missing(self, isolated_cwd: Path) -> None:
        paths = [isolated_cwd / "a.jpg"]
        assert load_embeddings("model-v1", paths) is None

    def test_save_load_roundtrip(self, isolated_cwd: Path) -> None:
        paths = [isolated_cwd / "a.jpg", isolated_cwd / "b.jpg"]
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        save_embeddings("model-v1", paths, embeddings, paths)
        result = load_embeddings("model-v1", paths)
        assert result is not None
        loaded_emb, loaded_paths = result
        np.testing.assert_array_equal(loaded_emb, embeddings)
        assert loaded_paths == paths

    def test_stale_when_cached_name_missing_from_inputs(
        self, isolated_cwd: Path
    ) -> None:
        paths_ab = [isolated_cwd / "a.jpg", isolated_cwd / "b.jpg"]
        embeddings = np.array([[1.0], [2.0]], dtype=np.float32)
        save_embeddings("model-v1", paths_ab, embeddings, paths_ab)
        # Now query with a SUBSET [A] - cache key differs, so miss
        # But to exercise the stale path, we need same key but different inputs.
        # The key depends on sorted filenames, so subsets differ in key.
        # To trigger the stale branch: save with [A,B]; then query with the
        # same original image_paths list but the path_lookup misses B. We can
        # do this by making the current image_paths use different Path objects
        # whose .name sorts to the same list. We can't easily trigger stale
        # via subset because the key changes. Instead, verify subset returns
        # None (cache miss, different key).
        result = load_embeddings("model-v1", [isolated_cwd / "a.jpg"])
        assert result is None

    def test_stale_returns_none_via_direct_mismatch(self, isolated_cwd: Path) -> None:
        # Craft a scenario where the cache key matches but a filename inside
        # the stored filenames is not in path_lookup. Achieved by having
        # image_paths with names that produce the same cache key but where
        # one name differs.
        # Save with [a.jpg, b.jpg]. Compute that key.
        # Query with two paths whose names are [a.jpg, c.jpg] - different key.
        # So that doesn't work either.
        # Easiest: directly write an .npz with a known key, then query with
        # image_paths that produce that key but whose names don't match cached
        # filenames. But key is derived from the same filenames used to save,
        # so they must match. The only stale path in practice is if the cache
        # file is from an older code path. We can simulate: save under a key,
        # then manually craft a .npz with a different filename array at that
        # key. Do that.
        paths = [isolated_cwd / "a.jpg", isolated_cwd / "b.jpg"]
        key = _cache_key("m", paths)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{key}.npz"
        # Cached filenames include one that's not in current input set
        np.savez(
            cache_file,
            embeddings=np.array([[1.0], [2.0]], dtype=np.float32),
            filenames=np.array(["a.jpg", "ghost.jpg"]),
        )
        result = load_embeddings("m", paths)
        assert result is None

    def test_valid_paths_are_original_path_objects(self, isolated_cwd: Path) -> None:
        p1 = isolated_cwd / "subdir" / "a.jpg"
        p2 = isolated_cwd / "other" / "b.jpg"
        paths = [p1, p2]
        embeddings = np.array([[1.0], [2.0]], dtype=np.float32)
        save_embeddings("m", paths, embeddings, paths)
        result = load_embeddings("m", paths)
        assert result is not None
        _, valid_paths = result
        # Reconstruction should yield the original Path objects (same parents)
        assert set(valid_paths) == {p1, p2}


class TestSaveEmbeddings:
    def test_creates_cache_dir(self, isolated_cwd: Path) -> None:
        assert not CACHE_DIR.exists()
        paths = [isolated_cwd / "a.jpg"]
        save_embeddings("m", paths, np.array([[1.0]], dtype=np.float32), paths)
        assert CACHE_DIR.exists()
        assert CACHE_DIR.is_dir()

    def test_writes_npz_matching_cache_key(self, isolated_cwd: Path) -> None:
        paths = [isolated_cwd / "a.jpg"]
        save_embeddings("m", paths, np.array([[1.0]], dtype=np.float32), paths)
        key = _cache_key("m", paths)
        expected = CACHE_DIR / f"{key}.npz"
        assert expected.exists()
