from pathlib import Path

from dtst.core.dedup import _dedup_sort_key, _UnionFind


class TestUnionFind:
    def test_initial_find_returns_self(self):
        uf = _UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_groups_initial_are_singletons(self):
        uf = _UnionFind(4)
        groups = uf.groups()
        assert len(groups) == 4
        for root, members in groups.items():
            assert members == [root]

    def test_union_merges_two_elements(self):
        uf = _UnionFind(3)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) != uf.find(0)

    def test_union_is_transitive(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)
        assert uf.find(0) == uf.find(1) == uf.find(2)
        assert uf.find(3) == uf.find(4)
        assert uf.find(0) != uf.find(3)

    def test_union_self_is_noop(self):
        uf = _UnionFind(3)
        uf.union(1, 1)
        assert uf.find(1) == 1

    def test_union_already_connected_is_noop(self):
        uf = _UnionFind(3)
        uf.union(0, 1)
        uf.union(0, 1)
        uf.union(1, 0)
        assert uf.find(0) == uf.find(1)

    def test_groups_after_unions(self):
        uf = _UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(3, 4)
        groups = uf.groups()
        # Expect 3 groups: {0,1}, {2,3,4}, {5}
        group_sets = sorted(tuple(sorted(members)) for members in groups.values())
        assert group_sets == [(0, 1), (2, 3, 4), (5,)]

    def test_groups_root_is_key(self):
        uf = _UnionFind(3)
        uf.union(0, 1)
        uf.union(1, 2)
        groups = uf.groups()
        # All entries should be under a single root, and that root should equal find(i)
        assert len(groups) == 1
        root = next(iter(groups.keys()))
        assert uf.find(0) == root
        assert sorted(groups[root]) == [0, 1, 2]

    def test_path_compression_flattens_chain(self):
        uf = _UnionFind(5)
        # Force a chain by unioning in order; with union-by-rank, roots shift,
        # so we manipulate the internal parents directly to construct a chain.
        uf._parent = [1, 2, 3, 4, 4]
        uf._rank = [0, 0, 0, 0, 0]
        root = uf.find(0)
        assert root == 4
        # After path compression, parents along the path should point closer to root
        # The loop compresses by pointing each node to its grandparent.
        assert uf._parent[0] != 1 or uf._parent[0] == 4
        # Re-finding should be cheap and still correct
        assert uf.find(0) == 4
        assert uf.find(1) == 4
        assert uf.find(2) == 4

    def test_path_compression_idempotent(self):
        uf = _UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(0, 2)
        r = uf.find(0)
        # After one find, all should agree, and a second call returns same root.
        assert uf.find(0) == r
        assert uf.find(1) == r
        assert uf.find(2) == r
        assert uf.find(3) == r

    def test_union_by_rank_equal_ranks_increments(self):
        uf = _UnionFind(2)
        # Both start with rank 0. After union, the surviving root's rank is 1.
        uf.union(0, 1)
        root = uf.find(0)
        assert uf._rank[root] == 1

    def test_union_by_rank_attaches_smaller_under_larger(self):
        uf = _UnionFind(4)
        # Build a tree of rank 1 rooted (effectively) at one element
        uf.union(0, 1)  # rank of root becomes 1
        root_ab = uf.find(0)
        # A lone node (rank 0) merged with that tree should attach under root_ab
        # and rank of root_ab should stay at 1.
        uf.union(root_ab, 2)
        assert uf.find(2) == root_ab
        assert uf._rank[root_ab] == 1
        # Merging the third lone node should also not increase rank beyond 1
        uf.union(root_ab, 3)
        assert uf.find(3) == root_ab
        assert uf._rank[root_ab] == 1

    def test_union_by_rank_merges_equal_rank_trees(self):
        uf = _UnionFind(4)
        uf.union(0, 1)  # tree A: rank 1
        uf.union(2, 3)  # tree B: rank 1
        uf.union(0, 2)  # equal ranks -> merged root rank becomes 2
        root = uf.find(0)
        assert uf.find(1) == root
        assert uf.find(2) == root
        assert uf.find(3) == root
        assert uf._rank[root] == 2


class TestDedupSortKey:
    def _info(self, w, h, fsize):
        return (w, h, fsize)

    def test_resolution_dominates_fsize(self):
        a = Path("a.jpg")
        b = Path("b.jpg")
        image_info = {
            a: self._info(1000, 1000, 100),  # higher res, tiny file
            b: self._info(100, 100, 10_000_000),  # low res, huge file
        }
        sidecars: dict[Path, dict] = {}
        ka = _dedup_sort_key(a, image_info, sidecars, prefer_upscaled=False)
        kb = _dedup_sort_key(b, image_info, sidecars, prefer_upscaled=False)
        assert ka > kb

    def test_fsize_breaks_resolution_tie(self):
        a = Path("a.jpg")
        b = Path("b.jpg")
        image_info = {
            a: self._info(500, 500, 200_000),
            b: self._info(500, 500, 100_000),
        }
        sidecars: dict[Path, dict] = {}
        ka = _dedup_sort_key(a, image_info, sidecars, prefer_upscaled=False)
        kb = _dedup_sort_key(b, image_info, sidecars, prefer_upscaled=False)
        assert ka > kb

    def test_blur_breaks_resolution_and_fsize_tie(self):
        a = Path("a.jpg")
        b = Path("b.jpg")
        image_info = {
            a: self._info(500, 500, 100_000),
            b: self._info(500, 500, 100_000),
        }
        sidecars = {
            a: {"metrics": {"blur": 120.0}},
            b: {"metrics": {"blur": 50.0}},
        }
        ka = _dedup_sort_key(a, image_info, sidecars, prefer_upscaled=False)
        kb = _dedup_sort_key(b, image_info, sidecars, prefer_upscaled=False)
        assert ka > kb

    def test_missing_sidecar_uses_zero_blur(self):
        a = Path("a.jpg")
        image_info = {a: self._info(100, 100, 1000)}
        sidecars: dict[Path, dict] = {}
        key = _dedup_sort_key(a, image_info, sidecars, prefer_upscaled=False)
        assert key == (1, 100 * 100, 1000, 0.0)

    def test_missing_metrics_section_uses_zero_blur(self):
        a = Path("a.jpg")
        image_info = {a: self._info(100, 100, 1000)}
        sidecars = {a: {"upscale": False}}
        key = _dedup_sort_key(a, image_info, sidecars, prefer_upscaled=False)
        assert key == (1, 10_000, 1000, 0.0)

    def test_prefer_upscaled_false_prefers_non_upscaled(self):
        # Default behavior: prefer_upscaled=False means non-upscaled wins.
        original = Path("orig.jpg")
        upscaled = Path("up.jpg")
        image_info = {
            original: self._info(100, 100, 1000),
            upscaled: self._info(100, 100, 1000),
        }
        sidecars = {
            original: {"metrics": {"blur": 0.0}},
            upscaled: {"upscale": True, "metrics": {"blur": 0.0}},
        }
        k_orig = _dedup_sort_key(original, image_info, sidecars, prefer_upscaled=False)
        k_up = _dedup_sort_key(upscaled, image_info, sidecars, prefer_upscaled=False)
        assert k_orig > k_up
        assert k_orig[0] == 1
        assert k_up[0] == 0

    def test_prefer_upscaled_true_prefers_upscaled(self):
        original = Path("orig.jpg")
        upscaled = Path("up.jpg")
        image_info = {
            original: self._info(100, 100, 1000),
            upscaled: self._info(100, 100, 1000),
        }
        sidecars = {
            original: {"metrics": {"blur": 0.0}},
            upscaled: {"upscale": True, "metrics": {"blur": 0.0}},
        }
        k_orig = _dedup_sort_key(original, image_info, sidecars, prefer_upscaled=True)
        k_up = _dedup_sort_key(upscaled, image_info, sidecars, prefer_upscaled=True)
        assert k_up > k_orig
        assert k_up[0] == 1
        assert k_orig[0] == 0

    def test_preference_dominates_resolution(self):
        # Even a much higher resolution loses to the preferred category.
        preferred = Path("small_but_preferred.jpg")
        other = Path("big_not_preferred.jpg")
        image_info = {
            preferred: self._info(10, 10, 100),  # tiny
            other: self._info(4000, 4000, 10_000_000),  # huge
        }
        # prefer_upscaled=False -> non-upscaled is preferred
        sidecars = {
            preferred: {},  # no upscale -> is_upscaled=0 -> preferred
            other: {"upscale": True},
        }
        k_pref = _dedup_sort_key(preferred, image_info, sidecars, prefer_upscaled=False)
        k_other = _dedup_sort_key(other, image_info, sidecars, prefer_upscaled=False)
        assert k_pref > k_other

    def test_key_tuple_shape(self):
        a = Path("a.jpg")
        image_info = {a: self._info(200, 100, 5000)}
        sidecars = {a: {"upscale": True, "metrics": {"blur": 42.5}}}
        key = _dedup_sort_key(a, image_info, sidecars, prefer_upscaled=True)
        assert key == (1, 200 * 100, 5000, 42.5)

    def test_falsy_upscale_treated_as_not_upscaled(self):
        a = Path("a.jpg")
        image_info = {a: self._info(10, 10, 10)}
        # upscale present but falsy
        sidecars = {a: {"upscale": False, "metrics": {"blur": 1.0}}}
        key = _dedup_sort_key(a, image_info, sidecars, prefer_upscaled=False)
        # Not upscaled + prefer_upscaled=False -> preference = 1 - 0 = 1
        assert key[0] == 1

    def test_sorted_group_picks_correct_winner(self):
        # Integration-style: reproduce the sort() call from dedup().
        a = Path("a.jpg")
        b = Path("b.jpg")
        c = Path("c.jpg")
        image_info = {
            a: self._info(500, 500, 100_000),
            b: self._info(1000, 1000, 50_000),  # highest res
            c: self._info(500, 500, 200_000),
        }
        sidecars: dict[Path, dict] = {}
        group = [a, b, c]
        group.sort(
            key=lambda p: _dedup_sort_key(p, image_info, sidecars, False),
            reverse=True,
        )
        assert group[0] == b  # highest resolution wins
