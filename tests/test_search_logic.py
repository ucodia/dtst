from dtst.core.search import _build_query_matrix, _dedup_results


class TestDedupResults:
    def test_empty_input_returns_empty_list(self):
        assert _dedup_results([]) == []

    def test_returns_list_not_dict(self):
        result = _dedup_results([{"url": "https://example.com/a.jpg"}])
        assert isinstance(result, list)

    def test_entries_without_url_are_dropped(self):
        results = [
            {"title": "no url here"},
            {"url": "https://example.com/a.jpg"},
            {"url": None, "title": "null url"},
            {"url": "", "title": "empty url"},
        ]
        deduped = _dedup_results(results)
        assert len(deduped) == 1
        assert deduped[0]["url"] == "https://example.com/a.jpg"

    def test_same_url_keeps_entry_with_more_non_null_fields(self):
        results = [
            {"url": "https://example.com/a.jpg", "title": None, "author": None},
            {"url": "https://example.com/a.jpg", "title": "Foo", "author": "Bar"},
        ]
        deduped = _dedup_results(results)
        assert len(deduped) == 1
        assert deduped[0]["title"] == "Foo"
        assert deduped[0]["author"] == "Bar"

    def test_same_url_richer_first_then_sparser_keeps_richer(self):
        results = [
            {"url": "https://example.com/a.jpg", "title": "Foo", "author": "Bar"},
            {"url": "https://example.com/a.jpg", "title": None, "author": None},
        ]
        deduped = _dedup_results(results)
        assert len(deduped) == 1
        assert deduped[0]["title"] == "Foo"
        assert deduped[0]["author"] == "Bar"

    def test_tie_keeps_the_first_entry(self):
        results = [
            {"url": "https://example.com/a.jpg", "title": "First"},
            {"url": "https://example.com/a.jpg", "title": "Second"},
        ]
        deduped = _dedup_results(results)
        assert len(deduped) == 1
        assert deduped[0]["title"] == "First"

    def test_distinct_urls_all_kept(self):
        results = [
            {"url": "https://example.com/a.jpg"},
            {"url": "https://example.com/b.jpg"},
            {"url": "https://example.com/c.jpg"},
        ]
        deduped = _dedup_results(results)
        urls = {r["url"] for r in deduped}
        assert urls == {
            "https://example.com/a.jpg",
            "https://example.com/b.jpg",
            "https://example.com/c.jpg",
        }

    def test_mixed_duplicates_and_missing_urls(self):
        results = [
            {"title": "no url"},
            {"url": "https://example.com/a.jpg", "title": "A1", "author": None},
            {"url": "https://example.com/b.jpg", "title": "B"},
            {"url": "https://example.com/a.jpg", "title": "A2", "author": "me"},
            {"url": None},
        ]
        deduped = _dedup_results(results)
        assert len(deduped) == 2
        by_url = {r["url"]: r for r in deduped}
        assert by_url["https://example.com/a.jpg"]["title"] == "A2"
        assert by_url["https://example.com/a.jpg"]["author"] == "me"
        assert by_url["https://example.com/b.jpg"]["title"] == "B"


class TestBuildQueryMatrix:
    def test_suffix_only_false_includes_bare_terms_and_combos(self):
        result = _build_query_matrix(
            ["cat", "dog"], ["photo", "portrait"], suffix_only=False
        )
        assert result == [
            "cat",
            "dog",
            "cat photo",
            "cat portrait",
            "dog photo",
            "dog portrait",
        ]

    def test_suffix_only_true_excludes_bare_terms(self):
        result = _build_query_matrix(
            ["cat", "dog"], ["photo", "portrait"], suffix_only=True
        )
        assert result == [
            "cat photo",
            "cat portrait",
            "dog photo",
            "dog portrait",
        ]

    def test_empty_suffix_in_list_is_skipped(self):
        result = _build_query_matrix(
            ["cat"], ["photo", "", "portrait"], suffix_only=False
        )
        assert result == ["cat", "cat photo", "cat portrait"]

    def test_empty_suffix_skipped_with_suffix_only(self):
        result = _build_query_matrix(["cat"], ["", "photo"], suffix_only=True)
        assert result == ["cat photo"]

    def test_empty_terms_with_suffix_only_false_returns_empty(self):
        result = _build_query_matrix([], ["photo", "portrait"], suffix_only=False)
        assert result == []

    def test_empty_terms_with_suffix_only_true_returns_empty(self):
        result = _build_query_matrix([], ["photo"], suffix_only=True)
        assert result == []

    def test_empty_suffixes_with_suffix_only_false_returns_bare_terms(self):
        result = _build_query_matrix(["cat", "dog"], [], suffix_only=False)
        assert result == ["cat", "dog"]

    def test_empty_suffixes_with_suffix_only_true_returns_empty(self):
        result = _build_query_matrix(["cat", "dog"], [], suffix_only=True)
        assert result == []

    def test_order_bare_terms_first_then_combinations(self):
        result = _build_query_matrix(["a", "b"], ["x", "y"], suffix_only=False)
        assert result[:2] == ["a", "b"]
        assert result[2:] == ["a x", "a y", "b x", "b y"]

    def test_default_suffix_only_is_false(self):
        result = _build_query_matrix(["cat"], ["photo"])
        assert result == ["cat", "cat photo"]

    def test_both_empty_returns_empty(self):
        assert _build_query_matrix([], [], suffix_only=False) == []
        assert _build_query_matrix([], [], suffix_only=True) == []
