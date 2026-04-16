import responses

from dtst.engines.brave import BRAVE_IMAGES_URL, BraveSearchEngine
from dtst.engines.flickr import FLICKR_LICENSES, FLICKR_REST, FlickrEngine
from dtst.engines.inaturalist import INAT_API, INaturalistEngine
from dtst.engines.serper import SERPER_IMAGES_URL, SerperEngine
from dtst.engines.wikimedia import (
    COMMONS_API,
    WikimediaEngine,
    _normalize_license,
    _strip_html,
)


class TestBrave:
    def test_no_api_key_returns_empty_without_http(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        engine = BraveSearchEngine(api_key="", delay=0)
        # No responses registered: if HTTP were attempted, it would raise ConnectionError.
        assert engine.search("cats", page=1) == []

    @responses.activate
    def test_happy_path_returns_results(self):
        responses.add(
            responses.GET,
            BRAVE_IMAGES_URL,
            json={
                "results": [
                    {
                        "title": "A cat",
                        "source": "example.com",
                        "properties": {
                            "url": "https://example.com/cat.jpg",
                            "width": 2000,
                            "height": 1500,
                        },
                    },
                    {
                        "title": "Tiny",
                        "source": "small.com",
                        "properties": {
                            "url": "https://small.com/tiny.jpg",
                            "width": 100,
                            "height": 100,
                        },
                    },
                ]
            },
            status=200,
        )
        engine = BraveSearchEngine(api_key="fake", delay=0, min_size=1024)
        results = engine.search("cats", page=1)
        assert len(results) == 1
        r = results[0]
        assert r["url"] == "https://example.com/cat.jpg"
        assert r["engine"] == "brave"
        assert r["query"] == "cats"
        assert r["width"] == 2000
        assert r["height"] == 1500
        assert r["title"] == "A cat"
        assert r["source_domain"] == "example.com"

    @responses.activate
    def test_empty_response_returns_empty_list(self):
        responses.add(
            responses.GET,
            BRAVE_IMAGES_URL,
            json={"results": []},
            status=200,
        )
        engine = BraveSearchEngine(api_key="fake", delay=0)
        assert engine.search("nothing", page=1) == []

    @responses.activate
    def test_malformed_entries_are_skipped(self):
        responses.add(
            responses.GET,
            BRAVE_IMAGES_URL,
            json={
                "results": [
                    "not a dict",
                    {"properties": {}},  # no url
                    {"properties": {"url": "https://ok.com/a.jpg"}},
                ]
            },
            status=200,
        )
        engine = BraveSearchEngine(api_key="fake", delay=0)
        results = engine.search("q", page=1)
        assert len(results) == 1
        assert results[0]["url"] == "https://ok.com/a.jpg"


class TestWikimedia:
    @responses.activate
    def test_happy_path_returns_results(self):
        responses.add(
            responses.GET,
            COMMONS_API,
            json={
                "query": {
                    "pages": {
                        "123": {
                            "title": "File:Cat.jpg",
                            "imageinfo": [
                                {
                                    "url": "https://upload.wikimedia.org/cat.jpg",
                                    "mime": "image/jpeg",
                                    "width": 2000,
                                    "height": 1500,
                                    "extmetadata": {
                                        "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                        "Artist": {"value": "<a href='#'>Jane Doe</a>"},
                                        "DateTimeOriginal": {
                                            "value": "<span>2020-01-01</span>"
                                        },
                                    },
                                }
                            ],
                        }
                    }
                }
            },
            status=200,
        )
        engine = WikimediaEngine(user_agent="test-agent", delay=0, min_size=1024)
        results = engine.search("cat", page=1)
        assert len(results) == 1
        r = results[0]
        assert r["url"] == "https://upload.wikimedia.org/cat.jpg"
        assert r["engine"] == "wikimedia"
        assert r["width"] == 2000
        assert r["height"] == 1500
        assert r["license"] == "cc-by-sa-4.0"
        assert r["source_domain"] == "commons.wikimedia.org"
        assert r["title"] == "Cat.jpg"
        assert r["author"] == "Jane Doe"
        assert r["date"] == "2020-01-01"

    @responses.activate
    def test_non_image_mime_skipped(self):
        responses.add(
            responses.GET,
            COMMONS_API,
            json={
                "query": {
                    "pages": {
                        "1": {
                            "title": "File:Some.pdf",
                            "imageinfo": [
                                {
                                    "url": "https://u.w/a.pdf",
                                    "mime": "application/pdf",
                                    "width": 2000,
                                    "height": 2000,
                                }
                            ],
                        },
                        "2": {
                            "title": "File:Small.jpg",
                            "imageinfo": [
                                {
                                    "url": "https://u.w/s.jpg",
                                    "mime": "image/jpeg",
                                    "width": 100,
                                    "height": 100,
                                }
                            ],
                        },
                    }
                }
            },
            status=200,
        )
        engine = WikimediaEngine(user_agent="ua", delay=0, min_size=1024)
        assert engine.search("q", page=1) == []


class TestWikimediaHelpers:
    def test_strip_html_removes_tags(self):
        assert _strip_html("<a href='x'>hi</a>") == "hi"

    def test_strip_html_whitespace(self):
        assert _strip_html("  plain  ") == "plain"

    def test_strip_html_nested(self):
        assert _strip_html("<span><b>bold</b> text</span>") == "bold text"

    def test_normalize_license_spaces_to_dashes(self):
        assert _normalize_license("CC BY-SA 4.0") == "cc-by-sa-4.0"

    def test_normalize_license_strips_and_lowers(self):
        assert _normalize_license("  Public Domain  ") == "public-domain"


class TestInaturalist:
    def test_non_integer_query_returns_empty(self):
        engine = INaturalistEngine(delay=0)
        assert engine.search("not-an-int", page=1) == []

    @responses.activate
    def test_happy_path_rewrites_photo_size(self):
        responses.add(
            responses.GET,
            INAT_API,
            json={
                "results": [
                    {
                        "quality_grade": "research",
                        "observed_on": "2023-06-01",
                        "taxon": {"preferred_common_name": "Chanterelle"},
                        "photos": [
                            {
                                "url": "https://static.inaturalist.org/photos/123/medium.jpg",
                                "license_code": "cc-by",
                                "attribution": "(c) someone",
                            }
                        ],
                    }
                ]
            },
            status=200,
        )
        engine = INaturalistEngine(delay=0)
        results = engine.search("12345", page=1)
        assert len(results) == 1
        r = results[0]
        assert r["url"] == "https://static.inaturalist.org/photos/123/original.jpg"
        assert r["engine"] == "inaturalist"
        assert r["license"] == "cc-by"
        assert r["source_domain"] == "inaturalist.org"
        assert r["title"] == "Chanterelle"
        assert r["author"] == "(c) someone"
        assert r["date"] == "2023-06-01"
        assert r["quality_grade"] == "research"

    @responses.activate
    def test_empty_results_returns_empty(self):
        responses.add(
            responses.GET,
            INAT_API,
            json={"results": []},
            status=200,
        )
        engine = INaturalistEngine(delay=0)
        assert engine.search("42", page=1) == []


class TestFlickr:
    def test_no_api_key_returns_empty_without_http(self, monkeypatch):
        monkeypatch.delenv("FLICKR_API_KEY", raising=False)
        engine = FlickrEngine(api_key="")
        assert engine.search("cats", page=1) == []

    def test_flickr_licenses_constant_lookup(self):
        assert FLICKR_LICENSES[0] == "all-rights-reserved"
        assert FLICKR_LICENSES[4] == "cc-by-2.0"
        assert FLICKR_LICENSES[9] == "cc0-1.0"
        assert FLICKR_LICENSES[10] == "public-domain-mark"
        assert 99 not in FLICKR_LICENSES

    @responses.activate
    def test_happy_path_returns_results(self):
        responses.add(
            responses.GET,
            FLICKR_REST,
            json={
                "stat": "ok",
                "photos": {
                    "photo": [
                        {
                            "title": "Cat photo",
                            "url_o": "https://live.staticflickr.com/orig.jpg",
                            "width_o": 3000,
                            "height_o": 2000,
                            "license": "4",
                            "ownername": "Owner Name",
                            "datetaken": "2022-05-01 12:00:00",
                        },
                        {
                            "title": "Small photo",
                            "url_o": "https://live.staticflickr.com/small.jpg",
                            "width_o": 100,
                            "height_o": 100,
                            "license": "4",
                        },
                    ]
                },
            },
            status=200,
        )
        engine = FlickrEngine(api_key="fake", min_size=1024)
        results = engine.search("cats", page=1)
        assert len(results) == 1
        r = results[0]
        assert r["url"] == "https://live.staticflickr.com/orig.jpg"
        assert r["engine"] == "flickr"
        assert r["width"] == 3000
        assert r["height"] == 2000
        assert r["license"] == "cc-by-2.0"
        assert r["source_domain"] == "flickr.com"
        assert r["title"] == "Cat photo"
        assert r["author"] == "Owner Name"
        assert r["date"] == "2022-05-01 12:00:00"

    @responses.activate
    def test_flickr_api_error_status_returns_empty(self):
        responses.add(
            responses.GET,
            FLICKR_REST,
            json={"stat": "fail", "message": "Invalid API Key"},
            status=200,
        )
        engine = FlickrEngine(api_key="fake")
        assert engine.search("q", page=1) == []


class TestSerper:
    def test_no_api_key_returns_empty_without_http(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        engine = SerperEngine(api_key="")
        assert engine.search("cats", page=1) == []

    @responses.activate
    def test_happy_path_returns_results(self):
        responses.add(
            responses.POST,
            SERPER_IMAGES_URL,
            json={
                "images": [
                    {
                        "title": "A cat",
                        "imageUrl": "https://example.com/cat.jpg",
                        "imageWidth": 2000,
                        "imageHeight": 1500,
                        "link": "https://host.example.com/page",
                    },
                    {
                        "title": "Tiny",
                        "imageUrl": "https://example.com/tiny.jpg",
                        "imageWidth": 50,
                        "imageHeight": 50,
                    },
                ]
            },
            status=200,
        )
        engine = SerperEngine(api_key="fake", min_size=1024)
        results = engine.search("cats", page=1)
        assert len(results) == 1
        r = results[0]
        assert r["url"] == "https://example.com/cat.jpg"
        assert r["engine"] == "serper"
        assert r["width"] == 2000
        assert r["height"] == 1500
        assert r["title"] == "A cat"
        assert r["source_domain"] == "host.example.com"

    @responses.activate
    def test_missing_images_key_returns_empty(self):
        responses.add(
            responses.POST,
            SERPER_IMAGES_URL,
            json={},
            status=200,
        )
        engine = SerperEngine(api_key="fake")
        assert engine.search("q", page=1) == []
