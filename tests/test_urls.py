from urllib.parse import parse_qs, urlparse

import pytest

from dtst.urls import canonicalize_image_url, clean_image_url


class TestCleanImageUrl:
    def test_no_query_returns_verbatim(self):
        url = "https://example.com/image.jpg"
        assert clean_image_url(url) == url

    def test_no_query_with_path_only(self):
        url = "https://example.com/path/to/image.png"
        assert clean_image_url(url) == url

    def test_only_resize_params_stripped(self):
        url = "https://example.com/image.jpg?w=100&h=200&quality=80"
        result = clean_image_url(url)
        parsed = urlparse(result)
        assert parsed.query == ""
        assert parsed.scheme == "https"
        assert parsed.netloc == "example.com"
        assert parsed.path == "/image.jpg"

    def test_mixed_params_keeps_non_resize(self):
        url = "https://example.com/image.jpg?id=abc123&w=500&token=xyz&quality=90"
        result = clean_image_url(url)
        params = parse_qs(urlparse(result).query)
        assert params == {"id": ["abc123"], "token": ["xyz"]}

    @pytest.mark.parametrize(
        "domain",
        ["media.gettyimages.com", "lookaside.fbsbx.com", "media.licdn.com"],
    )
    def test_keep_all_domains_returned_verbatim(self, domain):
        url = f"https://{domain}/image.jpg?w=100&h=200&quality=80"
        assert clean_image_url(url) == url

    def test_only_non_resize_params_returned_verbatim(self):
        url = "https://example.com/image.jpg?id=abc123&token=xyz"
        assert clean_image_url(url) == url

    @pytest.mark.parametrize(
        "resize_param",
        ["w", "h", "quality", "format", "auto", "crop"],
    )
    def test_individual_resize_params_stripped(self, resize_param):
        url = f"https://example.com/image.jpg?id=keep&{resize_param}=somevalue"
        result = clean_image_url(url)
        params = parse_qs(urlparse(result).query)
        assert params == {"id": ["keep"]}
        assert resize_param not in params

    @pytest.mark.parametrize(
        "resize_param,value",
        [
            ("w", "800"),
            ("h", "600"),
            ("width", "800"),
            ("height", "600"),
            ("q", "90"),
            ("quality", "85"),
            ("format", "jpeg"),
            ("fm", "webp"),
            ("auto", "compress"),
            ("webp", "1"),
            ("crop", "entropy"),
            ("strip", "all"),
            ("fit", "crop"),
            ("dpr", "2"),
            ("resize", "fill"),
        ],
    )
    def test_each_resize_param_alone_strips_query(self, resize_param, value):
        url = f"https://example.com/img.jpg?{resize_param}={value}"
        result = clean_image_url(url)
        assert urlparse(result).query == ""

    def test_fragment_preserved_when_stripped(self):
        url = "https://example.com/image.jpg?w=100&id=abc#anchor"
        result = clean_image_url(url)
        parsed = urlparse(result)
        assert parsed.fragment == "anchor"
        assert parse_qs(parsed.query) == {"id": ["abc"]}

    def test_fragment_preserved_when_nothing_changes(self):
        url = "https://example.com/image.jpg?id=abc#anchor"
        assert clean_image_url(url) == url

    def test_path_and_scheme_preserved(self):
        url = "http://example.com/deep/nested/path/image.jpg?w=100&id=abc"
        result = clean_image_url(url)
        parsed = urlparse(result)
        assert parsed.scheme == "http"
        assert parsed.netloc == "example.com"
        assert parsed.path == "/deep/nested/path/image.jpg"

    def test_https_scheme_preserved(self):
        url = "https://example.com/image.jpg?w=100&id=abc"
        result = clean_image_url(url)
        assert urlparse(result).scheme == "https"

    def test_multi_value_non_resize_params_preserved(self):
        url = "https://example.com/image.jpg?tag=a&tag=b&w=100"
        result = clean_image_url(url)
        params = parse_qs(urlparse(result).query)
        assert params == {"tag": ["a", "b"]}


class TestCanonicalizeImageUrl:
    def test_no_query_returns_verbatim(self):
        url = "https://example.com/image.jpg"
        assert canonicalize_image_url(url) == url

    def test_inner_https_url_returned_decoded(self):
        inner = "https://cdn.example.com/real/image.jpg"
        url = "https://proxy.example.com/fetch?url=https%3A%2F%2Fcdn.example.com%2Freal%2Fimage.jpg"
        assert canonicalize_image_url(url) == inner

    def test_inner_http_url_returned_decoded(self):
        inner = "http://cdn.example.com/image.png"
        url = "https://proxy.example.com/fetch?url=http%3A%2F%2Fcdn.example.com%2Fimage.png"
        assert canonicalize_image_url(url) == inner

    def test_non_http_scheme_returns_original(self):
        url = "https://proxy.example.com/fetch?url=javascript%3Aalert(1)"
        assert canonicalize_image_url(url) == url

    def test_ftp_scheme_returns_original(self):
        url = "https://proxy.example.com/fetch?url=ftp%3A%2F%2Fexample.com%2Ffile"
        assert canonicalize_image_url(url) == url

    def test_multiple_url_params_returns_original(self):
        url = "https://proxy.example.com/fetch?url=https%3A%2F%2Fa.com%2Fx.jpg&url=https%3A%2F%2Fb.com%2Fy.jpg"
        assert canonicalize_image_url(url) == url

    def test_query_without_url_key_returns_original(self):
        url = "https://proxy.example.com/fetch?id=abc&format=jpeg"
        assert canonicalize_image_url(url) == url

    def test_whitespace_stripped_from_inner(self):
        # Encode inner URL with leading/trailing whitespace (via %20)
        url = "https://proxy.example.com/fetch?url=%20%20https%3A%2F%2Fcdn.example.com%2Fimage.jpg%20%20"
        assert canonicalize_image_url(url) == "https://cdn.example.com/image.jpg"

    def test_empty_url_param_returns_original(self):
        # parse_qs with keep_blank_values=False drops empty; "url" won't be present
        url = "https://proxy.example.com/fetch?url="
        assert canonicalize_image_url(url) == url

    def test_plain_non_encoded_inner_url(self):
        # Even without encoding, parse_qs should split on =
        inner = "https://cdn.example.com/image.jpg"
        url = f"https://proxy.example.com/fetch?url={inner}"
        # parse_qs will take everything up to next & - since there's no &, full inner
        assert canonicalize_image_url(url) == inner
