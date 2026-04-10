from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse


RESIZE_PARAMS = frozenset(
    {
        "w",
        "h",
        "width",
        "height",
        "q",
        "quality",
        "format",
        "fm",
        "auto",
        "webp",
        "crop",
        "strip",
        "fit",
        "dpr",
        "resize",
        "lossy",
        "vtcrop",
        "cs",
        "disable",
        "smart",
        "get_thumbnail",
    }
)

KEEP_ALL_DOMAINS = frozenset(
    {
        "media.gettyimages.com",
        "lookaside.fbsbx.com",
        "media.licdn.com",
    }
)


def clean_image_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.query:
        return url

    domain = parsed.hostname or ""
    if domain in KEEP_ALL_DOMAINS:
        return url

    params = parse_qs(parsed.query, keep_blank_values=True)
    kept = {k: v for k, v in params.items() if k not in RESIZE_PARAMS}

    if len(kept) == len(params):
        return url

    cleaned = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode([(k, v) for k, vals in kept.items() for v in vals]),
            parsed.fragment,
        )
    )
    return cleaned


def canonicalize_image_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.query:
        return url
    params = parse_qs(parsed.query, keep_blank_values=False)
    inner = params.get("url")
    if not inner or len(inner) != 1:
        return url
    raw = unquote(inner[0]).strip()
    if not raw.startswith("http://") and not raw.startswith("https://"):
        return url
    return raw
