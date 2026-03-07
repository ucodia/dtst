from urllib.parse import parse_qs, unquote, urlparse


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
