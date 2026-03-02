import logging
import os

import requests

from dtst.engines.base import SearchEngine

logger = logging.getLogger(__name__)

FLICKR_REST = "https://api.flickr.com/services/rest/"
MIN_SIZE = 1024


class FlickrEngine(SearchEngine):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("FLICKR_API_KEY", "")

    @property
    def name(self) -> str:
        return "flickr"

    def search(self, query: str, page: int) -> list[str]:
        if not self._api_key:
            logger.warning("FLICKR_API_KEY not set; skipping Flickr")
            return []
        extras = "url_o,url_k,url_h,url_m"
        params = {
            "method": "flickr.photos.search",
            "api_key": self._api_key,
            "text": query,
            "extras": extras,
            "per_page": 100,
            "page": page,
            "format": "json",
            "nojsoncallback": 1,
        }
        try:
            r = requests.get(FLICKR_REST, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("Flickr request failed for %r page %s: %s", query, page, e)
            return []
        if data.get("stat") != "ok":
            logger.warning("Flickr API error for %r: %s", query, data.get("message", data))
            return []
        photos = data.get("photos", {}).get("photo") or []
        urls: list[str] = []
        for p in photos:
            if not isinstance(p, dict):
                continue
            url = (
                p.get("url_o")
                or p.get("url_k")
                or p.get("url_h")
                or p.get("url_m")
            )
            if not url:
                continue
            w = p.get("width_o") or p.get("width_k") or p.get("width_h")
            h = p.get("height_o") or p.get("height_k") or p.get("height_h")
            if w is not None and h is not None:
                try:
                    if max(int(w), int(h)) < MIN_SIZE:
                        continue
                except (TypeError, ValueError):
                    pass
            urls.append(url)
        return urls
