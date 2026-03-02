import logging
import os

import requests

from dtst.engines.base import SearchEngine

logger = logging.getLogger(__name__)

BING_IMAGES_URL = "https://api.bing.microsoft.com/v7.0/images/search"
MIN_SIZE = 1024


class BingEngine(SearchEngine):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("BING_SEARCH_API_KEY", "")

    @property
    def name(self) -> str:
        return "bing"

    def search(self, query: str, page: int) -> list[str]:
        if not self._api_key:
            logger.warning("BING_SEARCH_API_KEY not set; skipping Bing")
            return []
        count = 35
        offset = (page - 1) * count
        params = {
            "q": query,
            "count": count,
            "offset": offset,
            "imageType": "Photo",
            "size": "Large",
        }
        headers = {"Ocp-Apim-Subscription-Key": self._api_key}
        try:
            r = requests.get(
                BING_IMAGES_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("Bing request failed for %r page %s: %s", query, page, e)
            return []
        values = data.get("value") or []
        if not isinstance(values, list):
            return []
        urls: list[str] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            url = item.get("contentUrl")
            if not url:
                continue
            w = item.get("width")
            h = item.get("height")
            if w is not None and h is not None:
                try:
                    if max(int(w), int(h)) < MIN_SIZE:
                        continue
                except (TypeError, ValueError):
                    pass
            urls.append(url)
        return urls
