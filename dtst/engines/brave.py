import logging
import os
import time

import requests

from dtst.engines.base import SearchEngine

logger = logging.getLogger(__name__)

BRAVE_IMAGES_URL = "https://api.search.brave.com/res/v1/images/search"
MIN_SIZE = 1024
PER_PAGE = 100


class BraveSearchEngine(SearchEngine):
    def __init__(
        self,
        api_key: str | None = None,
        delay: float = 1.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self._delay = delay

    @property
    def name(self) -> str:
        return "brave"

    def search(self, query: str, page: int) -> list[str]:
        if not self._api_key:
            logger.warning("BRAVE_API_KEY not set; skipping Brave")
            return []
        time.sleep(self._delay)
        params = {
            "q": query,
            "count": PER_PAGE,
            "offset": (page - 1) * PER_PAGE,
            "country": "ALL",
            "search_lang": "en",
            "safesearch": "off",
        }
        headers = {"X-Subscription-Token": self._api_key}
        try:
            r = requests.get(
                BRAVE_IMAGES_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("Brave request failed for %r page %s: %s", query, page, e)
            return []
        results = data.get("results") or []
        if not isinstance(results, list):
            return []
        urls: list[str] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            props = result.get("properties") or {}
            url = props.get("url") if isinstance(props, dict) else None
            if not url:
                continue
            w = props.get("width")
            h = props.get("height")
            if w is not None and h is not None:
                try:
                    if max(int(w), int(h)) < MIN_SIZE:
                        continue
                except (TypeError, ValueError):
                    pass
            urls.append(url)
        return urls
