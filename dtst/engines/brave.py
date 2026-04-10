import logging
import os
import time

from dtst.engines.base import SearchEngine

logger = logging.getLogger(__name__)

BRAVE_IMAGES_URL = "https://api.search.brave.com/res/v1/images/search"
PER_PAGE = 100


class BraveSearchEngine(SearchEngine):
    def __init__(
        self,
        api_key: str | None = None,
        delay: float = 1.0,
        *,
        min_size: int = 1024,
        retries: int = 3,
        timeout: int | float = 30,
    ) -> None:
        super().__init__(min_size=min_size, retries=retries, timeout=timeout)
        self._api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self._delay = delay

    @property
    def name(self) -> str:
        return "brave"

    def search(self, query: str, page: int) -> list[dict]:
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
        r = self._session.get(
            BRAVE_IMAGES_URL,
            params=params,
            headers=headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        results_list = data.get("results") or []
        if not isinstance(results_list, list):
            return []
        results: list[dict] = []
        for result in results_list:
            if not isinstance(result, dict):
                continue
            props = result.get("properties") or {}
            url = props.get("url") if isinstance(props, dict) else None
            if not url:
                continue
            w: int | None = None
            h: int | None = None
            if isinstance(props, dict):
                pw = props.get("width")
                ph = props.get("height")
                if pw is not None and ph is not None:
                    try:
                        w = int(pw)
                        h = int(ph)
                        if max(w, h) < self.min_size:
                            continue
                    except (TypeError, ValueError):
                        pass
            results.append(
                self._make_result(
                    url=url,
                    query=query,
                    width=w,
                    height=h,
                    title=result.get("title"),
                    source_domain=result.get("source"),
                )
            )
        return results
