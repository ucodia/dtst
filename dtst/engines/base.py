from abc import ABC, abstractmethod
from datetime import datetime, timezone

import requests
from urllib3.util.retry import Retry


class SearchEngine(ABC):
    def __init__(
        self,
        min_size: int = 1024,
        retries: int = 3,
        timeout: int | float = 30,
    ) -> None:
        self.min_size = min_size
        self._timeout = timeout
        retry = Retry(
            total=retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def search(self, query: str, page: int) -> list[dict]:
        pass

    def _make_result(
        self,
        *,
        url: str,
        query: str,
        width: int | None = None,
        height: int | None = None,
        license: str | None = None,
        source_domain: str | None = None,
        title: str | None = None,
        author: str | None = None,
        date: str | None = None,
    ) -> dict:
        return {
            "url": url,
            "engine": self.name,
            "query": query,
            "width": width,
            "height": height,
            "license": license,
            "source_domain": source_domain,
            "title": title,
            "author": author,
            "date": date,
            "found_at": datetime.now(timezone.utc).isoformat(),
        }
