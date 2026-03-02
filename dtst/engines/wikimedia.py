import logging
import os
import time
import requests

from dtst.engines.base import SearchEngine

logger = logging.getLogger(__name__)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
MIN_SIZE = 1024


class WikimediaEngine(SearchEngine):
    def __init__(
        self,
        user_agent: str | None = None,
        delay: float = 0.2,
    ) -> None:
        self._user_agent = user_agent or os.environ.get(
            "WIKIMEDIA_USER_AGENT",
            "dtst/1.0 (https://github.com/dtst)",
        )
        self._delay = delay

    @property
    def name(self) -> str:
        return "wikimedia"

    def search(self, query: str, page: int) -> list[str]:
        time.sleep(self._delay)
        gsrlimit = 50
        gsroffset = (page - 1) * gsrlimit
        params = {
            "action": "query",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": gsrlimit,
            "gsroffset": gsroffset,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "format": "json",
        }
        headers = {"User-Agent": self._user_agent}
        try:
            r = requests.get(
                COMMONS_API,
                params=params,
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(
                "Wikimedia request failed for %r page %s: %s", query, page, e
            )
            return []
        urls: list[str] = []
        pages = data.get("query", {}).get("pages") or {}
        if not isinstance(pages, dict):
            return []
        for _pid, p in pages.items():
            if not isinstance(p, dict):
                continue
            info_list = p.get("imageinfo")
            if not info_list or not isinstance(info_list, list):
                continue
            for info in info_list:
                if not isinstance(info, dict):
                    continue
                mime = info.get("mime") or ""
                if not str(mime).lower().startswith("image/"):
                    continue
                url = info.get("url")
                if not url:
                    continue
                w = info.get("width")
                h = info.get("height")
                if w is not None and h is not None:
                    try:
                        if max(int(w), int(h)) < MIN_SIZE:
                            continue
                    except (TypeError, ValueError):
                        pass
                urls.append(url)
        return urls
