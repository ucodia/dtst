import logging
import re
import time

from dtst.engines.base import SearchEngine
from dtst.user_agent import get_user_agent

logger = logging.getLogger(__name__)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _normalize_license(raw: str) -> str:
    return raw.strip().lower().replace(" ", "-")


class WikimediaEngine(SearchEngine):
    def __init__(
        self,
        user_agent: str | None = None,
        delay: float = 0.2,
        *,
        min_size: int = 1024,
        retries: int = 3,
        timeout: int | float = 30,
    ) -> None:
        super().__init__(min_size=min_size, retries=retries, timeout=timeout)
        self._user_agent = user_agent or get_user_agent()
        self._delay = delay

    @property
    def name(self) -> str:
        return "wikimedia"

    def search(self, query: str, page: int) -> list[dict]:
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
            "iiprop": "url|size|mime|extmetadata",
            "format": "json",
        }
        headers = {"User-Agent": self._user_agent}
        r = self._session.get(
            COMMONS_API,
            params=params,
            headers=headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        results: list[dict] = []
        pages = data.get("query", {}).get("pages") or {}
        if not isinstance(pages, dict):
            return []
        for _pid, p in pages.items():
            if not isinstance(p, dict):
                continue
            page_title = p.get("title") or ""
            if page_title.startswith("File:"):
                page_title = page_title[5:]
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
                        if max(int(w), int(h)) < self.min_size:
                            continue
                    except (TypeError, ValueError):
                        pass

                ext = info.get("extmetadata") or {}
                license_str = None
                author = None
                date = None

                license_entry = ext.get("LicenseShortName")
                if isinstance(license_entry, dict) and license_entry.get("value"):
                    license_str = _normalize_license(license_entry["value"])

                artist_entry = ext.get("Artist")
                if isinstance(artist_entry, dict) and artist_entry.get("value"):
                    author = _strip_html(artist_entry["value"])

                date_entry = ext.get("DateTimeOriginal")
                if isinstance(date_entry, dict) and date_entry.get("value"):
                    date = _strip_html(date_entry["value"])

                results.append(
                    self._make_result(
                        url=url,
                        query=query,
                        width=int(w) if w is not None else None,
                        height=int(h) if h is not None else None,
                        license=license_str,
                        source_domain="commons.wikimedia.org",
                        title=page_title or None,
                        author=author,
                        date=date,
                    )
                )
        return results
