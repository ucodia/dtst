import logging
import os

from dtst.engines.base import SearchEngine

logger = logging.getLogger(__name__)

FLICKR_REST = "https://api.flickr.com/services/rest/"

FLICKR_LICENSES = {
    0: "all-rights-reserved",
    1: "cc-by-nc-sa-2.0",
    2: "cc-by-nc-2.0",
    3: "cc-by-nc-nd-2.0",
    4: "cc-by-2.0",
    5: "cc-by-sa-2.0",
    6: "cc-by-nd-2.0",
    7: "no-known-restrictions",
    8: "us-government-work",
    9: "cc0-1.0",
    10: "public-domain-mark",
}

SIZE_VARIANTS = ("o", "k", "h")


class FlickrEngine(SearchEngine):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_size: int = 1024,
        retries: int = 3,
        timeout: int | float = 30,
    ) -> None:
        super().__init__(min_size=min_size, retries=retries, timeout=timeout)
        self._api_key = api_key or os.environ.get("FLICKR_API_KEY", "")

    @property
    def name(self) -> str:
        return "flickr"

    def search(self, query: str, page: int) -> list[dict]:
        if not self._api_key:
            logger.warning("FLICKR_API_KEY not set; skipping Flickr")
            return []
        extras = "url_o,url_k,url_h,url_m,license,owner_name,date_taken,tags,o_dims"
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
        r = self._session.get(FLICKR_REST, params=params, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("stat") != "ok":
            logger.warning("Flickr API error for %r: %s", query, data.get("message", data))
            return []
        photos = data.get("photos", {}).get("photo") or []
        results: list[dict] = []
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
            w: int | None = None
            h: int | None = None
            for variant in SIZE_VARIANTS:
                wv = p.get(f"width_{variant}")
                hv = p.get(f"height_{variant}")
                if wv is not None and hv is not None:
                    try:
                        w = int(wv)
                        h = int(hv)
                    except (TypeError, ValueError):
                        w, h = None, None
                    break
            if w is not None and h is not None:
                try:
                    if max(w, h) < self.min_size:
                        continue
                except (TypeError, ValueError):
                    pass

            license_id = p.get("license")
            license_str = None
            if license_id is not None:
                try:
                    license_str = FLICKR_LICENSES.get(int(license_id))
                except (TypeError, ValueError):
                    pass

            results.append(self._make_result(
                url=url,
                query=query,
                width=w,
                height=h,
                license=license_str,
                source_domain="flickr.com",
                title=p.get("title"),
                author=p.get("ownername"),
                date=p.get("datetaken"),
            ))
        return results
