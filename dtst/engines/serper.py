import logging
import os
from urllib.parse import urlparse

from dtst.engines.base import SearchEngine

logger = logging.getLogger(__name__)

SERPER_IMAGES_URL = "https://google.serper.dev/images"


class SerperEngine(SearchEngine):
    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_size: int = 1024,
        retries: int = 3,
        timeout: int | float = 30,
    ) -> None:
        super().__init__(min_size=min_size, retries=retries, timeout=timeout)
        self._api_key = api_key or os.environ.get("SERPER_API_KEY", "")

    @property
    def name(self) -> str:
        return "serper"

    def search(self, query: str, page: int) -> list[dict]:
        if not self._api_key:
            logger.warning("SERPER_API_KEY not set; skipping Serper")
            return []
        payload: dict = {"q": query, "num": 100}
        if page > 1:
            payload["page"] = page
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        r = self._session.post(
            SERPER_IMAGES_URL,
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        images = data.get("images") or []
        if not isinstance(images, list):
            return []
        results: list[dict] = []
        for img in images:
            if not isinstance(img, dict):
                continue
            url = (
                img.get("imageUrl") or img.get("image") or img.get("original_image_url")
            )
            if not url:
                continue
            w = img.get("imageWidth") or img.get("original_image_width")
            h = img.get("imageHeight") or img.get("original_image_height")
            w_int: int | None = None
            h_int: int | None = None
            if w is not None and h is not None:
                try:
                    w_int = int(w)
                    h_int = int(h)
                    if max(w_int, h_int) < self.min_size:
                        continue
                except (TypeError, ValueError):
                    pass

            source_domain = None
            link = img.get("link")
            if link:
                source_domain = urlparse(link).netloc or None

            results.append(
                self._make_result(
                    url=url,
                    query=query,
                    width=w_int,
                    height=h_int,
                    title=img.get("title"),
                    source_domain=source_domain,
                )
            )
        return results
