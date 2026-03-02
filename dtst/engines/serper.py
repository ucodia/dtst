import logging
import os

import requests

from dtst.engines.base import SearchEngine

logger = logging.getLogger(__name__)

SERPER_IMAGES_URL = "https://google.serper.dev/images"
MIN_SIZE = 1024


class SerperEngine(SearchEngine):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("SERPER_API_KEY", "")

    @property
    def name(self) -> str:
        return "serper"

    def search(self, query: str, page: int) -> list[str]:
        if not self._api_key:
            logger.warning("SERPER_API_KEY not set; skipping Serper")
            return []
        payload: dict = {"q": query, "num": 10}
        if page > 1:
            payload["page"] = page
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(SERPER_IMAGES_URL, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning("Serper request failed for %r page %s: %s", query, page, e)
            return []
        images = data.get("images") or []
        if not isinstance(images, list):
            return []
        urls: list[str] = []
        for img in images:
            if not isinstance(img, dict):
                continue
            url = img.get("imageUrl") or img.get("image") or img.get("original_image_url")
            if not url:
                continue
            w = img.get("imageWidth") or img.get("original_image_width")
            h = img.get("imageHeight") or img.get("original_image_height")
            if w is not None and h is not None:
                try:
                    if max(int(w), int(h)) < MIN_SIZE:
                        continue
                except (TypeError, ValueError):
                    pass
            urls.append(url)
        return urls
