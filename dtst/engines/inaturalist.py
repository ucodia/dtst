import logging
import re
import time

from dtst.engines.base import SearchEngine
from dtst.user_agent import get_user_agent

logger = logging.getLogger(__name__)

INAT_API = "https://api.inaturalist.org/v1/observations"
DEFAULT_PER_PAGE = 200

PHOTO_SIZE_RE = re.compile(r"/(square|medium|small|thumb|large)\.")


class INaturalistEngine(SearchEngine):
    def __init__(
        self,
        delay: float = 1.0,
        per_page: int = DEFAULT_PER_PAGE,
        *,
        min_size: int = 1024,
        retries: int = 3,
        timeout: int | float = 30,
    ) -> None:
        super().__init__(min_size=min_size, retries=retries, timeout=timeout)
        self._delay = delay
        self._per_page = min(per_page, 200)

    @property
    def name(self) -> str:
        return "inaturalist"

    def search(self, query: str, page: int) -> list[dict]:
        time.sleep(self._delay)

        try:
            taxon_id = int(query)
        except (TypeError, ValueError):
            logger.warning("Invalid taxon_id %r; skipping", query)
            return []

        params: dict = {
            "taxon_id": taxon_id,
            "page": page,
            "per_page": self._per_page,
            "photos": "true",
            "order_by": "id",
            "order": "asc",
        }

        headers = {"User-Agent": get_user_agent()}
        r = self._session.get(
            INAT_API,
            params=params,
            headers=headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()

        observations = data.get("results") or []
        results: list[dict] = []

        for obs in observations:
            if not isinstance(obs, dict):
                continue

            taxon = obs.get("taxon") or {}
            taxon_name = taxon.get("preferred_common_name") or taxon.get("name")
            observed_on = obs.get("observed_on")
            quality_grade = obs.get("quality_grade")

            photos = obs.get("photos") or []
            for photo in photos:
                if not isinstance(photo, dict):
                    continue
                url = photo.get("url")
                if not url:
                    continue

                url = PHOTO_SIZE_RE.sub("/original.", url)

                license_code = photo.get("license_code")
                attribution = photo.get("attribution")

                result = self._make_result(
                    url=url,
                    query=query,
                    license=license_code,
                    source_domain="inaturalist.org",
                    title=taxon_name,
                    author=attribution,
                    date=observed_on,
                )
                result["quality_grade"] = quality_grade
                results.append(result)

        return results
