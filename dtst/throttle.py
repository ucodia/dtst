import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainPolicy:
    max_connections: int
    request_delay: float


DOMAIN_LIMITS: dict[str, DomainPolicy] = {
    "upload.wikimedia.org": DomainPolicy(max_connections=2, request_delay=0.7),
    "live.staticflickr.com": DomainPolicy(max_connections=6, request_delay=0.0),
    "inaturalist-open-data.s3.amazonaws.com": DomainPolicy(
        max_connections=4, request_delay=0.2
    ),
    "static.inaturalist.org": DomainPolicy(max_connections=4, request_delay=0.2),
}

DEFAULT_POLICY = DomainPolicy(max_connections=8, request_delay=0.0)

CIRCUIT_BREAKER_THRESHOLD = 5


class DomainThrottler:
    def __init__(
        self,
        limits: dict[str, DomainPolicy] | None = None,
        default: DomainPolicy = DEFAULT_POLICY,
    ):
        self._limits = limits or DOMAIN_LIMITS
        self._default = default
        self._semaphores: dict[str, threading.Semaphore] = {}
        self._last_request: dict[str, float] = {}
        self._consecutive_429s: dict[str, int] = {}
        self._tripped: dict[str, bool] = {}
        self._lock = threading.Lock()

    def _get_policy(self, domain: str) -> DomainPolicy:
        return self._limits.get(domain, self._default)

    def _get_semaphore(self, domain: str) -> threading.Semaphore:
        with self._lock:
            if domain not in self._semaphores:
                policy = self._get_policy(domain)
                self._semaphores[domain] = threading.Semaphore(policy.max_connections)
            return self._semaphores[domain]

    def acquire(self, domain: str) -> None:
        sem = self._get_semaphore(domain)
        sem.acquire()
        policy = self._get_policy(domain)
        if policy.request_delay > 0:
            with self._lock:
                last = self._last_request.get(domain, 0.0)
                now = time.monotonic()
                wait = policy.request_delay - (now - last)
                if wait > 0:
                    time.sleep(wait)
                self._last_request[domain] = time.monotonic()

    def release(self, domain: str) -> None:
        sem = self._get_semaphore(domain)
        sem.release()

    def record_429(self, domain: str) -> None:
        with self._lock:
            count = self._consecutive_429s.get(domain, 0) + 1
            self._consecutive_429s[domain] = count
            if count >= CIRCUIT_BREAKER_THRESHOLD and not self._tripped.get(
                domain, False
            ):
                self._tripped[domain] = True
                logger.warning(
                    "Circuit breaker tripped for %s -- skipping remaining URLs (re-run to retry)",
                    domain,
                )

    def record_success(self, domain: str) -> None:
        with self._lock:
            self._consecutive_429s[domain] = 0

    def is_tripped(self, domain: str) -> bool:
        with self._lock:
            return self._tripped.get(domain, False)

    def tripped_domains(self) -> list[str]:
        with self._lock:
            return [d for d, t in self._tripped.items() if t]

    def active_policies(self) -> dict[str, DomainPolicy]:
        return dict(self._limits)
