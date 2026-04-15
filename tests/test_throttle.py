import threading
import time

import pytest

from dtst.throttle import (
    CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_POLICY,
    DOMAIN_LIMITS,
    DomainPolicy,
    DomainThrottler,
)


# ---------------------------------------------------------------------------
# Policy lookup
# ---------------------------------------------------------------------------


def test_known_domain_returns_specific_policy():
    throttler = DomainThrottler()
    policy = throttler._get_policy("upload.wikimedia.org")
    assert policy == DOMAIN_LIMITS["upload.wikimedia.org"]
    assert policy.max_connections == 2
    assert policy.request_delay == 0.7


def test_unknown_domain_returns_default_policy():
    throttler = DomainThrottler()
    policy = throttler._get_policy("unknown.example.com")
    assert policy == DEFAULT_POLICY


def test_custom_limits_override_built_in():
    custom = {"custom.example.com": DomainPolicy(max_connections=1, request_delay=0.0)}
    throttler = DomainThrottler(limits=custom)
    assert throttler._get_policy("custom.example.com").max_connections == 1
    # Built-in domain no longer matches -> default.
    assert throttler._get_policy("upload.wikimedia.org") == DEFAULT_POLICY


# ---------------------------------------------------------------------------
# Semaphore behavior
# ---------------------------------------------------------------------------


def test_acquire_release_on_unknown_domain():
    throttler = DomainThrottler()
    throttler.acquire("unknown.example.com")
    throttler.release("unknown.example.com")


def test_max_connections_respected():
    custom = {"test.example.com": DomainPolicy(max_connections=2, request_delay=0.0)}
    throttler = DomainThrottler(limits=custom)

    throttler.acquire("test.example.com")
    throttler.acquire("test.example.com")

    # Third acquire should block.
    def third_acquire():
        throttler.acquire("test.example.com")

    t = threading.Thread(target=third_acquire)
    t.start()
    t.join(timeout=0.1)
    assert t.is_alive(), "third acquire should be blocked when max_connections=2"

    # Release one -- thread should now proceed.
    throttler.release("test.example.com")
    t.join(timeout=1.0)
    assert not t.is_alive(), "third acquire should unblock after release"

    # Clean up.
    throttler.release("test.example.com")
    throttler.release("test.example.com")


# ---------------------------------------------------------------------------
# Request delay
# ---------------------------------------------------------------------------


def test_no_delay_returns_quickly():
    custom = {"fast.example.com": DomainPolicy(max_connections=4, request_delay=0.0)}
    throttler = DomainThrottler(limits=custom)

    start = time.monotonic()
    throttler.acquire("fast.example.com")
    throttler.release("fast.example.com")
    throttler.acquire("fast.example.com")
    throttler.release("fast.example.com")
    elapsed = time.monotonic() - start
    assert elapsed < 0.05, f"expected near-instant, got {elapsed:.3f}s"


def test_request_delay_enforced_between_acquires():
    custom = {"test.example.com": DomainPolicy(max_connections=4, request_delay=0.1)}
    throttler = DomainThrottler(limits=custom)

    throttler.acquire("test.example.com")
    throttler.release("test.example.com")

    start = time.monotonic()
    throttler.acquire("test.example.com")
    elapsed = time.monotonic() - start
    throttler.release("test.example.com")

    assert 0.08 <= elapsed <= 0.25, (
        f"expected ~0.1s delay between acquires, got {elapsed:.3f}s"
    )


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_fresh_throttler_not_tripped():
    throttler = DomainThrottler()
    assert throttler.is_tripped("any.example.com") is False
    assert throttler.tripped_domains() == []


def test_record_429_below_threshold_not_tripped():
    throttler = DomainThrottler()
    for _ in range(CIRCUIT_BREAKER_THRESHOLD - 1):
        throttler.record_429("x.example.com")
    assert throttler.is_tripped("x.example.com") is False
    assert "x.example.com" not in throttler.tripped_domains()


def test_record_429_at_threshold_trips():
    throttler = DomainThrottler()
    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        throttler.record_429("x.example.com")
    assert throttler.is_tripped("x.example.com") is True
    assert "x.example.com" in throttler.tripped_domains()


def test_record_success_resets_counter():
    throttler = DomainThrottler()
    for _ in range(CIRCUIT_BREAKER_THRESHOLD - 1):
        throttler.record_429("x.example.com")
    throttler.record_success("x.example.com")
    for _ in range(CIRCUIT_BREAKER_THRESHOLD - 1):
        throttler.record_429("x.example.com")
    # Total 429s = 8, but counter was reset so we never hit the threshold in a row.
    assert throttler.is_tripped("x.example.com") is False


def test_record_success_does_not_untrip():
    throttler = DomainThrottler()
    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        throttler.record_429("x.example.com")
    assert throttler.is_tripped("x.example.com") is True
    throttler.record_success("x.example.com")
    assert throttler.is_tripped("x.example.com") is True


def test_tripped_domains_only_includes_tripped():
    throttler = DomainThrottler()
    throttler.record_429("a.example.com")
    throttler.record_429("b.example.com")
    assert throttler.tripped_domains() == []

    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        throttler.record_429("c.example.com")
    assert throttler.tripped_domains() == ["c.example.com"]


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------


def test_concurrent_record_429_trips_safely():
    throttler = DomainThrottler()
    errors = []

    def worker():
        try:
            for _ in range(5):
                throttler.record_429("concurrent.example.com")
        except Exception as e:  # pragma: no cover - defensive
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert throttler.is_tripped("concurrent.example.com") is True


def test_concurrent_acquire_release():
    custom = {"conc.example.com": DomainPolicy(max_connections=4, request_delay=0.0)}
    throttler = DomainThrottler(limits=custom)
    errors = []

    def worker():
        try:
            for _ in range(10):
                throttler.acquire("conc.example.com")
                throttler.release("conc.example.com")
        except Exception as e:  # pragma: no cover - defensive
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# ---------------------------------------------------------------------------
# active_policies
# ---------------------------------------------------------------------------


def test_active_policies_returns_copy():
    custom = {"a.example.com": DomainPolicy(max_connections=3, request_delay=0.0)}
    throttler = DomainThrottler(limits=custom)

    policies = throttler.active_policies()
    assert policies == custom

    policies["evil.example.com"] = DomainPolicy(max_connections=99, request_delay=99.0)
    policies.pop("a.example.com", None)

    # Internal state untouched.
    assert throttler._get_policy("a.example.com").max_connections == 3
    assert throttler._get_policy("evil.example.com") == DEFAULT_POLICY


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
