"""Login brute-force guard (T081).

A sliding fixed-window counter per client IP in Redis (INCR + EXPIRE). Kept
deliberately tiny — the Cloudflare Tunnel edge also rate-limits in production;
this is defense in depth for the login path specifically.

Fail-open: any Redis error returns "allowed" rather than locking everyone out —
availability beats a self-inflicted outage, and a stuffed login is still gated at
the edge. ponytail: per-IP fixed window; swap for a token bucket if bursts matter.
"""

from __future__ import annotations

LOGIN_PATH_SUFFIX = "/auth/jwt/login"


def register_attempt(redis, ip: str, *, limit: int, window: int) -> bool:
    """Count one login attempt for ``ip``; return True while under ``limit``.

    ``limit <= 0`` disables the guard (always True).
    """
    if limit <= 0:
        return True
    key = f"login-attempts:{ip}"
    try:
        count = redis.incr(key)
        if count == 1:
            redis.expire(key, window)
        return count <= limit
    except Exception:
        return True


if __name__ == "__main__":  # runnable self-check (Constitution V)
    class _FakeRedis:
        def __init__(self) -> None:
            self.store: dict[str, int] = {}

        def incr(self, key: str) -> int:
            self.store[key] = self.store.get(key, 0) + 1
            return self.store[key]

        def expire(self, key: str, ttl: int) -> None:
            pass

    r = _FakeRedis()
    allowed = [register_attempt(r, "1.2.3.4", limit=3, window=60) for _ in range(4)]
    assert allowed == [True, True, True, False], allowed
    # A different IP has its own budget.
    assert register_attempt(r, "5.6.7.8", limit=3, window=60) is True
    # Disabled guard never blocks.
    assert all(register_attempt(r, "9.9.9.9", limit=0, window=60) for _ in range(100))

    class _BrokenRedis:
        def incr(self, key: str):
            raise ConnectionError("redis down")

    # Fail-open: a broken cache must not lock users out.
    assert register_attempt(_BrokenRedis(), "1.2.3.4", limit=3, window=60) is True
    print("ratelimit self-check OK")
