"""Fixed-window rate limiter backed by Redis.

Every LLM call costs money, so a public chat endpoint needs a limit. A
per-pod counter would allow N x limit with N replicas; a counter in Redis
is shared by all replicas, so the limit holds cluster-wide.

    key = documind:rl:<client>:<current minute>
    INCR key            -> how many requests this minute
    EXPIRE key 70       -> Redis cleans up old windows by itself

If Redis is down we FAIL OPEN (allow the request): losing rate limiting for
a minute is better than taking the whole product down.
"""
import logging
import time

import redis.asyncio as redis

from .logging_setup import log_extra

log = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, client: redis.Redis, per_minute: int) -> None:
        self.r = client
        self.limit = per_minute

    async def check(self, client_key: str) -> tuple[bool, int, int]:
        """Returns (allowed, remaining, seconds_until_reset)."""
        if self.limit <= 0:
            return True, -1, 0
        now = time.time()
        window = int(now // 60)
        reset = 60 - int(now % 60)
        key = f"documind:rl:{client_key}:{window}"
        try:
            count = await self.r.incr(key)
            if count == 1:
                await self.r.expire(key, 70)
        except redis.RedisError as exc:
            log.warning("rate limiter unavailable, allowing request", extra=log_extra(error=str(exc)))
            return True, -1, 0
        return count <= self.limit, max(0, self.limit - count), reset
