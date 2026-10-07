"""Conversation memory in Redis.

Why Redis and not a Python dict? agent-service runs as several replicas
behind a Service. Consecutive questions from one user can land on different
pods, so the conversation must live outside the pods. Redis is shared by all
replicas and survives pod restarts.

Layout: one Redis LIST per session, key documind:session:<id>, each element a
JSON message {"role": ..., "content": ...}. We keep only the last N messages
(LTRIM) and expire idle sessions (EXPIRE) - bounded memory, bounded tokens.
Only the user's questions and the final answers are stored, not the tool
chatter: history is re-sent to the LLM on every turn and tokens cost money.
"""
import json
import logging

import redis.asyncio as redis

from .logging_setup import log_extra

log = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, client: redis.Redis, ttl_s: int, max_messages: int) -> None:
        self.r = client
        self.ttl = ttl_s
        self.max = max_messages

    @staticmethod
    def key(session_id: str) -> str:
        return f"documind:session:{session_id}"

    async def load(self, session_id: str) -> list[dict]:
        try:
            raw = await self.r.lrange(self.key(session_id), -self.max, -1)
            return [json.loads(x) for x in raw]
        except redis.RedisError as exc:
            # degrade gracefully: answer without history rather than failing
            log.warning("memory unavailable, continuing without history",
                        extra=log_extra(error=str(exc)))
            return []

    async def append(self, session_id: str, *messages: dict) -> None:
        k = self.key(session_id)
        try:
            async with self.r.pipeline(transaction=True) as pipe:
                pipe.rpush(k, *[json.dumps(msg, ensure_ascii=False) for msg in messages])
                pipe.ltrim(k, -self.max, -1)
                pipe.expire(k, self.ttl)
                await pipe.execute()
        except redis.RedisError as exc:
            log.warning("could not save history", extra=log_extra(error=str(exc)))

    async def clear(self, session_id: str) -> bool:
        return bool(await self.r.delete(self.key(session_id)))
