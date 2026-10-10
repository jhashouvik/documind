"""Where LangGraph keeps its memory.

    checkpointer  (BaseCheckpointSaver)  SHORT-term memory, per thread (= chat
                  session). A snapshot of the graph state after every super-step.
                  Enables: multi-turn chat, pause/resume (human-in-the-loop),
                  time travel, and surviving a pod restart mid-conversation.
    store         (BaseStore)  LONG-term memory: JSON documents in namespaces,
                  shared across threads, e.g. ("documind","users",<id>,"memories").

With DATABASE_URL set both live in Postgres (tables are created by setup() at
startup, idempotently). Without it we fall back to in-memory versions: fine for
tests and a laptop, wrong for production (lost on restart, not shared by replicas).
"""
import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from .logging_setup import log_extra

log = logging.getLogger(__name__)


@dataclass
class Persistence:
    checkpointer: object
    store: object
    kind: str                                   # "postgres" | "memory"
    ping: Callable[[], Awaitable[bool]]


async def _always_ok() -> bool:
    return True


def in_memory() -> Persistence:
    return Persistence(InMemorySaver(), InMemoryStore(), "memory", _always_ok)


@asynccontextmanager
async def open_persistence(settings):
    if not settings.database_url:
        log.warning("DATABASE_URL not set: conversation state is IN MEMORY (lost on restart)")
        yield in_memory()
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    # autocommit + dict_row + no prepared statements: what the LangGraph savers expect
    # (prepare_threshold=None also keeps it working behind PgBouncer)
    pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=settings.db_pool_size,
                               open=False, kwargs={"autocommit": True, "prepare_threshold": None,
                                                   "row_factory": dict_row})
    await pool.open(wait=False)

    async def ping() -> bool:
        try:
            async with pool.connection(timeout=2) as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001
            return False

    try:
        # no start-up order between pods: wait for Postgres instead of crashing
        for attempt in range(1, settings.db_connect_retries + 1):
            if await ping():
                break
            log.warning("postgres not reachable yet", extra=log_extra(attempt=attempt))
            await asyncio.sleep(settings.db_retry_delay_s)
        else:
            raise RuntimeError(f"Postgres not reachable after {settings.db_connect_retries} attempts")
        saver, store = AsyncPostgresSaver(pool), AsyncPostgresStore(pool)
        await saver.setup()                      # creates/migrates the checkpoint tables
        await store.setup()
        log.info("persistence ready", extra=log_extra(kind="postgres"))
        yield Persistence(saver, store, "postgres", ping)
    finally:
        await pool.close()
