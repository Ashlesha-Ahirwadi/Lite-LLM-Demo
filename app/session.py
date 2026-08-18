"""Redis-backed session store for conversation history (Section 9).

Storage model — one key per response id, holding the *full* running history
up to and including that turn:

    key:   session:resp_<uuid>
    value: JSON {"messages": [{"role", "content"}, ...], "model": "<requested model>"}
    TTL:   settings.session_ttl_seconds (default 3600s)

We store the whole history per id (rather than a linked list of single
turns pointing at their parent) so a follow-up read is a single GET with no
chain-walking, at the cost of duplicating earlier turns across keys. For an
in-memory, TTL-expiring store where a conversation is bounded and short-
lived, that trade favors read simplicity/latency over storage — see
DESIGN.md.
"""
import json

import redis.asyncio as redis
from pydantic import BaseModel

from app.config import settings
from app.models import Message


class SessionNotFoundError(Exception):
    """Raised when a `previous_response_id` doesn't resolve to a live session.

    Covers both "never existed" and "existed but TTL'd out" — Redis makes
    those indistinguishable on GET, and the client should get the same
    error either way (Section 10: 404, invalid_request_error).
    """

    def __init__(self, response_id: str) -> None:
        self.response_id = response_id
        super().__init__(f"No session found for response id: {response_id}")


class StoredSession(BaseModel):
    messages: list[Message]
    model: str


_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """Lazily create the module-level Redis client (one per process)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


def _key(response_id: str) -> str:
    return f"session:{response_id}"


async def save_session(response_id: str, messages: list[Message], model: str) -> None:
    """Persist a conversation's history under its response id, with TTL.

    Callers are responsible for deciding *whether* to call this at all —
    `store: false` (Section 7.1) means skip the call entirely so the id is
    never referenceable.
    """
    client = get_redis_client()
    session = StoredSession(messages=messages, model=model)
    await client.set(_key(response_id), session.model_dump_json(), ex=settings.session_ttl_seconds)


async def load_session(response_id: str) -> StoredSession:
    """Load a prior conversation's history. Raises SessionNotFoundError if missing/expired."""
    client = get_redis_client()
    raw = await client.get(_key(response_id))
    if raw is None:
        raise SessionNotFoundError(response_id)
    return StoredSession.model_validate(json.loads(raw))


async def close_redis_client() -> None:
    """Close the pooled connection on app shutdown."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
