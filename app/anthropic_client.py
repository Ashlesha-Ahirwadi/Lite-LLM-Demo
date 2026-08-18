"""Thin async httpx wrapper around Anthropic's Messages API.

Raw HTTP on purpose (not the Anthropic SDK) so the request/response shapes
the gateway depends on stay visible in transform.py rather than hidden
behind an SDK's own models (Section 5).
"""
import httpx

from app.config import settings

_ANTHROPIC_VERSION = "2023-06-01"
_MESSAGES_PATH = "/v1/messages"


class AnthropicAPIError(Exception):
    """Wraps any failure to get a successful response from Anthropic.

    Carries enough detail for server-side logging, but callers (the route
    handler / errors.py) decide what, if anything, gets echoed back to the
    client — Section 10 requires upstream failures surface as a clean 502
    that doesn't leak Anthropic's raw error body.
    """

    def __init__(self, message: str, *, status_code: int | None = None, anthropic_body: str | None = None) -> None:
        self.status_code = status_code
        self.anthropic_body = anthropic_body
        super().__init__(message)


_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Lazily create the module-level httpx client (one connection pool per process)."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            base_url=settings.anthropic_base_url,
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
        )
    return _http_client


async def close_http_client() -> None:
    """Close the pooled connection on app shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


async def call_messages_api(body: dict) -> dict:
    """POST a Messages API request body to Anthropic and return the parsed JSON reply.

    Raises AnthropicAPIError on network failure or any non-2xx response.
    """
    client = get_http_client()
    headers = {
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    try:
        response = await client.post(_MESSAGES_PATH, json=body, headers=headers)
    except httpx.RequestError as exc:
        raise AnthropicAPIError(f"Could not reach Anthropic: {exc}") from exc

    if response.status_code >= 400:
        raise AnthropicAPIError(
            f"Anthropic returned HTTP {response.status_code}",
            status_code=response.status_code,
            anthropic_body=response.text,
        )

    return response.json()
