"""FastAPI app and route wiring: POST /v1/responses (Section 6 flow, Section 7).

Business logic here raises typed exceptions (SessionNotFoundError,
AnthropicAPIError, HTTPException) and never builds an error response body
itself -- errors.py's registered handlers are the single place that shapes
those into the OpenAI-compatible error envelope (Section 10).
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.anthropic_client import call_messages_api, close_http_client
from app.config import settings
from app.errors import register_exception_handlers
from app.models import ResponsesRequest, ResponsesResponse
from app.session import close_redis_client, load_session, save_session
from app.transform import (
    anthropic_response_to_message,
    build_anthropic_request,
    build_responses_response,
    new_response_id,
    normalize_input,
)

_STATIC_DIR = Path(__file__).parent / "static"

# Demo-console mode device (see app/static/index.html's mode dropdown): both
# strings resolve to the same real ANTHROPIC_MODEL via settings.resolve_model
# (neither is in MODEL_MAP), so the only thing "anthropic-direct" changes is
# whether history gets reconstructed below -- it is not a different backend.
_DIRECT_MODEL_ID = "anthropic-direct"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_redis_client()
    await close_http_client()


app = FastAPI(title="Anthropic-backed OpenAI Responses gateway", lifespan=lifespan)
register_exception_handlers(app)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def demo_console() -> FileResponse:
    """Serves the throwaway browser demo harness (app/static/index.html).

    Purely presentational: it's a static file talking to /v1/responses on
    the same origin, no different from curl_examples.sh or the SDK demo --
    it doesn't touch transform/session/anthropic_client at all.
    """
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/v1/responses", response_model=ResponsesResponse)
async def create_response(request: ResponsesRequest) -> ResponsesResponse:
    # Section 7.1: stream is a stretch goal, MVP rejects it outright.
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not yet supported")

    # 1-2. Reconstruct history (Section 9 follow-up flow) and append this turn.
    # "anthropic-direct" is the one exception: it deliberately skips
    # reconstruction so only the newest turn goes to Anthropic, even if a
    # previous_response_id was sent -- that's what makes it "forget" in the
    # demo console. Everything after this point is one shared code path.
    reconstruct_history = request.model != _DIRECT_MODEL_ID
    if reconstruct_history and request.previous_response_id is not None:
        prior = await load_session(request.previous_response_id)  # raises SessionNotFoundError
        history = list(prior.messages)
    else:
        history = []
    history = history + normalize_input(request.input)

    # 3-4. Transform -> Anthropic request, call Anthropic (Section 8.1).
    anthropic_model = settings.resolve_model(request.model)
    max_tokens = request.max_output_tokens or settings.default_max_tokens
    anthropic_body = build_anthropic_request(
        history=history,
        anthropic_model=anthropic_model,
        max_tokens=max_tokens,
        instructions=request.instructions,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    anthropic_response = await call_messages_api(anthropic_body)

    # 5-7. Transform reply -> OpenAI shape, persist new history under a new id (Section 8.2, 9).
    response_id = new_response_id()
    new_history = history + [anthropic_response_to_message(anthropic_response)]
    if request.store:
        await save_session(response_id, new_history, model=request.model)

    # 8. Return, echoing the client-requested model (not the resolved Anthropic model id).
    return build_responses_response(
        anthropic_response=anthropic_response,
        response_id=response_id,
        model=request.model,
        previous_response_id=request.previous_response_id,
    )
