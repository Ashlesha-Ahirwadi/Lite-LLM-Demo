"""OpenAI-compatible error shapes and FastAPI exception handlers (Section 10).

Every error response, regardless of cause, gets the same envelope:

    { "error": { "message": ..., "type": ..., "param": ..., "code": null } }

The handlers here are the single place that translates internal exceptions
(SessionNotFoundError, AnthropicAPIError, pydantic validation errors, plain
HTTPExceptions) into that shape -- route/business logic just raises the
exception that matches what went wrong and doesn't know about response
formatting.
"""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.anthropic_client import AnthropicAPIError
from app.session import SessionNotFoundError


def error_body(message: str, *, type_: str, param: str | None = None, code: str | None = None) -> dict:
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(SessionNotFoundError)
    async def handle_session_not_found(request: Request, exc: SessionNotFoundError) -> JSONResponse:
        # Covers both "never existed" and "TTL expired" (Section 9/10 -- Redis
        # makes those indistinguishable, and the client sees the same error).
        return JSONResponse(
            status_code=404,
            content=error_body(
                f"Unknown or expired previous_response_id: '{exc.response_id}'",
                type_="invalid_request_error",
                param="previous_response_id",
            ),
        )

    @app.exception_handler(AnthropicAPIError)
    async def handle_anthropic_error(request: Request, exc: AnthropicAPIError) -> JSONResponse:
        # Deliberately generic: exc.anthropic_body may contain Anthropic's raw
        # error text, which Section 10 says must not leak to the client.
        return JSONResponse(
            status_code=502,
            content=error_body(
                "The upstream model provider returned an error.",
                type_="api_error",
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Covers missing `input`, empty `input`, missing `model`, wrong types, etc.
        first = exc.errors()[0] if exc.errors() else {}
        param = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or None
        message = first.get("msg", "Invalid request")
        return JSONResponse(
            status_code=400,
            content=error_body(message, type_="invalid_request_error", param=param),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Catches raise HTTPException(...) calls in route code, e.g. the
        # stream: true rejection (Section 7.1 / 10).
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(str(exc.detail), type_="invalid_request_error"),
        )
