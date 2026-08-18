"""OpenAI Responses <-> Anthropic Messages mapping (Section 8).

Every function here is pure: plain data in, plain data out, no network and
no reads from app.config/settings. Callers (main.py) are responsible for
resolving things like "which Anthropic model id" or "what's the default
max_tokens" from config and passing the resolved values in. That split is
what makes these functions unit-testable without mocking anything.
"""
import time
import uuid

from app.models import (
    ContentPart,
    IncompleteDetails,
    InputItem,
    Message,
    OutputMessage,
    OutputTextContent,
    ResponsesInput,
    ResponsesResponse,
    Usage,
)

# Anthropic stop_reason -> OpenAI Responses status (Section 8.2).
_INCOMPLETE_STOP_REASON = "max_tokens"
_INCOMPLETE_REASON_DETAIL = "max_output_tokens"


def _extract_text(content: str | list[ContentPart]) -> str:
    """Collapse a content part list down to its text (Section 7.2).

    Only `input_text` / `output_text` parts carry text for MVP; anything
    else (e.g. a future `input_image`) is silently skipped rather than
    erroring, since supporting non-text parts is explicitly out of scope
    (Section 4).
    """
    if isinstance(content, str):
        return content
    return "".join(part.text or "" for part in content if part.type in ("input_text", "output_text"))


def normalize_input(input_value: ResponsesInput) -> list[Message]:
    """Turn a request's `input` (string or item array) into a message list.

    - A bare string becomes a single user message.
    - An array of {role, content} items is mapped item-by-item, with
      multipart content flattened to text.
    """
    if isinstance(input_value, str):
        return [Message(role="user", content=input_value)]

    messages: list[Message] = []
    for item in input_value:
        assert isinstance(item, InputItem)
        messages.append(Message(role=item.role, content=_extract_text(item.content)))
    return messages


def build_anthropic_request(
    *,
    history: list[Message],
    anthropic_model: str,
    max_tokens: int,
    instructions: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict:
    """Build an Anthropic /v1/messages request body (Section 8.1).

    `max_tokens` is required by Anthropic, so the caller must always supply
    a concrete int — either the client's `max_output_tokens` or the
    configured default (Section 7.1's "if absent, set a default").
    """
    body: dict = {
        "model": anthropic_model,
        "messages": [{"role": m.role, "content": m.content} for m in history],
        "max_tokens": max_tokens,
    }
    if instructions is not None:
        body["system"] = instructions
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    return body


def _extract_assistant_text(anthropic_content: list[dict]) -> str:
    """Join Anthropic's text content blocks into one string.

    Anthropic can return multiple `text`-type blocks; non-text blocks
    (e.g. tool_use) don't occur in MVP since tool calling is out of scope
    (Section 4), but are skipped defensively rather than erroring.
    """
    return "".join(block.get("text", "") for block in anthropic_content if block.get("type") == "text")


def anthropic_response_to_message(anthropic_response: dict) -> Message:
    """Extract the assistant's reply as a normalized Message, for history append."""
    text = _extract_assistant_text(anthropic_response.get("content", []))
    return Message(role="assistant", content=text)


def build_responses_response(
    *,
    anthropic_response: dict,
    response_id: str,
    model: str,
    previous_response_id: str | None,
) -> ResponsesResponse:
    """Map an Anthropic /v1/messages response into an OpenAI Responses response (Section 8.2)."""
    text = _extract_assistant_text(anthropic_response.get("content", []))
    usage_in = anthropic_response.get("usage", {})
    input_tokens = usage_in.get("input_tokens", 0)
    output_tokens = usage_in.get("output_tokens", 0)

    stop_reason = anthropic_response.get("stop_reason")
    if stop_reason == _INCOMPLETE_STOP_REASON:
        status = "incomplete"
        incomplete_details = IncompleteDetails(reason=_INCOMPLETE_REASON_DETAIL)
    else:
        status = "completed"
        incomplete_details = None

    return ResponsesResponse(
        id=response_id,
        created_at=int(time.time()),
        status=status,
        model=model,
        output=[OutputMessage(content=[OutputTextContent(text=text)])],
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        previous_response_id=previous_response_id,
        incomplete_details=incomplete_details,
    )


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"
