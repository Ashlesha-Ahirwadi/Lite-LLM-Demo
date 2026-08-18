"""Pydantic models for the OpenAI Responses API subset we implement (Section 7).

Three groups of models live here:
  - Request models: what a client sends to POST /v1/responses.
  - Internal models: the normalized shape we pass between transform/session/anthropic_client.
  - Response models: what we send back, shaped like OpenAI's Responses API.

Normalizing `input` (string-or-array -> internal message list) is a pure
transformation, not a schema concern, so that logic lives in transform.py,
not here.
"""
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ContentPart(BaseModel):
    """One part of a multi-part input/output content list.

    OpenAI defines several part types (input_text, input_image, output_text,
    refusal, ...). MVP only needs the two text-bearing ones; other types are
    accepted (extra fields allowed) but their text is simply absent, so
    normalization skips them. See Section 7.2 and Non-goals (Section 4).
    """

    model_config = {"extra": "allow"}

    type: str
    text: Optional[str] = None


class InputItem(BaseModel):
    """One entry in an array-form `input` (Section 7.2).

    Only user/assistant are accepted here: system-level instructions have a
    single dedicated path (the top-level `instructions` field, Section 7.1),
    matching Anthropic's own `system` being a top-level, non-message field.
    """

    role: Literal["user", "assistant"]
    content: Union[str, list[ContentPart]]


# A request's `input` is either a bare string (shorthand for one user
# message) or an explicit list of role/content items.
ResponsesInput = Union[str, list[InputItem]]


class ResponsesRequest(BaseModel):
    """POST /v1/responses request body (the subset defined in Section 7.1)."""

    model: str
    input: ResponsesInput
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    store: bool = True
    stream: Optional[bool] = False
    metadata: Optional[dict[str, Any]] = None

    @field_validator("input")
    @classmethod
    def input_not_empty(cls, value: ResponsesInput) -> ResponsesInput:
        if isinstance(value, str) and not value.strip():
            raise ValueError("input must not be empty")
        if isinstance(value, list) and len(value) == 0:
            raise ValueError("input must not be an empty list")
        return value


# ---------------------------------------------------------------------------
# Internal models (used by transform.py, session.py, anthropic_client.py)
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """One normalized turn in a conversation history.

    This is the shape stored in Redis and the shape transform.py builds
    Anthropic `messages` payloads from — the common currency between the
    OpenAI-shaped world and the Anthropic-shaped world.
    """

    role: Literal["user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class OutputTextContent(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class OutputMessage(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[OutputTextContent]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class IncompleteDetails(BaseModel):
    reason: str


class ResponsesResponse(BaseModel):
    """POST /v1/responses response body (Section 7.3)."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["completed", "incomplete"]
    model: str
    output: list[OutputMessage]
    usage: Usage
    previous_response_id: Optional[str] = None
    incomplete_details: Optional[IncompleteDetails] = None

    # Not in the PRD's example body, but required by the official OpenAI SDK's
    # client-side Response model (openai.types.responses.Response) to parse a
    # response at all. Fixed to their "no tools" values since tool/function
    # calling is out of scope for MVP (Section 4) -- see Section 11.4's
    # acceptance criterion that the unmodified SDK must work against us.
    tools: list = Field(default_factory=list)
    tool_choice: Literal["none"] = "none"
    parallel_tool_calls: bool = False
