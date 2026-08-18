"""Pure unit tests for app/transform.py -- no network, no Redis, no Anthropic."""
from app.models import InputItem, Message
from app.transform import (
    anthropic_response_to_message,
    build_anthropic_request,
    build_responses_response,
    new_response_id,
    normalize_input,
)

# ---------------------------------------------------------------------------
# normalize_input (Section 7.2)
# ---------------------------------------------------------------------------


def test_normalize_input_string_becomes_single_user_message():
    result = normalize_input("hello there")
    assert result == [Message(role="user", content="hello there")]


def test_normalize_input_array_of_string_content():
    result = normalize_input([InputItem(role="user", content="hi")])
    assert result == [Message(role="user", content="hi")]


def test_normalize_input_array_preserves_roles_and_order():
    result = normalize_input(
        [
            InputItem(role="user", content="what's the capital of France?"),
            InputItem(role="assistant", content="Paris."),
            InputItem(role="user", content="and its population?"),
        ]
    )
    assert [m.role for m in result] == ["user", "assistant", "user"]
    assert result[1].content == "Paris."


def test_normalize_input_flattens_multipart_text_content():
    result = normalize_input(
        [InputItem(role="user", content=[{"type": "input_text", "text": "hello "}, {"type": "input_text", "text": "world"}])]
    )
    assert result == [Message(role="user", content="hello world")]


def test_normalize_input_skips_non_text_parts():
    result = normalize_input(
        [
            InputItem(
                role="user",
                content=[
                    {"type": "input_image", "image_url": "https://example.com/x.png"},
                    {"type": "input_text", "text": "describe this"},
                ],
            )
        ]
    )
    assert result == [Message(role="user", content="describe this")]


# ---------------------------------------------------------------------------
# build_anthropic_request (Section 8.1)
# ---------------------------------------------------------------------------


def test_build_anthropic_request_maps_history_and_required_fields():
    history = [Message(role="user", content="hi")]
    body = build_anthropic_request(history=history, anthropic_model="claude-3-5-haiku-20241022", max_tokens=1024)

    assert body["model"] == "claude-3-5-haiku-20241022"
    assert body["max_tokens"] == 1024
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "system" not in body
    assert "temperature" not in body
    assert "top_p" not in body


def test_build_anthropic_request_includes_optional_fields_when_present():
    history = [Message(role="user", content="hi")]
    body = build_anthropic_request(
        history=history,
        anthropic_model="claude-3-5-haiku-20241022",
        max_tokens=512,
        instructions="You are terse.",
        temperature=0.2,
        top_p=0.9,
    )

    assert body["system"] == "You are terse."
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9


def test_build_anthropic_request_multiturn_history_order_preserved():
    history = [
        Message(role="user", content="fact: my name is Ada"),
        Message(role="assistant", content="Got it."),
        Message(role="user", content="what's my name?"),
    ]
    body = build_anthropic_request(history=history, anthropic_model="m", max_tokens=100)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]


# ---------------------------------------------------------------------------
# build_responses_response / anthropic_response_to_message (Section 8.2)
# ---------------------------------------------------------------------------


def _anthropic_response(text="hi back", stop_reason="end_turn", input_tokens=5, output_tokens=7):
    return {
        "id": "msg_abc123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def test_build_responses_response_maps_text_and_usage():
    resp = build_responses_response(
        anthropic_response=_anthropic_response(text="Paris."),
        response_id="resp_1",
        model="gpt-4o",
        previous_response_id=None,
    )
    assert resp.output[0].content[0].text == "Paris."
    assert resp.usage.input_tokens == 5
    assert resp.usage.output_tokens == 7
    assert resp.usage.total_tokens == 12
    assert resp.status == "completed"
    assert resp.incomplete_details is None
    assert resp.model == "gpt-4o"
    assert resp.id == "resp_1"


def test_build_responses_response_end_turn_is_completed():
    resp = build_responses_response(
        anthropic_response=_anthropic_response(stop_reason="end_turn"),
        response_id="resp_2",
        model="gpt-4o",
        previous_response_id="resp_1",
    )
    assert resp.status == "completed"
    assert resp.previous_response_id == "resp_1"


def test_build_responses_response_max_tokens_is_incomplete():
    resp = build_responses_response(
        anthropic_response=_anthropic_response(stop_reason="max_tokens"),
        response_id="resp_3",
        model="gpt-4o",
        previous_response_id=None,
    )
    assert resp.status == "incomplete"
    assert resp.incomplete_details is not None
    assert resp.incomplete_details.reason == "max_output_tokens"


def test_build_responses_response_joins_multiple_text_blocks():
    ar = _anthropic_response()
    ar["content"] = [{"type": "text", "text": "part one. "}, {"type": "text", "text": "part two."}]
    resp = build_responses_response(anthropic_response=ar, response_id="resp_4", model="m", previous_response_id=None)
    assert resp.output[0].content[0].text == "part one. part two."


def test_anthropic_response_to_message_extracts_assistant_reply():
    msg = anthropic_response_to_message(_anthropic_response(text="the answer is 42"))
    assert msg == Message(role="assistant", content="the answer is 42")


def test_new_response_id_has_expected_prefix_and_is_unique():
    a, b = new_response_id(), new_response_id()
    assert a.startswith("resp_")
    assert b.startswith("resp_")
    assert a != b
