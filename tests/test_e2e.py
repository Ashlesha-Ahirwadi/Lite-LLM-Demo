"""Multi-turn flow test (Section 11.2): turn 1 establishes a fact, turn 2 (via
`previous_response_id`, with no client-side history) relies on it. Anthropic is
mocked; Redis is faked (fakeredis) so this runs with no live services.
"""
import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.session as session_module


@pytest.fixture(autouse=True)
def fake_redis():
    session_module._redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield
    session_module._redis_client = None


def _fake_anthropic_response(text: str, input_tokens: int = 10, output_tokens: int = 5) -> dict:
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def test_multiturn_conversation_remembers_established_fact(monkeypatch):
    calls = []

    async def fake_call_messages_api(body):
        calls.append(body)
        if len(calls) == 1:
            return _fake_anthropic_response("Got it, your favorite color is teal.")
        # Prove the reply can only be right because the gateway sent full
        # history -- the client never repeated "teal" itself.
        history_text = " ".join(m["content"] for m in body["messages"])
        assert "teal" in history_text
        return _fake_anthropic_response("Your favorite color is teal.")

    monkeypatch.setattr(main_module, "call_messages_api", fake_call_messages_api)
    client = TestClient(main_module.app)

    resp1 = client.post("/v1/responses", json={"model": "gpt-4o", "input": "my favorite color is teal"})
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["output"][0]["content"][0]["text"] == "Got it, your favorite color is teal."
    assert body1["previous_response_id"] is None
    response_id_1 = body1["id"]
    assert response_id_1.startswith("resp_")

    resp2 = client.post(
        "/v1/responses",
        json={"model": "gpt-4o", "input": "what's my favorite color?", "previous_response_id": response_id_1},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert "teal" in body2["output"][0]["content"][0]["text"]
    assert body2["previous_response_id"] == response_id_1
    assert body2["id"] != response_id_1

    # Second Anthropic call carried the full reconstructed history: turn 1's
    # user msg + assistant reply, then turn 2's new user msg -- in order.
    second_request_messages = calls[1]["messages"]
    assert [m["role"] for m in second_request_messages] == ["user", "assistant", "user"]


def test_store_false_response_is_not_referenceable(monkeypatch):
    async def fake_call_messages_api(body):
        return _fake_anthropic_response("ephemeral reply")

    monkeypatch.setattr(main_module, "call_messages_api", fake_call_messages_api)
    client = TestClient(main_module.app)

    resp1 = client.post("/v1/responses", json={"model": "gpt-4o", "input": "hello", "store": False})
    assert resp1.status_code == 200
    response_id_1 = resp1.json()["id"]

    resp2 = client.post(
        "/v1/responses",
        json={"model": "gpt-4o", "input": "still there?", "previous_response_id": response_id_1},
    )
    # Not persisted (store: false), so the id isn't referenceable -- OpenAI-shaped 404.
    assert resp2.status_code == 404
    assert resp2.json()["error"]["type"] == "invalid_request_error"


def test_anthropic_direct_mode_ignores_previous_response_id(monkeypatch):
    """The demo console's "Direct to model" mode: even with a previous_response_id,
    only the newest turn goes to Anthropic -- that's what makes it "forget".
    """
    calls = []

    async def fake_call_messages_api(body):
        calls.append(body)
        return _fake_anthropic_response("ok")

    monkeypatch.setattr(main_module, "call_messages_api", fake_call_messages_api)
    client = TestClient(main_module.app)

    resp1 = client.post("/v1/responses", json={"model": "anthropic-gateway", "input": "my favorite color is teal"})
    assert resp1.status_code == 200
    response_id_1 = resp1.json()["id"]

    resp2 = client.post(
        "/v1/responses",
        json={
            "model": "anthropic-direct",
            "input": "what's my favorite color?",
            "previous_response_id": response_id_1,
        },
    )
    assert resp2.status_code == 200
    assert resp2.json()["model"] == "anthropic-direct"

    second_request_messages = calls[1]["messages"]
    assert second_request_messages == [{"role": "user", "content": "what's my favorite color?"}]


def test_anthropic_direct_and_gateway_resolve_to_the_same_anthropic_model(monkeypatch):
    """The dropdown selects gateway behavior, not a different backend model."""
    captured_models = []

    async def fake_call_messages_api(body):
        captured_models.append(body["model"])
        return _fake_anthropic_response("ok")

    monkeypatch.setattr(main_module, "call_messages_api", fake_call_messages_api)
    client = TestClient(main_module.app)

    client.post("/v1/responses", json={"model": "anthropic-direct", "input": "hi"})
    client.post("/v1/responses", json={"model": "anthropic-gateway", "input": "hi"})

    assert captured_models[0] == captured_models[1] == main_module.settings.anthropic_model


def test_max_output_tokens_defaults_when_absent(monkeypatch):
    captured = {}

    async def fake_call_messages_api(body):
        captured["max_tokens"] = body["max_tokens"]
        return _fake_anthropic_response("ok")

    monkeypatch.setattr(main_module, "call_messages_api", fake_call_messages_api)
    client = TestClient(main_module.app)

    resp = client.post("/v1/responses", json={"model": "gpt-4o", "input": "hi"})
    assert resp.status_code == 200
    assert captured["max_tokens"] == main_module.settings.default_max_tokens
