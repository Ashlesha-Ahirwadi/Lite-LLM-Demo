"""Section 10: every error case returns the OpenAI-shaped envelope
{ "error": { "message", "type", "param", "code" } } with the right HTTP status.
"""
import fakeredis.aioredis
import httpx
import pytest
from fastapi.testclient import TestClient

import app.anthropic_client as anthropic_client_module
import app.main as main_module
import app.session as session_module


@pytest.fixture(autouse=True)
def fake_redis():
    session_module._redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield
    session_module._redis_client = None


@pytest.fixture(autouse=True)
def reset_http_client():
    yield
    anthropic_client_module._http_client = None


@pytest.fixture
def client():
    return TestClient(main_module.app)


def _assert_openai_error_shape(body: dict, *, type_: str) -> None:
    assert "error" in body
    error = body["error"]
    assert set(error.keys()) == {"message", "type", "param", "code"}
    assert error["type"] == type_
    assert isinstance(error["message"], str) and error["message"]


def test_missing_input_returns_400(client):
    resp = client.post("/v1/responses", json={"model": "gpt-4o"})
    assert resp.status_code == 400
    _assert_openai_error_shape(resp.json(), type_="invalid_request_error")


def test_empty_input_returns_400(client):
    resp = client.post("/v1/responses", json={"model": "gpt-4o", "input": ""})
    assert resp.status_code == 400
    _assert_openai_error_shape(resp.json(), type_="invalid_request_error")


def test_stream_true_returns_400(client):
    resp = client.post("/v1/responses", json={"model": "gpt-4o", "input": "hi", "stream": True})
    assert resp.status_code == 400
    body = resp.json()
    _assert_openai_error_shape(body, type_="invalid_request_error")
    assert "stream" in body["error"]["message"].lower()


def test_unknown_previous_response_id_returns_404(client):
    resp = client.post(
        "/v1/responses",
        json={"model": "gpt-4o", "input": "hi", "previous_response_id": "resp_does_not_exist"},
    )
    assert resp.status_code == 404
    body = resp.json()
    _assert_openai_error_shape(body, type_="invalid_request_error")
    assert body["error"]["param"] == "previous_response_id"


def test_upstream_anthropic_error_returns_502_without_leaking_raw_body(client, monkeypatch):
    def error_handler(request):
        return httpx.Response(500, text="super secret internal anthropic stack trace")

    anthropic_client_module._http_client = httpx.AsyncClient(
        base_url="https://api.anthropic.com", transport=httpx.MockTransport(error_handler)
    )

    resp = client.post("/v1/responses", json={"model": "gpt-4o", "input": "hi"})
    assert resp.status_code == 502
    body = resp.json()
    _assert_openai_error_shape(body, type_="api_error")
    assert "super secret internal anthropic stack trace" not in resp.text
