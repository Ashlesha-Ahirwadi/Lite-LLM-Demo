# Universal `/v1/responses` Gateway (Anthropic-backed)

A self-hostable API gateway that exposes an **OpenAI-compatible `/v1/responses`
endpoint**, but serves every request from **Anthropic's Messages API**.

OpenAI's `/v1/responses` API is *stateful*: a client sends only the newest
turn plus a `previous_response_id`, and OpenAI reconstructs the earlier
conversation server-side. Anthropic — like almost every other provider —
only exposes a *stateless* API, where the client must resend the full
message history on every call. For a gateway to offer one unified
`/v1/responses` endpoint across providers, the gateway itself has to own
the session state for the providers that lack it. This project is exactly
that slice, built as a deliberately scoped model of a real production
concern: OpenAI's Responses spec on the front, Anthropic's Messages API on
the back, and a Redis-backed session-emulation layer in between.

## Architecture

```
Client (curl / OpenAI SDK)
        │  POST /v1/responses   (OpenAI Responses shape)
        ▼
┌─────────────────────────────────────────────┐
│                 GATEWAY (FastAPI)            │
│                                              │
│  1. Validate request (Pydantic)              │
│  2. If previous_response_id present:         │
│        load prior history ───────────────────┼──► Redis (GET)
│  3. Append new input to history              │
│  4. Transform → Anthropic Messages request   │
│  5. Call Anthropic ──────────────────────────┼──► api.anthropic.com/v1/messages
│  6. Transform reply → OpenAI Response shape   │
│  7. Store updated history under new resp id ─┼──► Redis (SET + TTL)
│  8. Return response with new `id`            │
└─────────────────────────────────────────────┘
```

Each box in that flow is its own module, kept deliberately narrow:

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, route wiring — the flow above, nothing else |
| `app/models.py` | Pydantic request/response/internal schemas |
| `app/transform.py` | Pure OpenAI ⇄ Anthropic mapping functions (no I/O) |
| `app/session.py` | Redis get/set of conversation history + response id generation |
| `app/anthropic_client.py` | Raw `httpx` calls to Anthropic's Messages API |
| `app/errors.py` | OpenAI-compatible error shapes + FastAPI exception handlers |
| `app/config.py` | Env-driven settings |

## Quickstart

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

docker compose up
```

Then, in another terminal, run the headline proof — the official OpenAI
Python SDK, unmodified, pointed at this gateway instead of `api.openai.com`:

```bash
pip install -r requirements-dev.txt
python examples/openai_sdk_demo.py
```

It runs a two-turn conversation (`client.responses.create(...)` twice, the
second call using `previous_response_id`) and asserts the second reply
correctly recalls a fact from the first turn — proving the Redis-backed
session state works from the client's point of view, even though the
client never resent any history.

For a plain-curl version of the same two-turn flow, see
`examples/curl_examples.sh`.

## Try it in the browser

`docker compose up`, then open **http://localhost:8000** — this page is a throwaway demo harness for visualizing the gateway's session-threading in a browser, not part of the product. Its mode dropdown ("Direct to model" vs. "Through gateway") is itself a demonstration device: both options call the exact same Anthropic model, so any difference in whether a follow-up remembers earlier context comes entirely from the gateway's session layer, not the model.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

- `tests/test_transform.py` — pure unit tests for the OpenAI ⇄ Anthropic
  mapping functions, no network or Redis involved.
- `tests/test_e2e.py` — the multi-turn proof (Anthropic mocked, Redis faked
  via `fakeredis`): turn 1 establishes a fact, turn 2 relies on it via
  `previous_response_id` with no client-side history, plus `store: false`
  and default-`max_output_tokens` behavior.
- `tests/test_errors.py` — every error case from the spec (missing/empty
  `input`, `stream: true`, unknown `previous_response_id`, upstream
  Anthropic failure) returns the correct status and OpenAI-shaped error
  body, and that upstream error details never leak to the client.

## Configuration

All via environment variables (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — (required) | Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | Backend model used when the requested model isn't in `MODEL_MAP` |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Overridable for testing against a mock |
| `REDIS_URL` | `redis://localhost:6379/0` | Session store connection string |
| `SESSION_TTL_SECONDS` | `3600` | How long a `previous_response_id` stays referenceable |
| `DEFAULT_MAX_TOKENS` | `1024` | Used when a request omits `max_output_tokens` |
| `MODEL_MAP` | `{}` | JSON map of requested model name → Anthropic model id |

## Design decisions

See [DESIGN.md](./DESIGN.md) for the reasoning behind the session storage
model, why Redis instead of Postgres, how `instructions` and `store` are
handled, and a handful of other calls made along the way.

## What I'd build next

Deliberately out of scope for this MVP (see `PRD.md` Section 12), roughly
in the order I'd tackle them:

1. **Streaming (`stream: true`)** — translate Anthropic's SSE stream events
   into OpenAI Responses stream events. Currently rejected outright with a
   400.
2. **Tool/function calling** — `tools` and `tool_choice` passthrough with
   format translation. The response model already carries fixed "no tools"
   values (`tools: []`, `tool_choice: "none"`) so the OpenAI SDK parses
   responses cleanly today; real support means translating both directions.
3. **A second backend provider** (e.g. Gemini) behind the same endpoint, to
   actually prove the "provider-agnostic" framing rather than just leaving
   room for it. `anthropic_client.py` and the Anthropic-specific mapping in
   `transform.py` would need to become one of several provider adapters
   behind a common interface.
4. **`/v1/chat/completions`** alongside `/v1/responses`, to demonstrate the
   stateless-vs-stateful contrast side by side in the same codebase.
5. **Postgres as a durable audit log** of turns, contrasting with Redis's
   deliberately ephemeral, TTL-expiring role.
