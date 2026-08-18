# PRD — Universal `/v1/responses` Gateway (Anthropic-backed)

> **How to use this document:** Paste it into Claude Code as the spec for the project.
> Build the **MVP** first (Sections 1–11) and get it fully working and tested before
> touching **Stretch Goals** (Section 12). Keep the code readable and commented — the
> author needs to be able to explain every design decision out loud.

---

## 1. One-line summary

A self-hostable API gateway that exposes an **OpenAI-compatible `/v1/responses` endpoint**
but serves every request from **Anthropic's Messages API** — including the stateful,
multi-turn behavior (`previous_response_id`) that Anthropic has no native equivalent for.
The gateway emulates OpenAI's server-side session state itself, using **Redis**.

## 2. Why this exists (context)

OpenAI's `/v1/responses` API is **stateful**: a client can send only the newest turn plus a
`previous_response_id`, and OpenAI rebuilds the earlier conversation server-side. Almost every
other provider — Anthropic included — exposes only a **stateless** message-array API where the
client must resend the whole history each call.

For a gateway to offer *one* unified `/v1/responses` endpoint across *all* providers, the
**gateway itself must own the session state** for providers that lack it. This project builds
exactly that slice: OpenAI Responses spec on the front, Anthropic Messages on the back, with the
session-emulation layer in between. It is a deliberately scoped model of a real production concern.

## 3. Goals

- Accept requests in OpenAI Responses API shape and return responses in that same shape.
- Support **multi-turn conversations via `previous_response_id`**, with no client-side history.
- Translate faithfully to and from Anthropic's Messages API.
- Store and reconstruct session state in Redis, with TTL.
- Return **OpenAI-compatible error objects**.
- Be provable: the **official OpenAI Python SDK**, pointed at this gateway, should work unmodified.
- Run with **one command** (Docker Compose: app + Redis).

## 4. Non-goals (explicitly out of scope for MVP)

- Streaming responses (SSE) — stretch goal.
- Tool / function calling — stretch goal.
- More than one backend provider — stretch goal (design should not *prevent* it).
- Auth, billing, rate limiting, dashboards — not this project.
- Reproducing every field of the Responses spec — implement the core subset in Section 7.

## 5. Tech stack (must match)

- **Python 3.11+**
- **FastAPI** (+ `uvicorn`)
- **Redis** (session store)
- **Pydantic** (request/response models)
- **httpx** (async calls to Anthropic)
- **Docker + Docker Compose** (app + Redis)
- Anthropic accessed via **raw HTTP** (not the SDK) so the transformation work is visible.

## 6. Architecture

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

**Suggested file layout:**

```
app/
  main.py            # FastAPI app, route wiring
  models.py          # Pydantic models for Responses request/response
  transform.py       # OpenAI <-> Anthropic mapping (pure functions, unit-testable)
  session.py         # Redis get/set of conversation history + id generation
  anthropic_client.py# httpx call to Anthropic Messages API
  errors.py          # OpenAI-compatible error shapes + exception handlers
  config.py          # env-driven settings
tests/
  test_transform.py  # pure mapping tests (no network)
  test_e2e.py        # multi-turn flow (mock Anthropic or live)
examples/
  curl_examples.sh
  openai_sdk_demo.py # proves OpenAI SDK works against the gateway
docker-compose.yml
Dockerfile
README.md
DESIGN.md            # design decisions written in the author's own words
.env.example
```

## 7. Functional requirements — the endpoint

### 7.1 `POST /v1/responses`

Accept this subset of the OpenAI Responses request body:

| Field                  | Type              | Required | Handling                                                                 |
|------------------------|-------------------|----------|--------------------------------------------------------------------------|
| `model`                | string            | yes      | Map to an Anthropic model via config (Section 9). Unknown → default.     |
| `input`                | string OR array   | yes      | String = one user message. Array = list of input items (see 7.2).        |
| `instructions`         | string            | no       | Maps to Anthropic top-level `system`. **Does not persist across turns.** |
| `previous_response_id` | string            | no       | If present, load that conversation from Redis and prepend it.            |
| `max_output_tokens`    | int               | no       | Maps to Anthropic `max_tokens`. **If absent, set a default** (e.g. 1024).|
| `temperature`          | float             | no       | Pass through.                                                            |
| `top_p`                | float             | no       | Pass through.                                                            |
| `store`                | bool (default true)| no      | If false, do **not** persist history; returned id is non-referenceable.  |
| `stream`               | bool              | no       | MVP: reject with a clear error if true. (Stretch: implement.)            |
| `metadata`             | object            | no       | Store alongside session if convenient; otherwise ignore.                 |

### 7.2 Input item normalization

Normalize both input forms into an internal message list `[{role, content(text)}]`:
- A bare string → `[{role: "user", content: <string>}]`.
- An array of items → map each `{role, content}`; content may be a string or a list of
  content parts — extract the text (`input_text` / `output_text` types) into a single string for MVP.

### 7.3 Response body (returned to client)

Return an OpenAI-Responses-shaped object with at least:

```json
{
  "id": "resp_<uuid>",
  "object": "response",
  "created_at": 1730000000,
  "status": "completed",
  "model": "<the requested model>",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        { "type": "output_text", "text": "<assistant reply>" }
      ]
    }
  ],
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
  },
  "previous_response_id": "<echoed or null>"
}
```

## 8. Transformation rules (the core engineering)

Keep these as **pure functions** in `transform.py` so they can be unit-tested without network.

### 8.1 OpenAI Responses request → Anthropic Messages request

| OpenAI Responses           | Anthropic Messages         | Notes                                             |
|----------------------------|----------------------------|---------------------------------------------------|
| reconstructed history + `input` | `messages: [...]`     | Roles: `user` / `assistant`.                      |
| `instructions`             | `system`                   | Top-level string. Omit if absent.                 |
| `max_output_tokens`        | `max_tokens` (**required**)| Anthropic requires this — inject default if missing.|
| `temperature`, `top_p`     | same                       | Pass through.                                      |
| `model`                    | mapped model id            | Via config map; fallback to `ANTHROPIC_MODEL`.    |

### 8.2 Anthropic Messages response → OpenAI Responses response

| Anthropic                         | OpenAI Responses                                   |
|-----------------------------------|----------------------------------------------------|
| `content[].text` (text blocks)    | `output[0].content[0].text` (type `output_text`)   |
| `usage.input_tokens`              | `usage.input_tokens`                               |
| `usage.output_tokens`             | `usage.output_tokens` (+ compute `total_tokens`)   |
| `stop_reason == "max_tokens"`     | `status: "incomplete"` (+ `incomplete_details`)    |
| `stop_reason == "end_turn"`       | `status: "completed"`                              |
| new server-generated id           | `id: "resp_<uuid>"`                                |

## 9. Session management (the heart of the demo)

**Storage model.** For each stored response, save the *full running conversation up to and
including that turn* as a normalized message list, keyed by the response id:

```
Redis key:   session:resp_<uuid>
Redis value: JSON { "messages": [ {role, content}, ... ], "model": "<requested model>" }
TTL:         configurable, default 3600s
```

**First turn (no `previous_response_id`):**
1. history = normalized `input`
2. call Anthropic
3. new_history = history + assistant reply
4. `SET session:resp_A = new_history` (if `store != false`), with TTL
5. return response with `id = resp_A`

**Follow-up turn (`previous_response_id = resp_A`):**
1. `GET session:resp_A` → prior messages. If missing/expired → OpenAI 404-style error (Section 10).
2. history = prior messages + normalized new `input`
3. call Anthropic
4. new_history = history + assistant reply
5. `SET session:resp_B = new_history`, TTL
6. return response with `id = resp_B` (chaining continues indefinitely)

**Design decisions to document in DESIGN.md:**
- Why **Redis** over Postgres for session state (ephemeral, hot-path key lookups by id, native TTL).
- Why store the **whole history per response id** vs. a linked list of single turns (trade-off: simplicity/read-speed vs. storage; either is defensible — pick one and justify it).
- Why `instructions` are applied per-call and **not** persisted into history (matches OpenAI semantics).
- What `store: false` means and how expiry/TTL mirrors ephemeral server state.

## 10. Error handling (OpenAI-compatible)

All errors return this shape with an appropriate HTTP status:

```json
{ "error": { "message": "...", "type": "invalid_request_error", "param": "previous_response_id", "code": null } }
```

Required cases:
- Unknown/expired `previous_response_id` → 404, `type: "invalid_request_error"`.
- Missing `input` → 400.
- `stream: true` in MVP → 400 with a message saying streaming is not yet supported.
- Upstream Anthropic error → surface a 502 with a clean, OpenAI-shaped message (don't leak raw internals).

## 11. Testing & proof requirements

The demo is only convincing if it's provably correct. Include:

1. **`tests/test_transform.py`** — pure unit tests for both directions of the mapping (no network).
2. **`tests/test_e2e.py`** — a multi-turn flow: turn 1 establishes a fact, turn 2 (via
   `previous_response_id`) relies on that fact; assert the gateway remembered. Mock Anthropic where practical.
3. **`examples/curl_examples.sh`** — copy-pasteable two-turn conversation using `curl`.
4. **`examples/openai_sdk_demo.py`** — the headline proof: uses the **official OpenAI Python SDK**
   with `base_url` pointed at the gateway, calls `client.responses.create(...)` twice using
   `previous_response_id`, and shows it working against an Anthropic backend unmodified.

## 12. Stretch goals (only after MVP is solid)

- **Streaming** (`stream: true`) via SSE, translating Anthropic's stream events into OpenAI
  Responses stream events.
- **Function/tool calling** passthrough (`tools`, `tool_choice`) with format translation.
- **Second provider** (e.g. Gemini) behind the same endpoint, to prove "provider-agnostic."
- **`/v1/chat/completions`** endpoint too, to demonstrate understanding of the stateless-vs-stateful
  contrast side by side.
- **Postgres** as a durable audit log of turns (contrast with Redis's ephemeral role).

## 13. Deployment & config

**`docker compose up`** must start the gateway and Redis together.

Environment variables (`.env.example`):
- `ANTHROPIC_API_KEY` — required.
- `ANTHROPIC_MODEL` — default backend model id (a cheap current model is fine for the demo).
- `REDIS_URL` — e.g. `redis://redis:6379/0`.
- `SESSION_TTL_SECONDS` — default `3600`.
- `DEFAULT_MAX_TOKENS` — default `1024`.
- `MODEL_MAP` (optional) — JSON mapping requested model names → Anthropic model ids.

> Note: confirm the exact Anthropic model id string from Anthropic's current docs before running;
> set it via `ANTHROPIC_MODEL` rather than hardcoding.

## 14. README requirements

The README must include:
- The one-paragraph "what and why" (the stateful-vs-stateless problem).
- The architecture diagram (Section 6).
- Quickstart: `docker compose up`, then run `examples/openai_sdk_demo.py`.
- A **"Design decisions"** section (or link to DESIGN.md) written in the author's own words.
- A short "What I'd build next" section referencing the stretch goals.

## 15. Acceptance criteria (definition of done for MVP)

- [ ] `POST /v1/responses` accepts string and array `input` and returns a spec-shaped response.
- [ ] A follow-up request with `previous_response_id` correctly continues the conversation with no
      client-side history.
- [ ] Session history is stored in Redis with a TTL; expired/unknown ids return an OpenAI-shaped 404.
- [ ] `instructions`, `max_output_tokens`, `temperature`, `top_p`, `store` behave per Section 7.
- [ ] Errors follow the OpenAI error shape.
- [ ] The official OpenAI Python SDK, pointed at the gateway, runs a two-turn conversation successfully.
- [ ] `docker compose up` brings up app + Redis; the demo runs against it.
- [ ] Unit tests for transformations pass; the multi-turn e2e test passes.
- [ ] README + DESIGN.md explain the design decisions clearly.
