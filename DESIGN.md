# Design decisions

This is the "why," not the "what" — the code and README already cover what
each piece does. These are the calls I made along the way and the reasoning
behind them, in the order they came up while building.

## Why Redis for session state, not Postgres

The state I'm storing is exactly a hot-path key lookup by id ("give me the
history for `resp_A`"), it's meant to expire on its own after an hour, and
it doesn't need to survive a restart or be queried by anything other than
its id. That's Redis's whole design center: `GET`/`SET` by key, native TTL
via `EX`, in-memory speed. Postgres would work, but I'd be building TTL
expiry, id-based lookup, and probably a cleanup job myself on top of a
tool built for relational queries and durability I don't need. If this
gateway grew a requirement to *audit* every turn ever sent — for
compliance, debugging, analytics — that's a genuinely different job
(durable, queryable, append-only) and Postgres would be the right answer
for *that*, sitting alongside Redis rather than replacing it. That's why
"Postgres as a durable audit log" is in the stretch goals instead of being
folded into the session store.

## Why the whole history per response id, not a linked list of turns

Each Redis key (`session:resp_<uuid>`) holds the *entire* reconstructed
conversation up to that turn, not just that turn's message with a pointer
to its parent. The alternative — store one turn per key, with each turn
pointing at `previous_response_id` — uses less storage per key, but a
10-turn conversation's follow-up request would mean 10 sequential Redis
round trips to walk the chain back to the start before I could even build
the Anthropic request.

I picked whole-history-per-key because reads are on the request's critical
path (every single `/v1/responses` call with a `previous_response_id` has
to reconstruct history before it can call Anthropic), and writes are not
particularly expensive relatively speaking — a `SET` of one JSON blob,
even a duplicated one, is still one round trip and Redis is fast enough
that the extra bytes barely register for a TTL-bounded chat-length history.
I'm trading storage for a guarantee that "load history" is always exactly
one `GET`, regardless of how long the conversation has gotten. For a
demo/MVP of a chat gateway, that trade is the right one; it stops being
obviously right if conversations regularly ran into the thousands of turns,
at which point the duplicated-storage cost would start to dominate and the
linked-list approach (or a hybrid — snapshot every N turns) would be worth
revisiting.

## Why `instructions` are applied per-call, not persisted into history

`instructions` maps to Anthropic's top-level `system` field, and it's
*not* saved into the message list that gets stored in Redis. This matches
what OpenAI actually does: `instructions` is a per-request steering
knob, not a conversational turn — if you don't resend it on turn 2, it's
gone for turn 2. Baking it into the stored history would silently change
that contract (an instruction given once would keep influencing every
future turn whether or not the caller meant it to), and it would also
blur the one place in this codebase where "system-level" content is
supposed to live. Anthropic itself treats `system` the same way — a
top-level field, not a `messages` entry — so this isn't fighting the
backend, it's the same shape on both sides. It's also why array-form
`input` items only accept `role: "user" | "assistant"` (see
`InputItem` in `app/models.py`): if `"system"` were allowed there too,
there'd be two different paths for the same concept, and it'd be
ambiguous which one wins.

## What `store: false` means, and why TTL is the right model for "ephemeral"

`store: false` means: don't call `save_session` at all. The response still
gets a normal `resp_<uuid>` id, but since nothing was ever written under
that key, a follow-up request using it as `previous_response_id` gets
exactly the same 404 as an id that expired or never existed — Redis makes
"never existed" and "TTL'd out" indistinguishable on a `GET`, and I
decided that's fine, because from the client's perspective both cases mean
the same thing: *this conversation isn't available to continue from
server-side state*. That's also why `SessionNotFoundError` doesn't
distinguish the two cases internally — there's no information to
distinguish them with, and inventing a fake distinction (e.g. tracking
"this id existed once" separately) would add a second source of truth for
no behavioral difference to the client.

TTL is the mechanism for ephemeral state generally, not just for
`store: false`: every stored session expires after `SESSION_TTL_SECONDS`
(default 3600s) whether or not the client cares. That mirrors how OpenAI's
own server-side response IDs aren't kept forever either — this is
emulating bounded, temporary server state, not building a permanent chat
history store.

## Why raw `httpx` instead of the Anthropic SDK

The PRD calls for this explicitly (Section 5), and having built it, I'd
make the same call independently: the entire point of `transform.py` is
that the OpenAI-shape-to-Anthropic-shape mapping is visible and
unit-testable as plain dicts. Going through the Anthropic SDK would mean
building Anthropic SDK model objects in `transform.py`, which hides the
actual wire shape behind another library's abstractions — exactly the
layer this project exists to make explicit. Raw `httpx` keeps
`anthropic_client.py` to "POST this dict, return that dict" (see
`call_messages_api`), so the request/response shapes I'm reasoning about
in `transform.py` are the literal JSON, not an SDK's interpretation of it.

## Why `transform.py` has no I/O at all

Every function in `transform.py` takes plain data in and returns plain
data out — no `import app.config`, no `import app.session`, no network.
Concretely: `build_anthropic_request` takes `max_tokens` and
`anthropic_model` as required arguments rather than reading them from
`settings` itself. That's what makes `tests/test_transform.py` able to
assert "given this history and these fields, the Anthropic body looks like
*this*" without mocking Redis, httpx, or environment variables — the
tests read as pure input/output examples. The cost is that `main.py` has
to do the resolving (`settings.resolve_model(...)`,
`request.max_output_tokens or settings.default_max_tokens`) before calling
into `transform.py`, but that's a small, honest cost for keeping the
actual mapping logic — the part most likely to have subtle bugs — trivial
to test.

## Why errors funnel through one set of handlers, not scattered `try/except`

`app/main.py` never builds an error response. It raises whatever exception
matches what actually went wrong — `SessionNotFoundError` from
`session.py`, `AnthropicAPIError` from `anthropic_client.py`,
`RequestValidationError` from Pydantic itself, or a plain `HTTPException`
for the `stream: true` rejection — and `app/errors.py` is the only place
that knows how to turn any of those into the OpenAI error envelope
(`{"error": {"message", "type", "param", "code"}}`). That split means
adding a new failure mode later is "raise the right exception," not
"remember to format the error correctly at every call site." It's also
why `AnthropicAPIError` carries `anthropic_body` (the raw upstream error
text) as an attribute rather than putting it straight into the exception
message: the handler in `errors.py` deliberately never surfaces it to the
client (Section 10's "don't leak raw internals"), but it's still there for
server-side logging if I want it later.

## Why the response always echoes the client's *requested* model, not the resolved Anthropic model id

If a client sends `"model": "gpt-4o"` and `MODEL_MAP` resolves that to
`claude-3-5-sonnet-...` for the actual Anthropic call, the response still
says `"model": "gpt-4o"`. That's what a real OpenAI-compatible endpoint
does — echo back what was asked for — and it's also what keeps the
`MODEL_MAP` indirection invisible to the client, which is the point of
having it: from the outside, this still looks like a normal
`/v1/responses` endpoint that happens to run on different infrastructure.

## Why `tools: []`, `tool_choice: "none"`, `parallel_tool_calls: false` are fixed fields on every response

These aren't in the PRD's example response body, but I found out the hard
way (running the actual OpenAI SDK demo) that they're non-optional fields
on the SDK's own client-side `Response` model — without them, the
official SDK can't even parse a response from this gateway, which would
have quietly broken Section 11.4's acceptance criterion. Since tool
calling is explicitly out of scope for MVP, the honest values are the
"no tools" ones: an empty tools list, `tool_choice: "none"`, and
`parallel_tool_calls: false`. They're fixed, not configurable, because
there's currently no code path that could make them anything else be
true.

## Why `requirements.txt` and `requirements-dev.txt` are split, and the Dockerfile only installs the former

`pytest`, `openai`, and `fakeredis` are needed to develop and test this
project, but the running gateway never imports any of them. Installing
them into the container image would just be dead weight in every deploy.
I confirmed this split is actually safe (not just assumed) by installing
*only* `requirements.txt` into a clean virtualenv and importing
`app.main` successfully — if the app secretly depended on something in
the dev file, that would have failed immediately instead of surfacing
later as a mysterious container crash.

## Why the Redis and httpx clients are lazy, module-level singletons

Both `app/session.py` and `app/anthropic_client.py` create their client
(`redis.asyncio.Redis`, `httpx.AsyncClient`) once, on first use, and reuse
it for the life of the process — rather than opening a new connection per
request. Both libraries are built around connection pooling for exactly
this reason; recreating the client per-request would throw that pooling
away and add connection-setup latency to every single call. The FastAPI
`lifespan` in `main.py` closes both on shutdown so the process doesn't
leak sockets when it exits.
