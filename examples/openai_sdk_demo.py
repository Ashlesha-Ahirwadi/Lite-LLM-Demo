"""Proves the official OpenAI Python SDK works against this gateway, unmodified.

This is the headline demo (Section 11.4): point the OpenAI SDK's `base_url`
at the gateway instead of api.openai.com, then run a two-turn conversation
using `client.responses.create(..., previous_response_id=...)`. The SDK has
no idea the replies are actually coming from Anthropic -- that's the point.

Usage:
    docker compose up -d
    python examples/openai_sdk_demo.py

Requires the gateway to be running (default http://localhost:8000) with a
valid ANTHROPIC_API_KEY configured server-side. The SDK's own api_key is
never checked by the gateway -- any non-empty string works -- since auth is
explicitly out of scope for MVP (Section 4).
"""
import os

from openai import OpenAI

GATEWAY_BASE_URL = os.environ.get("GATEWAY_BASE_URL", "http://localhost:8000/v1")


def main() -> None:
    client = OpenAI(api_key="unused-gateway-does-not-check-this", base_url=GATEWAY_BASE_URL)

    print(f"--- Turn 1: establishing a fact (gateway at {GATEWAY_BASE_URL}) ---")
    response1 = client.responses.create(
        model="gpt-4o",
        input="My favorite programming language is Rust. Remember that.",
    )
    print("assistant:", response1.output_text)
    print("response id:", response1.id)

    print("\n--- Turn 2: relying on it, via previous_response_id (no history resent) ---")
    response2 = client.responses.create(
        model="gpt-4o",
        input="What's my favorite programming language?",
        previous_response_id=response1.id,
    )
    print("assistant:", response2.output_text)
    print("previous_response_id:", response2.previous_response_id)

    assert "rust" in response2.output_text.lower(), (
        "Expected the gateway's Redis-backed session state to carry turn 1's "
        "fact into turn 2 without the client resending it."
    )
    print("\nOK: multi-turn state was carried server-side by the gateway, via Anthropic underneath.")


if __name__ == "__main__":
    main()
