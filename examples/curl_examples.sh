#!/usr/bin/env bash
# Copy-pasteable two-turn conversation against the gateway via curl (Section 11.3).
# Requires the gateway running locally (default http://localhost:8000), e.g. via
# `docker compose up`.
set -euo pipefail

BASE_URL="${GATEWAY_BASE_URL:-http://localhost:8000}"

echo "--- Turn 1: establishing a fact ---"
TURN1=$(curl -sS "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gpt-4o",
        "input": "My favorite programming language is Rust. Remember that."
      }')
echo "$TURN1" | python3 -m json.tool

RESPONSE_ID=$(echo "$TURN1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "response id: $RESPONSE_ID"

echo
echo "--- Turn 2: relying on it via previous_response_id (no history resent) ---"
curl -sS "$BASE_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"gpt-4o\",
        \"input\": \"What's my favorite programming language?\",
        \"previous_response_id\": \"$RESPONSE_ID\"
      }" | python3 -m json.tool
