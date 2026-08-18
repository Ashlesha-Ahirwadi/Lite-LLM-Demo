"""Env-driven settings for the gateway.

Centralized here so every other module imports one `settings` object instead
of calling `os.environ` scattered around the codebase.
"""
import json
import os


class Settings:
    def __init__(self) -> None:
        self.anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
        self.anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.anthropic_base_url: str = os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        )
        self.redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.session_ttl_seconds: int = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))
        self.default_max_tokens: int = int(os.environ.get("DEFAULT_MAX_TOKENS", "1024"))

        # Optional JSON map of requested model name -> Anthropic model id.
        # Falls back to `anthropic_model` for anything not listed.
        raw_model_map = os.environ.get("MODEL_MAP", "")
        self.model_map: dict[str, str] = json.loads(raw_model_map) if raw_model_map else {}

    def resolve_model(self, requested_model: str) -> str:
        return self.model_map.get(requested_model, self.anthropic_model)


settings = Settings()
