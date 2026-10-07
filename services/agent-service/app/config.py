"""Configuration for agent-service (12-factor: everything from env vars).

In Kubernetes: non-secret values come from the `agent-config` ConfigMap and
OPENROUTER_API_KEY comes from the `documind-llm` Secret.
"""
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- identity -----------------------------------------------------------
    app_name: str = "agent-service"
    app_version: str = "1.0.0"
    track: str = "stable"                     # stable | canary (shown in the UI badge)
    log_level: str = "INFO"

    # --- LLM via OpenRouter (OpenAI-compatible API) --------------------------------
    openrouter_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"     # default; must support tool calling
    # models the UI may choose from (comma separated). Keep it short: every entry
    # is a cost decision. Check tool support on openrouter.ai/models.
    llm_models: str = "openai/gpt-4o-mini,openai/gpt-4.1-mini,meta-llama/llama-3.3-70b-instruct"
    llm_fallback_models: str = ""             # comma separated, tried by OpenRouter in order
    llm_temperature: float = 0.2
    llm_max_tokens: int = 900
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    app_url: str = "http://documind.localtest.me"   # sent as HTTP-Referer (OpenRouter attribution)

    # --- agent behaviour -----------------------------------------------------------
    prompt_version: Literal["v1", "v2"] = "v1"
    agent_max_steps: int = 5                  # max LLM round-trips per question
    max_steps_limit: int = 8                  # upper bound a request may ask for
    max_tokens_limit: int = 2000              # upper bound a request may ask for
    tool_result_max_chars: int = 6000         # protects the context window
    trace_content: bool = False               # put prompts/answers into trace spans (privacy!)
    max_message_chars: int = 4000

    # --- knowledge-service (Service 1) --------------------------------------------
    knowledge_url: str = "http://knowledge-service:8001"   # Kubernetes DNS name
    knowledge_timeout_s: float = 15.0
    search_top_k: int = 4
    max_search_top_k: int = 10
    score_threshold: float | None = None      # None = knowledge-service default
    max_upload_mb: int = 20

    # --- Redis: conversation memory + rate limiting -----------------------------------
    redis_url: str = "redis://redis:6379/0"
    session_ttl_s: int = 24 * 3600
    history_max_messages: int = 12            # 6 question/answer pairs
    rate_limit_per_minute: int = 20           # per client IP, 0 disables

    @property
    def allowed_llm_models(self) -> list[str]:
        models = [m.strip() for m in self.llm_models.split(",") if m.strip()]
        if self.llm_model not in models:
            models.insert(0, self.llm_model)
        return models

    @property
    def fallback_models(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
