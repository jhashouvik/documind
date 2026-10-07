"""Configuration for knowledge-service.

Every setting can be overridden with an environment variable of the same name
in upper case (12-factor app). In Kubernetes these come from the
`knowledge-config` ConfigMap and the `documind-llm` Secret.
"""
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- identity -----------------------------------------------------------
    app_name: str = "knowledge-service"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # --- vector database (Qdrant) --------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_prefix: str = "documind"        # one collection per embedding model
    qdrant_connect_retries: int = 30           # startup: Qdrant may start after us
    qdrant_retry_delay_s: float = 2.0

    # --- embeddings via OpenRouter ----------------------------------------------
    # openrouter = real embedding models through https://openrouter.ai/api/v1/embeddings
    # hash       = tiny deterministic fake for unit tests and offline demos only
    embedding_provider: Literal["openrouter", "hash"] = "openrouter"
    openrouter_api_key: SecretStr = SecretStr("")
    embeddings_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-small"          # default
    # allow-list the UI may choose from (comma separated). Each model = own collection.
    embedding_models: str = ("openai/text-embedding-3-small,openai/text-embedding-3-large,"
                             "qwen/qwen3-embedding-0.6b")
    embed_batch_size: int = 64                 # texts per API request
    embed_timeout_s: float = 30.0
    embed_max_retries: int = 3                 # on 429 / 5xx / network errors
    hash_dim: int = 384                        # vector size of the hash provider

    # --- chunking (defaults + limits the API enforces) --------------------------------
    chunk_size: int = 800                      # characters (~200 tokens)
    chunk_overlap: int = 120
    min_chunk_size: int = 200
    max_chunk_size: int = 4000

    # --- API limits ---------------------------------------------------------------
    max_upload_mb: int = 20
    default_top_k: int = 5
    max_top_k: int = 20
    score_threshold: float = 0.25              # drop weak matches (cosine similarity)
    app_url: str = "http://documind.localtest.me"   # OpenRouter attribution header

    @property
    def allowed_embedding_models(self) -> list[str]:
        models = [m.strip() for m in self.embedding_models.split(",") if m.strip()]
        if self.embedding_model not in models:
            models.insert(0, self.embedding_model)
        return models


@lru_cache
def get_settings() -> Settings:
    return Settings()
