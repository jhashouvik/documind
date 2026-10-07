"""Per-request settings chosen in the UI's settings panel.

The browser may ask for a different LLM, temperature, number of retrieved
passages, similarity threshold... but the SERVER decides what is allowed:
models come from an allow-list and every number has a hard upper bound.
Otherwise anyone could pick the most expensive model with 100k output tokens.
"""
from dataclasses import asdict, dataclass

from pydantic import BaseModel, Field


class ChatOptions(BaseModel):
    llm_model: str | None = Field(default=None, examples=["openai/gpt-4o-mini"])
    temperature: float | None = Field(default=None, ge=0, le=1.5)
    max_tokens: int | None = Field(default=None, ge=50)
    max_steps: int | None = Field(default=None, ge=1)
    prompt_version: str | None = Field(default=None, pattern="^v[12]$")
    top_k: int | None = Field(default=None, ge=1)
    score_threshold: float | None = Field(default=None, ge=0, le=0.95)
    embedding_model: str | None = None


@dataclass(frozen=True)
class RunConfig:
    llm_model: str
    temperature: float
    max_tokens: int
    max_steps: int
    prompt_version: str
    top_k: int
    score_threshold: float | None
    embedding_model: str | None

    def as_dict(self) -> dict:
        return asdict(self)


class OptionError(ValueError):
    pass


def resolve(opts: ChatOptions | None, s) -> RunConfig:
    o = opts or ChatOptions()
    model = o.llm_model or s.llm_model
    if model not in s.allowed_llm_models:
        raise OptionError(f"llm_model must be one of {s.allowed_llm_models}")
    if o.max_tokens is not None and o.max_tokens > s.max_tokens_limit:
        raise OptionError(f"max_tokens must be <= {s.max_tokens_limit}")
    if o.max_steps is not None and o.max_steps > s.max_steps_limit:
        raise OptionError(f"max_steps must be <= {s.max_steps_limit}")
    if o.top_k is not None and o.top_k > s.max_search_top_k:
        raise OptionError(f"top_k must be <= {s.max_search_top_k}")
    return RunConfig(
        llm_model=model,
        temperature=s.llm_temperature if o.temperature is None else o.temperature,
        max_tokens=o.max_tokens or s.llm_max_tokens,
        max_steps=o.max_steps or s.agent_max_steps,
        prompt_version=o.prompt_version or s.prompt_version,
        top_k=o.top_k or s.search_top_k,
        score_threshold=s.score_threshold if o.score_threshold is None else o.score_threshold,
        embedding_model=o.embedding_model,
    )


def agent_options(s) -> dict:
    """What the UI needs to draw the generation/retrieval controls."""
    return {
        "llm_models": s.allowed_llm_models, "default_llm_model": s.llm_model,
        "temperature": {"default": s.llm_temperature, "min": 0, "max": 1.5, "step": 0.1},
        "max_tokens": {"default": s.llm_max_tokens, "min": 100, "max": s.max_tokens_limit, "step": 50},
        "max_steps": {"default": s.agent_max_steps, "min": 1, "max": s.max_steps_limit, "step": 1},
        "top_k": {"default": s.search_top_k, "min": 1, "max": s.max_search_top_k, "step": 1},
        "prompt_versions": ["v1", "v2"], "default_prompt_version": s.prompt_version,
    }
