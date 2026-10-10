"""Chat models for every role in the agent graph (LangChain `BaseChatModel`s).

Each node of the graph asks the ModelHub for a model by ROLE:

    writer                       -> the model chosen in the UI, its temperature and max_tokens
    supervisor planner analyst   -> the chosen model, temperature 0 (decisions, not prose)
    librarian
    grader rewriter grounding    -> LLM_SMALL_MODEL if set (cheap "judge" calls), temperature 0
    memory

OpenRouter speaks the OpenAI API, so we use `langchain_openai.ChatOpenAI` and
only change `base_url`. It is imported lazily: unit tests inject fake models
and never need it (and it pulls in tiktoken, a compiled extension).

UsageCallback is a LangChain callback handler. Passed once in the run config,
LangChain calls it for EVERY model call anywhere in the graph, including the
nested create_agent sub-agents, which is how we count tokens per turn.
"""
import logging
import time
from typing import Any
from uuid import UUID

import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import LLMResult

from . import metrics as m

log = logging.getLogger(__name__)

PRIMARY_ROLES = {"writer", "supervisor", "planner", "analyst", "librarian"}


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def friendly_error(exc: Exception) -> LLMError:
    """Translate SDK exceptions into messages that tell you what to fix."""
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, openai.AuthenticationError):
        return LLMError("OpenRouter rejected the API key (401). Check the documind-llm Secret.", 401)
    if isinstance(exc, openai.RateLimitError):
        return LLMError("OpenRouter rate limit reached (429). Wait and retry, or use another model.", 429)
    if isinstance(exc, openai.NotFoundError):
        return LLMError("Model not found or it does not support tool calling (404). "
                        "Change LLM_MODEL to a model that lists 'tools' on openrouter.ai/models.", 404)
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 402:
        return LLMError("OpenRouter account has insufficient credits (402).", 402)
    if isinstance(exc, openai.APITimeoutError):
        return LLMError("The LLM did not answer in time (timeout).", 504)
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("Cannot reach OpenRouter. Check the pod's internet/DNS access "
                        "(egress NetworkPolicy?).", 503)
    if isinstance(exc, openai.APIStatusError):
        return LLMError(f"OpenRouter error {exc.status_code}: {exc.message}", exc.status_code)
    return LLMError(f"LLM call failed: {type(exc).__name__}: {exc}")


def is_transient(exc: Exception) -> bool:
    """LangGraph RetryPolicy predicate: retry a node only for errors that can
    go away by themselves (never for a bad key or a missing model)."""
    return isinstance(exc, (openai.APIConnectionError, openai.RateLimitError,
                            openai.InternalServerError))


class ModelHub:
    def __init__(self, settings) -> None:
        self.s = settings
        self._cache: dict[tuple, BaseChatModel] = {}

    def chat(self, cfg, role: str) -> BaseChatModel:
        if role in PRIMARY_ROLES or not self.s.llm_small_model:
            model = cfg.llm_model
        else:
            model = self.s.llm_small_model
        temperature = cfg.temperature if role == "writer" else 0.0
        max_tokens = (cfg.max_tokens if role == "writer" else
                      self.s.llm_judge_max_tokens if role == "grounding" else self.s.llm_aux_max_tokens)
        key = (model, temperature, max_tokens)
        if key not in self._cache:
            self._cache[key] = self._build(model, temperature, max_tokens)
        return self._cache[key]

    def structured(self, cfg, role: str, schema):
        """A runnable returning {"raw": AIMessage, "parsed": ..., "parsing_error": ...}
        for the Pydantic `schema`. include_raw=True: decide() validates the raw
        tool-call arguments itself (with run-time context) and can send errors
        back to the model. method="function_calling" works with every
        OpenRouter model that supports tools (json_schema mode does not)."""
        return self.chat(cfg, role).with_structured_output(schema, method="function_calling",
                                                           include_raw=True)

    def _build(self, model: str, temperature: float, max_tokens: int) -> BaseChatModel:
        from langchain_openai import ChatOpenAI   # lazy: see module docstring
        extra = {}
        fallbacks = [f for f in self.s.fallback_models if f != model]
        if fallbacks:
            extra["extra_body"] = {"models": [model, *fallbacks]}   # OpenRouter-specific
        return ChatOpenAI(
            model=model, temperature=temperature, max_tokens=max_tokens,
            api_key=self.s.openrouter_api_key.get_secret_value() or "missing",
            base_url=self.s.llm_base_url, timeout=self.s.llm_timeout_s,
            max_retries=self.s.llm_max_retries,          # SDK retries 429/5xx with backoff
            stream_usage=True,                           # token usage also when streaming
            default_headers={"HTTP-Referer": self.s.app_url, "X-Title": "DocuMind"},
            **extra)


class UsageCallback(BaseCallbackHandler):
    """Collects tokens, latency and time-to-first-token of every LLM call in one turn."""

    run_inline = True            # call us in the event loop, not in a thread pool

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.models: set[str] = set()
        self._start: dict[UUID, tuple[float, str]] = {}
        self._first_token: set[UUID] = set()

    def on_chat_model_start(self, serialized: dict, messages, *, run_id: UUID, **kw: Any) -> None:
        params = kw.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or (kw.get("metadata") or {}).get(
            "ls_model_name") or "unknown"
        self._start[run_id] = (time.perf_counter(), model)

    def on_llm_new_token(self, token: str, *, run_id: UUID, **kw: Any) -> None:
        if run_id not in self._first_token and run_id in self._start:
            self._first_token.add(run_id)
            t0, model = self._start[run_id]
            m.LLM_TTFT.labels(model).observe(time.perf_counter() - t0)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kw: Any) -> None:
        t0, model = self._start.pop(run_id, (time.perf_counter(), "unknown"))
        self._first_token.discard(run_id)
        self.calls += 1
        self.models.add(model)
        m.LLM_REQUESTS.labels(model, "ok").inc()
        m.LLM_LATENCY.labels(model).observe(time.perf_counter() - t0)
        for gens in response.generations:
            for g in gens:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                pin, pout = usage.get("input_tokens") or 0, usage.get("output_tokens") or 0
                self.prompt_tokens += pin
                self.completion_tokens += pout
                m.LLM_TOKENS.labels(model, "prompt").inc(pin)
                m.LLM_TOKENS.labels(model, "completion").inc(pout)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw: Any) -> None:
        _, model = self._start.pop(run_id, (0, "unknown"))
        m.LLM_REQUESTS.labels(model, "error").inc()

    @property
    def usage(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}
