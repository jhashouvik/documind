"""LLM client for OpenRouter.

OpenRouter exposes an OpenAI-compatible API, so we use the official `openai`
SDK and only change base_url. Everything is STREAMED: the user sees words as
soon as the model produces them.

Streaming + tool calling: when the model decides to call a tool, the stream
contains no text; instead it sends the tool name and the JSON arguments in
fragments (deltas), each tagged with an `index`. We glue those fragments back
together per index. At the end of the stream we know whether this round
produced text (the final answer) or tool calls (the agent must act and ask
the model again).
"""
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

import openai

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # raw JSON string produced by the model

    def as_message(self) -> dict:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class LLMEvent:
    kind: Literal["content", "end"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None
    model: str | None = None
    finish_reason: str | None = None


class LLMError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class LLMClient(Protocol):
    model: str

    def stream_chat(self, messages: list[dict], tools: list[dict] | None, model: str | None = None,
                    temperature: float | None = None, max_tokens: int | None = None
                    ) -> AsyncIterator[LLMEvent]: ...


def _friendly_error(exc: Exception) -> LLMError:
    """Translate SDK exceptions into messages that tell you what to fix."""
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
    return LLMError(f"LLM call failed: {exc}")


class OpenRouterClient:
    def __init__(self, settings, http_client=None) -> None:
        self.model = settings.llm_model
        self.fallbacks = settings.fallback_models
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._client = openai.AsyncOpenAI(
            api_key=settings.openrouter_api_key.get_secret_value() or "missing",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,   # SDK retries 429/5xx with backoff
            default_headers={"HTTP-Referer": settings.app_url, "X-Title": "DocuMind"},
            http_client=http_client,             # injectable for tests
        )

    async def stream_chat(self, messages: list[dict], tools: list[dict] | None, model: str | None = None,
                          temperature: float | None = None, max_tokens: int | None = None
                          ) -> AsyncIterator[LLMEvent]:
        model = model or self.model
        kwargs: dict = {"model": model, "messages": messages, "stream": True,
                        "temperature": self.temperature if temperature is None else temperature,
                        "max_tokens": max_tokens or self.max_tokens,
                        "stream_options": {"include_usage": True}}
        if tools:
            kwargs["tools"] = tools
        fallbacks = [f for f in self.fallbacks if f != model]
        if fallbacks:
            # OpenRouter-specific: try these models if the primary one fails
            kwargs["extra_body"] = {"models": [model, *fallbacks]}

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            acc: dict[int, dict] = {}
            usage = None
            finish = None
            async for chunk in stream:
                model = chunk.model or model
                if chunk.usage:
                    usage = {"prompt_tokens": chunk.usage.prompt_tokens,
                             "completion_tokens": chunk.usage.completion_tokens}
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish = choice.finish_reason or finish
                delta = choice.delta
                if delta.content:
                    yield LLMEvent("content", text=delta.content)
                for tc in delta.tool_calls or []:
                    slot = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name and not slot["name"]:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        except openai.OpenAIError as exc:
            raise _friendly_error(exc) from exc

        calls = [ToolCall(id=s["id"] or f"call_{i}", name=s["name"], arguments=s["arguments"] or "{}")
                 for i, s in sorted(acc.items())]
        yield LLMEvent("end", tool_calls=calls, usage=usage, model=model, finish_reason=finish)


def parse_arguments(raw: str) -> dict:
    """Models occasionally emit invalid JSON; the caller turns this into an
    error message the model can read and correct on the next round."""
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return value
