from dataclasses import replace

import httpx
import openai
import pytest

from app.config import Settings
from app.models import LLMError, ModelHub, friendly_error, is_transient
from app.options import resolve


def test_roles_get_the_right_model_and_settings():
    pytest.importorskip("langchain_openai", exc_type=ImportError)   # e.g. tiktoken DLL blocked on Windows
    s = Settings(openrouter_api_key="k", llm_small_model="openai/gpt-4.1-nano", llm_fallback_models="x/y")
    hub = ModelHub(s)
    cfg = replace(resolve(None, s), temperature=0.7, max_tokens=321)
    writer, grader = hub.chat(cfg, "writer"), hub.chat(cfg, "grader")
    assert (writer.model_name, writer.temperature, writer.max_tokens) == ("openai/gpt-4o-mini", 0.7, 321)
    assert (grader.model_name, grader.temperature) == ("openai/gpt-4.1-nano", 0.0)
    assert hub.chat(cfg, "supervisor").model_name == "openai/gpt-4o-mini"     # decisions: main model
    assert hub.chat(cfg, "writer") is writer                                  # cached
    assert writer.openai_api_base == "https://openrouter.ai/api/v1"
    assert writer.extra_body == {"models": ["openai/gpt-4o-mini", "x/y"]}     # OpenRouter fallbacks


def test_friendly_errors_and_retry_predicate():
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    auth = openai.AuthenticationError("bad key", response=resp, body=None)
    assert friendly_error(auth).status == 401 and "API key" in str(friendly_error(auth))
    conn = openai.APIConnectionError(request=req)
    assert friendly_error(conn).status == 503
    assert is_transient(conn) and not is_transient(auth)
    same = LLMError("x", 418)
    assert friendly_error(same) is same
