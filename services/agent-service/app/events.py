"""The streaming contract: every Server-Sent Event as a Pydantic model.

AgentEvent is a DISCRIMINATED UNION: the `type` field says which model a
payload is, so Pydantic picks the right class directly instead of trying them
one by one:

    EVENT_ADAPTER.validate_json('{"type": "token", "text": "Hi", "draft": 1}')  -> TokenEvent

The runner only ever yields these models, so a typo in a field name fails in
the tests, not in the browser. GET /v1/events/schema serves the JSON Schema
of the union (e.g. to generate TypeScript types for a frontend).
"""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .schemas import SourceOut


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def sse(self) -> str:
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"   # type: ignore[attr-defined]


class SessionEvent(_Event):
    type: Literal["session"] = "session"
    session_id: str
    version: str
    track: str
    model: str
    engine: str = "langgraph"
    mode: Literal["new", "resume", "replay"]


class StepEvent(_Event):
    """Progress of one agent. Built by emit() inside nodes and tools."""
    type: Literal["step"] = "step"
    agent: str = Field(min_length=1)
    action: str = Field(min_length=1)
    detail: str = ""
    status: Literal["ok", "error", "info"] = "ok"
    draft: int | None = Field(default=None, ge=1)


class TokenEvent(_Event):
    type: Literal["token"] = "token"
    text: str
    draft: int = Field(ge=1)


class InterruptAction(BaseModel):
    name: str
    args: dict[str, Any]
    description: str | None = None


class InterruptEvent(_Event):
    type: Literal["interrupt"] = "interrupt"
    session_id: str
    actions: list[InterruptAction] = Field(min_length=1)
    allowed: list[Literal["approve", "reject", "edit"]]


class Usage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)


class FindingOut(BaseModel):
    agent: str
    task: str
    result: str


class DoneEvent(_Event):
    type: Literal["done"] = "done"
    status: Literal["done"] = "done"
    session_id: str
    answer: str
    citations: list[SourceOut]
    sources: list[SourceOut]
    steps: int = Field(ge=0)
    usage: Usage
    llm_calls: int = Field(ge=0)
    model: str
    version: str
    track: str
    took_ms: int = Field(ge=0)
    settings: dict[str, Any]
    plan: list[str]
    findings: list[FindingOut]
    grounded: bool | None
    drafts: int = Field(ge=0)


class ErrorEvent(_Event):
    type: Literal["error"] = "error"
    message: str
    status: int | None = None


AgentEvent = Annotated[SessionEvent | StepEvent | TokenEvent | InterruptEvent | DoneEvent | ErrorEvent,
                       Field(discriminator="type")]
EVENT_ADAPTER: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
