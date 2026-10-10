from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .options import ChatOptions

SESSION_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000,
                         examples=["What is the TIV threshold for referral to the chief underwriter?"])
    session_id: str | None = Field(default=None, pattern=SESSION_PATTERN,
                                   description="Reuse to continue a conversation (= LangGraph thread_id)")
    user_id: str | None = Field(default=None, pattern=SESSION_PATTERN,
                                description="Stable id of the user: enables long-term memory across sessions")
    options: ChatOptions | None = Field(default=None, description="Per-request settings (UI panel)")


class _Resume(BaseModel):
    model_config = ConfigDict(extra="forbid")       # e.g. "args" on an approve is a client bug
    session_id: str = Field(pattern=SESSION_PATTERN)


class ApproveDecision(_Resume):
    decision: Literal["approve"]


class RejectDecision(_Resume):
    decision: Literal["reject"]
    message: str | None = Field(default=None, max_length=500, description="Why it was rejected (optional)")


class EditDecision(_Resume):
    decision: Literal["edit"]
    args: dict[str, Any] = Field(description="Replacement arguments for the paused tool call. "
                                             "Validated against that tool's own Pydantic schema.")


# A discriminated union as a request body: the "decision" field selects the model,
# so {"decision": "reject", "message": ...} and {"decision": "edit", "args": {...}}
# are each validated against exactly the right shape.
ResumeDecision = Annotated[ApproveDecision | RejectDecision | EditDecision, Field(discriminator="decision")]


class ReplayRequest(BaseModel):
    checkpoint_id: str = Field(min_length=1, max_length=100)


class SourceOut(BaseModel):
    source_id: str
    doc_id: str
    filename: str
    page: int
    score: float
    text: str


class ChatResponse(BaseModel):
    status: Literal["done", "interrupted"] = "done"
    session_id: str
    answer: str = ""
    citations: list[SourceOut] = []
    sources: list[SourceOut] = []
    steps: int = 0
    trace: list[dict] = []
    usage: dict = {}
    llm_calls: int = 0
    model: str = ""
    version: str = ""
    track: str = ""
    took_ms: int = 0
    settings: dict = {}
    plan: list[str] = []
    findings: list[dict] = []
    grounded: bool | None = None
    drafts: int = 0
    interrupt: dict | None = None


class InfoResponse(BaseModel):
    service: str
    version: str
    track: str
    model: str
    fallback_models: list[str]
    prompt_version: str
    max_steps: int
    engine: str
    persistence: str
    agents: list[str]
