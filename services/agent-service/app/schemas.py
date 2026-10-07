from pydantic import BaseModel, Field

from .options import ChatOptions

SESSION_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000,
                         examples=["What is the TIV threshold for referral to the chief underwriter?"])
    session_id: str | None = Field(default=None, pattern=SESSION_PATTERN,
                                   description="Reuse to continue a conversation")
    options: ChatOptions | None = Field(default=None, description="Per-request settings (UI panel)")


class SourceOut(BaseModel):
    source_id: str
    doc_id: str
    filename: str
    page: int
    score: float
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[SourceOut]
    sources: list[SourceOut]
    steps: int
    trace: list[dict]
    usage: dict
    model: str
    version: str
    track: str
    took_ms: int
    settings: dict


class InfoResponse(BaseModel):
    service: str
    version: str
    track: str
    model: str
    fallback_models: list[str]
    prompt_version: str
    max_steps: int
