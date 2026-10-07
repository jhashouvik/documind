"""Request/response models. FastAPI uses them for validation and to generate
the OpenAPI docs you can browse at /docs."""
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    status: str = Field(description="indexed | reindexed | duplicate")
    pages: int
    chunks: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    took_ms: int


class TextIngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200, examples=["meeting-notes.md"])
    text: str = Field(min_length=1, max_length=2_000_000)
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_model: str | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    content_type: str | None = None
    pages: int | None = None
    total_chunks: int | None = None
    size_bytes: int | None = None
    uploaded_at: str | None = None
    embedding_model: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["What is the TIV referral limit?"])
    top_k: int | None = Field(default=None, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    doc_ids: list[str] | None = Field(default=None, description="Restrict to these documents")
    embedding_model: str | None = Field(default=None, description="Which knowledge base (model) to search")


class SearchResult(BaseModel):
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    embedding_model: str
    results: list[SearchResult]
    took_ms: int


class CollectionInfo(BaseModel):
    collection: str
    points: int


class InfoResponse(BaseModel):
    service: str
    version: str
    embedding_provider: str
    default_embedding_model: str
    collections: list[CollectionInfo]
    chunk_size: int
    chunk_overlap: int


class Range(BaseModel):
    default: float
    min: float
    max: float
    step: float


class OptionsResponse(BaseModel):
    """Everything the UI needs to draw its settings controls."""
    embedding_models: list[str]
    default_embedding_model: str
    chunk_size: Range
    chunk_overlap: Range
    top_k: Range
    score_threshold: Range
