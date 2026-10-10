"""Runtime context: the dependencies of one graph run.

LangGraph separates two things:
  * STATE   - data that changes and is checkpointed (messages, sources...)
  * CONTEXT - things a run needs but that must NOT be saved: HTTP clients,
              model objects, settings. Passed as `graph.astream(..., context=ctx)`
              and read in a node as `runtime.context`, or in a tool as
              `runtime.context` via the ToolRuntime argument.
"""
from dataclasses import dataclass

from ..config import Settings
from ..knowledge_client import KnowledgeClient
from ..models import ModelHub
from ..options import RunConfig


@dataclass
class AgentContext:
    cfg: RunConfig
    settings: Settings
    models: ModelHub
    knowledge: KnowledgeClient
    user_id: str | None = None
