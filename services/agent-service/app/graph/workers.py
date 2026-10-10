"""Two workers built with LangChain's prebuilt `create_agent` (a ReAct loop:
model -> tools -> model ... until the model answers without a tool call).

Middleware is how LangChain 1.x customises that loop without rewriting it:

  ModelCallLimitMiddleware   stop a sub-agent that keeps calling the model
  HumanInTheLoopMiddleware   PAUSE before `delete_document` runs. Internally it
                             calls LangGraph's interrupt(): the whole graph stops,
                             its state is checkpointed, and the HTTP request ends.
                             A later request resumes it with the human's decision
                             (approve / reject), possibly on another pod.

Agents are compiled once per model object and cached (compiling a graph is
cheap, but not free). The cache keeps the model itself next to the agent: an
id() alone could be reused by a new object after the old one is gone.
"""
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ModelCallLimitMiddleware

from .context import AgentContext
from .prompts import ANALYST, LIBRARIAN
from .tools import calculator, delete_document, list_documents

_CACHE: dict[tuple[str, int], tuple[object, object]] = {}


def _cached(role: str, model, build):
    key = (role, id(model))
    hit = _CACHE.get(key)
    if hit is None or hit[0] is not model:
        _CACHE[key] = (model, build())
    return _CACHE[key][1]


def analyst_agent(model):
    return _cached("analyst", model, lambda: create_agent(
            model, [calculator], system_prompt=ANALYST, context_schema=AgentContext, name="analyst",
            middleware=[ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")]))


def librarian_agent(model):
    return _cached("librarian", model, lambda: create_agent(
            model, [list_documents, delete_document], system_prompt=LIBRARIAN,
            context_schema=AgentContext, name="librarian",
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={"delete_document": {"allowed_decisions": ["approve", "edit", "reject"],
                                                      "description": "Delete a document from the knowledge base"}},
                    description_prefix="The librarian wants to run"),
                ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
            ]))
