"""The DocuMind multi-agent system, built with LangGraph + LangChain.

    state.py      the shared state, reducers and structured-output schemas
    context.py    per-run dependencies (not persisted): models, clients, settings
    prompts.py    the system prompt of every agent
    tools.py      LangChain tools (calculator, list/delete documents)
    research.py   corrective-RAG subgraph (retrieve -> grade -> rewrite -> retrieve)
    workers.py    analyst + librarian: prebuilt `create_agent` ReAct agents
    nodes.py      supervisor, planner, writer, grounding check, memory
    builder.py    wires the nodes into the top-level StateGraph
    runner.py     runs/resumes the graph and turns its stream into SSE events
"""
