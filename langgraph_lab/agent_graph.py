"""YOUR TASK, part 2 (day 12): build the SAME agent as a LangGraph graph.

Run the tests any time:   uv run pytest -q -k graph

LangGraph is an open-source (MIT) Python library for stateful agents. The ideas:

    STATE          the data that flows through the graph (messages and an audit list)
    NODES          plain functions that read the state and return changes to it
    EDGES          which node runs next; a CONDITIONAL edge chooses the next node
    CHECKPOINTER   saves the state after each step so a graph can pause and resume
    INTERRUPT      pauses the graph and waits for a human; resume with Command(resume=...)

Read, in this order:
  https://docs.langchain.com/oss/python/langgraph/quickstart
  https://docs.langchain.com/oss/python/langgraph/interrupts
  https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT

The picture to build:

    START -> llm_call --(model asked for tools?)--> tool_node -> llm_call ...
                      \\--(no)--> END
"""

import operator
from typing import Annotated, TypedDict

from llm import ChatModel
from tools import Tool


class AgentState(TypedDict):
    """The state. `Annotated[..., operator.add]` means new items are APPENDED, not replaced.

    Keep secrets out of state: the checkpointer stores it (Playbook ORCH-06).
    """

    messages: Annotated[list[dict], operator.add]
    audit: Annotated[list[str], operator.add]


def build_graph(model: ChatModel, tools: dict[str, Tool] | None = None):
    """Create and compile the graph.

    Hints:
      1. Write a `llm_call(state)` node: ask the model, return {"messages": [reply]}.
      2. Write a `tool_node(state)` node: run each requested tool like the plain agent did. For a tool
         with `needs_approval`, call `interrupt({...})` and only run the tool if the answer is True.
         (The node restarts from the top when resumed, so only the side effect must wait for the answer.)
      3. Write `should_continue(state)`: return "tool_node" when the last message has tool calls, else END.
      4. Build a `StateGraph(AgentState)`, add the nodes and edges, then compile with a checkpointer
         (`InMemorySaver` is fine for the lab; production needs a durable one).
    """
    raise NotImplementedError("Build and compile the graph (see the hints above).")


def start(graph, question: str, thread_id: str, max_steps: int = 10) -> dict:
    """Run until the answer, or until a human decision is needed.

    Return {"status": "done", "answer": ..., "audit": [...]} or
           {"status": "needs_approval", "request": <what the human must approve>}.
    Pass `{"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps}` as the config
    on EVERY call (Playbook ORCH-04). A paused run shows up as "__interrupt__" in the result.
    """
    raise NotImplementedError("Run the graph with a thread ID and a recursion limit.")


def resume(graph, thread_id: str, approved: bool, max_steps: int = 10) -> dict:
    """Give the human's decision to a paused graph (same thread ID) and carry on.

    Hint: invoke the graph with `Command(resume=approved)` and the same config.
    Return the same shape as `start`.
    """
    raise NotImplementedError("Resume a paused graph with the human's decision.")
