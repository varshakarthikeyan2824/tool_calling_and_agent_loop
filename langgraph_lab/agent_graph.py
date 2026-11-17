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

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from llm import ChatModel
from tools import Tool, build_tools, validate_arguments


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
    tools = tools or build_tools()

    def llm_call(state: AgentState) -> dict:
        reply = model.chat(state["messages"], [t.schema() for t in tools.values()])
        return {"messages": [reply]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tool_node" if last.get("tool_calls") else END

    def tool_node(state: AgentState) -> dict:
        new_messages = []
        new_audit = []
        for call in state["messages"][-1]["tool_calls"]:
            name, args = call["name"], call["arguments"]
            tool = tools.get(name)
            if tool is None:
                content, line = f"Unknown tool: {name}", f"{name}: unknown tool"
            elif problem := validate_arguments(tool, args):
                content, line = f"Error: {problem}", f"{name}: invalid arguments ({problem})"
            else:
                approved = True
                if tool.needs_approval:
                    approved = interrupt({"tool": name, "preview": tool.preview(args)})
                if not approved:
                    content, line = "Denied: a human did not approve this action.", f"{name}: denied by human"
                else:
                    try:
                        content, line = tool.run(args), f"{name}: ran with {args}"
                    except Exception as exc:  # noqa: BLE001 (tool crash must not kill the loop)
                        content, line = f"Error: {exc}", f"{name}: failed ({exc})"
            new_messages.append(
                {"role": "tool", "tool_call_id": call["id"], "name": name, "content": content}
            )
            new_audit.append(line)
        return {"messages": new_messages, "audit": new_audit}

    builder = StateGraph(AgentState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    return builder.compile(checkpointer=InMemorySaver())


def start(graph, question: str, thread_id: str, max_steps: int = 10) -> dict:
    """Run until the answer, or until a human decision is needed.

    Return {"status": "done", "answer": ..., "audit": [...]} or
           {"status": "needs_approval", "request": <what the human must approve>}.
    Pass `{"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps}` as the config
    on EVERY call (Playbook ORCH-04). A paused run shows up as "__interrupt__" in the result.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps}
    result = graph.invoke({"messages": [{"role": "user", "content": question}], "audit": []}, config)
    if "__interrupt__" in result:
        return {"status": "needs_approval", "request": result["__interrupt__"][0].value}
    return {"status": "done", "answer": result["messages"][-1]["content"], "audit": result["audit"]}


def resume(graph, thread_id: str, approved: bool, max_steps: int = 10) -> dict:
    """Give the human's decision to a paused graph (same thread ID) and carry on.

    Hint: invoke the graph with `Command(resume=approved)` and the same config.
    Return the same shape as `start`.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps}
    result = graph.invoke(Command(resume=approved), config)
    if "__interrupt__" in result:
        return {"status": "needs_approval", "request": result["__interrupt__"][0].value}
    return {"status": "done", "answer": result["messages"][-1]["content"], "audit": result["audit"]}
