"""YOUR TASK, part 1 (day 12): build the agent loop in plain Python.

Open TASK.md first. The tests in test_agents.py describe exactly what this function must do.
Run them any time:   uv run pytest -q -k plain

An agent is a loop:

    ask the model -> does it want a tool? -> run the tool -> give the result back -> repeat

The dangerous word is "repeat", so the loop needs hard limits (Playbook AG-04) and anything that
changes data needs a human yes (AG-03).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from memory import MemoryStore
from tools import build_tools, validate_arguments

THREADS: dict[str, list[dict]] = {}


@dataclass
class AgentResult:
    answer: str
    steps: int
    stopped_by: str = "answer"  # "answer", "step_limit" or "time_limit"
    audit: list[str] = field(default_factory=list)  # a record of everything the agent did (AG-11)


def _run_with_timeout(tool, arguments, timeout):
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return executor.submit(tool.run, arguments).result(timeout=timeout)
    finally:
        executor.shutdown(wait=False)  # do not wait for a stuck tool


PRICE_PER_1K_TOKENS = 0.002  # made-up price for the lab


def _estimate_tokens(messages) -> int:
    # Rough rule: about 4 characters per token. Real clients report exact usage.
    return sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or "")) for m in messages) // 4


def run_agent(
    question,
    model,
    approve,
    tools=None,
    max_steps=6,
    max_seconds=30.0,
    thread_id=None,
    user_id=None,
    memory: MemoryStore | None = None,
    tool_timeout: float = 5.0,
    max_tokens: int = 20_000,
    max_cost: float = 0.50,
) -> AgentResult:
    """Run the loop and return an AgentResult.

    Hints (read them one at a time; try before you read the next):
      1. `tools` defaults to `build_tools()` from tools.py. `messages` starts as the user's question.
      2. Each round: call `model.chat(messages, [tool.schema() for tool in tools.values()])`.
      3. If the reply has no tool calls it is the final answer: return it.
      4. For each tool call: check the tool exists, check the arguments with `validate_arguments`,
         ask `approve(tool.preview(arguments))` when `tool.needs_approval`, then run it.
         Errors and denials go back to the model as tool messages, never as exceptions.
      5. Write one line to the audit list for every call (what ran, or why it did not).
      6. Stop after `max_steps` rounds or `max_seconds` seconds and say which limit stopped you.
    """
    tools = tools or build_tools()
    messages = THREADS.setdefault(thread_id, []) if thread_id else []
    messages = THREADS.setdefault(thread_id, []) if thread_id else []
    if memory and user_id and not messages:
        facts = memory.all_for(user_id)
        if facts:
            messages.append({"role": "system", "content": f"Known about this user: {facts}"})
    messages.append({"role": "user", "content": question})
    audit = []
    started = time.monotonic()
    tokens_used = 0

    for step in range(1, max_steps + 1):
        if time.monotonic() - started > max_seconds:
            return AgentResult(answer="", steps=step - 1, stopped_by="time_limit", audit=audit)

        reply = model.chat(messages, [t.schema() for t in tools.values()])
        tokens_used += _estimate_tokens([reply])
        tokens_used += _estimate_tokens(messages)  # each call re-sends the whole history
        if tokens_used > max_tokens:
            return AgentResult(answer="", steps=step - 1, stopped_by="token_limit", audit=audit)
        if tokens_used / 1000 * PRICE_PER_1K_TOKENS > max_cost:
            return AgentResult(answer="", steps=step - 1, stopped_by="cost_limit", audit=audit)
        messages.append(reply)

        if not reply["tool_calls"]:
            return AgentResult(answer=reply["content"], steps=step, stopped_by="answer", audit=audit)

        for call in reply["tool_calls"]:
            tool = tools.get(call["name"])
            if tool is None:
                result = f"Unknown tool: {call['name']}"
                audit.append(f"{call['name']}: unknown tool")
            else:
                problem = validate_arguments(tool, call["arguments"])
                if problem:
                    result = f"Error: {problem}"
                    audit.append(f"{call['name']}: invalid arguments ({problem})")
                elif tool.needs_approval and not approve(tool.preview(call["arguments"])):
                    result = "Denied: a human did not approve this action."
                    audit.append(f"{call['name']}: denied by human")
                else:
                    try:
                        result = _run_with_timeout(tool, call["arguments"], tool_timeout)
                        audit.append(f"{call['name']}: ran with {call['arguments']}")
                    except TimeoutError:
                        result = f"Error: {call['name']} timed out after {tool_timeout}s"
                        audit.append(f"{call['name']}: timed out after {tool_timeout}s")
                    except Exception as exc:  # noqa: BLE001 - a tool crash must go back to the model, never kill the loop
                        result = f"Error: {exc}"
                        audit.append(f"{call['name']}: failed ({exc})")
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": result}
            )

    return AgentResult(answer=reply["content"], steps=step, stopped_by="step_limit", audit=audit)
