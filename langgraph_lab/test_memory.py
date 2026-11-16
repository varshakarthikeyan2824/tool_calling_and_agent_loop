import pytest

from agent_plain import run_agent
from llm import final_answer
from memory import MemoryStore


def test_memory_is_scoped_per_user():
    store = MemoryStore()
    store.remember("alice", "floor", "2")
    assert store.recall("alice", "floor") == "2"
    assert store.recall("bob", "floor") is None


def test_memory_refuses_secrets():
    with pytest.raises(ValueError):
        MemoryStore().remember("alice", "api_key", "abc")


def test_agent_sees_the_users_memory_but_not_other_users():
    seen = []

    class Spy:
        def chat(self, messages, tools):
            seen.append(list(messages))
            return final_answer("ok")

    store = MemoryStore()
    store.remember("alice", "floor", "2")
    run_agent("hi", Spy(), approve=lambda p: True, user_id="alice", memory=store)
    run_agent("hi", Spy(), approve=lambda p: True, user_id="bob", memory=store)
    assert "floor" in str(seen[0])
    assert "floor" not in str(seen[1])
