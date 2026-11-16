"""A small memory store, scoped per user. Never store secrets here (Playbook ORCH-06)."""

FORBIDDEN_WORDS = ("password", "secret", "token", "api_key")


class MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, str]] = {}

    def remember(self, user_id: str, key: str, value: str) -> None:
        if any(word in key.lower() for word in FORBIDDEN_WORDS):
            raise ValueError("Do not store secrets in memory.")
        self._data.setdefault(user_id, {})[key] = value

    def recall(self, user_id: str, key: str) -> str | None:
        return self._data.get(user_id, {}).get(key)

    def all_for(self, user_id: str) -> dict[str, str]:
        return dict(self._data.get(user_id, {}))
