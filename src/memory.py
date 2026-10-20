from repositories import ChatMessageRepository
from database import transaction

# ── Manager: API curat pentru restul aplicației. Nu știe de SQL.
class PersistentMemory:
    def __init__(self, window: int = 10):
        self.window = window

    def load_messages(self, session_id: str) -> list[dict]:
        with transaction() as db:
            rows = ChatMessageRepository(db).latest(session_id, self.window)
            return [{"role": m.role, "content": m.content} for m in rows]

    def save_message(self, session_id: str, role: str, content: str) -> None:
        with transaction() as db:     # tranzacție atomică
            ChatMessageRepository(db).add(session_id, role, content)
