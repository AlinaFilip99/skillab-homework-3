"""
Conversation Memory - Persistență în PostgreSQL.

Usage:
    from src.optimize.memory import ConversationMemory
    from src.db.database import transaction

    with transaction() as session:
        memory = ConversationMemory("user-123", session)
        memory.add("user", "Ce este o licitație?")
        memory.add("assistant", "O licitație este...")

        messages = memory.get_messages(limit=10)
"""
import logging
from typing import Literal
from uuid import uuid4

from sqlalchemy.orm import Session

from src.config import get_settings
from src.db.repositories import ConversationRepository

logger = logging.getLogger(__name__)
_settings = get_settings()

Role = Literal["user", "assistant", "system"]


class ConversationMemory:
    """
    Memory persistentă pentru conversații.

    Stochează mesajele în PostgreSQL și le recuperează per session_id.
    Necesită un SQLAlchemy session activ.
    """

    __slots__ = ("session_id", "_repo")

    def __init__(self, session_id: str | None, db_session: Session):
        """
        Args:
            session_id: ID unic pentru conversație (se generează dacă None)
            db_session: SQLAlchemy session activ
        """
        self.session_id = session_id or uuid4().hex
        self._repo = ConversationRepository(db_session)

    def add(self, role: Role, content: str) -> None:
        """Adaugă un mesaj în memory."""
        self._repo.add(self.session_id, role, content)
        logger.debug(f"[MEMORY] +{role} → {self.session_id[:8]}")

    def add_exchange(self, user_message: str, assistant_message: str) -> None:
        """Adaugă un exchange complet (user + assistant)."""
        self._repo.add(self.session_id, "user", user_message)
        self._repo.add(self.session_id, "assistant", assistant_message)

    def get_messages(self, limit: int | None = None) -> list[dict[str, str]]:
        """
        Returnează mesajele din conversație.

        Args:
            limit: Numărul maxim de mesaje recente (None = toate)

        Returns:
            Lista de mesaje [{"role": ..., "content": ...}, ...]
        """
        messages = self._repo.get_messages(self.session_id, limit)
        return [{"role": m.role, "content": m.content} for m in messages]

    def get_context(self, max_messages: int = 10) -> list[dict[str, str]]:
        """
        Returnează ultimele N mesaje pentru context LLM.

        Asigură că începe cu un mesaj user (cerință pentru majoritatea LLM-urilor).
        """
        messages = self.get_messages(limit=max_messages)

        # Sari peste mesajele non-user de la început
        while messages and messages[0]["role"] != "user":
            messages = messages[1:]

        return messages

    def count(self) -> int:
        """Numărul total de mesaje din conversație."""
        return self._repo.count(self.session_id)

    def clear(self) -> int:
        """Șterge toate mesajele din conversație."""
        deleted = self._repo.clear(self.session_id)
        logger.info(f"[MEMORY] Cleared {deleted} messages from {self.session_id[:8]}")
        return deleted


def cleanup_old_conversations(db_session: Session, days: int | None = None) -> int:
    """
    Șterge conversațiile mai vechi de N zile.

    Args:
        db_session: SQLAlchemy session
        days: Număr de zile (default: din config)

    Returns:
        Numărul de mesaje șterse
    """
    days = days or _settings.memory_ttl_days
    repo = ConversationRepository(db_session)
    deleted = repo.delete_older_than(days)
    logger.info(f"[MEMORY] Cleanup: {deleted} messages older than {days} days")
    return deleted
