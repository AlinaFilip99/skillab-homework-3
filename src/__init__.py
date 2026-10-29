"""
Document Analyst - Multi-agent system for SEAP public procurement.

Usage:
    from app import ChatSession

    session = ChatSession()
    response = session.ask("Ce este o licitație deschisă?")
    print(response.answer)

CLI:
    python -m src.main
"""
# from src.chat import ChatSession, ChatResponse
from src.config import Settings, get_settings

__all__ = [
    # "ChatSession",
    # "ChatResponse",
    "Settings",
    "get_settings",
]
