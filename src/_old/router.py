"""Rutare simplă: clasifică întrebarea și apelează agentul potrivit."""
from pathlib import Path
import joblib

from src._old.state import OrchestratorState
from src._old.orchestrator import Orchestrator
from src._old.analyst_agent import AnalystAgent
from skillab import get_llm
from skillab.llm.base import LLMProvider


class Router:
    def __init__(self, orchestrator: Orchestrator, analyst: AnalystAgent, llm: LLMProvider | None = None):
        self.orchestrator = orchestrator
        self.analyst = analyst
        self.classifier = joblib.load('intent_classifier.joblib')
        # Construim graful o singură dată.
        self.rag_app = orchestrator.build_graph()
        self.llm = llm or get_llm()

    def _build_messages(self, user_prompt: str, system_prompt: str) -> list[dict]:
        """Build messages list with optional caching context."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def ask(self, question: str, system_prompt: str, session_id: str = "default") -> dict:
        intent = self.classifier.predict([question])[0]

        if intent in ("search", "extract"):
            result = self.rag_app.invoke(OrchestratorState(query=question))
            return {
                "intent": intent,
                "agent": "orchestrator",
                "status": result.get("status", "unknown"),
                "answer": result.get("answer", ""),
            }

        if intent in ("analyze", "report"):
            # În implementarea atașată, AnalystAgent.chat primește session_id.
            try:
                result = self.analyst.chat(question, session_id=session_id)
            except TypeError:
                # Compatibil și cu versiuni în care chat(question) este suficient.
                result = self.analyst.chat(question)

            if isinstance(result, dict):
                return {
                    "intent": intent,
                    "agent": "analyst",
                    "status": result.get("status", "unknown"),
                    "answer": result.get("answer", ""),
                }

            return {
                "intent": intent,
                "agent": "analyst",
                "status": getattr(result, "status", "unknown"),
                "answer": getattr(result, "answer", str(result)),
            }
        messages= self._build_messages(question, system_prompt)
        result = self.llm.generate_sync(messages)

        return {
            "intent": intent,
            "agent": "default",
            "status": "default",
            "answer": result or '',
        }
