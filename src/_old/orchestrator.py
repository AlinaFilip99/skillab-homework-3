"""
Orchestrator - Supervizor care coordonează RAG Agent

Flow:
    call_rag → evaluate ──┬──→ answer → END
         ↑                │
         │   can_answer   │
         │   = false      │
         └────────────────┘

"""
import logging
import re
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, START, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry

from src._old.state import OrchestratorState, OrchestratorFeedback
from src._old.rag_agent import RAGAgent, RAGAgentConfig
from src._old.cache import SmartCache

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class OrchestratorConfig:
    max_iterations: int = 3
    min_score: float = 0.25


class Orchestrator:
    """
    Supervizor care coordonează RAG Agent.

    Flow:
        1. Apelează RAG Agent pentru căutare
        2. Evaluează: pot răspunde cu aceste chunks?
        3. Dacă DA → generează răspuns
        4. Dacă NU → trimite feedback la RAG Agent, repeat
    """

    def __init__(
        self,
        cache: SmartCache | None = None,
        config: OrchestratorConfig | None = None,
        llm: LLMProvider | None = None,
        llm_caching_enabled: bool = False,
        llm_caching_context: str = "",
    ):
        self.config = config or OrchestratorConfig()
        self.llm = llm or get_llm()
        self.llm_caching_enabled = llm_caching_enabled
        self.llm_caching_context = llm_caching_context
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # RAG Agent - graf separat
        rag_config = RAGAgentConfig()
        rag_config.default_threshold = self.config.min_score
        self.rag = RAGAgent(
            self.llm, 
            rag_config,
            cache,
            llm_caching_enabled=llm_caching_enabled,
            llm_caching_context=llm_caching_context,)

    def _build_messages(self, user_prompt: str) -> list[dict]:
        """Build messages list with optional caching context."""
        messages = []
        if self.llm_caching_enabled and self.llm_caching_context:
            messages.append({"role": "system", "content": self.llm_caching_context})
        messages.append({"role": "user", "content": user_prompt})
        return messages
    
    # === NODES ===

    def node_call_rag(self, state: OrchestratorState) -> dict:
        """Apelează RAG Agent pentru căutare. COMPLET - nu modifica."""
        logger.info(f"[CALL_RAG] iter {state.iteration + 1}")

        # Apelează RAG Agent cu query și feedback (dacă există)
        rag_result = self.rag.run(
            query=state.query,
            feedback=state.feedback,  # None prima dată
        )

        return {
            "rag_result": rag_result["result"],
            "iteration": state.iteration + 1,
        }

    def node_evaluate(self, state: OrchestratorState) -> dict:
        """Evaluează dacă contextul RAG este suficient pentru a răspunde."""
        logger.info(f"[EVALUATE] iter {state.iteration}")

        # 1. Construim contextul din rezultatele RAG.
        results = state.rag_result.results if state.rag_result else []

        context = "\n\n".join(
            f"[{r.file_name}]\n{r.content}"
            for r in results
        )

        # 2. Pregătim promptul de evaluare.
        rag_result = state.rag_result

        prompt = self.prompts.render(
            "rag_evaluate",
            query=state.query,
            context=context or "Nu au fost găsite rezultate relevante.",
            max_score=rag_result.max_score if rag_result else 0.0,
            avg_score=rag_result.avg_score if rag_result else 0.0,
        )

        # 3. Apelăm LLM-ul.
        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)

        # 4. Extragem răspunsul text.
        if isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            response_text = response.get("content", "")
        else:
            response_text = getattr(response, "content", str(response))

        response_text = response_text.strip()

        # Eliminăm delimitatoarele Markdown, dacă există.
        match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            response_text = match.group(1).strip()

        # 5. Validăm JSON-ul folosind modelul Pydantic.
        feedback = OrchestratorFeedback.model_validate_json(response_text)

        logger.info(
            "[EVALUATE] can_answer=%s, missing_info=%s",
            feedback.can_answer,
            feedback.missing_info,
        )

        return {"feedback": feedback}

    def node_answer(self, state: OrchestratorState) -> dict:
        """Generează răspunsul final pe baza contextului RAG."""
        logger.info("[ANSWER]")

        # 1. Construim contextul din rezultatele RAG.
        results = state.rag_result.results if state.rag_result else []

        context = "\n\n".join(
            f"[{r.file_name}]\n{r.content}"
            for r in results
        )

        # Dacă nu există rezultate, nu inventăm context.
        if not context.strip():
            return {
                "answer": (
                    "Nu am găsit suficiente informații relevante "
                    "pentru a răspunde la întrebare."
                ),
                "status": "failed",
            }

        # 2. Pregătim promptul de răspuns.
        prompt = self.prompts.render(
            "rag_answer",
            query=state.query,
            context=context,
        )

        # 3. Apelăm LLM-ul.
        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)

        # 4. Extragem răspunsul text.
        if isinstance(response, str):
            answer = response
        elif isinstance(response, dict):
            answer = response.get("content", "")
        else:
            answer = getattr(response, "content", str(response))

        answer = answer.strip()

        # 5. Determinăm statusul rezultatului.
        if not answer:
            status = "failed"
        elif state.feedback and state.feedback.can_answer:
            status = "success"
        else:
            # Există un răspuns, dar evaluarea nu l-a confirmat
            # ca fiind suficient de bine susținut.
            status = "partial"

        logger.info("[ANSWER] status=%s", status)

        return {
            "answer": answer,
            "status": status,
        }

    # === ROUTING ===

    def _should_continue(self, state: OrchestratorState) -> Literal["call_rag", "answer"]:
        """Decide dacă continuăm căutarea sau răspundem."""
        if state.feedback and state.feedback.can_answer:
            return "answer"
        if state.iteration >= self.config.max_iterations:
            logger.info(f"[ROUTING] Max iterations ({self.config.max_iterations}) reached")
            return "answer"
        return "call_rag"

    # === GRAPH ===

    def build_graph(self):
        """Construiește graful Orchestrator."""
        graph = StateGraph(OrchestratorState)

        graph.add_node("call_rag", self.node_call_rag)
        graph.add_node("evaluate", self.node_evaluate)
        graph.add_node("answer", self.node_answer)

        graph.add_edge(START, "call_rag")
        graph.add_edge("call_rag", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {"call_rag": "call_rag", "answer": "answer"}
        )
        graph.add_edge("answer", END)

        return graph.compile()
