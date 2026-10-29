"""
RAG Agent - Graf pentru căutare în pgvector

Flow:
    START → refine → search → END

- refine: dacă are feedback, rafinează query-ul
- search: caută în pgvector (COMPLET)
"""
import json
import logging
import re
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry
from src._old.cache import SmartCache

from src._old.state import (
    RAGAgentState,
    RAGSearchResult,
    SearchResultItem,
    RefinedQuery,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class RAGAgentConfig:
    top_k: int = 5
    default_threshold: float = 0.25


class RAGAgent:
    """
    Agent pentru căutare în pgvector.

    Flow:
        refine → search

    Dacă primește feedback de la Orchestrator, rafinează query-ul
    înainte de a căuta.
    """

    def __init__(
        self,
        llm: LLMProvider,
        cache: SmartCache,
        config: RAGAgentConfig | None = None,
        llm_caching_enabled: bool = False,
        llm_caching_context: str = "",
    ):
        self.llm = llm
        self.cache = cache
        self.llm_caching_enabled = llm_caching_enabled
        self.llm_caching_context = llm_caching_context
        self.config = config or RAGAgentConfig()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))
        self.graph = self._build_graph()

    def _build_messages(self, user_prompt: str) -> list[dict]:
        """Build messages list with optional caching context."""
        messages = []
        if self.llm_caching_enabled and self.llm_caching_context:
            messages.append({"role": "system", "content": self.llm_caching_context})
        messages.append({"role": "user", "content": user_prompt})
        return messages
    
    # === NODES ===

    def node_refine(self, state: RAGAgentState) -> dict:
        """Rafinează query-ul dacă avem feedback de la Orchestrator."""
        logger.info(f"[REFINE] feedback={state.feedback is not None}")

        # Prima căutare: nu avem feedback, deci păstrăm query-ul original.
        if state.feedback is None:
            return {"refined": RefinedQuery(query=state.query)}

        # 1. Construim un rezumat al rezultatelor căutării precedente.
        result = state.result

        if result and result.results:
            found_summary = "\n".join(
                f"- {item.file_name} (score={item.score:.3f}): "
                f"{item.summary or item.content[:300]}"
                for item in result.results
            )
            max_score = result.max_score
            avg_score = result.avg_score
        else:
            found_summary = "Nu au fost găsite rezultate relevante."
            max_score = 0.0
            avg_score = 0.0

        # 2. Extragem feedback-ul, indiferent dacă este dict sau obiect.
        feedback = state.feedback

        def get_feedback_value(name, default=None):
            if isinstance(feedback, dict):
                return feedback.get(name, default)
            return getattr(feedback, name, default)

        can_answer = get_feedback_value("can_answer", False)
        missing_info = get_feedback_value("missing_info", "")
        suggestion = get_feedback_value("suggestion", "")

        # 3. Construim promptul folosind valorile cerute de rag_refine.yaml.
        prompt = self.prompts.render(
            "rag_refine",
            original_query=state.query,
            current_query=state.current_query,
            found_summary=found_summary,
            max_score=max_score,
            avg_score=avg_score,
            current_threshold=state.current_threshold
            or self.config.default_threshold,
            can_answer=can_answer,
            missing_info=missing_info,
            suggestion=suggestion,
        )

        # 4. Apelăm LLM-ul.
        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)

        # Acceptăm atât un răspuns text, cât și un obiect cu atributul content.
        response_text = (
            response if isinstance(response, str)
            else response.content
        )

        # 5. Eliminăm eventualele delimitatoare Markdown din JSON.
        response_text = response_text.strip()
        response_text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            response_text,
            flags=re.IGNORECASE,
        ).strip()

        # 6. Validăm răspunsul cu modelul Pydantic.
        refined = RefinedQuery.model_validate_json(response_text)

        logger.info(
            "[REFINE] query='%s', threshold=%s",
            refined.query,
            refined.threshold,
        )

        # 7. Returnăm actualizarea stării grafului.
        return {"refined": refined}


    def node_search(self, state: RAGAgentState) -> dict:
        """Caută chunks similare în pgvector. COMPLET - nu modifica."""
        from src._old.database import transaction
        from src._old.rag_service import RAGService

        query = state.current_query
        threshold = state.current_threshold or self.config.default_threshold

        logger.info(f"[SEARCH] '{query}' (top_k={self.config.top_k}, threshold={threshold})")

        # 1. CHECK CACHE — întrebare similară deja răspunsă?
        cached = self.cache.get(query)
        if cached is not None:
            return {
                "result": json.loads(cached)
                }                        # HIT · $0 · ~10ms
        
        # 2. CALL LLM — doar la MISS
        with transaction() as db:
            rag = RAGService(db)
            results = rag.search(query, top_k=self.config.top_k, threshold=threshold)

        # Transformă în SearchResultItem
        items = [
            SearchResultItem(
                content=chunk.content,
                summary=chunk.summary or "",
                file_name=chunk.file_name,
                score=score,
            )
            for chunk, score in results
        ]

        # Calculează statistici
        scores = [item.score for item in items]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        result = RAGSearchResult(
            query_used=query,
            results=items,
            max_score=max_score,
            avg_score=avg_score,
        )

        # 3. SAVE CACHE — pentru viitoare query-uri similare
        self.cache.set(query, json.dumps(result))
        
        return {
            "result": result
        }

    # === GRAPH ===

    def _build_graph(self):
        """Construiește graful RAG Agent."""
        graph = StateGraph(RAGAgentState)

        graph.add_node("refine", self.node_refine)
        graph.add_node("search", self.node_search)

        graph.add_edge(START, "refine")
        graph.add_edge("refine", "search")
        graph.add_edge("search", END)

        return graph.compile()

    def run(self, query: str, feedback=None) -> RAGAgentState:
        """Execută agentul."""
        initial = RAGAgentState(query=query, feedback=feedback)
        return self.graph.invoke(initial)
