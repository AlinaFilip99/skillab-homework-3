"""RAG Agent - Graf pentru căutare în pgvector."""
import json
import logging

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.db.database import transaction
from src.llm import LLMProvider, get_llm
from src.prompts import get_prompts
from src.rag import RAGService

from .state import RAGAgentState, RAGSearchResult, RefinedQuery, SearchResultItem
from .utils import extract_json

logger = logging.getLogger(__name__)
_settings = get_settings()


class RAGAgent:
    def __init__(
        self,
        llm: LLMProvider | None = None,
        top_k: int | None = None,
        default_threshold: float | None = None,
        llm_caching_enabled: bool = False,
        llm_caching_context: str = "",
    ):
        self.llm = llm or get_llm()
        self.top_k = top_k or _settings.rag_top_k
        self.default_threshold = default_threshold or _settings.rag_threshold
        self.llm_caching_enabled = llm_caching_enabled
        self.llm_caching_context = llm_caching_context
        self.prompts = get_prompts()
        self.graph = self._build_graph()

    def _build_messages(self, user_prompt: str) -> list[dict]:
        """Build messages list with optional caching context."""
        messages = []
        if self.llm_caching_enabled and self.llm_caching_context:
            messages.append({"role": "system", "content": self.llm_caching_context})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def node_refine(self, state: RAGAgentState) -> dict:
        logger.info(f"[RAG:REFINE] feedback={state.feedback is not None}")

        if state.feedback is None:
            return {"refined": RefinedQuery(query=state.query)}

        found_summary = "Nimic găsit."
        if state.result and state.result.results:
            summaries = [
                f"- [{r.file_name}] (score={r.score:.2f}): {r.content[:100]}..."
                for r in state.result.results[:3]
            ]
            found_summary = "\n".join(summaries)

        prompt = self.prompts.render(
            "rag_refine",
            original_query=state.query,
            current_query=state.current_query,
            found_summary=found_summary,
            max_score=state.result.max_score if state.result else 0,
            avg_score=state.result.avg_score if state.result else 0,
            current_threshold=state.current_threshold or self.default_threshold,
            can_answer=state.feedback.can_answer,
            missing_info=state.feedback.missing_info,
            suggestion=state.feedback.suggestion,
        )

        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)

        try:
            data = json.loads(extract_json(response.content))
            refined = RefinedQuery(
                query=data.get("query", state.query),
                threshold=data.get("threshold"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse refine response: {e}")
            refined = RefinedQuery(query=state.query)

        logger.info(f"[RAG:REFINE] refined query: {refined.query}")
        return {"refined": refined, "llm_usages": state.llm_usages + [response.usage]}

    def node_search(self, state: RAGAgentState) -> dict:
        query = state.current_query
        threshold = state.current_threshold or self.default_threshold

        logger.info(f"[RAG:SEARCH] '{query}' (top_k={self.top_k}, threshold={threshold})")

        with transaction() as session:
            rag = RAGService(session)
            results = rag.search(query, top_k=self.top_k, threshold=threshold)
            items = [
                SearchResultItem(
                    content=chunk.content,
                    summary=chunk.summary or "",
                    file_name=chunk.file_name,
                    score=score,
                )
                for chunk, score in results
            ]

        scores = [item.score for item in items]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(f"[RAG:SEARCH] found {len(items)} chunks (max={max_score:.2f})")

        return {
            "result": RAGSearchResult(
                query_used=query,
                results=items,
                max_score=max_score,
                avg_score=avg_score,
            )
        }

    def _build_graph(self):
        graph = StateGraph(RAGAgentState)
        graph.add_node("refine", self.node_refine)
        graph.add_node("search", self.node_search)
        graph.add_edge(START, "refine")
        graph.add_edge("refine", "search")
        graph.add_edge("search", END)
        return graph.compile()

    def run(self, query: str, feedback=None) -> RAGAgentState:
        initial = RAGAgentState(query=query, feedback=feedback)
        result = self.graph.invoke(initial)
        return RAGAgentState.model_validate(result)
