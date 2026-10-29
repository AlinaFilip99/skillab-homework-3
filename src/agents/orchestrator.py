"""Orchestrator - Supervizor care coordonează RAG Agent."""
import json
import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.llm import LLMProvider, get_llm
from src.prompts import get_prompts

from .rag import RAGAgent
from .state import OrchestratorFeedback, OrchestratorState
from .utils import extract_json

logger = logging.getLogger(__name__)
_settings = get_settings()


class Orchestrator:
    def __init__(
        self,
        llm: LLMProvider | None = None,
        max_iterations: int | None = None,
        llm_caching_enabled: bool = False,
        llm_caching_context: str = "",
    ):
        self.llm = llm or get_llm()
        self.max_iterations = max_iterations or _settings.max_iterations
        self.prompts = get_prompts()
        self.llm_caching_enabled = llm_caching_enabled
        self.llm_caching_context = llm_caching_context
        self.rag = RAGAgent(
            llm=self.llm,
            llm_caching_enabled=llm_caching_enabled,
            llm_caching_context=llm_caching_context,
        )
        self.graph = self._build_graph()

    def node_call_rag(self, state: OrchestratorState) -> dict:
        logger.info(f"[ORCH:CALL_RAG] iter {state.iteration + 1}")
        rag_result = self.rag.run(query=state.query, feedback=state.feedback)
        # Collect LLM usages from RAG agent
        new_usages = state.llm_usages + rag_result.llm_usages
        return {"rag_result": rag_result.result, "iteration": state.iteration + 1, "llm_usages": new_usages}

    def _build_messages(self, user_prompt: str) -> list[dict]:
        """Build messages list with optional caching context."""
        messages = []
        if self.llm_caching_enabled and self.llm_caching_context:
            messages.append({"role": "system", "content": self.llm_caching_context})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def node_evaluate(self, state: OrchestratorState) -> dict:
        logger.info(f"[ORCH:EVALUATE] iter {state.iteration}")

        if state.rag_result and state.rag_result.results:
            context = "\n\n".join(f"[{r.file_name}]\n{r.content}" for r in state.rag_result.results)
            max_score = state.rag_result.max_score
            avg_score = state.rag_result.avg_score
        else:
            context = "Nu am găsit informații relevante."
            max_score = avg_score = 0.0

        prompt = self.prompts.render(
            "rag_evaluate",
            query=state.query,
            context=context,
            max_score=max_score,
            avg_score=avg_score,
        )

        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)

        try:
            data = json.loads(extract_json(response.content))
            feedback = OrchestratorFeedback(
                can_answer=data.get("can_answer", False),
                missing_info=data.get("missing_info") or "",
                wrong_context=data.get("wrong_context") or "",
                suggestion=data.get("suggestion") or "",
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse evaluate response: {e}")
            feedback = OrchestratorFeedback(can_answer=False, missing_info="Failed to evaluate")

        logger.info(f"[ORCH:EVALUATE] can_answer={feedback.can_answer}")
        return {"feedback": feedback, "llm_usages": state.llm_usages + [response.usage]}

    def node_answer(self, state: OrchestratorState) -> dict:
        logger.info("[ORCH:ANSWER]")

        if state.rag_result and state.rag_result.results:
            context = "\n\n".join(f"[{r.file_name}]\n{r.content}" for r in state.rag_result.results)
        else:
            context = "Nu am găsit informații relevante."

        prompt = self.prompts.render("rag_answer", query=state.query, context=context)
        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)
        answer = response.content

        if state.feedback and state.feedback.can_answer:
            status = "success"
        elif state.rag_result and state.rag_result.results:
            status = "partial"
        else:
            status = "failed"

        return {"answer": answer, "status": status, "llm_usages": state.llm_usages + [response.usage]}

    def _should_continue(self, state: OrchestratorState) -> Literal["call_rag", "answer"]:
        if state.feedback and state.feedback.can_answer:
            return "answer"
        if state.iteration >= self.max_iterations:
            logger.info(f"[ORCH:ROUTING] Max iterations ({self.max_iterations}) reached")
            return "answer"
        return "call_rag"

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)

        graph.add_node("call_rag", self.node_call_rag)
        graph.add_node("evaluate", self.node_evaluate)
        graph.add_node("answer", self.node_answer)

        graph.add_edge(START, "call_rag")
        graph.add_edge("call_rag", "evaluate")
        graph.add_conditional_edges("evaluate", self._should_continue)
        graph.add_edge("answer", END)

        return graph.compile()

    def run(self, query: str) -> OrchestratorState:
        initial = OrchestratorState(query=query)
        result = self.graph.invoke(initial)
        return OrchestratorState.model_validate(result)
