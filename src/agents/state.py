"""Pydantic State Models pentru agenți."""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.metrics import LLMUsage


# RAG Agent
class SearchResultItem(BaseModel):
    content: str
    summary: str
    file_name: str
    score: float


class RAGSearchResult(BaseModel):
    query_used: str
    results: list[SearchResultItem] = Field(default_factory=list)
    max_score: float = 0.0
    avg_score: float = 0.0


class RefinedQuery(BaseModel):
    query: str
    threshold: float | None = None


class OrchestratorFeedback(BaseModel):
    can_answer: bool
    missing_info: str = ""
    wrong_context: str = ""
    suggestion: str = ""


class RAGAgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    feedback: OrchestratorFeedback | None = None
    refined: RefinedQuery | None = None
    result: RAGSearchResult | None = None
    llm_usages: list[LLMUsage] = Field(default_factory=list)

    @property
    def current_query(self) -> str:
        return self.refined.query if self.refined else self.query

    @property
    def current_threshold(self) -> float | None:
        return self.refined.threshold if self.refined else None


class OrchestratorState(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    query: str
    rag_result: RAGSearchResult | None = None
    feedback: OrchestratorFeedback | None = None
    iteration: int = 0
    answer: str = ""
    status: Literal["pending", "success", "partial", "failed"] = "pending"
    llm_usages: list[LLMUsage] = Field(default_factory=list)


# NL2SQL Agent
class NL2SQLState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    table_name: str = ""
    schema_context: dict = Field(default_factory=dict)
    sql_query: str = ""
    is_valid: bool = False
    validation_error: str = ""
    result: pd.DataFrame = Field(default_factory=pd.DataFrame)
    execution_error: str = ""
    retry_count: int = 0
    max_retries: int = 2
    status: Literal["pending", "success", "failed"] = "pending"
    llm_usages: list[LLMUsage] = Field(default_factory=list)


# Analyst Agent
class QueryStep(BaseModel):
    id: str
    action: Literal["query"] = "query"
    table: str
    sub_question: str


class ToolStep(BaseModel):
    id: str
    action: Literal["tool"] = "tool"
    tool_name: str
    input_steps: list[str] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)


PlanStep = QueryStep | ToolStep


class StepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: str
    action: str
    description: str
    status: Literal["success", "failed"]
    row_count: int = 0
    error: str = ""


class AnalystState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    reasoning: str = ""
    plan: list[QueryStep | ToolStep] = Field(default_factory=list)
    current_step: int = 0
    slices: dict[str, pd.DataFrame] = Field(default_factory=dict)
    step_results: list[StepResult] = Field(default_factory=list)
    answer: str = ""
    status: Literal["pending", "success", "failed", "no_plan"] = "pending"
    llm_usages: list[LLMUsage] = Field(default_factory=list)
