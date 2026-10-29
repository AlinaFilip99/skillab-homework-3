"""Metrics Models - Dataclasses for tracking LLM usage and chat metrics."""
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """LLM usage metrics from a single generation call."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "latency_ms": self.latency_ms,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class GuardrailsMetrics:
    """Guardrails check metrics."""

    checked: bool = False
    safe: bool = True
    blocked_category: str | None = None

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "safe": self.safe,
            "blocked_category": self.blocked_category,
        }


@dataclass(frozen=True, slots=True)
class CacheMetrics:
    """Cache hit/miss metrics."""

    semantic_hit: bool = False
    prompt_cache_hit: bool = False

    def to_dict(self) -> dict:
        return {
            "semantic_hit": self.semantic_hit,
            "prompt_cache_hit": self.prompt_cache_hit,
        }


@dataclass(frozen=True, slots=True)
class MemoryMetrics:
    """Conversation memory metrics."""

    messages_count: int = 0
    session_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "messages_count": self.messages_count,
            "session_id": self.session_id,
        }


@dataclass(frozen=True, slots=True)
class AgentMetrics:
    """Agent execution metrics."""

    agent_type: str | None = None
    iterations: int = 0
    sql_query: str | None = None
    retry_count: int = 0
    steps_count: int = 0

    def to_dict(self) -> dict:
        return {
            "type": self.agent_type,
            "iterations": self.iterations,
            "sql_query": self.sql_query,
            "retry_count": self.retry_count,
            "steps_count": self.steps_count,
        }


@dataclass(slots=True)
class ChatMetrics:
    """Aggregated metrics for a complete chat response."""

    llm: list[LLMUsage] = field(default_factory=list)
    guardrails: GuardrailsMetrics = field(default_factory=GuardrailsMetrics)
    cache: CacheMetrics = field(default_factory=CacheMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    agent: AgentMetrics = field(default_factory=AgentMetrics)

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.llm)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.llm)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_latency_ms(self) -> int:
        return sum(u.latency_ms for u in self.llm)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(u.cache_read_tokens for u in self.llm)

    def add_llm_usage(self, usage: LLMUsage) -> None:
        """Add LLM usage to the metrics."""
        self.llm.append(usage)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Get primary LLM info (first call or aggregate)
        primary_llm = self.llm[0] if self.llm else None

        return {
            "llm": {
                "provider": primary_llm.provider if primary_llm else None,
                "model": primary_llm.model if primary_llm else None,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cache_read_tokens": self.total_cache_read_tokens,
                "latency_ms": self.total_latency_ms,
                "calls_count": len(self.llm),
            },
            "guardrails": self.guardrails.to_dict(),
            "cache": self.cache.to_dict(),
            "memory": self.memory.to_dict(),
            "agent": self.agent.to_dict(),
        }
