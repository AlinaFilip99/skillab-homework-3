"""LLM Provider - Abstract interface."""
import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from src.metrics import LLMUsage

Message = dict[str, str]


@dataclass(slots=True)
class LLMResponse:
    """Response from LLM generation with usage metrics."""

    content: str
    usage: LLMUsage

    def __str__(self) -> str:
        return self.content


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    _sync_loop = None
    _sync_loop_lock = threading.Lock()

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self._model = model or self.default_model
        self.temperature = temperature
        self._client = None
        self._client_lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._model

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        pass

    @abstractmethod
    def _get_client(self):
        pass

    @abstractmethod
    async def generate(self, messages: list[Message], cache_system: bool = False) -> LLMResponse:
        """
        Generate response.

        Args:
            messages: Lista de mesaje
            cache_system: Cache system prompt (doar Anthropic, ignorat altfel)

        Returns:
            LLMResponse with content and usage metrics
        """
        pass

    @abstractmethod
    async def stream(self, messages: list[Message], cache_system: bool = False) -> AsyncIterator[str]:
        """
        Stream response.

        Args:
            messages: Lista de mesaje
            cache_system: Cache system prompt (doar Anthropic, ignorat altfel)
        """
        pass

    @classmethod
    def _get_sync_loop(cls):
        with cls._sync_loop_lock:
            if cls._sync_loop is None or cls._sync_loop.is_closed():
                cls._sync_loop = asyncio.new_event_loop()
            return cls._sync_loop

    def generate_sync(self, messages: list[Message], cache_system: bool = False) -> LLMResponse:
        """Synchronous wrapper for generate()."""
        return self._get_sync_loop().run_until_complete(self.generate(messages, cache_system))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
