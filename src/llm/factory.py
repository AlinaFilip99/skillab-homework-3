"""LLM Factory - Provider implementations."""
import json
import logging
import time

import anthropic
import httpx
from google import genai

from src.config import get_settings
from src.metrics import LLMUsage

from .base import LLMProvider, LLMResponse, Message

logger = logging.getLogger(__name__)
_settings = get_settings()

PROVIDER_ALIASES = {"gemini": "google", "local": "ollama"}


class AnthropicProvider(LLMProvider):
    """Claude via Anthropic SDK."""

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-5"

    def is_configured(self) -> bool:
        return bool(_settings.anthropic_api_key)

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                self._client = anthropic.AsyncAnthropic()
        return self._client

    def _prepare_messages(
        self,
        messages: list[Message],
        cache_system: bool = False,
    ) -> tuple[str | list[dict], list[dict]]:
        """
        Prepară mesajele pentru Anthropic API.

        Args:
            messages: Lista de mesaje
            cache_system: Dacă True, system prompt e formatat cu cache_control

        Returns:
            (system, formatted_messages)
        """
        system_text = ""
        formatted = []

        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                formatted.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # Format system pentru caching
        # TTL "1h" pentru cache de 1 oră (default e doar 5 minute)
        if cache_system and system_text:
            system = [{
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }]
        else:
            system = system_text

        return system, formatted

    async def generate(self, messages: list[Message], cache_system: bool = False) -> LLMResponse:
        client = self._get_client()
        system, msgs = self._prepare_messages(messages, cache_system)

        if cache_system:
            system_len = len(str(system)) if system else 0
            logger.info(f"[ANTHROPIC] Prompt caching enabled (1h TTL), system_len={system_len} chars")

        start_time = time.perf_counter()
        response = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system or [],
            messages=msgs,
            temperature=self.temperature,
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Extract usage metrics
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", 0)
        cache_creation_tokens = getattr(response.usage, "cache_creation_input_tokens", 0)

        # Log cache stats
        if cache_system and cache_read_tokens > 0:
            logger.debug(
                f"[CACHE] Anthropic prompt cache: "
                f"read={cache_read_tokens}, created={cache_creation_tokens}"
            )

        usage = LLMUsage(
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            latency_ms=latency_ms,
        )

        return LLMResponse(content=response.content[0].text, usage=usage)

    async def stream(self, messages: list[Message], cache_system: bool = False):
        client = self._get_client()
        system, msgs = self._prepare_messages(messages, cache_system)

        async with client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=system or [],
            messages=msgs,
            temperature=self.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text


class GoogleProvider(LLMProvider):
    """Gemini via Google GenAI SDK."""

    @property
    def name(self) -> str:
        return "google"

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    def is_configured(self) -> bool:
        return bool(_settings.google_api_key)

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                self._client = genai.Client(api_key=_settings.google_api_key)
        return self._client

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        contents = []
        system_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                text = f"{system_text}\n\n{content}" if system_text else content
                system_text = ""
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
        return contents

    async def generate(self, messages: list[Message], cache_system: bool = False) -> LLMResponse:
        # cache_system ignorat - Gemini nu suportă prompt caching în acest mod
        client = self._get_client()
        contents = self._prepare_messages(messages)

        start_time = time.perf_counter()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config={"temperature": self.temperature},
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Extract usage from usage_metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        usage = LLMUsage(
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

        return LLMResponse(content=response.text, usage=usage)

    async def stream(self, messages: list[Message], cache_system: bool = False):
        # cache_system ignorat - Gemini nu suportă prompt caching în acest mod
        client = self._get_client()
        contents = self._prepare_messages(messages)
        async for chunk in await client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config={"temperature": self.temperature},
        ):
            if chunk.text:
                yield chunk.text


class OllamaProvider(LLMProvider):
    """Local models via Ollama HTTP API."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return "llama3.2"

    @property
    def base_url(self) -> str:
        return _settings.ollama_base_url

    def is_configured(self) -> bool:
        return True

    def _get_client(self):
        with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        return self._client

    def _prepare_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in messages]

    async def generate(self, messages: list[Message], cache_system: bool = False) -> LLMResponse:
        # cache_system ignorat - Ollama nu suportă prompt caching
        client = self._get_client()

        start_time = time.perf_counter()
        response = await client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": self._prepare_messages(messages),
                "stream": False,
                "options": {"temperature": self.temperature},
            },
        )
        response.raise_for_status()
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        data = response.json()

        # Ollama returns prompt_eval_count and eval_count
        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        usage = LLMUsage(
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

        return LLMResponse(content=data["message"]["content"], usage=usage)

    async def stream(self, messages: list[Message], cache_system: bool = False):
        # cache_system ignorat - Ollama nu suportă prompt caching
        client = self._get_client()
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": self.model,
                "messages": self._prepare_messages(messages),
                "stream": True,
                "options": {"temperature": self.temperature},
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if content := data.get("message", {}).get("content"):
                        yield content


_PROVIDERS = {
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "ollama": OllamaProvider,
}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> LLMProvider:
    """Get LLM provider instance."""
    provider = provider or _settings.llm_provider
    provider = PROVIDER_ALIASES.get(provider, provider)

    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(_PROVIDERS.keys())}")

    model = model or _settings.llm_model
    instance = _PROVIDERS[provider](model=model, temperature=temperature)

    if not instance.is_configured():
        raise ValueError(f"Provider '{provider}' not configured.")

    logger.info(f"Created: {instance}")
    return instance
