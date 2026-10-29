"""
LLM Module - Provider-agnostic LLM abstraction.

Usage:
    from src.llm import get_llm

    # Auto-detect provider from environment
    llm = get_llm()

    # Explicit provider
    llm = get_llm("anthropic")
    llm = get_llm("google", model="gemini-1.5-pro")
    llm = get_llm("ollama", model="mistral")

    # Sync (runs async internally)
    response = llm.generate_sync(messages)

    # Async
    response = await llm.generate(messages)

    # Streaming
    async for chunk in llm.stream(messages):
        print(chunk, end="")

Messages format:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
"""
from .base import LLMProvider, LLMResponse, Message
from .factory import get_llm, AnthropicProvider, GoogleProvider, OllamaProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Message",
    "get_llm",
    "AnthropicProvider",
    "GoogleProvider",
    "OllamaProvider",
]
