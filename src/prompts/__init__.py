"""Prompts Module."""
from .registry import PromptRegistry, get_prompts
from .template import PromptTemplate

__all__ = ["PromptTemplate", "PromptRegistry", "get_prompts"]
