"""Prompt Template."""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from jinja2 import Environment, meta

_jinja_env = Environment()


@lru_cache(maxsize=128)
def _compile_template(prompt: str):
    return _jinja_env.from_string(prompt)


@dataclass(frozen=True)
class PromptTemplate:
    """Immutable prompt template from YAML."""

    name: str
    version: str
    prompt: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.prompt:
            raise ValueError("PromptTemplate requires name and prompt")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        known = {"name", "version", "prompt", "description"}
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            prompt=data.get("prompt", ""),
            description=data.get("description", ""),
            metadata={k: v for k, v in data.items() if k not in known},
        )

    def render(self, **variables) -> str:
        return _compile_template(self.prompt).render(**variables)

    def get_variables(self) -> set[str]:
        return meta.find_undeclared_variables(_jinja_env.parse(self.prompt))
