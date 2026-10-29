"""Prompt Registry."""
import logging
from functools import lru_cache
from pathlib import Path

import yaml

from .template import PromptTemplate

logger = logging.getLogger(__name__)


class PromptRegistry:
    """Load and manage YAML prompt templates."""

    def __init__(self, folder: str):
        self._templates: dict[str, PromptTemplate] = {}
        self._load_folder(folder)

    def _load_folder(self, folder: str) -> None:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Prompt folder not found: {folder}")

        for path in folder_path.rglob("*.yaml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if data and isinstance(data, dict):
                    template = PromptTemplate.from_dict(data)
                    self._templates[template.name] = template
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {len(self._templates)} prompts from {folder}")

    def get(self, name: str) -> PromptTemplate:
        if name not in self._templates:
            raise KeyError(f"Prompt '{name}' not found")
        return self._templates[name]

    def render(self, name: str, **variables) -> str:
        return self.get(name).render(**variables)

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def __len__(self) -> int:
        return len(self._templates)


@lru_cache
def get_prompts() -> PromptRegistry:
    """Get singleton prompt registry."""
    return PromptRegistry("src/prompts/library")
