"""
Structured Extraction Service using native LLM SDKs.

Extracts structured data from document text using LLM providers
with their native structured output capabilities.
"""
import json
import logging
from typing import Any

import anthropic
import httpx
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.config import get_settings

from .schemas import SCHEMA_REGISTRY, get_schema

logger = logging.getLogger(__name__)
_settings = get_settings()

EXTRACTION_PROMPT = """Extract structured data from the following document.
Return ONLY valid JSON matching the required schema. Do not include any explanation.

Document:
{text}

Extract the data according to the schema and return as JSON."""


class ExtractionService:
    """Service for extracting structured data from documents using LLMs."""

    def __init__(self, provider: str | None = None):
        self.provider = provider or _settings.llm_provider
        self._client = None

    def _get_anthropic_client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic()

    def _get_google_client(self) -> genai.Client:
        return genai.Client(api_key=_settings.google_api_key)

    def _get_ollama_client(self) -> httpx.Client:
        return httpx.Client(base_url=_settings.ollama_base_url, timeout=120.0)

    def _schema_to_json_schema(self, schema: type[BaseModel]) -> dict:
        """Convert Pydantic model to JSON schema for LLM."""
        return schema.model_json_schema()

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((anthropic.RateLimitError, httpx.HTTPStatusError)),
    )
    def _extract_anthropic(self, text: str, schema: type[BaseModel]) -> dict[str, Any]:
        """Extract using Anthropic's tool use for structured output."""
        client = self._get_anthropic_client()
        json_schema = self._schema_to_json_schema(schema)

        # Use tool use for structured extraction
        tool = {
            "name": "extract_data",
            "description": f"Extract {schema.__name__} data from the document",
            "input_schema": json_schema,
        }

        # Use extraction_model if set, otherwise fall back to llm_model
        model = _settings.extraction_model or _settings.llm_model or "claude-sonnet-4-5"

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            tools=[tool],
            tool_choice={"type": "tool", "name": "extract_data"},
            messages=[
                {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}
            ],
        )

        # Extract tool use result
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_data":
                return block.input

        raise ValueError("No structured output returned from Anthropic")

    def _extract_google(self, text: str, schema: type[BaseModel]) -> dict[str, Any]:
        """Extract using Google Gemini's structured output."""
        client = self._get_google_client()

        response = client.models.generate_content(
            model=_settings.llm_model or "gemini-2.5-flash",
            contents=EXTRACTION_PROMPT.format(text=text),
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )

        return json.loads(response.text)

    def _extract_ollama(self, text: str, schema: type[BaseModel]) -> dict[str, Any]:
        """Extract using Ollama's JSON mode."""
        client = self._get_ollama_client()
        json_schema = self._schema_to_json_schema(schema)

        # Build prompt with schema hint
        prompt = f"""{EXTRACTION_PROMPT.format(text=text)}

JSON Schema to follow:
{json.dumps(json_schema, indent=2)}

Return valid JSON only:"""

        response = client.post(
            "/api/generate",
            json={
                "model": _settings.llm_model or "llama3.2",
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            },
        )
        response.raise_for_status()
        data = response.json()

        return json.loads(data["response"])

    def extract(self, text: str, schema_type: str) -> dict[str, Any]:
        """
        Extract structured data from text.

        Args:
            text: Document text to extract from
            schema_type: Type of schema ("invoice" or "contract")

        Returns:
            Extracted data as dictionary
        """
        schema = get_schema(schema_type)
        provider = self.provider

        logger.info(f"Extracting {schema_type} using {provider}")

        if provider == "anthropic":
            result = self._extract_anthropic(text, schema)
        elif provider in ("google", "gemini"):
            result = self._extract_google(text, schema)
        elif provider in ("ollama", "local"):
            result = self._extract_ollama(text, schema)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Validate against schema
        validated = schema.model_validate(result)
        logger.info(f"Extracted and validated {schema_type}")

        return validated.model_dump()

    def extract_and_validate(self, text: str, schema_type: str) -> BaseModel:
        """
        Extract and return validated Pydantic model.

        Args:
            text: Document text to extract from
            schema_type: Type of schema ("invoice" or "contract")

        Returns:
            Validated Pydantic model instance
        """
        schema = get_schema(schema_type)
        data = self.extract(text, schema_type)
        return schema.model_validate(data)
