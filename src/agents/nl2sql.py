"""NL2SQL Agent - Transformă întrebări în SQL și execută."""
import json
import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd
import sqlparse
from langgraph.graph import END, StateGraph
from sqlalchemy import text

from src.config import get_settings
from src.db.database import transaction
from src.llm import LLMProvider, get_llm
from src.prompts import get_prompts

from .state import NL2SQLState

logger = logging.getLogger(__name__)
_settings = get_settings()

DANGEROUS_PATTERNS = [
    r";\s*DROP", r";\s*DELETE", r";\s*UPDATE", r";\s*INSERT",
    r";\s*ALTER", r";\s*CREATE", r";\s*TRUNCATE", r"--", r"/\*",
]


def _clean_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text.strip()


def _is_safe_sql(sql: str) -> tuple[bool, str]:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, sql.upper(), re.IGNORECASE):
            return False, f"Dangerous pattern: {pattern}"
    return True, ""


def _is_valid_select(sql: str) -> tuple[bool, str]:
    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            return False, "Could not parse SQL"
        stmt = parsed[0]
        if stmt.get_type() != "SELECT":
            return False, f"Only SELECT allowed, got: {stmt.get_type()}"
        return True, ""
    except Exception as e:
        return False, f"Parse error: {e}"


class NL2SQLAgent:
    def __init__(
        self,
        table_name: str,
        schema: dict | None = None,
        schema_path: str | None = None,
        llm: LLMProvider | None = None,
        max_retries: int | None = None,
        llm_caching_enabled: bool = False,
        llm_caching_context: str = "",
    ):
        self.table_name = table_name
        self.llm = llm or get_llm()
        self.max_retries = max_retries or _settings.max_retries
        self.llm_caching_enabled = llm_caching_enabled
        self.llm_caching_context = llm_caching_context
        self.prompts = get_prompts()

        if schema:
            self.schema = schema
        elif schema_path:
            self.schema = json.loads(Path(schema_path).read_text())
        else:
            self.schema = {}

        self.graph = self._build_graph()

    def _build_messages(self, user_prompt: str) -> list[dict]:
        """Build messages list with optional caching context."""
        messages = []
        if self.llm_caching_enabled and self.llm_caching_context:
            messages.append({"role": "system", "content": self.llm_caching_context})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def node_get_context(self, state: NL2SQLState) -> dict:
        return {"schema_context": self.schema, "table_name": self.table_name}

    def node_generate_sql(self, state: NL2SQLState) -> dict:
        logger.info(f"[NL2SQL:GENERATE] {state.question}")

        prompt = self.prompts.render(
            "nl2sql_generate",
            table_name=self.table_name,
            table_description=self.schema.get("description", ""),
            columns=self.schema.get("columns", {}),
            business_rules=self.schema.get("business_rules", {}),
            question=state.question,
        )

        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)
        sql = _clean_sql(response.content)

        logger.info(f"[NL2SQL:GENERATE] SQL: {sql[:100]}...")
        return {"sql_query": sql, "llm_usages": state.llm_usages + [response.usage]}

    def node_validate_sql(self, state: NL2SQLState) -> dict:
        logger.info(f"[NL2SQL:VALIDATE] {state.sql_query[:50]}...")

        is_safe, safety_error = _is_safe_sql(state.sql_query)
        if not is_safe:
            return {"is_valid": False, "validation_error": safety_error}

        is_valid, parse_error = _is_valid_select(state.sql_query)
        if not is_valid:
            return {"is_valid": False, "validation_error": parse_error}

        return {"is_valid": True, "validation_error": ""}

    def node_execute_sql(self, state: NL2SQLState) -> dict:
        logger.info("[NL2SQL:EXECUTE]")
        try:
            with transaction() as session:
                result = session.execute(text(state.sql_query))
                df = pd.DataFrame(result.mappings().all())
            logger.info(f"[NL2SQL:EXECUTE] {len(df)} rows returned")
            return {"result": df, "execution_error": "", "status": "success"}
        except Exception as e:
            logger.warning(f"[NL2SQL:EXECUTE] Error: {e}")
            return {"execution_error": str(e)}

    def node_handle_error(self, state: NL2SQLState) -> dict:
        logger.info(f"[NL2SQL:ERROR] retry {state.retry_count + 1}/{self.max_retries}")
        new_retry = state.retry_count + 1

        if new_retry >= self.max_retries:
            return {"retry_count": new_retry, "status": "failed"}

        error_msg = state.validation_error or state.execution_error

        prompt = self.prompts.render(
            "nl2sql_error",
            table_name=self.table_name,
            question=state.question,
            failed_sql=state.sql_query,
            error_message=error_msg,
            columns=self.schema.get("columns", {}),
        )

        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)
        new_sql = _clean_sql(response.content)

        logger.info(f"[NL2SQL:ERROR] Corrected SQL: {new_sql[:100]}...")
        return {
            "sql_query": new_sql,
            "retry_count": new_retry,
            "is_valid": False,
            "validation_error": "",
            "execution_error": "",
            "llm_usages": state.llm_usages + [response.usage],
        }

    def _route_after_validate(self, state: NL2SQLState) -> Literal["execute_sql", "handle_error"]:
        return "execute_sql" if state.is_valid else "handle_error"

    def _route_after_execute(self, state: NL2SQLState) -> Literal["__end__", "handle_error"]:
        return END if not state.execution_error else "handle_error"

    def _route_after_error(self, state: NL2SQLState) -> Literal["generate_sql", "__end__"]:
        return "generate_sql" if state.retry_count < self.max_retries else END

    def _build_graph(self):
        graph = StateGraph(NL2SQLState)

        graph.add_node("get_context", self.node_get_context)
        graph.add_node("generate_sql", self.node_generate_sql)
        graph.add_node("validate_sql", self.node_validate_sql)
        graph.add_node("execute_sql", self.node_execute_sql)
        graph.add_node("handle_error", self.node_handle_error)

        graph.set_entry_point("get_context")
        graph.add_edge("get_context", "generate_sql")
        graph.add_edge("generate_sql", "validate_sql")
        graph.add_conditional_edges("validate_sql", self._route_after_validate)
        graph.add_conditional_edges("execute_sql", self._route_after_execute)
        graph.add_conditional_edges("handle_error", self._route_after_error)

        return graph.compile()

    def run(self, question: str) -> NL2SQLState:
        initial = NL2SQLState(question=question, table_name=self.table_name, max_retries=self.max_retries)
        result = self.graph.invoke(initial)
        return NL2SQLState.model_validate(result)
