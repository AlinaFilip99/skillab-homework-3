"""
NL2SQL Agent
"""
import json
import logging
import re
from pathlib import Path
from typing import Literal
from sqlalchemy import text

import pandas as pd
import sqlparse
from langgraph.graph import StateGraph, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry
from database import transaction

from state import NL2SQLState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class NL2SQLAgent:
    """
    Flow:
        get_context → generate_sql → validate → execute
                                        ↓
                                    handle_error
    """

    def __init__(
        self,
        table_name: str,
        schema_path: str,
        db_url: str = "postgresql+psycopg2://demo:demo123@localhost:5433/rag_demo",
        max_retries: int = 2,
        llm: LLMProvider | None = None,
    ):
        self.table_name = table_name
        self.db_url = db_url
        self.max_retries = max_retries
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # Load schema
        self.schema = json.loads(Path(schema_path).read_text())

        self.graph = self._build_graph()

    # === NODES ===

    def node_get_context(self, state: NL2SQLState) -> dict:
        """COMPLET."""
        return {
            "schema_context": self.schema,
            "table_name": self.table_name,
        }

    def node_generate_sql(self, state: NL2SQLState) -> dict:
        """
        Generează SQL folosind prompt-ul nl2sql_generate și LLM-ul.
        """
        logger.info(f"[GENERATE] {state.question}")

        # 1. Renderează prompt-ul "nl2sql_generate"
        prompt = self.prompts.render(
            "nl2sql_generate",
            table_name=self.table_name,
            table_description=self.schema.get("table_description", ""),
            columns=self.schema.get("columns", {}),
            business_rules=self.schema.get("business_rules", {}),
            question=state.question,
        )

        # 2. Apelează LLM
        response = self.llm.generate_sync([{"role": "user", "content":prompt}])
        sql_text = response.text if hasattr(response, "text") else str(response)

        # 3. Curăță răspunsul (elimină blocurile ```sql ... ```)
        sql_clean = re.sub(r"```sql\n?(.*?)\n?```", r"\1", sql_text, flags=re.DOTALL)
        sql_clean = re.sub(r"```\n?(.*?)\n?```", r"\1", sql_clean, flags=re.DOTALL).strip()

        # 4. Returnează query-ul SQL generat
        return {"sql_query": sql_clean}

    def node_validate_sql(self, state: NL2SQLState) -> dict:
        """
        Validează query-ul SQL (securitate, parsare și tip SELECT).
        """
        sql = state.sql_query
        logger.info(f"[VALIDATE] {sql[:50]}...")

        # 1. Check pentru SQL injection / operațiuni periculoase
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"]
        sql_upper = sql.upper()
        for keyword in dangerous_keywords:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                return {"is_valid": False, "validation_error": f"Operațiune SQL interzisă detectată: {keyword}"}

        # Verifică instrucțiuni multiple separate prin punct și virgulă
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        if len(statements) > 1:
            return {"is_valid": False, "validation_error": "Nu sunt permise instrucțiuni SQL multiple."}

        # 2. & 3. Parsează cu sqlparse și verifică că e SELECT
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return {"is_valid": False, "validation_error": "Query-ul SQL nu a putut fi parsat."}
            
            stmt = parsed[0]
            if stmt.get_type() != 'SELECT':
                return {"is_valid": False, "validation_error": f"Sunt permise doar interogări de tip SELECT, s-a primit: {stmt.get_type()}"}
        except Exception as e:
            return {"is_valid": False, "validation_error": f"Eroare de parsare SQL: {str(e)}"}

        # 4. Return status valid
        return {"is_valid": True, "validation_error": ""}

    def node_execute_sql(self, state: NL2SQLState) -> dict:
        """
        Execută query-ul SQL în baza de date și returnează rezultatul ca DataFrame.
        """
        logger.info("[EXECUTE]")

        try:
            with transaction() as session:
                result = session.execute(text(state.sql_query))
                df = pd.DataFrame(result.mappings().all())
            return {"result": df, "status": "success", "execution_error": None}
        except Exception as e:
            logger.error(f"Eroare la execuția SQL: {e}")
            return {"result": pd.DataFrame(), "execution_error": str(e), "status": "failed"}

    def node_handle_error(self, state: NL2SQLState) -> dict:
        """
        Gestionează erorile și reîncearcă generarea corectată a SQL-ului.
        """
        logger.info("[ERROR]")

        # 1. Incrementează retry_count
        new_retry_count = state.retry_count + 1

        # 2. Dacă a depășit sau egalat max_retries, oprește execuția
        if new_retry_count >= self.max_retries:
            return {"retry_count": new_retry_count, "status": "failed"}

        error_message = state.execution_error or state.validation_error or "Eroare necunoscută"

        # 3. Renderează prompt-ul "nl2sql_error"
        prompt = self.prompts.render(
            "nl2sql_error",
            table_name=self.table_name,
            question=state.question,
            failed_sql=state.sql_query,
            error_message=error_message,
            columns=self.schema.get("columns", {}),
        )

        # 4. Apelează LLM pentru SQL corectat
        response = self.llm.generate_sync([{"role": "user", "content":prompt}])
        sql_text = response.text if hasattr(response, "text") else str(response)

        # Curăță răspunsul
        sql_clean = re.sub(r"```sql\n?(.*?)\n?```", r"\1", sql_text, flags=re.DOTALL)
        sql_clean = re.sub(r"```\n?(.*?)\n?```", r"\1", sql_clean, flags=re.DOTALL).strip()

        # 5. Returnează noul SQL și numărul actualizat de reîncercări
        return {
            "sql_query": sql_clean,
            "retry_count": new_retry_count,
            "execution_error": None,
            "validation_error": None,
            "is_valid": True,
        }

    # === ROUTING ===

    def _route_after_validate(self, state: NL2SQLState) -> str:
        return "execute_sql" if state.is_valid else "handle_error"

    def _route_after_execute(self, state: NL2SQLState) -> str:
        return END if not state.execution_error else "handle_error"

    def _route_after_error(self, state: NL2SQLState) -> str:
        return "generate_sql" if state.retry_count < self.max_retries else END

    # === GRAPH ===

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
        graph.add_conditional_edges("validate_sql", self._route_after_validate, ["execute_sql", "handle_error"])
        graph.add_conditional_edges("execute_sql", self._route_after_execute, [END, "handle_error"])
        graph.add_conditional_edges("handle_error", self._route_after_error, ["generate_sql", END])

        return graph.compile()

    def run(self, question: str) -> NL2SQLState:
        """Execută agentul."""
        initial = NL2SQLState(
            question=question,
            table_name=self.table_name,
            max_retries=self.max_retries,
        )
        return self.graph.invoke(initial)
