"""Analyst Agent - Planificare și execuție multi-step queries."""
import json
import logging
from pathlib import Path

import pandas as pd
from langgraph.graph import END, StateGraph

from src.config import get_settings
from src.llm import LLMProvider, get_llm
from src.prompts import get_prompts
from src.tools import (
    call_tool,
    data_tools,  # noqa: F401
    get_tools_catalog,
)

from .nl2sql import NL2SQLAgent
from .state import AnalystState, QueryStep, StepResult, ToolStep
from .utils import extract_json

logger = logging.getLogger(__name__)
_settings = get_settings()


class AnalystAgent:
    def __init__(
        self,
        tables_config: dict[str, dict],
        llm: LLMProvider | None = None,
        llm_caching_enabled: bool = False,
        llm_caching_context: str = "",
    ):
        self.llm = llm or get_llm()
        self.llm_caching_enabled = llm_caching_enabled
        self.llm_caching_context = llm_caching_context
        self.prompts = get_prompts()

        self.tables_info = []
        self.sql_agents: dict[str, NL2SQLAgent] = {}

        for table_name, config in tables_config.items():
            if "schema_path" in config:
                schema = json.loads(Path(config["schema_path"]).read_text())
            else:
                schema = config.get("schema", {})

            self.tables_info.append({
                "name": table_name,
                "description": schema.get("description", ""),
                "columns": list(schema.get("columns", {}).keys()),
            })

            self.sql_agents[table_name] = NL2SQLAgent(
                table_name=table_name,
                schema=schema,
                llm=self.llm,
                llm_caching_enabled=llm_caching_enabled,
                llm_caching_context=llm_caching_context,
            )

        self.tools_catalog = get_tools_catalog()
        self.graph = self._build_graph()

    def _build_messages(self, user_prompt: str) -> list[dict]:
        """Build messages list with optional caching context."""
        messages = []
        if self.llm_caching_enabled and self.llm_caching_context:
            messages.append({"role": "system", "content": self.llm_caching_context})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def node_make_plan(self, state: AnalystState) -> dict:
        logger.info(f"[ANALYST:PLAN] {state.question}")

        prompt = self.prompts.render(
            "analyst_plan",
            tables=self.tables_info,
            tools_catalog=self.tools_catalog,
            question=state.question,
            history=[],
        )

        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)
        new_usages = state.llm_usages + [response.usage]

        try:
            data = json.loads(extract_json(response.content))
            reasoning = data.get("reasoning", "")
            steps_raw = data.get("steps", [])

            plan = []
            for s in steps_raw:
                if s.get("action") == "query":
                    plan.append(QueryStep(id=s["id"], table=s["table"], sub_question=s["sub_question"]))
                elif s.get("action") == "tool":
                    plan.append(ToolStep(
                        id=s["id"],
                        tool_name=s["tool_name"],
                        input_steps=s.get("input_steps", []),
                        params=s.get("params", {}),
                    ))

            logger.info(f"[ANALYST:PLAN] {len(plan)} steps created")
            return {"reasoning": reasoning, "plan": plan, "current_step": 0, "llm_usages": new_usages}

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse plan: {e}")
            return {"reasoning": "Failed to create plan", "plan": [], "status": "no_plan", "llm_usages": new_usages}

    def node_execute_step(self, state: AnalystState) -> dict:
        step = state.plan[state.current_step]
        logger.info(f"[ANALYST:EXECUTE] step {step.id}: {step.action}")

        new_usages = list(state.llm_usages)

        if step.action == "query":
            result, df, step_usages = self._execute_query(step)
            new_usages.extend(step_usages)
        elif step.action == "tool":
            result, df = self._execute_tool(step, state.slices)
        else:
            result = StepResult(
                step_id=step.id, action=step.action,
                description=f"Unknown: {step.action}", status="failed",
                error=f"Action '{step.action}' not supported",
            )
            df = None

        new_slices = dict(state.slices)
        if df is not None:
            new_slices[step.id] = df

        return {
            "slices": new_slices,
            "step_results": state.step_results + [result],
            "current_step": state.current_step + 1,
            "llm_usages": new_usages,
        }

    def _execute_query(self, step: QueryStep) -> tuple[StepResult, pd.DataFrame | None, list]:
        agent = self.sql_agents.get(step.table)
        if not agent:
            return StepResult(
                step_id=step.id, action=step.action, description=step.sub_question,
                status="failed", error=f"No agent for table '{step.table}'",
            ), None, []

        try:
            nl2sql_result = agent.run(step.sub_question)
        except Exception as e:
            logger.exception(f"NL2SQL failed for step {step.id}")
            return StepResult(
                step_id=step.id, action=step.action, description=step.sub_question,
                status="failed", error=str(e),
            ), None, []

        if nl2sql_result.status != "success":
            return StepResult(
                step_id=step.id, action=step.action, description=step.sub_question,
                status="failed", error=nl2sql_result.execution_error or nl2sql_result.validation_error,
            ), None, nl2sql_result.llm_usages

        return StepResult(
            step_id=step.id, action=step.action, description=step.sub_question,
            status="success", row_count=len(nl2sql_result.result),
        ), nl2sql_result.result, nl2sql_result.llm_usages

    def _execute_tool(self, step: ToolStep, slices: dict[str, pd.DataFrame]) -> tuple[StepResult, pd.DataFrame | None]:
        for input_id in step.input_steps:
            if input_id not in slices:
                return StepResult(
                    step_id=step.id, action=step.action, description=step.tool_name,
                    status="failed", error=f"Input '{input_id}' not found",
                ), None

        inputs = [slices[input_id] for input_id in step.input_steps]

        try:
            result_df = call_tool(step.tool_name, inputs=inputs, **step.params)
        except Exception as e:
            logger.exception(f"Tool {step.tool_name} failed")
            return StepResult(
                step_id=step.id, action=step.action, description=step.tool_name,
                status="failed", error=str(e),
            ), None

        return StepResult(
            step_id=step.id, action=step.action, description=f"{step.tool_name}({step.params})",
            status="success", row_count=len(result_df),
        ), result_df

    def node_synthesize(self, state: AnalystState) -> dict:
        logger.info("[ANALYST:SYNTHESIZE]")

        final_data = ""
        if state.plan and state.slices:
            last_step_id = state.plan[-1].id
            if last_step_id in state.slices:
                final_data = state.slices[last_step_id].head(10).to_string()

        prompt = self.prompts.render(
            "analyst_synthesize",
            question=state.question,
            reasoning=state.reasoning,
            results=state.step_results,
            final_data=final_data,
        )

        messages = self._build_messages(prompt)
        response = self.llm.generate_sync(messages, cache_system=self.llm_caching_enabled)
        answer = response.content

        failed_steps = [r for r in state.step_results if r.status == "failed"]
        if not state.plan:
            status = "no_plan"
        elif failed_steps:
            status = "failed"
        else:
            status = "success"

        return {"answer": answer, "status": status, "llm_usages": state.llm_usages + [response.usage]}

    def _route_after_plan(self, state: AnalystState) -> str:
        return "execute_step" if state.plan else "synthesize"

    def _route_after_execute(self, state: AnalystState) -> str:
        return "execute_step" if state.current_step < len(state.plan) else "synthesize"

    def _build_graph(self):
        graph = StateGraph(AnalystState)

        graph.add_node("make_plan", self.node_make_plan)
        graph.add_node("execute_step", self.node_execute_step)
        graph.add_node("synthesize", self.node_synthesize)

        graph.set_entry_point("make_plan")
        graph.add_conditional_edges("make_plan", self._route_after_plan)
        graph.add_conditional_edges("execute_step", self._route_after_execute)
        graph.add_edge("synthesize", END)

        return graph.compile()

    def run(self, question: str) -> AnalystState:
        initial = AnalystState(question=question)
        result = self.graph.invoke(initial)
        return AnalystState.model_validate(result)
