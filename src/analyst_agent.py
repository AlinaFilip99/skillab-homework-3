"""
Analyst Agent 
"""
import json
import logging
from pathlib import Path

import pandas as pd
from langgraph.graph import StateGraph, END
from skillab import get_llm
from skillab.llm.base import LLMProvider
from skillab.prompts import PromptRegistry
from skillab.tools import ToolWrapper

from state import AnalystState, QueryStep, ToolStep, StepResult
from nl2sql_agent import NL2SQLAgent
from dataclasses import is_dataclass, asdict
import re

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class AnalystAgent:
    """
    Flow:
        make_plan → execute_step (loop) → synthesize

    Plan steps:
        - QueryStep: query pe tabel via NL2SQL
        - ToolStep: apel tool din registry (join_data, filter_data)

    Rezultatele sunt stocate în state.slices[step.id] = DataFrame
    """

    def __init__(
        self,
        tables_config: dict[str, dict],
        db_url: str = "postgresql+psycopg2://demo:demo123@localhost:5433/rag_demo",
        llm: LLMProvider | None = None,
    ):
        self.db_url = db_url
        self.llm = llm or get_llm()
        self.prompts = PromptRegistry(str(PROMPTS_DIR))

        # Load table info + create NL2SQL agents
        self.tables_info = []
        self.sql_agents: dict[str, NL2SQLAgent] = {}

        for table_name, config in tables_config.items():
            schema = json.loads(Path(config["schema_path"]).read_text())
            self.tables_info.append({
                "name": table_name,
                "description": schema.get("description", ""),
                "columns": list(schema.get("columns", {}).keys()),
            })
            self.sql_agents[table_name] = NL2SQLAgent(
                table_name=table_name,
                schema_path=config["schema_path"],
                db_url=db_url,
                llm=self.llm,
            )

        # Tool catalog pentru prompt
        self.tools_catalog = ToolWrapper.to_prompt_string()

        self.graph = self._build_graph()

    # === NODES ===

    def node_make_plan(self, state: AnalystState) -> dict:
        """Create and validate an execution plan."""
        logger.info("[PLAN] %s", state.question)

        try:
            prompt = self.prompts.render(
                "analyst_plan",
                tables= self.tables_info,
                tools_catalog= self.tools_catalog,
                question= state.question,
                history= []
            )

            response = self.llm.generate_sync([{"role": "user", "content":prompt}])
            if not isinstance(response, str):
                response = str(response)

            # Accept JSON wrapped in Markdown fences.
            response = response.strip()
            response = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                response,
                flags=re.IGNORECASE,
            ).strip()

            plan_data = json.loads(response)

            if not isinstance(plan_data, dict):
                raise ValueError("The plan response must be a JSON object.")

            reasoning = plan_data.get("reasoning", "")
            raw_steps = plan_data.get("steps")

            if not isinstance(raw_steps, list):
                raise ValueError("The plan must contain a 'steps' list.")

            if not raw_steps:
                raise ValueError("The plan contains no steps.")

            steps = []
            seen_ids = set()
            available_ids = set()
            available_tables = set(self.sql_agents)

            for raw_step in raw_steps:
                if not isinstance(raw_step, dict):
                    raise ValueError("Every plan step must be a JSON object.")

                step_id = raw_step.get("id")
                action = raw_step.get("action")

                if not isinstance(step_id, str) or not step_id.strip():
                    raise ValueError("Every step must have a non-empty ID.")

                if step_id in seen_ids:
                    raise ValueError(f"Duplicate step ID: {step_id}")

                if action == "query":
                    table = raw_step.get("table")

                    if table not in available_tables:
                        raise ValueError(
                            f"Step '{step_id}' references unknown table '{table}'."
                        )

                    if not raw_step.get("sub_question"):
                        raise ValueError(
                            f"Query step '{step_id}' needs a sub_question."
                        )

                    step = QueryStep(**raw_step)

                elif action == "tool":
                    input_steps = raw_step.get("input_steps", [])

                    if not isinstance(input_steps, list) or not input_steps:
                        raise ValueError(
                            f"Tool step '{step_id}' needs input_steps."
                        )

                    # Steps may only depend on previously declared steps.
                    unknown_ids = set(input_steps) - available_ids
                    if unknown_ids:
                        raise ValueError(
                            f"Step '{step_id}' references unavailable "
                            f"input steps: {sorted(unknown_ids)}"
                        )

                    if not raw_step.get("tool_name"):
                        raise ValueError(
                            f"Tool step '{step_id}' needs a tool_name."
                        )

                    params = raw_step.get("params", {})
                    if not isinstance(params, dict):
                        raise ValueError(
                            f"Step '{step_id}' params must be an object."
                        )

                    step = ToolStep(**raw_step)

                else:
                    raise ValueError(
                        f"Unsupported action '{action}' in step '{step_id}'."
                    )

                steps.append(step)
                seen_ids.add(step_id)
                available_ids.add(step_id)

            return {
                "reasoning": reasoning,
                "plan": steps,
                "current_step": 0,
                "slices": {},
                "step_results": [],
            }

        except Exception:
            logger.exception("Failed to create execution plan")
            return {
                "reasoning": "Plan generation failed.",
                "plan": [],
                "current_step": 0,
                "slices": {},
                "step_results": [],
                "answer": "I couldn't create a valid execution plan for this question.",
                "status": "failed",
            }

    def node_execute_step(self, state: AnalystState) -> dict:
        """Execută pasul curent din plan și stochează rezultatul în slices."""
        step_idx = state.current_step
        step = state.plan[step_idx]
        logger.info(f"[EXECUTE] step {step.id}: {step.action}")

        if step.action == "query":
            result, df = self._execute_query(step)
        elif step.action == "tool":
            result, df = self._execute_tool(step, state.slices)
        else:
            result = StepResult(
                step_id=step.id,
                action=step.action,
                description=f"Unknown action: {step.action}",
                status="failed",
                error=f"Action '{step.action}' not supported",
            )
            df = None

        # Update slices dacă execuția a reușit
        new_slices = dict(state.slices)
        if df is not None:
            new_slices[step.id] = df

        return {
            "slices": new_slices,
            "step_results": state.step_results + [result],
            "current_step": step_idx + 1,
        }

    def _execute_query(self, step: QueryStep) -> tuple[StepResult, pd.DataFrame | None]:
        """Execută query pe tabel via NL2SQL agent."""
        agent = self.sql_agents.get(step.table)
        if not agent:
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=step.sub_question,
                status="failed",
                error=f"No agent for table '{step.table}'",
            ), None

        try:
            nl2sql_result = agent.run(step.sub_question)
        except Exception as e:
            logger.exception(f"NL2SQL failed for step {step.id}")
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=step.sub_question,
                status="failed",
                error=str(e),
            ), None

        if nl2sql_result.status != "success":
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=step.sub_question,
                status="failed",
                error=nl2sql_result.execution_error or nl2sql_result.validation_error,
            ), None

        return StepResult(
            step_id=step.id,
            action=step.action,
            description=step.sub_question,
            status="success",
            row_count=len(nl2sql_result.result),
        ), nl2sql_result.result

    def _execute_tool(
        self,
        step: ToolStep,
        slices: dict[str, pd.DataFrame],
    ) -> tuple[StepResult, pd.DataFrame | None]:
        """Execută un tool din registry folosind slices."""
        # Validare input_steps - verifică că ID-urile există în slices
        for input_id in step.input_steps:
            if input_id not in slices:
                return StepResult(
                    step_id=step.id,
                    action=step.action,
                    description=f"{step.tool_name}",
                    status="failed",
                    error=f"Input step '{input_id}' not found in slices",
                ), None

        # Construiește argumentele pentru tool (convenție: input_dfs + params)
        input_dfs = [slices[input_id] for input_id in step.input_steps]
        args = {"input_dfs": input_dfs, **step.params}

        # Execută tool
        try:
            result_df = ToolWrapper.call(step.tool_name, args)
        except NotImplementedError as e:
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=f"{step.tool_name}",
                status="failed",
                error=f"Tool not implemented: {e}",
            ), None
        except Exception as e:
            logger.exception(f"Tool {step.tool_name} failed")
            return StepResult(
                step_id=step.id,
                action=step.action,
                description=f"{step.tool_name}",
                status="failed",
                error=str(e),
            ), None

        return StepResult(
            step_id=step.id,
            action=step.action,
            description=f"{step.tool_name}({step.params})",
            status="success",
            row_count=len(result_df),
        ), result_df


    def node_synthesize(self, state: AnalystState) -> dict:
        """Synthesize execution results into a final answer."""
        logger.info("[SYNTHESIZE]")

        try:
            # Get the DataFrame produced by the last plan step, if available.
            final_df = None
            if state.plan:
                final_df = state.slices.get(state.plan[-1].id)

            # Keep the prompt compact and readable.
            final_data = (
                final_df.head(10).to_string(index=False)
                if final_df is not None and not final_df.empty
                else None
            )

            # Convert result objects into dictionaries for the prompt.
            results = []
            for result in state.step_results:
                if hasattr(result, "model_dump"):
                    item = result.model_dump()
                elif hasattr(result, "dict"):
                    item = result.dict()
                elif is_dataclass(result):
                    item = asdict(result)
                else:
                    item = {
                        "step_id": result.step_id,
                        "action": result.action,
                        "description": result.description,
                        "status": result.status,
                        "row_count": getattr(result, "row_count", None),
                        "error": getattr(result, "error", None),
                    }

                results.append(item)

            prompt = self.prompts.render(
                "analyst_synthesize",
                question= state.question,
                reasoning= state.reasoning or "",
                results= results,
                final_data= final_data
            )

            answer = self.llm.generate_sync([{"role": "user", "content":prompt}])

            if not isinstance(answer, str):
                answer = str(answer)

            answer = answer.strip()
            if not answer:
                raise ValueError("The LLM returned an empty answer.")

            successful_steps = sum(
                1 for result in state.step_results
                if result.status == "success"
            )

            return {
                "answer": answer,
                "status": "success" if successful_steps else "failed",
            }

        except Exception:
            logger.exception("Failed to synthesize analyst response")
            return {
                "answer": (
                    "I couldn't synthesize the results into a final answer. "
                    "Please check the execution logs."
                ),
                "status": "failed",
            }


    # === ROUTING ===

    def _route_after_plan(self, state: AnalystState) -> str:
        return "execute_step" if state.plan else "synthesize"

    def _route_after_execute(self, state: AnalystState) -> str:
        return "execute_step" if state.current_step < len(state.plan) else "synthesize"

    # === GRAPH ===

    def _build_graph(self):
        graph = StateGraph(AnalystState)

        graph.add_node("make_plan", self.node_make_plan)
        graph.add_node("execute_step", self.node_execute_step)
        graph.add_node("synthesize", self.node_synthesize)

        graph.set_entry_point("make_plan")
        graph.add_conditional_edges("make_plan", self._route_after_plan, ["execute_step", "synthesize"])
        graph.add_conditional_edges("execute_step", self._route_after_execute, ["execute_step", "synthesize"])
        graph.add_edge("synthesize", END)

        return graph.compile()

    def chat(self, question: str) -> AnalystState:
        """Execută agentul."""
        initial = AnalystState(question=question)
        return self.graph.invoke(initial)
