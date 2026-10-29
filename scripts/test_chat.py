"""Test manual pentru agenți."""
import os
import rootutils
rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from dotenv import load_dotenv
load_dotenv()

from src.llm import get_llm
from src.agents import Orchestrator, AnalystAgent
from src.prompts import get_prompts
from src.agents import route
from src.guardrails import PromptValidator

# Alias-uri pentru provideri (gemini -> google, ollama -> local)
_PROVIDER_ALIASES = {
    "gemini": "google",
    "ollama": "local",
}

# Mapping provider -> env var pentru model
_MODEL_ENV_VARS = {
    "google": "GOOGLE_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
    "local": "OLLAMA_MODEL",
}

prompts = get_prompts()

def _resolve_provider(provider: str | None) -> str | None:
    """Rezolvă alias-uri (gemini -> google)."""
    if not provider:
        return None
    return _PROVIDER_ALIASES.get(provider.lower(), provider.lower())

def _get_model_from_env(provider: str | None) -> str | None:
    """Citește model din env var specific provider-ului."""
    resolved = _resolve_provider(provider)
    if not resolved:
        return None
    env_var = _MODEL_ENV_VARS.get(resolved, f"{resolved.upper()}_MODEL")
    return os.getenv("LLM_MODEL") or os.getenv(env_var)

# Rezolvă provider și model
LLM_PROVIDER = _resolve_provider(os.getenv("LLM_PROVIDER"))
LLM_MODEL = _get_model_from_env(os.getenv("LLM_PROVIDER"))

def ask_orchestrator(query: str):
    print("\n=== Orchestrator (RAG + LLM) ===")
    orch = Orchestrator()
    result = orch.run(query)
    print(f"  Status: {result.status}")
    return result.answer


def ask_analyst(query: str):
    print("\n=== Analyst Agent ===")
    agent = AnalystAgent(tables_config={
        "achizitii_directe": {"schema_path": "data/schemas/achizitii_directe.json"},
        "anunturi_initiere": {"schema_path": "data/schemas/anunturi_initiere.json"},
    })
    result = agent.run(query)
    print(f"  Status: {result.status}")
    print(f"  Plan: {len(result.plan)} steps")
    return result.answer

def _build_messages(self, user_prompt: str, system_prompt: str = "") -> list[dict]:
    """Build messages list."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def ask_agent(query: str, history: list):
    # Creează LLM explicit din config
    llm = get_llm(provider=LLM_PROVIDER, model=LLM_MODEL)
    print(f"Using LLM: {LLM_PROVIDER} / {llm.model}")

    system_prompt = prompts.render("qa_system", history=history)

    messages = _build_messages(query, system_prompt)
    response = llm.generate_sync(messages)
    answer = response.content

    return answer

if __name__ == "__main__":
    history = []

    print("Chat SEAP — scrie 'exit' pentru a închide.")
    while True:
        question = input("\nTu: ").strip()
        if question.lower() in {"exit", "quit", "stop"}:
            break
        if not question:
            continue
        try:

            # validate query
            v = PromptValidator(use_llm=False)
            r = v.validate(question)
            
            if not r.passed:
                print("\nPrompt injection detected")
                continue
            
            intent = route(question)

            if intent =='rag':
                result = ask_orchestrator(question)
            elif intent == 'sql':
                result = ask_analyst(question)
            else:
                result = ask_agent(question, history)

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result})
            print(f"\nAgent: {result['answer']}")
        except Exception as exc:
            print(f"Eroare: {exc}")

