"""Entry point simplu pentru chatul SEAP."""
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "skillab-py" / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from skillab import get_llm
from src._old.orchestrator import Orchestrator
from src._old.analyst_agent import AnalystAgent
from src._old.router import Router
from src._old.chat import Chat

# Păstrează alias-urile și variabilele de mediu din exemplul existent.
ALIASES = {"gemini": "google", "ollama": "local"}
MODEL_ENV = {
    "google": "GOOGLE_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
    "local": "OLLAMA_MODEL",
}


def make_llm():
    provider = os.getenv("LLM_PROVIDER")
    provider = ALIASES.get(provider.lower(), provider.lower()) if provider else None
    model_env = MODEL_ENV.get(provider, f"{provider.upper()}_MODEL") if provider else None
    model = os.getenv("LLM_MODEL") or (os.getenv(model_env) if model_env else None)
    return get_llm(provider=provider, model=model)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


    from src._old.cache import SmartCache
    llm = make_llm()
    cache = SmartCache()
    orchestrator = Orchestrator(llm=llm, cache=cache)

    data_dir = ROOT / "data" / "nl2sql_agent"
    tables_config = {
        "achizitii_directe": {
            "schema_path": str(data_dir / "schema_achizitii_directe.json"),
            "business_path": str(data_dir / "business_achizitii_directe.json"),
        },
        "anunturi_initiere": {
            "schema_path": str(data_dir / "schema_anunturi_initiere.json"),
            "business_path": str(data_dir / "business_anunturi_initiere.json"),
        },
    }

    # IMPORTANT: actualul analyst_agent.py atașat cere și PersistentMemory.
    # Configurează memory conform implementării existente în proiectul tău.
    from src._old.memory import PersistentMemory
    memory = PersistentMemory()
    analyst = AnalystAgent(tables_config=tables_config, memory=memory, llm=llm)

    router = Router(
        orchestrator=orchestrator,
        analyst=analyst,
        llm=llm
    )
    chat = Chat(router)
    chat.run()


if __name__ == "__main__":
    main()
