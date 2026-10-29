"""Test manual pentru agenți."""
import rootutils
rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.db.database import transaction
from src.rag import RAGService
from src.agents import Orchestrator, NL2SQLAgent, AnalystAgent


def test_rag():
    print("\n=== RAG Service ===")
    with transaction() as session:
        rag = RAGService(session)
        results = rag.search("contract", top_k=3)
        for chunk, score in results:
            print(f"  [{score:.2f}] {chunk.file_name}: {chunk.content[:80]}...")


def test_orchestrator():
    print("\n=== Orchestrator (RAG + LLM) ===")
    orch = Orchestrator()
    result = orch.run("Ce documente avem despre contracte?")
    print(f"  Status: {result.status}")
    print(f"  Answer: {result.answer[:200]}...")


def test_nl2sql():
    print("\n=== NL2SQL Agent ===")
    agent = NL2SQLAgent(
        table_name="achizitii_directe",
        schema_path="data/schemas/achizitii_directe.json",
    )
    result = agent.run("Care sunt cele mai mari 5 achiziții?")
    print(f"  Status: {result.status}")
    print(f"  SQL: {result.sql_query}")
    if result.status == "success":
        print(f"  Rows: {len(result.result)}")
        print(result.result.head())


def test_analyst():
    print("\n=== Analyst Agent ===")
    agent = AnalystAgent(tables_config={
        "achizitii_directe": {"schema_path": "data/schemas/achizitii_directe.json"},
        "anunturi_initiere": {"schema_path": "data/schemas/anunturi_initiere.json"},
    })
    result = agent.run("Care sunt top 3 furnizori după valoare?")
    print(f"  Status: {result.status}")
    print(f"  Plan: {len(result.plan)} steps")
    print(f"  Answer: {result.answer[:200]}...")


if __name__ == "__main__":
    import sys

    tests = {
        "rag": test_rag,
        "orch": test_orchestrator,
        "sql": test_nl2sql,
        "analyst": test_analyst,
    }

    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name in tests:
            tests[name]()
        else:
            print(f"Unknown: {name}. Use: {list(tests.keys())}")
    else:
        print("Usage: python scripts/test_agents.py <rag|orch|sql|analyst>")
        print("\nRunning all...")
        for name, fn in tests.items():
            try:
                fn()
            except Exception as e:
                print(f"  ERROR: {e}")
