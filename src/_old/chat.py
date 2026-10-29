"""Buclă Q&A minimală, cu istoric și prompt qa_system."""
from pathlib import Path
import yaml
from jinja2 import Template
from src._old.router import Router
from skillab.prompts import PromptRegistry

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

class Chat:
    def __init__(self, router: Router):
        self.router = router
        self.prompts = PromptRegistry(str(PROMPTS_DIR))
        self.history = []

    def ask(self, question: str, session_id: str = "default") -> dict:
        # Randăm promptul cu istoricul, păstrând structura din qa_system.yaml.
        system_prompt = self.prompts.render("qa_system", history=self.history)

        result = self.router.ask(question, system_prompt, session_id=session_id)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": result["answer"]})
        return result

    def run(self):
        print("Chat SEAP — scrie 'exit' pentru a închide.")
        while True:
            question = input("\nTu: ").strip()
            if question.lower() in {"exit", "quit", "stop"}:
                break
            if not question:
                continue
            try:
                result = self.ask(question)
                print(f"\n[{result['agent']} / {result['intent']}] {result['status']}")
                print(result["answer"])
            except Exception as exc:
                print(f"Eroare: {exc}")
