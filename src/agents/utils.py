"""Agent Utilities."""
import re


def extract_json(text: str) -> str:
    """Extrage JSON din markdown code block sau text."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return text
