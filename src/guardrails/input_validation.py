from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os

import mimetypes
import fitz  # pymupdf
import unicodedata

from PyPDF2 import PdfReader

import re
from guardrails import Guard
from guardrails_ai.toxic_language import ToxicLanguage
from guardrails_ai.nsfw_text import NSFWText
from .prompt_injection import INJECTION_PATTERNS, PromptValidator

class FilterResult(Enum):
    PASSED = "passed"
    BLOCKED_FILE = "blocked_file"
    BLOCKED_CONTENT = "blocked_content"


@dataclass
class Output:
    status: FilterResult
    text: str = ""
    metadata: dict | None = None
    reason: str = ""

MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_PAGES = 500
MAX_TEXT_LENGTH = 1_000_000        # 1M caractere

ALLOWED_TYPES = {
    "application/pdf": [".pdf"],
    "text/plain": [".txt", ".csv"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
}


def validate_file(filepath: str) -> tuple[bool, str]:
    # 1. Size check — prima barieră, cea mai ieftină.
    if os.path.getsize(filepath) > MAX_FILE_SIZE:
        return False, f"Fișier prea mare (>{MAX_FILE_SIZE // 1024 // 1024}MB)"                           

    # 2. Magic bytes — tipul REAL al fișierului, nu ce zice extensia.
    mime, _ = mimetypes.guess_type(filepath)
    if mime not in ALLOWED_TYPES:
        return False, f"Tip neacceptat: {mime}"

    # 3. Extension match — extensia trebuie să corespundă tipului real.
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ALLOWED_TYPES[mime]:
        return False, f"Extensie {ext} nu se potrivește cu {mime}"

    return True, "OK"


def safe_extract_pdf(filepath: str) -> str:
    """Extrage text cu limite — oprește bombele de decompresie."""
    doc = fitz.open(filepath)
    if doc.page_count > MAX_PAGES:
        raise ValueError(f"Prea multe pagini: {doc.page_count} > {MAX_PAGES}")

    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) > MAX_TEXT_LENGTH:    # un PDF mic poate exploda în text
            raise ValueError(f"Text prea lung (>{MAX_TEXT_LENGTH} chars) — posibil bombă")
    return text

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"]

def normalize_text(text: str) -> str:
    # 1. Normalizare unicode (NFC) — unifică reprezentări echivalente.
    text = unicodedata.normalize("NFC", text)
    # 2. Scoatem caracterele zero-width (invizibile).
    for zw in ZERO_WIDTH:
        text = text.replace(zw, "")
    return text

def detect_injection_regex(text: str) -> tuple[bool, list[str]]:
    matches = [p for p in INJECTION_PATTERNS if re.search(p, text)]
    return len(matches) > 0, matches


def sanitize_metadata(filepath: str) -> dict:
    meta = PdfReader(filepath).metadata or {}
    sanitized = {}
    for key, value in meta.items():
        clean_key = key.replace("/", "")          # PyPDF2 prefixează cu "/"
        if isinstance(value, str):
            if detect_injection_regex(value)[0]:   # câmp suspect → redactăm
                sanitized[clean_key] = "[REDACTED]"
            else:
                sanitized[clean_key] = value[:200]  # +trunchiem ca limită
    return sanitized

guard = Guard().use(
    ToxicLanguage(threshold=0.8, on_fail="exception")).use(
    NSFWText(threshold=0.8, on_fail="exception"),
)

def check_content_safety(text: str) -> tuple[bool, str | None]:
    """Întoarce (e_sigur, motiv_dacă_nu)."""
    try:
        guard.validate(text)
        return True, None
    except Exception as e:
        return False, str(e)


class ContentFilter:
    def __init__(self, use_llm: bool = False):
        self.content_validator = PromptValidator(use_llm=use_llm)

    def filter(self, filepath: str) -> Output:
        # 1. File validation
        ok, msg = validate_file(filepath)
        if not ok:
            return Output(FilterResult.BLOCKED_FILE, reason=msg)

        # 2. Safe extraction
        try:
            text = safe_extract_pdf(filepath)
        except ValueError as e:
            return Output(FilterResult.BLOCKED_FILE, reason=str(e))

        # 3. Normalize & check injection
        text = normalize_text(text)
        inj = self.content_validator.validate(text)
        if not inj.passed:
            return Output(FilterResult.BLOCKED_CONTENT, reason=f"injection: {inj.method}")

        # 4. Toxic check
        safe, reason = check_content_safety(text)
        if not safe:
            return Output(FilterResult.BLOCKED_CONTENT, reason=reason)

        # 5. Sanitize metadata & return PASSED
        return Output(FilterResult.PASSED, text=text, metadata=sanitize_metadata(filepath))


# if __name__ == "__main__":
#     import sys

#     path = sys.argv[1] if len(sys.argv) > 1 else "tests/valid_invoice.pdf"
#     result = ContentFilter().filter(path)
#     print(f"status: {result.status.value}")
#     print(f"reason: {result.reason or '—'}")
