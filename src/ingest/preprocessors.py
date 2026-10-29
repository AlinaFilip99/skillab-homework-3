import re
import unicodedata
from typing import Callable

PreprocessorFunc = Callable[[str], str]
PREPROCESSORS: dict[str, PreprocessorFunc] = {}


def register_preprocessor(name: str):
    def decorator(func: PreprocessorFunc) -> PreprocessorFunc:
        PREPROCESSORS[name] = func
        return func
    return decorator


def get_preprocessor(name: str) -> PreprocessorFunc:
    if name not in PREPROCESSORS:
        raise ValueError(f"Unknown preprocessor: {name}. Available: {list(PREPROCESSORS.keys())}")
    return PREPROCESSORS[name]


def preprocess(text: str, preprocessor_names: list[str]) -> str:
    for name in preprocessor_names:
        func = get_preprocessor(name)
        text = func(text)
    return text


@register_preprocessor("normalize_whitespace")
def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@register_preprocessor("normalize_diacritics")
def normalize_diacritics(text: str) -> str:
    replacements = {
        "ş": "ș", "Ş": "Ș",
        "ţ": "ț", "Ţ": "Ț",
        "ã": "ă", "Ã": "Ă",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


@register_preprocessor("fix_ocr_artifacts")
def fix_ocr_artifacts(text: str) -> str:
    text = re.sub(r"[•●○■□▪▫]", "-", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[|¦│┃]", " ", text)
    return text


@register_preprocessor("remove_headers_footers")
def remove_headers_footers(text: str) -> str:
    lines = text.split("\n")
    if len(lines) < 5:
        return text

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(pagina|page|pag\.?)\s*\d+", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\d+\s*(/|din|of)\s*\d+$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^-+\s*\d+\s*-+$", stripped):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


@register_preprocessor("normalize_unicode")
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


@register_preprocessor("clean_seap")
def clean_seap(text: str) -> str:
    text = normalize_whitespace(text)
    text = normalize_diacritics(text)
    text = fix_ocr_artifacts(text)
    return text
