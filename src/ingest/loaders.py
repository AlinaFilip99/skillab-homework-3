import logging
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from src.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

# Magic bytes signatures for file type validation
MAGIC_SIGNATURES = {
    "pdf": b"%PDF",
    "docx": b"PK\x03\x04",  # ZIP format (OOXML)
}


def validate_file(path: Path) -> None:
    """Validate file size and type before loading.

    Raises:
        ValueError: If file is too large or has invalid magic bytes.
        FileNotFoundError: If file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    max_size = _settings.max_file_size_mb * 1024 * 1024
    size = path.stat().st_size
    if size > max_size:
        raise ValueError(
            f"File too large: {size} bytes > {max_size} bytes ({_settings.max_file_size_mb} MB)"
        )

    ext = path.suffix.lower().lstrip(".")
    if ext in MAGIC_SIGNATURES:
        with open(path, "rb") as f:
            header = f.read(8)
        expected = MAGIC_SIGNATURES[ext]
        if not header.startswith(expected):
            raise ValueError(f"Invalid {ext} file: magic bytes mismatch")

LoaderFunc = Callable[[Path], list[Document]]
LOADERS: dict[str, LoaderFunc] = {}


def register_loader(*extensions: str):
    def decorator(func: LoaderFunc) -> LoaderFunc:
        for ext in extensions:
            LOADERS[ext.lower().lstrip(".")] = func
        return func
    return decorator


def get_loader(path: Path) -> LoaderFunc:
    ext = path.suffix.lower().lstrip(".")
    if ext not in LOADERS:
        raise ValueError(f"No loader for .{ext}. Supported: {list(LOADERS.keys())}")
    return LOADERS[ext]


def load(path: Path) -> list[Document]:
    validate_file(path)
    loader = get_loader(path)
    docs = loader(path)
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("file_name", path.name)
    return docs


@register_loader("txt", "md")
def load_text(path: Path) -> list[Document]:
    content = path.read_text(encoding="utf-8")
    return [Document(page_content=content, metadata={"page": 0})]


@register_loader("pdf")
def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"page": i}))
    return docs if docs else [Document(page_content="", metadata={"page": 0})]


@register_loader("docx")
def load_docx(path: Path) -> list[Document]:
    from unstructured.partition.docx import partition_docx

    elements = partition_docx(str(path))
    text = "\n\n".join(str(el) for el in elements)
    return [Document(page_content=text, metadata={"page": 0})]
