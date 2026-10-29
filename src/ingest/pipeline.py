import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document

from src.config import get_settings
from src.db.database import transaction
from src.db.models import ExtractedContract, ExtractedInvoice
from src.db.repositories import (
    DocumentChunkRepository,
    ExtractedContractRepository,
    ExtractedInvoiceRepository,
)
from src.rag.service import RAGService

from .chunking import ChunkConfig, chunk, get_config, validate_chunk
from .extraction import ExtractionService
from .loaders import load
from .preprocessors import preprocess
from .schemas import SCHEMA_REGISTRY

logger = logging.getLogger(__name__)
_settings = get_settings()


def compute_content_hash(text: str) -> str:
    """Compute MD5 hash of content for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()

# Checkpoint step types
StepType = Literal["loaded", "preprocessed", "extracted", "chunked", "stored"]
CHECKPOINT_DIR = Path(".checkpoints")


@dataclass
class PipelineCheckpoint:
    """State checkpoint for pipeline resumption."""
    file_name: str
    step: StepType
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineCheckpoint":
        return cls(**d)


class CheckpointManager:
    """Manages pipeline checkpoints for resumable ingestion."""

    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _get_checkpoint_path(self, file_name: str, step: StepType) -> Path:
        safe_name = file_name.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"{safe_name}_{step}.json"

    def save(
        self,
        file_name: str,
        step: StepType,
        data: dict[str, Any],
    ) -> PipelineCheckpoint:
        """Save a checkpoint for a specific step."""
        checkpoint = PipelineCheckpoint(
            file_name=file_name,
            step=step,
            timestamp=datetime.now().isoformat(),
            data=data,
        )
        path = self._get_checkpoint_path(file_name, step)
        path.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2))
        logger.debug(f"Checkpoint saved: {file_name} @ {step}")
        return checkpoint

    def load(self, file_name: str, step: StepType) -> PipelineCheckpoint | None:
        """Load a checkpoint if it exists."""
        path = self._get_checkpoint_path(file_name, step)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return PipelineCheckpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid checkpoint {path}: {e}")
            return None

    def get_latest_step(self, file_name: str) -> StepType | None:
        """Get the latest completed step for a file."""
        steps: list[StepType] = ["stored", "chunked", "extracted", "preprocessed", "loaded"]
        for step in steps:
            if self.load(file_name, step):
                return step
        return None

    def cleanup(self, file_name: str) -> int:
        """Remove all checkpoints for a file."""
        steps: list[StepType] = ["loaded", "preprocessed", "extracted", "chunked", "stored"]
        removed = 0
        for step in steps:
            path = self._get_checkpoint_path(file_name, step)
            if path.exists():
                path.unlink()
                removed += 1
        return removed

    def cleanup_all(self) -> int:
        """Remove all checkpoints."""
        removed = 0
        for path in self.checkpoint_dir.glob("*.json"):
            path.unlink()
            removed += 1
        return removed

    def cleanup_expired(self, max_age_hours: int | None = None) -> int:
        """Remove checkpoints older than TTL."""
        max_age = max_age_hours if max_age_hours is not None else _settings.checkpoint_ttl_hours
        cutoff = datetime.now() - timedelta(hours=max_age)
        removed = 0
        for path in self.checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                timestamp = datetime.fromisoformat(data["timestamp"])
                if timestamp < cutoff:
                    path.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError):
                path.unlink()  # Remove corrupt checkpoints
                removed += 1
        return removed

    def list_checkpoints(self) -> list[PipelineCheckpoint]:
        """List all checkpoints."""
        checkpoints = []
        for path in self.checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                checkpoints.append(PipelineCheckpoint.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        return checkpoints

    def get_status(self, file_name: str) -> dict[str, Any]:
        """Get checkpoint status for a file."""
        steps: list[StepType] = ["loaded", "preprocessed", "extracted", "chunked", "stored"]
        status: dict[str, Any] = {"file_name": file_name, "steps": {}}
        for step in steps:
            cp = self.load(file_name, step)
            if cp:
                status["steps"][step] = {
                    "timestamp": cp.timestamp,
                    "has_data": bool(cp.data),
                }
        status["latest_step"] = self.get_latest_step(file_name)
        return status


@dataclass
class IngestResult:
    file_name: str
    success: bool
    chunks: int = 0
    chars: int = 0
    pages: int = 0
    doc_type: str = "default"
    extracted_data: dict[str, Any] | None = None
    error: str | None = None
    chunks_data: list[dict] | None = None  # Populated in dry_run mode

    def __repr__(self) -> str:
        if self.success:
            extra = " + extracted" if self.extracted_data else ""
            dry_run = " [DRY RUN]" if self.chunks_data is not None else ""
            return f"IngestResult({self.file_name}: {self.chunks} chunks, {self.chars} chars{extra}{dry_run})"
        return f"IngestResult({self.file_name}: FAILED - {self.error})"


@dataclass
class BatchResult:
    results: dict[str, IngestResult]

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results.values() if r.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results.values() if not r.success)

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.results.values() if r.success)

    @property
    def total_chars(self) -> int:
        return sum(r.chars for r in self.results.values() if r.success)

    def __repr__(self) -> str:
        return f"BatchResult({self.success_count} ok, {self.failed_count} failed, {self.total_chunks} chunks)"


class DocumentPipeline:

    def __init__(
        self,
        doc_type: str = "default",
        config: ChunkConfig | None = None,
        use_checkpoints: bool = False,
        keep_checkpoints: bool = True,
        extract_schema: str | None = None,
        dry_run: bool = False,
        generate_embeddings: bool = False,
    ):
        self.doc_type = doc_type
        self.config = config or get_config(doc_type)
        self.use_checkpoints = use_checkpoints
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_mgr = CheckpointManager() if use_checkpoints else None
        self.extract_schema = extract_schema
        self.extractor = ExtractionService() if extract_schema else None
        self.dry_run = dry_run
        self.generate_embeddings = generate_embeddings

    def _apply_preprocessing(self, docs: list[Document]) -> list[Document]:
        if not self.config.preprocessors:
            return docs

        processed = []
        for doc in docs:
            text = preprocess(doc.page_content, self.config.preprocessors)
            processed.append(Document(page_content=text, metadata=doc.metadata.copy()))
        return processed

    def _add_chunk_metadata(self, chunks: list[Document], file_name: str) -> list[Document]:
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["chunk_total"] = total
            c.metadata["doc_type"] = self.doc_type
            # Add provenance metadata
            c.metadata["page"] = c.metadata.get("page", 0)
            c.metadata["source_file"] = file_name
        return chunks

    def _to_db_format(self, chunks: list[Document], file_name: str) -> list[dict]:
        return [
            {
                "content": c.page_content,
                "chunk_type": "paragraph",
                "summary": "",
                "metadata_json": json.dumps({
                    k: v for k, v in c.metadata.items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }),
                "content_hash": compute_content_hash(c.page_content),
            }
            for c in chunks
        ]

    def _store_extracted(
        self,
        session,
        file_name: str,
        data: dict[str, Any],
    ) -> None:
        """Store extracted data to appropriate table."""
        if self.extract_schema == "invoice":
            repo = ExtractedInvoiceRepository(session)
            # Delete existing extraction for this file
            repo.delete_by_file(file_name)
            # Create new record
            invoice = ExtractedInvoice(
                file_name=file_name,
                numar=data.get("numar"),
                data=data.get("data"),
                furnizor=data.get("furnizor"),
                client=data.get("client"),
                produse_json=json.dumps(data.get("produse", []), ensure_ascii=False),
                subtotal=data.get("subtotal"),
                tva=data.get("tva"),
                total=data.get("total"),
            )
            repo.add(invoice)
            logger.info(f"Stored extracted invoice for {file_name}")

        elif self.extract_schema == "contract":
            repo = ExtractedContractRepository(session)
            # Delete existing extraction for this file
            repo.delete_by_file(file_name)
            # Create new record
            contract = ExtractedContract(
                file_name=file_name,
                numar=data.get("numar"),
                data_incheiere=data.get("data_incheiere"),
                prestator=data.get("prestator"),
                beneficiar=data.get("beneficiar"),
                valoare=data.get("valoare"),
                durata_luni=data.get("durata_luni"),
                obligatii_json=json.dumps(data.get("obligatii_prestator", []), ensure_ascii=False),
            )
            repo.add(contract)
            logger.info(f"Stored extracted contract for {file_name}")

    def _serialize_docs(self, docs: list[Document]) -> list[dict]:
        """Serialize documents for checkpoint storage."""
        return [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]

    def _deserialize_docs(self, data: list[dict]) -> list[Document]:
        """Deserialize documents from checkpoint."""
        return [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in data
        ]

    def ingest(self, path: str | Path, resume: bool = False) -> IngestResult:
        path = Path(path)
        file_name = path.name

        if not path.exists():
            return IngestResult(file_name, False, error="File not found")

        try:
            docs: list[Document] = []
            chunks: list[Document] = []
            pages = 0
            extracted_data: dict[str, Any] | None = None
            start_step: StepType | None = None

            # Check for resume from checkpoint
            if resume and self.checkpoint_mgr:
                latest = self.checkpoint_mgr.get_latest_step(file_name)
                if latest:
                    start_step = latest
                    logger.info(f"Resuming {file_name} from step: {latest}")

            # Step 1: Load
            if start_step in (None,):
                docs = load(path)
                pages = len(docs)

                if not docs or not any(d.page_content.strip() for d in docs):
                    return IngestResult(file_name, False, error="No content extracted")

                if self.checkpoint_mgr:
                    self.checkpoint_mgr.save(file_name, "loaded", {
                        "docs": self._serialize_docs(docs),
                        "pages": pages,
                    })
            elif start_step == "loaded":
                cp = self.checkpoint_mgr.load(file_name, "loaded")  # type: ignore
                docs = self._deserialize_docs(cp.data["docs"])  # type: ignore
                pages = cp.data["pages"]  # type: ignore

            # Step 2: Preprocess
            if start_step in (None, "loaded"):
                docs = self._apply_preprocessing(docs)

                if self.checkpoint_mgr:
                    self.checkpoint_mgr.save(file_name, "preprocessed", {
                        "docs": self._serialize_docs(docs),
                        "pages": pages,
                    })
            elif start_step == "preprocessed":
                cp = self.checkpoint_mgr.load(file_name, "preprocessed")  # type: ignore
                docs = self._deserialize_docs(cp.data["docs"])  # type: ignore
                pages = cp.data["pages"]  # type: ignore

            # Step 3: Extract (optional - only if extract_schema is set)
            if self.extractor and self.extract_schema:
                if start_step in (None, "loaded", "preprocessed"):
                    # Combine all document text for extraction
                    full_text = "\n\n".join(d.page_content for d in docs)
                    extracted_data = self.extractor.extract(full_text, self.extract_schema)

                    if self.checkpoint_mgr:
                        self.checkpoint_mgr.save(file_name, "extracted", {
                            "docs": self._serialize_docs(docs),
                            "pages": pages,
                            "extracted_data": extracted_data,
                        })
                elif start_step == "extracted":
                    cp = self.checkpoint_mgr.load(file_name, "extracted")  # type: ignore
                    docs = self._deserialize_docs(cp.data["docs"])  # type: ignore
                    pages = cp.data["pages"]  # type: ignore
                    extracted_data = cp.data.get("extracted_data")  # type: ignore

            # Step 4: Chunk
            if start_step in (None, "loaded", "preprocessed", "extracted"):
                chunks = chunk(docs, self.config)
                chunks = self._add_chunk_metadata(chunks, file_name)
                # Filter out invalid chunks
                original_count = len(chunks)
                chunks = [c for c in chunks if validate_chunk(c.page_content)]
                if len(chunks) < original_count:
                    logger.debug(
                        f"Filtered {original_count - len(chunks)} invalid chunks"
                    )

                if self.checkpoint_mgr:
                    self.checkpoint_mgr.save(file_name, "chunked", {
                        "chunks": self._serialize_docs(chunks),
                        "pages": pages,
                        "extracted_data": extracted_data,
                    })
            elif start_step == "chunked":
                cp = self.checkpoint_mgr.load(file_name, "chunked")  # type: ignore
                chunks = self._deserialize_docs(cp.data["chunks"])  # type: ignore
                pages = cp.data["pages"]  # type: ignore
                extracted_data = cp.data.get("extracted_data")  # type: ignore

            # Step 5: Store (skip in dry_run mode)
            total_chars = sum(len(c.page_content) for c in chunks)
            db_chunks = self._to_db_format(chunks, file_name)

            if self.dry_run:
                if self.generate_embeddings:
                    from src.rag.service import get_embedding_model
                    texts = [c["content"] for c in db_chunks]
                    embeddings = get_embedding_model().encode(texts, normalize_embeddings=True)
                    for db_chunk, emb in zip(db_chunks, embeddings):
                        db_chunk["embedding"] = emb.tolist()

                extra_info = f" + extracted {self.extract_schema}" if extracted_data else ""
                logger.info(f"[DRY RUN] Would store {len(db_chunks)} chunks for {file_name}{extra_info}")
                return IngestResult(
                    file_name=file_name,
                    success=True,
                    chunks=len(db_chunks),
                    chars=total_chars,
                    pages=pages,
                    doc_type=self.doc_type,
                    extracted_data=extracted_data,
                    chunks_data=db_chunks,
                )

            with transaction() as session:
                rag = RAGService(session)
                deleted = rag.delete_file(file_name)
                if deleted:
                    logger.debug(f"Replaced {deleted} existing chunks")
                count = rag.add_chunks_batch(file_name, db_chunks)

                # Store extracted data if available
                if extracted_data:
                    self._store_extracted(session, file_name, extracted_data)

            if self.checkpoint_mgr:
                self.checkpoint_mgr.save(file_name, "stored", {
                    "chunks": count,
                    "chars": total_chars,
                    "pages": pages,
                    "extracted_data": extracted_data,
                })

            extra_info = f" + extracted {self.extract_schema}" if extracted_data else ""
            logger.info(f"Ingested {file_name}: {count} chunks, {total_chars} chars{extra_info}")

            # Cleanup checkpoints after success (unless keep_checkpoints is True)
            if self.checkpoint_mgr and not self.keep_checkpoints:
                self.checkpoint_mgr.cleanup(file_name)

            return IngestResult(
                file_name=file_name,
                success=True,
                chunks=count,
                chars=total_chars,
                pages=pages,
                doc_type=self.doc_type,
                extracted_data=extracted_data,
            )

        except Exception as e:
            logger.exception(f"Failed to ingest {file_name}")
            return IngestResult(file_name, False, error=str(e))

    def ingest_batch(self, paths: list[str | Path]) -> BatchResult:
        results = {}
        for p in paths:
            path = Path(p)
            results[path.name] = self.ingest(path)
        return BatchResult(results)

    def ingest_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
    ) -> BatchResult:
        directory = Path(directory)
        extensions = extensions or ["pdf", "txt", "md", "docx"]

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = []
        for ext in extensions:
            ext = ext.lstrip(".")
            files.extend(directory.glob(f"*.{ext}"))

        return self.ingest_batch(files)


def ingest(
    path: str | Path,
    doc_type: str = "default",
    config: ChunkConfig | None = None,
    extract_schema: str | None = None,
) -> IngestResult:
    pipeline = DocumentPipeline(doc_type, config, extract_schema=extract_schema)
    return pipeline.ingest(path)


def ingest_batch(
    paths: list[str | Path],
    doc_type: str = "default",
    config: ChunkConfig | None = None,
    extract_schema: str | None = None,
) -> BatchResult:
    pipeline = DocumentPipeline(doc_type, config, extract_schema=extract_schema)
    return pipeline.ingest_batch(paths)


def ingest_directory(
    directory: str | Path,
    doc_type: str = "default",
    extensions: list[str] | None = None,
    config: ChunkConfig | None = None,
    extract_schema: str | None = None,
) -> BatchResult:
    pipeline = DocumentPipeline(doc_type, config, extract_schema=extract_schema)
    return pipeline.ingest_directory(directory, extensions)


def ingest_dry_run(
    path: str | Path,
    doc_type: str = "default",
    config: ChunkConfig | None = None,
    extract_schema: str | None = None,
    generate_embeddings: bool = False,
) -> IngestResult:
    """Ingest without writing to DB. Returns chunks for inspection."""
    pipeline = DocumentPipeline(
        doc_type,
        config,
        extract_schema=extract_schema,
        dry_run=True,
        generate_embeddings=generate_embeddings,
    )
    return pipeline.ingest(path)
