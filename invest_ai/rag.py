"""Docling PDF ingestion and local Qdrant retrieval for FCA reference material."""

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

logger = logging.getLogger("LBG-InvestAI")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "rag_sources"
DEFAULT_QDRANT_DIR = PROJECT_ROOT / "data" / "qdrant"
DEFAULT_COLLECTION = "fca_guidance"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
POINT_NAMESPACE = uuid.UUID("6bf7295f-df8f-4a7c-afb4-12d08a031f85")


class RagIndexNotReadyError(RuntimeError):
    """Raised when retrieval is requested before source PDFs are indexed."""


@dataclass(frozen=True)
class ChunkRecord:
    """Text chunk plus provenance used in Qdrant payloads and citations."""

    text: str
    source_id: str
    document_title: str
    publication: str
    source_url: str
    page_numbers: List[int]
    section: str
    element_types: List[str]
    chunk_index: int

    def payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_id": self.source_id,
            "document_title": self.document_title,
            "publication": self.publication,
            "source_url": self.source_url,
            "page_numbers": self.page_numbers,
            "section": self.section,
            "element_types": self.element_types,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class RetrievedPassage:
    """A retrieved chunk with stable citation metadata."""

    citation_id: str
    text: str
    document_title: str
    publication: str
    source_url: str
    page_numbers: List[int]
    section: str
    score: float

    def prompt_block(self) -> str:
        pages = ", ".join(str(page) for page in self.page_numbers) or "unknown"
        section = f" — {self.section}" if self.section else ""
        return (
            f"[{self.citation_id}] {self.document_title} ({self.publication}), "
            f"page(s) {pages}{section}\n{self.text}"
        )

    def citation(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "document_title": self.document_title,
            "publication": self.publication,
            "source_url": self.source_url,
            "page_numbers": self.page_numbers,
            "section": self.section,
            "score": round(self.score, 4),
        }


SOURCE_CATALOG: dict[str, dict[str, str]] = {
    "fca-consumer-duty-fg22-5.pdf": {
        "title": "Final non-Handbook Guidance for firms on the Consumer Duty",
        "publication": "FCA FG22/5 (July 2022)",
        "url": "https://www.fca.org.uk/publication/finalised-guidance/fg22-5.pdf",
    },
    "fca-vulnerable-customers-fg21-1.pdf": {
        "title": "Guidance for firms on the fair treatment of vulnerable customers",
        "publication": "FCA FG21/1 (February 2021)",
        "url": "https://www.fca.org.uk/publication/finalised-guidance/fg21-1.pdf",
    },
    "fca-pension-transfer-suitability-fg21-3.pdf": {
        "title": "Advising on pension transfers",
        "publication": "FCA FG21/3 (March 2021)",
        "url": "https://www.fca.org.uk/publication/finalised-guidance/fg21-3.pdf",
    },
}


def _chunk_to_record(chunk: Any, source_path: Path, chunk_index: int) -> ChunkRecord:
    source = SOURCE_CATALOG.get(
        source_path.name,
        {
            "title": source_path.stem.replace("-", " ").title(),
            "publication": "Uncatalogued PDF",
            "url": "",
        },
    )
    metadata = chunk.meta
    pages = sorted(
        {
            int(provenance.page_no)
            for item in metadata.doc_items
            for provenance in item.prov
            if provenance.page_no is not None
        }
    )
    element_types = sorted(
        {
            str(item.label.value)
            for item in metadata.doc_items
            if getattr(item, "label", None) is not None
        }
    )
    headings = [str(heading) for heading in (metadata.headings or []) if heading]
    content = str(chunk.text).strip()
    if not content:
        raise ValueError(
            f"Docling returned an empty chunk for {source_path.name} "
            f"(chunk {chunk_index})."
        )

    return ChunkRecord(
        text=content,
        source_id=source_path.name,
        document_title=source["title"],
        publication=source["publication"],
        source_url=source["url"],
        page_numbers=pages,
        section=" > ".join(headings),
        element_types=element_types,
        chunk_index=chunk_index,
    )


class QdrantRagIndex:
    """Persistent local Qdrant index using local FastEmbed embeddings."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.storage_path = Path(
            storage_path
            or os.getenv("QDRANT_PATH", str(DEFAULT_QDRANT_DIR))
        ).expanduser().resolve()
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", DEFAULT_COLLECTION
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.client = QdrantClient(path=str(self.storage_path))
        self._embedder: Optional[TextEmbedding] = None
        self._vector_size: Optional[int] = None

    def _ensure_embedder_and_collection(self) -> None:
        if self._embedder is None:
            self._embedder = TextEmbedding(model_name=self.embedding_model_name)
            probe_vector = next(iter(self._embedder.embed(["embedding size probe"])))
            self._vector_size = len(probe_vector)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_size or 0,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            collection = self.client.get_collection(self.collection_name)
            vectors = collection.config.params.vectors
            if not isinstance(vectors, models.VectorParams):
                raise RuntimeError(
                    f"Qdrant collection '{self.collection_name}' uses an unsupported "
                    "named-vector configuration."
                )
            if vectors.size != self._vector_size:
                raise RuntimeError(
                    f"Vector size mismatch for collection '{self.collection_name}': "
                    f"stored={vectors.size}, current={self._vector_size}. Use a new "
                    "collection name or rebuild the local index."
                )

    def index_records(
        self, records: Iterable[ChunkRecord], batch_size: int = 32
    ) -> int:
        """Embed and upsert a document's chunks, replacing its prior indexed copy."""
        record_list = list(records)
        if not record_list:
            raise ValueError("No non-empty PDF chunks were produced for indexing.")

        self._ensure_embedder_and_collection()
        source_id = record_list[0].source_id
        if any(record.source_id != source_id for record in record_list):
            raise ValueError("One index_records call must contain a single source PDF.")

        vectors = list(
            self._embedder.embed(
                [record.text for record in record_list], batch_size=batch_size
            )
        )
        if len(vectors) != len(record_list):
            raise RuntimeError("Embedding model returned an unexpected vector count.")

        source_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_id",
                    match=models.MatchValue(value=source_id),
                )
            ]
        )
        points = [
            models.PointStruct(
                id=str(
                    uuid.uuid5(
                        POINT_NAMESPACE,
                        f"{record.source_id}:{record.chunk_index}:{hashlib.sha256(record.text.encode()).hexdigest()}",
                    )
                ),
                vector=vector.tolist(),
                payload=record.payload(),
            )
            for record, vector in zip(record_list, vectors)
        ]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=source_filter),
            wait=True,
        )
        for offset in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[offset : offset + batch_size],
                wait=True,
            )

        logger.info(
            "Indexed %d chunks from %s into Qdrant collection %s",
            len(points),
            source_id,
            self.collection_name,
        )
        return len(points)

    def index_pdf(self, pdf_path: Path, describe_pictures: bool = False) -> int:
        """Parse a PDF with Docling and index its structure-aware chunks."""
        pdf_path = Path(pdf_path).expanduser().resolve()
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=True,
            do_picture_description=describe_pictures,
            generate_picture_images=describe_pictures,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info("Parsing PDF with Docling: %s", pdf_path.name)
        conversion = converter.convert(pdf_path)
        if not conversion.document:
            raise RuntimeError(f"Docling did not produce a document for {pdf_path}.")

        chunks = list(HybridChunker().chunk(conversion.document))
        records = [
            _chunk_to_record(chunk, pdf_path, index)
            for index, chunk in enumerate(chunks)
        ]
        return self.index_records(records)

    def retrieve(self, query: str, limit: int = 4) -> List[RetrievedPassage]:
        if not query.strip():
            raise ValueError("RAG search query must not be empty.")
        if not self.client.collection_exists(self.collection_name):
            raise RagIndexNotReadyError(
                "The FCA knowledge index is missing. Stop the API server, run "
                "'python -m invest_ai.ingest', then restart the server."
            )
        count = self.client.count(collection_name=self.collection_name, exact=True).count
        if count == 0:
            raise RagIndexNotReadyError(
                "The FCA knowledge index is empty. Stop the API server, run "
                "'python -m invest_ai.ingest', then restart the server."
            )

        self._ensure_embedder_and_collection()
        query_vector = next(iter(self._embedder.query_embed([query]))).tolist()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        passages = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            passages.append(
                RetrievedPassage(
                    citation_id=f"S{rank}",
                    text=str(payload.get("text", "")),
                    document_title=str(payload.get("document_title", "FCA source")),
                    publication=str(payload.get("publication", "")),
                    source_url=str(payload.get("source_url", "")),
                    page_numbers=[
                        int(page) for page in payload.get("page_numbers", [])
                    ],
                    section=str(payload.get("section", "")),
                    score=float(point.score or 0.0),
                )
            )
        return passages

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(collection_name=self.collection_name, exact=True).count
