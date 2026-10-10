"""Offline tests for local Qdrant indexing and retrieval behavior."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from invest_ai.compliance import ComplianceGuardrailEngine
from invest_ai.contracts import CustomerContext
from invest_ai.graph import InvestAIGraph
from invest_ai.rag import (
    ChunkRecord,
    QdrantRagIndex,
    RagIndexNotReadyError,
    RetrievedPassage,
    _chunk_to_record,
)


class FakeEmbedding:
    def embed(self, texts, batch_size=256):
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        for text in texts:
            vector = (
                [1.0, 0.0, 0.0]
                if "pension" in text.lower()
                else [0.0, 1.0, 0.0]
            )
            yield np.asarray(vector, dtype=np.float32)

    def query_embed(self, queries):
        for query in queries:
            if not query:
                raise ValueError("query must not be empty")
            yield np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class FakeRouter:
    def classify(self, query):
        return (
            "INVESTMENT_GUIDANCE"
            if "pension" in query.lower()
            else "GENERAL_INQUIRY"
        )


class FakeAdvisor:
    def __init__(self):
        self.received_passages = []

    def respond(self, query, context, retrieved_passages):
        if not query:
            raise ValueError("query must not be empty")
        self.received_passages = retrieved_passages
        return f"General response for {context.investor_experience} investor."


class FakeRagIndex:
    def __init__(self):
        self.calls = []

    def retrieve(self, query):
        self.calls.append(query)
        return [
            RetrievedPassage(
                citation_id="S1",
                text="FCA guidance passage.",
                document_title="FCA reference",
                publication="FG00/0",
                source_url="https://example.test",
                page_numbers=[8],
                section="Pensions",
                score=0.9,
            )
        ]


class RagIndexTests(unittest.TestCase):
    def test_chunk_metadata_preserves_section_and_page(self):
        item = SimpleNamespace(
            label=SimpleNamespace(value="table"),
            prov=[SimpleNamespace(page_no=12)],
        )
        chunk = SimpleNamespace(
            text="Contribution type | Tax treatment",
            meta=SimpleNamespace(
                doc_items=[item],
                headings=["Pensions", "Tax treatment"],
            ),
        )

        record = _chunk_to_record(
            chunk,
            Path("fca-consumer-duty-fg22-5.pdf"),
            3,
        )

        self.assertEqual(record.page_numbers, [12])
        self.assertEqual(record.section, "Pensions > Tax treatment")
        self.assertEqual(record.element_types, ["table"])
        self.assertEqual(record.publication, "FCA FG22/5 (July 2022)")

    def test_local_qdrant_index_retrieve_and_replace_document(self):
        with TemporaryDirectory() as temp_dir:
            index = QdrantRagIndex(
                storage_path=Path(temp_dir),
                collection_name="unit_test",
            )
            index._embedder = FakeEmbedding()
            index._vector_size = 3
            try:
                first = ChunkRecord(
                    text="Pension transfer guidance and risks.",
                    source_id="demo.pdf",
                    document_title="Demo FCA guidance",
                    publication="FG00/0",
                    source_url="https://example.test/source",
                    page_numbers=[4],
                    section="Suitability",
                    element_types=["text", "table"],
                    chunk_index=0,
                )
                index.index_records([first])
                hits = index.retrieve("pension risks")

                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0].page_numbers, [4])
                self.assertEqual(hits[0].section, "Suitability")
                self.assertEqual(hits[0].citation_id, "S1")
                self.assertEqual(hits[0].citation()["source_url"], first.source_url)

                replacement = ChunkRecord(
                    text="Updated pension transfer guidance.",
                    source_id="demo.pdf",
                    document_title="Updated FCA guidance",
                    publication="FG00/1",
                    source_url="https://example.test/updated",
                    page_numbers=[5],
                    section="Updated section",
                    element_types=["text"],
                    chunk_index=0,
                )
                index.index_records([replacement])

                self.assertEqual(index.count(), 1)
                self.assertEqual(index.retrieve("pension risks")[0].text, replacement.text)
            finally:
                index.client.close()

    def test_retrieval_requires_an_index(self):
        index = QdrantRagIndex.__new__(QdrantRagIndex)
        index.collection_name = "empty"
        index.client = SimpleNamespace(
            collection_exists=lambda name: name != "empty"
        )
        with self.assertRaises(RagIndexNotReadyError):
            index.retrieve("pension guidance")

    def test_langgraph_retrieves_context_only_on_guidance_path(self):
        advisor = FakeAdvisor()
        rag_index = FakeRagIndex()
        graph = InvestAIGraph(
            router=FakeRouter(),
            advisor=advisor,
            compliance=ComplianceGuardrailEngine(),
            rag_index=rag_index,
        ).compiled
        context = CustomerContext(
            customer_id="TEST",
            age_band="35-50",
            investor_experience="Intermediate",
        ).model_dump()

        guidance = graph.invoke(
            {"query_text": "How does a pension work?", "context": context}
        )
        self.assertEqual(len(advisor.received_passages), 1)
        self.assertEqual(guidance["retrieved_sources"][0]["citation_id"], "S1")
        self.assertIn("educational guidance only", guidance["generated_guidance"])

        rejection = graph.invoke(
            {"query_text": "How do I cook rice?", "context": context}
        )
        self.assertEqual(rejection["intent"], "GENERAL_INQUIRY")
        self.assertIn("rejection_message", rejection)
        self.assertEqual(len(rag_index.calls), 1)


if __name__ == "__main__":
    unittest.main()
