"""CLI for parsing local FCA PDFs and indexing them into persistent Qdrant."""

import argparse
import logging
from pathlib import Path

from invest_ai.rag import DEFAULT_SOURCE_DIR, QdrantRagIndex


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse FCA PDFs with Docling and index them in local Qdrant."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing source PDFs (default: data/rag_sources)",
    )
    parser.add_argument(
        "--describe-pictures",
        action="store_true",
        help=(
            "Use Docling's local vision model to describe figures and charts. "
            "This can be slow and downloads additional model weights."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    pdf_paths = sorted(args.source_dir.glob("*.pdf"))
    if not pdf_paths:
        parser.error(f"No PDFs found in {args.source_dir.resolve()}")

    index = QdrantRagIndex()
    total_chunks = 0
    for pdf_path in pdf_paths:
        total_chunks += index.index_pdf(
            pdf_path,
            describe_pictures=args.describe_pictures,
        )
    print(
        f"Indexed {total_chunks} chunks from {len(pdf_paths)} PDFs "
        f"into Qdrant collection '{index.collection_name}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
