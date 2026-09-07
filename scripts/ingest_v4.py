"""Recria chunks.jsonl usando a ingestão V4.

Uso:
    python -m scripts.ingest_v4
    python -m scripts.ingest_v4 --directory pdfs --output chunks.jsonl
"""
from __future__ import annotations
import argparse
from pathlib import Path
from ingestion.chunker import ingest_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        default="pdfs",
        help="Pasta com PDF/DOCX. Subpastas também são lidas.",
    )
    parser.add_argument(
        "--output",
        default="chunks.jsonl",
        help="Arquivo JSONL de saída.",
    )
    args = parser.parse_args()

    chunks = ingest_directory(args.directory, args.output)
    print(f"{len(chunks)} chunks gerados em {args.output}")


if __name__ == "__main__":
    main()
