from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ingestion.metadata import build_embedding_text, build_metadata
from ingestion.parsers import parse_path
from ingestion.structure import StructuralBlock, build_structure


BOILERPLATE_PATTERNS = [
    r"Conte[uú]do licenciado para.{0,300}?(?:\d{3}\.\d{3}\.\d{3}-\d{2}|$)",
    r"Licenciado para.{0,300}?(?:\d{3}\.\d{3}\.\d{3}-\d{2}|$)",
    r"Esse material é rastreável.*?Código Penal Brasileiro\.",
    r"Compartilhar livros, imagens e aulas.*?Código Penal Brasileiro\.",
]


def clean_text(text: str) -> str:
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.I | re.S)
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?im)^\s*(?:p[aá]gina\s*)?\d{1,4}\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


DEFAULT_TARGET = {
    "pop": 1600,
    "scientific_article": 1800,
    "book": 1800,
    "lecture_slides": 2200,
    "clinical_case": 1500,
    "docx": 1600,
    "reference_pdf": 1400,
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(
        re.sub(r"\s+", " ", text).strip().lower().encode("utf-8")
    ).hexdigest()


def split_sentences(text: str) -> list[str]:
    return [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?])\s+", text.strip())
        if chunk.strip()
    ]


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()]

    sentences = split_sentences(text)
    result: list[str] = []
    current: list[str] = []
    size = 0

    for sentence in sentences:
        if current and size + len(sentence) + 1 > max_chars:
            result.append(" ".join(current).strip())
            current = []
            size = 0
        current.append(sentence)
        size += len(sentence) + 1

    if current:
        result.append(" ".join(current).strip())
    return result


def _make_children(block: StructuralBlock, target: int) -> list[str]:
    if block.block_type == "slide":
        return [block.text.strip()] if block.text.strip() else [block.title]

    paragraphs = [p.strip() for p in block.text.split("\n\n") if p.strip()]
    base = "\n\n".join(paragraphs)
    return _split_long(base, target)


def chunk_document(path: str | Path) -> list[dict]:
    document = parse_path(path)
    blocks = build_structure(document)
    chunks: list[dict] = []
    child_counter = 0

    for parent_index, block in enumerate(blocks):
        if not block.text.strip():
            continue

        parent_id = (
            f"{document.document_id}_p{parent_index:04d}_"
            f"{stable_hash(block.text)[:10]}"
        )

        children = _make_children(
            block,
            DEFAULT_TARGET.get(document.document_type, DEFAULT_TARGET["reference_pdf"]),
        )

        for child_index, child_text in enumerate(children):
            child_text = clean_text(child_text)
            if not child_text:
                continue
            content_hash = stable_hash(child_text)
            chunk_id = (
                f"{document.document_id}_c{child_counter:05d}_"
                f"{content_hash[:10]}"
            )
            metadata = build_metadata(
                document_type=document.document_type,
                source=document.source,
                title=document.title,
                section_path=block.section_path,
                text=child_text,
            )
            embedding_text = build_embedding_text(
                title=document.title,
                section_path=block.section_path,
                text=child_text,
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "child_index": child_index,
                    "document_id": document.document_id,
                    "text": child_text,
                    "embedding_text": embedding_text,
                    "source": document.source,
                    "page": block.page_start,
                    "page_start": block.page_start,
                    "page_end": block.page_end,
                    "char_count": len(child_text),
                    "content_hash": content_hash,
                    "forced_split": len(children) > 1,
                    "has_visual_content": block.has_visual_content,
                    **metadata,
                }
            )
            child_counter += 1

    # posições para compatibilidade com o retrieval existente
    for pos, chunk in enumerate(chunks):
        chunk["doc_pos"] = pos
        chunk["global_pos"] = pos

    return chunks


def ingest_directory(directory: str | Path, output_file: str | Path) -> list[dict]:
    directory = Path(directory)
    supported = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx"}
    )

    all_chunks: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for path in supported:
        try:
            chunks = chunk_document(path)
        except Exception as exc:
            print(f"[ERRO] {path.name}: {exc}")
            continue

        for chunk in chunks:
            key = (chunk["document_id"], chunk["content_hash"])
            if key in seen:
                continue
            seen.add(key)
            all_chunks.append(chunk)

    output = Path(output_file)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(__import__("json").dumps(chunk, ensure_ascii=False) + "\n")

    return all_chunks
