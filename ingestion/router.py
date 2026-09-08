from __future__ import annotations

from pathlib import Path
import re


def detect_document_type(
    path: str | Path,
    text_sample: str = "",
    page_count: int = 1,
    pages_with_images: int = 0,
) -> str:
    """Classifica o documento por estrutura/sinais, sem exigir um LLM."""
    suffix = Path(path).suffix.lower()
    if suffix == ".docx":
        text = text_sample.lower()
        if "procedimento operacional padrão" in text or re.search(r"\bpop\b", text):
            return "pop"
        return "docx"

    if suffix != ".pdf":
        return "unknown"

    text = text_sample.lower()
    name = Path(path).stem.lower()

    if (
        "procedimento operacional padrão" in text
        or re.search(r"\bpop\b", name)
        or "biossegurança e limpeza" in text
        or "responsável técnico" in text
    ):
        return "pop"

    if (
        "abstract" in text
        and ("references" in text or "referências" in text)
    ) or "doi.org/" in text or re.search(r"\bdoi\s*:", text):
        return "scientific_article"

    if (
        "caso clínico" in text
        or "casos clínicos" in text
        or re.search(r"\bpaciente\s+(do sexo|feminina|masculino)", text)
    ):
        return "clinical_case"

    slide_hints = (
        "slide", "slides", "aula", "aulas", "apresentação",
        "treinamento", "curso", "módulo",
    )
    if any(hint in name for hint in slide_hints):
        return "lecture_slides"

    # Heurística visual: PDFs com muitas páginas e alta densidade de imagens
    # tendem a ser material de slides. É deliberadamente conservadora.
    if page_count >= 8 and pages_with_images / max(page_count, 1) >= 0.65:
        return "lecture_slides"

    if (
        "capítulo" in text
        or "chapter" in text
        or "sumário" in text
        or "table of contents" in text
    ):
        return "book"

    return "reference_pdf"
