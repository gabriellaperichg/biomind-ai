from __future__ import annotations

import re
import unicodedata
from pathlib import Path


CONTENT_HINTS = (
    ("contraindication", ("contraindica", "contraindicado", "contraindicacao")),
    ("adverse_event", ("efeito adverso", "evento adverso", "complica", "reacao adversa")),
    ("procedure", ("procedimento", "aplicacao", "tecnica", "administracao", "diluicao")),
    ("treatment", ("tratamento", "terapia", "medicamento", "suplementa", "conduta")),
    ("diagnosis", ("diagnostico", "diagnóstico", "hipotese", "diferencial")),
    ("assessment", ("avaliacao", "avaliação", "anamnese", "exame", "investigacao", "tricoscopia")),
    ("mechanism", ("mecanismo", "fisiopatologia", "5-alfa", "receptor", "enzima")),
    ("classification", ("escala", "classificacao", "classificação", "estagio", "estágio")),
    ("case", ("caso clinico", "casos clinicos", "caso clínico", "paciente do sexo")),
    ("reference", ("referencias", "referências", "bibliografia")),
)


TOPICS = (
    ("androgenetic_alopecia", ("alopecia androgenetica", "alopecia androgenética", "aag")),
    ("alopecia_areata", ("alopecia areata",)),
    ("frontal_fibrosing_alopecia", ("alopecia frontal fibrosante",)),
    ("lichen_planopilaris", ("liquen plano pilar", "líquen plano pilar")),
    ("mesotherapy", ("mesoterapia",)),
    ("microneedling", ("microagulhamento",)),
    ("ozone", ("ozonio", "ozônio", "ozonoterapia")),
    ("anesthetic", ("anestesico", "anestésico",)),
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


def infer_content_type(text: str, section_path: list[str] | None = None) -> str:
    normalized = normalize(" ".join(section_path or []) + " " + text)
    for kind, hints in CONTENT_HINTS:
        if any(hint in normalized for hint in hints):
            return kind
    return "general"


def infer_topic(text: str, section_path: list[str] | None = None) -> str:
    normalized = normalize(" ".join(section_path or []) + " " + text)
    for topic, hints in TOPICS:
        if any(hint in normalized for hint in hints):
            return topic
    return "general_trichology"


def infer_evidence_type(document_type: str, text: str) -> str:
    normalized = normalize(text)
    if document_type == "scientific_article":
        if "randomized" in normalized or "randomizado" in normalized:
            return "primary_research"
        if "systematic review" in normalized or "revisao sistematica" in normalized:
            return "systematic_review"
        if "guideline" in normalized or "diretriz" in normalized:
            return "guideline"
        return "scientific_article"
    if document_type == "clinical_case":
        return "case_report"
    if document_type == "pop":
        return "internal_protocol"
    if document_type == "lecture_slides":
        return "educational"
    return "reference"


def build_metadata(
    *,
    document_type: str,
    source: str,
    title: str,
    section_path: list[str],
    text: str,
) -> dict[str, str | int | bool]:
    return {
        "document_type": document_type,
        "content_type": infer_content_type(text, section_path),
        "topic": infer_topic(text, section_path),
        "evidence_type": infer_evidence_type(document_type, text),
        "source": Path(source).name,
        "title": title[:500],
        "section_path": " > ".join(section_path)[:1000] or "general",
    }


def build_embedding_text(
    *,
    title: str,
    section_path: list[str],
    text: str,
    visual_summary: str = "",
) -> str:
    parts = [
        f"Título: {title}",
        f"Seção: {' > '.join(section_path) if section_path else 'geral'}",
        text.strip(),
    ]
    if visual_summary.strip():
        parts.append(f"Descrição visual: {visual_summary.strip()}")
    return "\n\n".join(part for part in parts if part.strip())
