"""
Camada de integração entre a aplicação e o biomind_core.

O objetivo é impedir que routers conheçam todos os detalhes do formato
retornado pelo núcleo RAG.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypedDict

import biomind_core as core


logger = logging.getLogger("biomind.services.rag")


class RagServiceError(RuntimeError):
    """Falha técnica ao executar ou interpretar o núcleo RAG."""


class NormalizedSource(TypedDict):
    source_number: int
    source_name: str
    pages: str | None
    similarity: float | None
    chunk_ids: list[str] | None


class NormalizedAnswer(TypedDict):
    status: str
    answer: str
    disclaimer: str | None
    best_similarity: float | None
    sources: list[NormalizedSource]
    model_name: str | None
    embedding_model: str | None
    prompt_version: str | None
    generation_time_ms: int


def answer_question(question: str) -> NormalizedAnswer:
    """
    Executa o RAG e devolve um formato estável para o restante da aplicação.
    """

    normalized_question = question.strip()

    if not normalized_question:
        return {
            "status": "vazio",
            "answer": "",
            "disclaimer": None,
            "best_similarity": None,
            "sources": [],
            "model_name": None,
            "embedding_model": None,
            "prompt_version": None,
            "generation_time_ms": 0,
        }

    started_at = time.perf_counter()

    try:
        raw_result = core.responder(normalized_question)
    except Exception as exc:
        logger.exception(
            "Falha técnica ao executar biomind_core.responder"
        )

        raise RagServiceError(
            "O núcleo do Biomind não conseguiu gerar a resposta."
        ) from exc

    measured_time_ms = round(
        (time.perf_counter() - started_at) * 1000
    )

    try:
        return normalize_result(
            raw_result,
            measured_time_ms=measured_time_ms,
        )
    except Exception as exc:
        logger.exception(
            "Formato inesperado retornado pelo biomind_core"
        )

        raise RagServiceError(
            "O núcleo do Biomind retornou uma resposta inválida."
        ) from exc


def check_rag_health() -> dict[str, Any]:
    """
    Encapsula o health check do núcleo.
    """

    try:
        result = core.health()
    except Exception as exc:
        logger.exception("Falha ao consultar biomind_core.health")

        raise RagServiceError(
            "Não foi possível verificar o núcleo do Biomind."
        ) from exc

    if not isinstance(result, dict):
        raise RagServiceError(
            "O health check retornou um formato inválido."
        )

    return result


def normalize_result(
    result: Any,
    *,
    measured_time_ms: int,
) -> NormalizedAnswer:
    """
    Aceita formatos comuns do core e converte para o formato da aplicação.
    """

    if isinstance(result, str):
        return {
            "status": "ok",
            "answer": result,
            "disclaimer": None,
            "best_similarity": None,
            "sources": [],
            "model_name": None,
            "embedding_model": None,
            "prompt_version": None,
            "generation_time_ms": measured_time_ms,
        }

    if hasattr(result, "model_dump"):
        result = result.model_dump()

    if not isinstance(result, dict):
        raise TypeError(
            "biomind_core.responder() deve retornar dict, modelo Pydantic "
            "ou texto."
        )

    status = str(result.get("status") or "ok")

    answer = str(
        result.get("answer")
        or result.get("resposta")
        or result.get("content")
        or result.get("conteudo")
        or result.get("texto")
        or result.get("message")
        or result.get("mensagem")
        or ""
    )

    if status == "ok" and not answer:
        status = "sem_material"

    raw_sources = (
        result.get("sources")
        or result.get("fontes")
        or []
    )

    if not isinstance(raw_sources, list):
        raw_sources = [raw_sources]

    sources = [
        normalize_source(source, number)
        for number, source in enumerate(raw_sources, start=1)
    ]

    best_similarity = first_present(
        result,
        "best_similarity",
        "best_sim",
        "melhor_similaridade",
    )

    generation_time = first_present(
        result,
        "generation_time_ms",
        "tempo_geracao_ms",
    )

    return {
        "status": status,
        "answer": answer,
        "disclaimer": optional_text(
            first_present(
                result,
                "disclaimer",
                "aviso",
                "aviso_legal",
            )
        ),
        "best_similarity": optional_float(best_similarity),
        "sources": sources,
        "model_name": optional_text(
            first_present(
                result,
                "model_name",
                "modelo",
            )
        ),
        "embedding_model": optional_text(
            first_present(
                result,
                "embedding_model",
                "modelo_embedding",
            )
        ),
        "prompt_version": optional_text(
            first_present(
                result,
                "prompt_version",
                "versao_prompt",
            )
        ),
        "generation_time_ms": (
            optional_int(generation_time)
            or measured_time_ms
        ),
    }


def normalize_source(
    source: Any,
    number: int,
) -> NormalizedSource:
    if isinstance(source, str):
        return {
            "source_number": number,
            "source_name": source,
            "pages": None,
            "similarity": None,
            "chunk_ids": None,
        }

    if hasattr(source, "model_dump"):
        source = source.model_dump()

    if not isinstance(source, dict):
        return {
            "source_number": number,
            "source_name": "Fonte não identificada",
            "pages": None,
            "similarity": None,
            "chunk_ids": None,
        }

    source_number = optional_int(
        first_present(
            source,
            "source_number",
            "n",
            "numero",
        )
    ) or number

    source_name = optional_text(
        first_present(
            source,
            "source_name",
            "source",
            "arquivo",
            "file",
            "document",
            "documento",
        )
    ) or "Fonte não identificada"

    pages = normalize_pages(
        first_present(
            source,
            "pages",
            "page",
            "paginas",
            "páginas",
        )
    )

    similarity = optional_float(
        first_present(
            source,
            "similarity",
            "sim",
            "score",
            "similaridade",
        )
    )

    raw_chunk_ids = first_present(
        source,
        "chunk_ids",
        "chunk_ids_json",
        "chunks",
    )

    chunk_ids: list[str] | None

    if raw_chunk_ids is None:
        chunk_ids = None
    elif isinstance(raw_chunk_ids, list):
        chunk_ids = [str(value) for value in raw_chunk_ids]
    else:
        chunk_ids = [str(raw_chunk_ids)]

    return {
        "source_number": source_number,
        "source_name": source_name,
        "pages": pages,
        "similarity": similarity,
        "chunk_ids": chunk_ids,
    }


def first_present(
    mapping: dict[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        value = mapping.get(key)

        if value is not None:
            return value

    return None


def normalize_pages(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)

    return str(value)


def optional_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    return text or None


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None
