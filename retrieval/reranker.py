from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("biomind.retranker")

MODEL_NAME = os.getenv(
    "BIOMIND_RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3",
)
ENABLED = os.getenv("BIOMIND_RERANKER_ENABLED", "1") == "1"
OFFLINE = os.getenv("BIOMIND_OFFLINE", "1") == "1"

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import CrossEncoder

    _model = CrossEncoder(
        MODEL_NAME,
        model_kwargs={"local_files_only": OFFLINE},
        tokenizer_kwargs={"local_files_only": OFFLINE},
    )
    return _model


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    if not ENABLED:
        return candidates[:top_n]

    try:
        model = _get_model()
        pairs = [(query, str(candidate["text"])) for candidate in candidates]
        scores = model.predict(
            pairs,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.warning(
            "Reranker indisponível; mantendo ranking híbrido. Motivo=%s",
            type(exc).__name__,
        )
        return candidates[:top_n]

    ranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        try:
            normalized = float(score)
        except (TypeError, ValueError):
            normalized = 0.0
        item["rerank_score"] = normalized
        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            float(item.get("rerank_score", 0.0)),
            float(item.get("hybrid_score", 0.0)),
        ),
        reverse=True,
    )
    return ranked[:top_n]
