from __future__ import annotations

from typing import Any

from rag.query import build_query_variants
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank


def retrieve(
    question: str,
    *,
    collection,
    final_k: int = 5,
    candidate_k: int = 30,
) -> list[dict[str, Any]]:
    variants = build_query_variants(question)
    candidates = hybrid_search(
        variants,
        collection,
        k_dense=candidate_k,
        k_bm25=candidate_k,
    )
    reranked = rerank(
        question,
        candidates,
        top_n=final_k,
    )
    return reranked
