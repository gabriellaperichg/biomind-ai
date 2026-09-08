from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from retrieval.bm25 import load_index
from retrieval.dense import embed_queries


BASE_DIR = Path(__file__).resolve().parents[1]


def _resolve_path(
    value: str | None,
    default: Path,
) -> Path:
    path = Path(value).expanduser() if value else default

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


DB_DIR = _resolve_path(
    os.getenv("BIOMIND_DB_DIR"),
    BASE_DIR / "biomind_db",
)

BM25_FILE = _resolve_path(
    os.getenv("BIOMIND_BM25_FILE"),
    DB_DIR / "bm25_index.json",
)


def _dense_candidates(
    queries: Sequence[str],
    col,
    k: int,
) -> dict[str, dict[str, Any]]:

    if not queries:
        return {}

    vectors = embed_queries(list(queries))

    result = col.query(
        query_embeddings=vectors,
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    out: dict[str, dict[str, Any]] = {}

    for qidx, query in enumerate(queries):

        ids = result.get("ids", [[]])[qidx]
        docs = result.get("documents", [[]])[qidx]
        metas = result.get("metadatas", [[]])[qidx]
        distances = result.get("distances", [[]])[qidx]

        for rank, (
            record_id,
            text,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                docs,
                metas,
                distances,
            ),
            start=1,
        ):

            if text is None or metadata is None:
                continue

            similarity = max(
                0.0,
                1.0 - float(distance),
            )

            record_id = str(record_id)

            current = out.get(record_id)

            if (
                current is None
                or similarity
                > current["semantic_similarity"]
            ):
                out[record_id] = {
                    "id": record_id,
                    "text": str(text),
                    "metadata": metadata,
                    "semantic_similarity": similarity,
                    "matched_query": query,
                    "dense_rank": rank,
                }

    return out


def hybrid_search(
    queries: Sequence[str],
    col,
    *,
    k_dense: int = 30,
    k_bm25: int = 30,
) -> list[dict[str, Any]]:
    """
    Combina busca semântica (Dense) e lexical (BM25)
    usando Reciprocal Rank Fusion (RRF).

    O BM25 é armazenado separadamente do banco
    principal da aplicação.
    """

    if not queries:
        return []

    dense = _dense_candidates(
        queries,
        col,
        k_dense,
    )

    bm25 = load_index(BM25_FILE)

    # -------------------------------------------------
    # Caso BM25 ainda não exista
    # -------------------------------------------------

    if bm25 is None:

        ranked = list(dense.values())

        ranked.sort(
            key=lambda item: item[
                "semantic_similarity"
            ],
            reverse=True,
        )

        for rank, item in enumerate(
            ranked,
            start=1,
        ):
            item["hybrid_score"] = (
                1.0 / (60 + rank)
            )

            item["bm25_score"] = 0.0

        return ranked

    # -------------------------------------------------
    # Fusão BM25 + Dense
    # -------------------------------------------------

    fused: dict[str, dict[str, Any]] = {}

    # -------------------------
    # BM25
    # -------------------------

    for query in queries:

        bm25_results = bm25.search(
            query,
            k=k_bm25,
        )

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            record_id = str(
                result["chunk_id"]
            )

            item = fused.setdefault(
                record_id,
                {
                    "id": record_id,
                    "text": result.get("text"),
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),
                    "bm25_score": 0.0,
                    "semantic_similarity": 0.0,
                    "hybrid_score": 0.0,
                    "matched_query": query,
                },
            )

            item["bm25_score"] = max(
                float(
                    item.get(
                        "bm25_score",
                        0.0,
                    )
                ),
                float(
                    result.get(
                        "score",
                        0.0,
                    )
                ),
            )

            item["hybrid_score"] += (
                1.0 / (60 + rank)
            )

            # Preserva texto e metadata vindos do BM25
            if not item.get("text"):
                item["text"] = result.get(
                    "text"
                )

            if not item.get("metadata"):
                item["metadata"] = result.get(
                    "metadata",
                    {},
                )

    # -------------------------
    # Dense
    # -------------------------

    dense_ranked = sorted(
        dense.values(),
        key=lambda item: item[
            "semantic_similarity"
        ],
        reverse=True,
    )

    for rank, result in enumerate(
        dense_ranked,
        start=1,
    ):

        record_id = result["id"]

        item = fused.setdefault(
            record_id,
            {
                "id": record_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "bm25_score": 0.0,
                "semantic_similarity": 0.0,
                "hybrid_score": 0.0,
                "matched_query": result[
                    "matched_query"
                ],
            },
        )

        item["text"] = result["text"]
        item["metadata"] = result[
            "metadata"
        ]

        item["semantic_similarity"] = result[
            "semantic_similarity"
        ]

        item["dense_rank"] = rank

        item["hybrid_score"] += (
            1.0 / (60 + rank)
        )

        item["matched_query"] = result[
            "matched_query"
        ]

    # -------------------------------------------------
    # Remove resultados incompletos
    # -------------------------------------------------

    results = [
        item
        for item in fused.values()
        if item.get("text") is not None
        and item.get("metadata") is not None
    ]

    # -------------------------------------------------
    # Ordenação final
    # -------------------------------------------------

    results.sort(
        key=lambda item: (
            float(
                item.get(
                    "hybrid_score",
                    0.0,
                )
            ),
            float(
                item.get(
                    "semantic_similarity",
                    0.0,
                )
            ),
            float(
                item.get(
                    "bm25_score",
                    0.0,
                )
            ),
        ),
        reverse=True,
    )

    return results