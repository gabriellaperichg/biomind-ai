"""
BioMind — Embeddings + Chroma + BM25 + Hybrid + Reranker.

Comandos:

    python embed.py build

    python embed.py ask "paciente com escamação e coceira no couro cabeludo"

Fluxo:

    documentos
        ↓
    chunks.jsonl
        ↓
    BGE-M3
        ↓
    Chroma
        +
    BM25
        ↓
    Hybrid / RRF
        ↓
    Reranker
        ↓
    janela de contexto
        ↓
    score de qualidade
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "ANONYMIZED_TELEMETRY",
    "FALSE",
)

import chromadb

from retrieval_quality import (
    build_embedding_text,
    enrich_chunk_metadata,
    score_candidate,
)

from retrieval.bm25 import build_index

from retrieval.hybrid import hybrid_search

from retrieval.reranker import rerank


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def _project_path(
    value: str,
    default: Path,
) -> str:

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    return str(path.resolve())


CHUNKS_FILE = _project_path(
    os.getenv(
        "BIOMIND_CHUNKS_FILE",
        "chunks.jsonl",
    ),
    BASE_DIR / "chunks.jsonl",
)


DB_DIR = _project_path(
    os.getenv(
        "BIOMIND_DB_DIR",
        "biomind_db",
    ),
    BASE_DIR / "biomind_db",
)


COLLECTION = os.getenv(
    "BIOMIND_COLLECTION",
    "casos",
)


MODEL_NAME = os.getenv(
    "BIOMIND_EMBED_MODEL",
    "BAAI/bge-m3",
)


EMBED_BATCH_SIZE = int(
    os.getenv(
        "BIOMIND_EMBED_BATCH_SIZE",
        "8",
    )
)


CHROMA_BATCH_SIZE = int(
    os.getenv(
        "BIOMIND_CHROMA_BATCH_SIZE",
        "200",
    )
)


# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------

TOP_K = int(
    os.getenv(
        "BIOMIND_TOP_K",
        "4",
    )
)


WINDOW = int(
    os.getenv(
        "BIOMIND_WINDOW",
        "0",
    )
)


# Quantidade de candidatos antes do reranker.
DENSE_CANDIDATES = int(
    os.getenv(
        "BIOMIND_DENSE_CANDIDATES",
        "30",
    )
)


SPARSE_CANDIDATES = int(
    os.getenv(
        "BIOMIND_SPARSE_CANDIDATES",
        "30",
    )
)


RERANK_CANDIDATES = int(
    os.getenv(
        "BIOMIND_RERANK_CANDIDATES",
        "10",
    )
)


# ------------------------------------------------------------
# Compatibilidade com configuração antiga
# ------------------------------------------------------------

CANDIDATE_MULTIPLIER = int(
    os.getenv(
        "BIOMIND_CANDIDATE_MULTIPLIER",
        "8",
    )
)


# Não definir arbitrariamente antes de termos
# um conjunto de avaliação real.
MIN_SIMILARITY: float | None = None


PREVIEW_CHARS = 1600


SCHEMA_VERSION = 3


# ============================================================
# CACHE
# ============================================================

_model = None
_client = None


# ============================================================
# MODELO
# ============================================================

def get_model():

    global _model

    if _model is None:

        from sentence_transformers import SentenceTransformer

        offline_only = (
            os.getenv(
                "BIOMIND_OFFLINE",
                "0",
            )
            == "1"
        )

        modo = (
            "somente cache local"
            if offline_only
            else "download/cache local"
        )

        print(
            f"Carregando {MODEL_NAME} "
            f"({modo})..."
        )

        _model = SentenceTransformer(
            MODEL_NAME,
            local_files_only=offline_only,
        )

    return _model


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_texts(
    texts: list[str],
    *,
    query: bool = False,
) -> list[list[float]]:

    if not texts:
        return []

    model = get_model()

    if (
        query
        and hasattr(
            model,
            "encode_query",
        )
    ):

        encoder = model.encode_query

    elif (
        not query
        and hasattr(
            model,
            "encode_document",
        )
    ):

        encoder = model.encode_document

    else:

        encoder = model.encode

    embeddings = encoder(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


# ============================================================
# CHROMA
# ============================================================

def get_client():

    global _client

    if _client is None:

        _client = chromadb.PersistentClient(
            path=DB_DIR,
        )

    return _client


def collection_names(client) -> set[str]:

    nomes = set()

    for collection in client.list_collections():

        nomes.add(
            getattr(
                collection,
                "name",
                str(collection),
            )
        )

    return nomes


def get_collection():

    client = get_client()

    if COLLECTION not in collection_names(
        client
    ):

        return None

    return client.get_collection(
        name=COLLECTION,
    )


# ============================================================
# LEITURA DOS CHUNKS
# ============================================================

def load_chunks() -> list[dict[str, Any]]:

    caminho = Path(
        CHUNKS_FILE
    )

    if not caminho.exists():

        raise FileNotFoundError(
            f"{CHUNKS_FILE} não encontrado "
            "— rode o processo de ingestão primeiro."
        )

    chunks: list[dict[str, Any]] = []

    ids_vistos: set[str] = set()

    with caminho.open(
        encoding="utf-8"
    ) as arquivo:

        for numero_linha, linha in enumerate(
            arquivo,
            start=1,
        ):

            if not linha.strip():
                continue

            try:

                chunk = json.loads(
                    linha
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"JSON inválido em "
                    f"{CHUNKS_FILE}, "
                    f"linha {numero_linha}: "
                    f"{exc}"
                ) from exc

            for campo in (
                "text",
                "source",
            ):

                if not chunk.get(
                    campo
                ):

                    raise ValueError(
                        f"Chunk da linha "
                        f"{numero_linha} "
                        f"não possui "
                        f"'{campo}'."
                    )

            # Compatibilidade com chunks antigos.
            chunk_id = str(
                chunk.get(
                    "chunk_id"
                )
                or f"legacy_{numero_linha:08d}"
            )

            document_id = str(
                chunk.get(
                    "document_id"
                )
                or Path(
                    str(
                        chunk["source"]
                    )
                ).stem.lower()
            )

            if chunk_id in ids_vistos:

                raise ValueError(
                    f"chunk_id duplicado: "
                    f"{chunk_id}"
                )

            ids_vistos.add(
                chunk_id
            )

            chunk["chunk_id"] = (
                chunk_id
            )

            chunk["document_id"] = (
                document_id
            )

            chunk["page_start"] = int(
                chunk.get(
                    "page_start",
                    chunk.get(
                        "page",
                        1,
                    ),
                )
            )

            chunk["page_end"] = int(
                chunk.get(
                    "page_end",
                    chunk["page_start"],
                )
            )

            # Enriquece chunks antigos.
            quality_metadata = (
                enrich_chunk_metadata(
                    str(
                        chunk["source"]
                    ),
                    str(
                        chunk["text"]
                    ),
                    str(
                        chunk.get(
                            "section"
                        )
                        or ""
                    )
                    or None,
                )
            )

            for key, value in (
                quality_metadata.items()
            ):

                chunk.setdefault(
                    key,
                    value,
                )

            chunk.setdefault(
                "embedding_text",
                build_embedding_text(
                    str(
                        chunk["source"]
                    ),
                    str(
                        chunk["text"]
                    ),
                    chunk,
                ),
            )

            chunks.append(
                chunk
            )

    if not chunks:

        raise ValueError(
            f"{CHUNKS_FILE} está vazio."
        )

    # Posição dentro de cada documento.
    posicoes: dict[
        str,
        int,
    ] = defaultdict(int)

    for global_pos, chunk in enumerate(
        chunks
    ):

        documento = str(
            chunk["document_id"]
        )

        chunk["doc_pos"] = (
            posicoes[documento]
        )

        chunk["global_pos"] = (
            global_pos
        )

        posicoes[documento] += 1

    return chunks


# ============================================================
# TEXTO DO EMBEDDING
# ============================================================

def embedding_text(
    chunk: dict[str, Any],
) -> str:

    return str(
        chunk.get(
            "embedding_text"
        )
        or chunk["text"]
    )


# ============================================================
# METADATA
# ============================================================

def metadata_for(
    chunk: dict[str, Any],
) -> dict[str, Any]:

    return {

        "chunk_id": str(
            chunk["chunk_id"]
        ),

        "document_id": str(
            chunk["document_id"]
        ),

        "source": str(
            chunk["source"]
        ),

        "doc_pos": int(
            chunk["doc_pos"]
        ),

        "global_pos": int(
            chunk["global_pos"]
        ),

        "page_start": int(
            chunk["page_start"]
        ),

        "page_end": int(
            chunk["page_end"]
        ),

        "char_count": int(
            chunk.get(
                "char_count",
                len(
                    str(
                        chunk["text"]
                    )
                ),
            )
        ),

        "content_hash": str(
            chunk.get(
                "content_hash",
                "",
            )
        ),

        "forced_split": bool(
            chunk.get(
                "forced_split",
                False,
            )
        ),

        "parent_id": str(
            chunk.get(
                "parent_id",
                "",
            )
        ),

        "parent_index": int(
            chunk.get(
                "parent_index",
                chunk.get(
                    "doc_pos",
                    0,
                ),
            )
        ),

        "child_index": int(
            chunk.get(
                "child_index",
                0,
            )
        ),

        "has_visual_content": bool(
            chunk.get(
                "has_visual_content",
                False,
            )
        ),

        "document_type": str(
            chunk.get(
                "document_type",
                "educational_material",
            )
        ),

        "content_type": str(
            chunk.get(
                "content_type",
                "general",
            )
        ),

        "topic": str(
            chunk.get(
                "topic",
                "general_trichology",
            )
        ),

        "evidence_type": str(
            chunk.get(
                "evidence_type",
                "reference",
            )
        ),

        "source_quality": str(
            chunk.get(
                "source_quality",
                "medium",
            )
        ),

        "section": str(
            chunk.get(
                "section",
                "general",
            )
        ),

        "section_path": str(
            chunk.get(
                "section_path",
                chunk.get(
                    "section",
                    "general",
                ),
            )
        ),

        "title": str(
            chunk.get(
                "title",
                Path(
                    str(
                        chunk["source"]
                    )
                ).stem,
            )
        )[:500],
    }


# ============================================================
# BATCHES
# ============================================================

def iter_batches(
    items: list[Any],
    size: int,
) -> Iterable[list[Any]]:

    for inicio in range(
        0,
        len(items),
        size,
    ):

        yield items[
            inicio:
            inicio + size
        ]


# ============================================================
# CONSTRUÇÃO SEGURA
# ============================================================

def create_build_collection(
    client,
    total: int,
):

    temp_name = (
        f"{COLLECTION}_building"
    )

    if temp_name in collection_names(
        client
    ):

        client.delete_collection(
            name=temp_name
        )

    return client.create_collection(
        name=temp_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME,
            "schema_version": SCHEMA_VERSION,
            "expected_chunks": total,
            "built_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    )


def swap_collection(
    client,
    nova_collection,
) -> None:

    """
    Troca a coleção somente depois
    que a nova coleção está completa.
    """

    backup_name = (
        f"{COLLECTION}_backup"
    )

    nomes = collection_names(
        client
    )

    if backup_name in nomes:

        client.delete_collection(
            name=backup_name
        )

        nomes.remove(
            backup_name
        )

    havia_anterior = (
        COLLECTION in nomes
    )

    if havia_anterior:

        anterior = (
            client.get_collection(
                name=COLLECTION
            )
        )

        anterior.modify(
            name=backup_name
        )

    try:

        nova_collection.modify(
            name=COLLECTION
        )

    except Exception:

        if (
            havia_anterior
            and backup_name
            in collection_names(client)
        ):

            client.get_collection(
                name=backup_name
            ).modify(
                name=COLLECTION
            )

        raise

    if (
        backup_name
        in collection_names(client)
    ):

        client.delete_collection(
            name=backup_name
        )


# ============================================================
# BUILD
# ============================================================

def build() -> None:

    try:

        chunks = load_chunks()

    except (
        FileNotFoundError,
        ValueError,
    ) as exc:

        print(exc)
        return

    client = get_client()

    temp = create_build_collection(
        client,
        len(chunks),
    )

    try:

        limite_chroma = int(
            client.get_max_batch_size()
        )

    except Exception:

        limite_chroma = (
            CHROMA_BATCH_SIZE
        )

    batch_size = max(
        1,
        min(
            CHROMA_BATCH_SIZE,
            limite_chroma,
        ),
    )

    print(
        f"Gerando embeddings de "
        f"{len(chunks)} chunks "
        f"em lotes de "
        f"{batch_size}..."
    )

    try:

        processados = 0

        for lote in iter_batches(
            chunks,
            batch_size,
        ):

            embeddings = embed_texts(
                [
                    embedding_text(
                        chunk
                    )
                    for chunk in lote
                ],
                query=False,
            )

            temp.add(

                ids=[
                    str(
                        chunk["chunk_id"]
                    )
                    for chunk in lote
                ],

                embeddings=embeddings,

                documents=[
                    str(
                        chunk["text"]
                    )
                    for chunk in lote
                ],

                metadatas=[
                    metadata_for(
                        chunk
                    )
                    for chunk in lote
                ],
            )

            processados += len(
                lote
            )

            print(
                f"  {processados}/"
                f"{len(chunks)} "
                f"indexados"
            )

        # ----------------------------------------------------
        # Verificação antes da troca
        # ----------------------------------------------------

        gravados = temp.count()

        if gravados != len(
            chunks
        ):

            raise RuntimeError(
                f"Índice incompleto: "
                f"esperado "
                f"{len(chunks)}, "
                f"gravado "
                f"{gravados}."
            )

        # ----------------------------------------------------
        # Só agora substituímos a coleção atual
        # ----------------------------------------------------

        swap_collection(
            client,
            temp,
        )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        bm25_path = Path(
            os.getenv(
                "BIOMIND_BM25_FILE",
                str(
                    Path(DB_DIR)
                    / "bm25_index.json"
                ),
            )
        )

        build_index(
            chunks,
            bm25_path,
        )

    except Exception as exc:

        print(
            f"Falha durante o build: "
            f"{exc}"
        )

        print(
            "O índice anterior foi "
            "preservado quando possível."
        )

        return

    print(
        f"{len(chunks)} chunks "
        f"indexados em "
        f"./{DB_DIR}/"
    )


# ============================================================
# VALIDAÇÃO DA COLEÇÃO
# ============================================================

def validate_collection_model(
    col,
) -> None:

    metadata = (
        col.metadata
        or {}
    )

    modelo = metadata.get(
        "embedding_model"
    )

    if (
        modelo
        and modelo != MODEL_NAME
    ):

        raise RuntimeError(
            f"O índice foi criado "
            f"com '{modelo}', "
            f"mas o script usa "
            f"'{MODEL_NAME}'. "
            f"Rode o build novamente."
        )


# ============================================================
# JANELA DE CONTEXTO
# ============================================================

def get_window(
    col,
    meta_central: dict[str, Any],
) -> list[dict[str, Any]]:

    documento = str(
        meta_central[
            "document_id"
        ]
    )

    posicao = int(
        meta_central[
            "doc_pos"
        ]
    )

    inicio = max(
        0,
        posicao - WINDOW,
    )

    fim = (
        posicao + WINDOW
    )

    resposta = col.get(

        where={
            "$and": [
                {
                    "document_id":
                    documento
                },
                {
                    "doc_pos": {
                        "$gte":
                        inicio
                    }
                },
                {
                    "doc_pos": {
                        "$lte":
                        fim
                    }
                },
            ]
        },

        include=[
            "documents",
            "metadatas",
        ],
    )

    registros = []

    for (
        record_id,
        documento_texto,
        metadata,
    ) in zip(

        resposta["ids"],

        resposta.get(
            "documents"
        )
        or [],

        resposta.get(
            "metadatas"
        )
        or [],
    ):

        if (
            documento_texto is None
            or metadata is None
        ):

            continue

        registros.append(
            {
                "id": record_id,
                "text": documento_texto,
                "metadata": metadata,
            }
        )

    registros.sort(
        key=lambda item:
        int(
            item["metadata"][
                "doc_pos"
            ]
        )
    )

    return registros


# ============================================================
# MERGE
# ============================================================

def merge_overlapping_chunks(
    texts: list[str],
) -> str:

    """
    Remove repetição causada pelo overlap
    dos chunks.
    """

    if not texts:
        return ""

    palavras = texts[0].split()

    for texto in texts[1:]:

        proximas = texto.split()

        max_overlap = min(
            120,
            len(palavras),
            len(proximas),
        )

        overlap = 0

        for tamanho in range(
            max_overlap,
            4,
            -1,
        ):

            if (
                palavras[
                    -tamanho:
                ]
                == proximas[
                    :tamanho
                ]
            ):

                overlap = tamanho
                break

        palavras.extend(
            proximas[
                overlap:
            ]
        )

    return " ".join(
        palavras
    )


# ============================================================
# PÁGINAS
# ============================================================

def format_page_ranges(
    metadatas: list[
        dict[str, Any]
    ],
) -> str:

    intervalos = sorted(

        (
            int(
                meta.get(
                    "page_start",
                    1,
                )
            ),

            int(
                meta.get(
                    "page_end",
                    meta.get(
                        "page_start",
                        1,
                    ),
                )
            ),
        )

        for meta in metadatas
    )

    mesclados: list[
        list[int]
    ] = []

    for inicio, fim in intervalos:

        if (
            not mesclados
            or inicio
            > mesclados[-1][1] + 1
        ):

            mesclados.append(
                [
                    inicio,
                    fim,
                ]
            )

        else:

            mesclados[-1][1] = max(
                mesclados[-1][1],
                fim,
            )

    return ", ".join(

        str(inicio)

        if inicio == fim

        else f"{inicio}-{fim}"

        for inicio, fim
        in mesclados
    )


# ============================================================
# RETRIEVAL HÍBRIDO
# ============================================================

def retrieve(
    pergunta: str,
) -> list[dict[str, Any]]:

    pergunta = pergunta.strip()

    if not pergunta:

        raise ValueError(
            "A pergunta está vazia."
        )

    col = get_collection()

    if (
        col is None
        or col.count() == 0
    ):

        raise RuntimeError(
            "Índice vazio — rode "
            "'python embed.py build' "
            "primeiro."
        )

    validate_collection_model(
        col
    )

    # --------------------------------------------------------
    # 1. HYBRID
    #
    # Dense:
    # entende significado.
    #
    # BM25:
    # encontra termos exatos.
    #
    # RRF:
    # combina os dois rankings.
    # --------------------------------------------------------

    candidatos = hybrid_search(

        [pergunta],

        col,

        k_dense=DENSE_CANDIDATES,

        k_bm25=SPARSE_CANDIDATES,
    )

    if not candidatos:
        return []

    print(
        f"\nHybrid encontrou "
        f"{len(candidatos)} candidatos."
    )

    # --------------------------------------------------------
    # 2. RERANKER
    #
    # O CrossEncoder olha diretamente para:
    #
    # pergunta + trecho
    #
    # e decide quais realmente são mais relevantes.
    #
    # Se o modelo estiver indisponível,
    # retrieval.reranker faz fallback para
    # o ranking híbrido.
    # --------------------------------------------------------

    candidatos = rerank(

        pergunta,

        candidatos,

        top_n=RERANK_CANDIDATES,
    )

    print(
        f"Reranker retornou "
        f"{len(candidatos)} candidatos."
    )

    if not candidatos:
        return []

    # --------------------------------------------------------
    # 3. JANELA + QUALIDADE
    # --------------------------------------------------------

    resultados: list[
        dict[str, Any]
    ] = []

    ids_usados: set[str] = set()

    for candidato in candidatos:

        record_id = str(
            candidato["id"]
        )

        metadata = (
            candidato.get(
                "metadata"
            )
            or {}
        )

        if (
            not metadata
            or record_id
            in ids_usados
        ):

            continue

        similaridade = float(
            candidato.get(
                "semantic_similarity",
                0.0,
            )
        )

        if (
            MIN_SIMILARITY
            is not None
            and similaridade
            < MIN_SIMILARITY
        ):

            continue

        # ----------------------------------------------------
        # Busca chunks vizinhos do mesmo documento.
        # ----------------------------------------------------

        janela = get_window(
            col,
            metadata,
        )

        if not janela:
            continue

        ids_janela = {
            str(
                item["id"]
            )
            for item in janela
        }

        if (
            ids_janela
            & ids_usados
        ):

            continue

        metadatas = [
            item["metadata"]
            for item in janela
        ]

        textos = [
            str(
                item["text"]
            )
            for item in janela
        ]

        texto_mesclado = (
            merge_overlapping_chunks(
                textos
            )
        )

        # ----------------------------------------------------
        # Score de qualidade já existente no projeto.
        # ----------------------------------------------------

        quality = score_candidate(

            pergunta,

            semantic_similarity=
                similaridade,

            source=str(
                metadata.get(
                    "source",
                    "",
                )
            ),

            text=texto_mesclado,

            metadata=metadata,
        )

        if quality.rejected:
            continue

        resultados.append(

            {
                "similarity":
                    similaridade,

                "hybrid_score":
                    float(
                        candidato.get(
                            "hybrid_score",
                            0.0,
                        )
                    ),

                "bm25_score":
                    float(
                        candidato.get(
                            "bm25_score",
                            0.0,
                        )
                    ),

                "rerank_score":
                    float(
                        candidato.get(
                            "rerank_score",
                            0.0,
                        )
                    ),

                "adjusted_score":
                    quality.adjusted_score,

                "source":
                    str(
                        metadata.get(
                            "source",
                            "",
                        )
                    ),

                "pages":
                    format_page_ranges(
                        metadatas
                    ),

                "text":
                    texto_mesclado,

                "chunk_ids":
                    [
                        str(
                            item["id"]
                        )
                        for item
                        in janela
                    ],

                "document_type":
                    quality.document_type,

                "content_type":
                    quality.content_type,

                "reasons":
                    list(
                        quality.reasons
                    ),
            }
        )

        ids_usados.update(
            ids_janela
        )

    # --------------------------------------------------------
    # 4. RANKING FINAL
    #
    # A qualidade continua sendo a camada final.
    # --------------------------------------------------------

    resultados.sort(

        key=lambda item: (

            float(
                item.get(
                    "adjusted_score",
                    0.0,
                )
            ),

            float(
                item.get(
                    "rerank_score",
                    0.0,
                )
            ),

            float(
                item.get(
                    "hybrid_score",
                    0.0,
                )
            ),

        ),

        reverse=True,
    )

    return resultados[
        :TOP_K
    ]


# ============================================================
# CLI
# ============================================================

def ask(
    pergunta: str,
) -> None:

    try:

        resultados = retrieve(
            pergunta
        )

    except (
        ValueError,
        RuntimeError,
    ) as exc:

        print(exc)

        return

    print(
        f'\nPergunta: "{pergunta}"\n'
        + "-" * 80
    )

    if not resultados:

        print(
            "Nenhum trecho passou "
            "pelos critérios "
            "de recuperação."
        )

        return

    for numero, resultado in enumerate(
        resultados,
        start=1,
    ):

        texto = str(
            resultado["text"]
        )

        previa = texto[
            :PREVIEW_CHARS
        ]

        print(

            f"\n▸ #{numero} "
            f"{resultado['source']} "
            f"· pág. "
            f"{resultado['pages']}"
        )

        print(

            f"semantic="
            f"{resultado['similarity']:.4f}"

            f" | hybrid="
            f"{resultado['hybrid_score']:.4f}"

            f" | bm25="
            f"{resultado['bm25_score']:.4f}"

            f" | rerank="
            f"{resultado['rerank_score']:.4f}"

            f" | final="
            f"{resultado['adjusted_score']:.4f}"
        )

        if resultado.get(
            "document_type"
        ):

            print(
                "document_type="
                f"{resultado['document_type']}"
            )

        if resultado.get(
            "content_type"
        ):

            print(
                "content_type="
                f"{resultado['content_type']}"
            )

        print()

        print(
            previa
            + (
                "…"
                if len(texto)
                > PREVIEW_CHARS
                else ""
            )
        )

    print(
        "\n" + "-" * 80
    )

    print(
        "Esses grupos de contexto "
        "podem ser enviados ao "
        "modelo gerador."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if (
        len(sys.argv) < 2
        or sys.argv[1]
        not in {
            "build",
            "ask",
        }
    ):

        print(
            "Uso:"
        )

        print(
            "  python embed.py build"
        )

        print(
            '  python embed.py ask '
            '"sua pergunta"'
        )

        return

    comando = sys.argv[1]

    if comando == "build":

        build()

        return

    pergunta = " ".join(
        sys.argv[2:]
    ).strip()

    if not pergunta:

        pergunta = (
            "escamação no couro cabeludo"
        )

    ask(
        pergunta
    )


if __name__ == "__main__":

    main()