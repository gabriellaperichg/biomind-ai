"""
Biomind — Etapa 2: embeddings + índice Chroma com janela de contexto (v2).

Comandos:
    python 02_embed.py build
    python 02_embed.py ask "paciente com escamação e coceira no couro cabeludo"

Privacidade:
- Chroma é executado localmente em ./biomind_db.
- O modelo é baixado do Hugging Face na primeira execução.
- Depois do primeiro download, defina BIOMIND_OFFLINE=1 para impedir acesso à rede.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Evita telemetria em versões antigas do Chroma. Versões atuais não coletam
# telemetria de produto, mas esta variável é mantida como defesa adicional.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb


CHUNKS_FILE = "chunks.jsonl"
DB_DIR = "biomind_db"
COLLECTION = "casos"

MODEL_NAME = "BAAI/bge-m3"
EMBED_BATCH_SIZE = 8       # aumente se houver RAM/VRAM disponível
CHROMA_BATCH_SIZE = 200

TOP_K = 4                  # quantidade de grupos de contexto devolvidos
WINDOW = 1                 # chunks vizinhos de cada lado
CANDIDATE_MULTIPLIER = 4   # compensa resultados adjacentes/duplicados

# Não invente um limiar sem medir em perguntas reais. Quando tiver um conjunto
# de avaliação, troque None pelo valor calibrado para o seu corpus.
MIN_SIMILARITY: float | None = None

PREVIEW_CHARS = 1600
SCHEMA_VERSION = 2

_model = None
_client = None


# ---------------------------------------------------------------------------
# Modelo e cliente
# ---------------------------------------------------------------------------

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        offline_only = os.getenv("BIOMIND_OFFLINE", "0") == "1"
        modo = "somente cache local" if offline_only else "download/cache local"
        print(f"Carregando {MODEL_NAME} ({modo})...")

        _model = SentenceTransformer(
            MODEL_NAME,
            local_files_only=offline_only,
        )
    return _model


def embed_texts(texts: list[str], *, query: bool = False) -> list[list[float]]:
    if not texts:
        return []

    model = get_model()

    # Sentence Transformers recente oferece métodos específicos para consulta
    # e documento. Para BGE-M3, sem prompts especiais, o resultado é compatível
    # com encode(); o fallback mantém compatibilidade com versões anteriores.
    if query and hasattr(model, "encode_query"):
        encoder = model.encode_query
    elif not query and hasattr(model, "encode_document"):
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


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_DIR)
    return _client


def collection_names(client) -> set[str]:
    nomes = set()
    for collection in client.list_collections():
        nomes.add(getattr(collection, "name", str(collection)))
    return nomes


def get_collection():
    client = get_client()
    if COLLECTION not in collection_names(client):
        return None
    return client.get_collection(name=COLLECTION)


# ---------------------------------------------------------------------------
# Leitura e validação dos chunks
# ---------------------------------------------------------------------------

def load_chunks() -> list[dict[str, Any]]:
    caminho = Path(CHUNKS_FILE)
    if not caminho.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} não encontrado — rode o 01_chunk.py primeiro."
        )

    chunks: list[dict[str, Any]] = []
    ids_vistos: set[str] = set()

    with caminho.open(encoding="utf-8") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            if not linha.strip():
                continue

            try:
                chunk = json.loads(linha)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido em {CHUNKS_FILE}, linha {numero_linha}: {exc}"
                ) from exc

            for campo in ("text", "source"):
                if not chunk.get(campo):
                    raise ValueError(
                        f"Chunk da linha {numero_linha} não possui '{campo}'."
                    )

            # Compatibilidade com versões antigas da Etapa 1.
            chunk_id = str(chunk.get("chunk_id") or f"legacy_{numero_linha:08d}")
            document_id = str(
                chunk.get("document_id")
                or Path(str(chunk["source"])).stem.lower()
            )

            if chunk_id in ids_vistos:
                raise ValueError(f"chunk_id duplicado: {chunk_id}")
            ids_vistos.add(chunk_id)

            chunk["chunk_id"] = chunk_id
            chunk["document_id"] = document_id
            chunk["page_start"] = int(chunk.get("page_start", chunk.get("page", 1)))
            chunk["page_end"] = int(chunk.get("page_end", chunk["page_start"]))
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"{CHUNKS_FILE} está vazio.")

    # A posição é por documento, não pelo corpus inteiro.
    posicoes: dict[str, int] = defaultdict(int)
    for global_pos, chunk in enumerate(chunks):
        documento = str(chunk["document_id"])
        chunk["doc_pos"] = posicoes[documento]
        chunk["global_pos"] = global_pos
        posicoes[documento] += 1

    return chunks


def embedding_text(chunk: dict[str, Any]) -> str:
    """Texto usado no embedding; aceita enriquecimento futuro da Etapa 1."""
    return str(chunk.get("embedding_text") or chunk["text"])


def metadata_for(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk["chunk_id"]),
        "document_id": str(chunk["document_id"]),
        "source": str(chunk["source"]),
        "doc_pos": int(chunk["doc_pos"]),
        "global_pos": int(chunk["global_pos"]),
        "page_start": int(chunk["page_start"]),
        "page_end": int(chunk["page_end"]),
        "char_count": int(chunk.get("char_count", len(str(chunk["text"])))),
        "content_hash": str(chunk.get("content_hash", "")),
        "forced_split": bool(chunk.get("forced_split", False)),
    }


def iter_batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    for inicio in range(0, len(items), size):
        yield items[inicio: inicio + size]


# ---------------------------------------------------------------------------
# Construção segura do índice
# ---------------------------------------------------------------------------

def create_build_collection(client, total: int):
    temp_name = f"{COLLECTION}_building"
    if temp_name in collection_names(client):
        client.delete_collection(name=temp_name)

    return client.create_collection(
        name=temp_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": MODEL_NAME,
            "schema_version": SCHEMA_VERSION,
            "expected_chunks": total,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def swap_collection(client, nova_collection) -> None:
    """Troca o índice somente depois de a nova coleção estar completa."""
    backup_name = f"{COLLECTION}_backup"
    nomes = collection_names(client)

    if backup_name in nomes:
        client.delete_collection(name=backup_name)
        nomes.remove(backup_name)

    havia_anterior = COLLECTION in nomes
    if havia_anterior:
        anterior = client.get_collection(name=COLLECTION)
        anterior.modify(name=backup_name)

    try:
        nova_collection.modify(name=COLLECTION)
    except Exception:
        # Tenta restaurar o índice anterior caso a renomeação falhe.
        if havia_anterior and backup_name in collection_names(client):
            client.get_collection(name=backup_name).modify(name=COLLECTION)
        raise

    if backup_name in collection_names(client):
        client.delete_collection(name=backup_name)


def build() -> None:
    try:
        chunks = load_chunks()
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return

    client = get_client()
    temp = create_build_collection(client, len(chunks))

    # Respeita o limite do cliente Chroma e evita lotes gigantes.
    try:
        limite_chroma = int(client.get_max_batch_size())
    except Exception:
        limite_chroma = CHROMA_BATCH_SIZE
    batch_size = max(1, min(CHROMA_BATCH_SIZE, limite_chroma))

    print(f"Gerando embeddings de {len(chunks)} chunks em lotes de {batch_size}...")

    try:
        processados = 0
        for lote in iter_batches(chunks, batch_size):
            embeddings = embed_texts(
                [embedding_text(chunk) for chunk in lote],
                query=False,
            )

            temp.add(
                ids=[str(chunk["chunk_id"]) for chunk in lote],
                embeddings=embeddings,
                documents=[str(chunk["text"]) for chunk in lote],
                metadatas=[metadata_for(chunk) for chunk in lote],
            )

            processados += len(lote)
            print(f"  {processados}/{len(chunks)} indexados")

        gravados = temp.count()
        if gravados != len(chunks):
            raise RuntimeError(
                f"Índice incompleto: esperado {len(chunks)}, gravado {gravados}."
            )

        swap_collection(client, temp)

    except Exception as exc:
        print(f"Falha durante o build: {exc}")
        print("O índice anterior foi preservado quando possível.")
        return

    print(f"{len(chunks)} chunks indexados em ./{DB_DIR}/")


# ---------------------------------------------------------------------------
# Recuperação com janela
# ---------------------------------------------------------------------------

def validate_collection_model(col) -> None:
    metadata = col.metadata or {}
    modelo = metadata.get("embedding_model")
    if modelo and modelo != MODEL_NAME:
        raise RuntimeError(
            f"O índice foi criado com '{modelo}', mas o script usa '{MODEL_NAME}'. "
            "Rode o build novamente."
        )


def get_window(col, meta_central: dict[str, Any]) -> list[dict[str, Any]]:
    documento = str(meta_central["document_id"])
    posicao = int(meta_central["doc_pos"])
    inicio = max(0, posicao - WINDOW)
    fim = posicao + WINDOW

    resposta = col.get(
        where={
            "$and": [
                {"document_id": documento},
                {"doc_pos": {"$gte": inicio}},
                {"doc_pos": {"$lte": fim}},
            ]
        },
        include=["documents", "metadatas"],
    )

    registros = []
    for record_id, documento_texto, metadata in zip(
        resposta["ids"],
        resposta.get("documents") or [],
        resposta.get("metadatas") or [],
    ):
        if documento_texto is None or metadata is None:
            continue
        registros.append(
            {
                "id": record_id,
                "text": documento_texto,
                "metadata": metadata,
            }
        )

    registros.sort(key=lambda item: int(item["metadata"]["doc_pos"]))
    return registros


def merge_overlapping_chunks(texts: list[str]) -> str:
    """Remove a repetição exata causada pelo overlap da Etapa 1."""
    if not texts:
        return ""

    palavras = texts[0].split()

    for texto in texts[1:]:
        proximas = texto.split()
        max_overlap = min(120, len(palavras), len(proximas))
        overlap = 0

        for tamanho in range(max_overlap, 4, -1):
            if palavras[-tamanho:] == proximas[:tamanho]:
                overlap = tamanho
                break

        palavras.extend(proximas[overlap:])

    return " ".join(palavras)


def format_page_ranges(metadatas: list[dict[str, Any]]) -> str:
    intervalos = sorted(
        (
            int(meta.get("page_start", 1)),
            int(meta.get("page_end", meta.get("page_start", 1))),
        )
        for meta in metadatas
    )

    mesclados: list[list[int]] = []
    for inicio, fim in intervalos:
        if not mesclados or inicio > mesclados[-1][1] + 1:
            mesclados.append([inicio, fim])
        else:
            mesclados[-1][1] = max(mesclados[-1][1], fim)

    return ", ".join(
        str(inicio) if inicio == fim else f"{inicio}-{fim}"
        for inicio, fim in mesclados
    )


def retrieve(pergunta: str) -> list[dict[str, Any]]:
    pergunta = pergunta.strip()
    if not pergunta:
        raise ValueError("A pergunta está vazia.")

    col = get_collection()
    if col is None or col.count() == 0:
        raise RuntimeError("Índice vazio — rode 'python 02_embed.py build' primeiro.")

    validate_collection_model(col)

    quantidade_candidatos = min(
        col.count(),
        max(TOP_K, TOP_K * CANDIDATE_MULTIPLIER),
    )

    resposta = col.query(
        query_embeddings=embed_texts([pergunta], query=True),
        n_results=quantidade_candidatos,
        include=["documents", "metadatas", "distances"],
    )

    resultados: list[dict[str, Any]] = []
    ids_usados: set[str] = set()

    for record_id, metadata, distancia in zip(
        resposta["ids"][0],
        resposta["metadatas"][0],
        resposta["distances"][0],
    ):
        if metadata is None or distancia is None:
            continue
        if record_id in ids_usados:
            continue

        similaridade = 1.0 - float(distancia)
        if MIN_SIMILARITY is not None and similaridade < MIN_SIMILARITY:
            continue

        janela = get_window(col, metadata)
        if not janela:
            continue

        ids_janela = {str(item["id"]) for item in janela}
        if ids_janela & ids_usados:
            continue

        metadatas = [item["metadata"] for item in janela]
        textos = [str(item["text"]) for item in janela]

        resultados.append(
            {
                "similarity": similaridade,
                "source": str(metadata["source"]),
                "pages": format_page_ranges(metadatas),
                "text": merge_overlapping_chunks(textos),
                "chunk_ids": [str(item["id"]) for item in janela],
            }
        )
        ids_usados.update(ids_janela)

        if len(resultados) >= TOP_K:
            break

    return resultados


def ask(pergunta: str) -> None:
    try:
        resultados = retrieve(pergunta)
    except (ValueError, RuntimeError) as exc:
        print(exc)
        return

    print(f'\nPergunta: "{pergunta}"\n' + "-" * 70)

    if not resultados:
        print("Nenhum trecho passou pelos critérios de recuperação.")
        return

    for resultado in resultados:
        texto = str(resultado["text"])
        previa = texto[:PREVIEW_CHARS]

        print(
            f"\n▸ {resultado['source']} · pág. {resultado['pages']} "
            f"· similaridade {resultado['similarity']:.3f}"
        )
        print(previa + ("…" if len(texto) > PREVIEW_CHARS else ""))

    print("\n" + "-" * 70)
    print("Esses grupos de contexto podem ser enviados ao modelo gerador de respostas.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"build", "ask"}:
        print(__doc__)
        return

    comando = sys.argv[1]
    if comando == "build":
        build()
        return

    pergunta = " ".join(sys.argv[2:]).strip()
    if not pergunta:
        pergunta = "escamação no couro cabeludo"
    ask(pergunta)


if __name__ == "__main__":
    main()