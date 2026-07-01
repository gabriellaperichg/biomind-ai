"""
Biomind — Etapa 2 do pipeline: embeddings + índice (Chroma) com janela de contexto.

  build : gera os embeddings dos chunks (chunks.jsonl) e grava no índice Chroma.
  ask   : busca os trechos mais parecidos e devolve cada um JUNTO com os
          vizinhos (contexto ao redor) e a citação (arquivo + página).

Uso:
  python 02_embed.py build
  python 02_embed.py ask "paciente com escamacao e coceira no couro cabeludo"

Tudo roda offline. O modelo bge-m3 baixa uma única vez na primeira execução
(~2GB) e depois nunca mais precisa de internet.
"""

import sys
import json
import chromadb

CHUNKS_FILE = "chunks.jsonl"
DB_DIR = "biomind_db"
COLLECTION = "casos"
TOP_K = 4        # quantos trechos a busca retorna
WINDOW = 1       # quantos chunks vizinhos incluir de cada lado (a "janela")

# --- embeddings locais (bge-m3) ---
_model = None


def embed_texts(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Carregando bge-m3 (baixa uma vez, depois roda offline)...")
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model.encode(texts, normalize_embeddings=True).tolist()


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def build():
    try:
        chunks = [json.loads(l) for l in open(CHUNKS_FILE, encoding="utf-8")]
    except FileNotFoundError:
        print(f"{CHUNKS_FILE} não encontrado — rode o 01_chunk.py primeiro.")
        return
    if not chunks:
        print("chunks.jsonl está vazio.")
        return

    # recria a coleção do zero pra não duplicar
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    print(f"Gerando embeddings de {len(chunks)} chunks...")
    embeddings = embed_texts([c["text"] for c in chunks])

    col.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": c["source"], "page": c["page"], "chunk_id": i}
            for i, c in enumerate(chunks)
        ],
    )
    print(f"{len(chunks)} chunks indexados em ./{DB_DIR}/")


def _pega(col, gid):
    r = col.get(ids=[str(gid)])
    if r["ids"]:
        return r["documents"][0], r["metadatas"][0]
    return None, None


def ask(pergunta):
    col = get_collection()
    if col.count() == 0:
        print("Índice vazio — rode 'python 02_embed.py build' primeiro.")
        return

    q_emb = embed_texts([pergunta])
    res = col.query(query_embeddings=q_emb, n_results=TOP_K)

    print(f'\nPergunta: "{pergunta}"\n' + "-" * 60)
    ja_mostrados = set()

    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        gid, src = meta["chunk_id"], meta["source"]
        if gid in ja_mostrados:
            continue

        # monta a JANELA: o trecho + vizinhos do mesmo documento
        partes = []
        for off in range(-WINDOW, WINDOW + 1):
            d, m = _pega(col, gid + off)
            if d and m and m["source"] == src:
                partes.append((m["chunk_id"], d, m["page"]))
                ja_mostrados.add(m["chunk_id"])

        partes.sort()
        texto = " (…) ".join(p[1] for p in partes)
        paginas = ", ".join(str(p) for p in sorted({p[2] for p in partes}))
        sim = 1 - dist

        print(f"\n▸ {src} · pág. {paginas} · similaridade {sim:.2f}")
        print(texto[:700] + ("…" if len(texto) > 700 else ""))

    print("\n" + "-" * 60)
    print("Estes são os trechos que o modelo receberia como contexto para orientar o caso.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "ask"):
        print(__doc__)
    elif sys.argv[1] == "build":
        build()
    else:
        ask(" ".join(sys.argv[2:]) or "escamacao no couro cabeludo")