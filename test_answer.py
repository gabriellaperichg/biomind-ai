"""
Biomind — Etapa 3 do pipeline: orientação com o modelo local (Ollama).

Junta tudo: busca os trechos relevantes (etapa 2), aplica um PISO DE RELEVÂNCIA
e, se houver material bom o suficiente, passa os trechos pro modelo local com o
prompt de "orienta, não diagnostica". Se o melhor trecho for fraco, ele NÃO
inventa — avisa que a base não tem material relevante.

Pré-requisitos:
  1. Ollama instalado e rodando (https://ollama.com)
  2. Um modelo baixado, ex.:  ollama pull llama3.1
  3. O índice já criado (python 02_embed.py build)

Uso:
  python 03_answer.py "paciente do sexo feminino, 29 anos, rarefacao no topo, deseja engravidar"
"""

import sys
import chromadb
import requests

DB_DIR = "biomind_db"
COLLECTION = "casos"
TOP_K = 4
WINDOW = 1
PISO_RELEVANCIA = 0.40           # similaridade mínima do MELHOR trecho pra tentar responder
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"        # troque pelo modelo que você baixou

SYSTEM_PROMPT = """Você é o Biomind, assistente de APOIO À DECISÃO CLÍNICA em tricologia, usado por biomédicas e tricologistas. Responda sempre em português do Brasil.

Seu papel é ORIENTAR A INVESTIGAÇÃO — nunca concluir por ela. Regras invioláveis:
- Use APENAS as informações dos TRECHOS DA BASE fornecidos. Não use conhecimento externo nem invente.
- Se os trechos não cobrirem o caso, diga com clareza que a base não tem material suficiente sobre isso — não force uma resposta.
- NUNCA dê diagnóstico definitivo nem prescreva tratamento, medicamento, dose ou conduta. Mesmo que um trecho traga condutas ou doses, não as repita como recomendação: aponte que a fonte traz condutas para este perfil e que a profissional deve revisá-las.
- Sugira caminhos de investigação, perguntas a fazer ao paciente, o que observar, e hipóteses a DIFERENCIAR (sempre como hipóteses a investigar, não como respostas).
- Cite as fontes usando os números dos trechos, ex.: [1], [2].
- Deixe claro que a decisão clínica é da profissional.

Estruture a resposta com estes cabeçalhos (use "### " e inclua só os relevantes):
### O que investigar primeiro
### Perguntas para o paciente
### O que observar
### Hipóteses a diferenciar (a investigar, não concluir)
### Sinais de alerta / quando encaminhar

Termine com uma linha curta lembrando que é orientação para investigação e que a decisão é da profissional."""

# --- embeddings locais (bge-m3), igual à etapa 2 ---
_model = None


def embed_texts(texts):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model.encode(texts, normalize_embeddings=True).tolist()


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def _pega(col, gid):
    r = col.get(ids=[str(gid)])
    if r["ids"]:
        return r["documents"][0], r["metadatas"][0]
    return None, None


def recuperar(pergunta):
    """Retorna [{text, source, pages, sim}] — cada trecho já com sua janela de contexto."""
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=embed_texts([pergunta]), n_results=TOP_K)
    trechos, ja = [], set()
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        gid, src = meta["chunk_id"], meta["source"]
        if gid in ja:
            continue
        partes = []
        for off in range(-WINDOW, WINDOW + 1):
            d, m = _pega(col, gid + off)
            if d and m and m["source"] == src:
                partes.append((m["chunk_id"], d, m["page"]))
                ja.add(m["chunk_id"])
        partes.sort()
        trechos.append({
            "text": " (…) ".join(p[1] for p in partes),
            "source": src,
            "pages": ", ".join(str(p) for p in sorted({p[2] for p in partes})),
            "sim": 1 - dist,
        })
    return trechos


def montar_contexto(trechos):
    blocos = []
    for i, t in enumerate(trechos, 1):
        blocos.append(f"[{i}] (fonte: {t['source']}, pág. {t['pages']})\n{t['text']}")
    return "\n\n".join(blocos)


def perguntar_ollama(system, user):
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def responder(pergunta):
    trechos = recuperar(pergunta)

    # PISO DE RELEVÂNCIA: se não achou nada ou o melhor é fraco, não inventa.
    if not trechos or trechos[0]["sim"] < PISO_RELEVANCIA:
        melhor = f"{trechos[0]['sim']:.2f}" if trechos else "nenhum"
        print("\nNão encontrei material suficientemente relevante na base sobre este caso "
              f"(melhor similaridade: {melhor}, piso: {PISO_RELEVANCIA:.2f}).")
        print("Registre o caso e considere acrescentar referências sobre o tema à base.")
        return

    contexto = montar_contexto(trechos)
    user = (f"CASO DESCRITO PELA PROFISSIONAL:\n{pergunta}\n\n"
            f"TRECHOS DA BASE:\n{contexto}\n\n"
            "Com base SOMENTE nos trechos acima, oriente o que a profissional deve investigar.")

    try:
        resposta = perguntar_ollama(SYSTEM_PROMPT, user)
    except requests.exceptions.RequestException as e:
        print(f"\nNão consegui falar com o Ollama em {OLLAMA_URL}.")
        print("Verifique se o Ollama está rodando e se o modelo foi baixado "
              f"(ollama pull {OLLAMA_MODEL}). Detalhe: {e}")
        return

    print("\n" + resposta)
    print("\n" + "-" * 60)
    print("Fontes consultadas:")
    for i, t in enumerate(trechos, 1):
        print(f"  [{i}] {t['source']} — pág. {t['pages']} (similaridade {t['sim']:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        responder(" ".join(sys.argv[1:]))