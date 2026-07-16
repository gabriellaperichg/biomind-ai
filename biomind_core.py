"""
Biomind — lógica central compartilhada (usada pelo CLI e pela interface web).

Faz a recuperação (com janela de contexto), aplica o piso de relevância e,
se houver material relevante, chama o modelo local (Ollama) com o prompt de
"orienta, não diagnostica". A função responder() devolve dados estruturados.
"""

import chromadb
import requests

# ------------------- configuração -------------------
DB_DIR = "biomind_db"
COLLECTION = "casos"
TOP_K = 4
WINDOW = 1
PISO_RELEVANCIA = 0.40          # calibre com o bge-m3 real (veja o README)
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"       # troque pelo modelo que você baixou

# Aviso fixo anexado a toda resposta bem-sucedida (não depende do modelo).
AVISO_LEGAL = (
    "⚠️ Observação: As informações apresentadas são baseadas em estudos, evidências "
    "científicas e boas práticas, tendo como finalidade fornecer sugestões e apoiar a "
    "análise do caso. Esta inteligência artificial não realiza diagnósticos, não define "
    "condutas e não substitui a avaliação clínica. A decisão final sobre a conduta a ser "
    "adotada é de responsabilidade exclusiva da biomédica responsável pelo acompanhamento "
    "do paciente."
)

SYSTEM_PROMPT = """Você é o Biomind, assistente de APOIO À DECISÃO CLÍNICA em tricologia, usado por biomédicas e tricologistas.

IDIOMA — regra crítica:
- Escreva TUDO em português do Brasil. Os trechos da base podem estar em inglês; TRADUZA os termos técnicos.
- Use a terminologia em português: "eflúvio telógeno" (não "telogen effluvium"), "fios velus" (não "vellus-like"), "hiperandrogenismo ovariano" (não "ovarian hyperandrogenism"), "acantose nigricante" (não "acanthosis nigricans"), "miniaturização" (não "miniaturization").
- Nenhuma palavra em inglês na resposta. Se um termo não tiver tradução consagrada, escreva em português e ponha o original entre parênteses.

SEU PAPEL — orientar a investigação, nunca concluir por ela:
- Use APENAS as informações dos TRECHOS DA BASE fornecidos. Não use conhecimento externo nem invente.
- Se os trechos não cobrirem o caso, diga com clareza que a base não tem material suficiente — não force uma resposta.
- NUNCA dê diagnóstico definitivo nem prescreva tratamento, medicamento, dose ou conduta. Se um trecho trouxer condutas ou doses, não as repita como recomendação: aponte que a fonte traz condutas para este perfil e que a profissional deve revisá-las.
- Cite as fontes com os números dos trechos: [1], [2].

QUALIDADE — o que faz uma boa orientação:
- Seja CONCRETA e ACIONÁVEL. Cada item deve dizer algo que a profissional pode de fato fazer ou perguntar.
- PROIBIDO escrever itens vagos ou circulares como "investigar como a condição afeta o quadro" ou "avaliar a situação do paciente". Se não tiver algo específico e apoiado nos trechos, omita o item.
- Máximo de 3 itens por seção. Prefira poucos itens fortes a muitos itens fracos.
- Cada item deve ser uma frase curta e direta, em português claro.
- Não repita no item o que já está no título da seção.

FORMATO — use "### " nos cabeçalhos, "- " nos itens, e inclua só as seções relevantes:
### O que investigar primeiro
### Perguntas para o paciente
### O que observar
### Hipóteses a diferenciar (a investigar, não concluir)
### Sinais de alerta / quando encaminhar

Termine com UMA linha curta lembrando que é orientação para investigação e que a decisão é da profissional."""

# ------------------- embeddings (bge-m3) -------------------
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
    """[{text, source, pages, sim}] — cada trecho já com sua janela de contexto."""
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
            "sim": round(1 - dist, 3),
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
        # temperatura baixa = mais fiel às fontes, menos "criatividade"
        "options": {"temperature": 0.2},
    }, timeout=600)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def responder(pergunta):
    """Devolve um dicionário estruturado com o resultado."""
    trechos = recuperar(pergunta)

    if not trechos or trechos[0]["sim"] < PISO_RELEVANCIA:
        melhor = trechos[0]["sim"] if trechos else None
        return {
            "status": "sem_material",
            "best_sim": melhor,
            "message": ("Não encontrei material suficientemente relevante na base sobre este "
                        "caso. Registre o caso e considere acrescentar referências sobre o tema."),
            "sources": [],
        }

    contexto = montar_contexto(trechos)
    user = (f"CASO DESCRITO PELA PROFISSIONAL:\n{pergunta}\n\n"
            f"TRECHOS DA BASE:\n{contexto}\n\n"
            "Com base SOMENTE nos trechos acima, oriente o que a profissional deve investigar.")
    try:
        resposta = perguntar_ollama(SYSTEM_PROMPT, user)
    except requests.exceptions.RequestException as e:
        return {
            "status": "erro_modelo",
            "message": (f"Não consegui falar com o Ollama em {OLLAMA_URL}. Verifique se ele "
                        f"está rodando e se o modelo foi baixado (ollama pull {OLLAMA_MODEL})."),
            "detail": str(e),
            "sources": [],
        }

    return {
        "status": "ok",
        "answer": resposta,
        "disclaimer": AVISO_LEGAL,
        "best_sim": trechos[0]["sim"],
        "sources": [
            {"n": i + 1, "source": t["source"], "pages": t["pages"], "sim": t["sim"]}
            for i, t in enumerate(trechos)
        ],
    }