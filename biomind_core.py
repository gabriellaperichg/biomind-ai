"""
Biomind — núcleo RAG local com recuperação clínica conservadora.

Princípios:
- embeddings, Chroma e Ollama executados na infraestrutura autorizada;
- recuperação em múltiplas consultas, reranking e diversidade de fontes;
- exclusão de documentos administrativos em perguntas clínicas gerais;
- proteção contra transferência de diagnóstico/conduta de outro caso;
- validação mínima da resposta antes de entregá-la à interface.
"""

from __future__ import annotations

import html
import importlib
import importlib.util
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Estas variáveis precisam existir antes de sentence-transformers / HF Hub
# serem importados. O import do modelo é feito de forma preguiçosa abaixo.
if os.getenv("BIOMIND_OFFLINE", "1") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
import requests

from retrieval_quality import (
    build_query_variants,
    content_similarity,
    enrich_chunk_metadata,
    score_candidate,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

AQUI = Path(__file__).resolve().parent


def _resolver_caminho_projeto(valor: str | None, padrao: Path) -> str:
    """Resolve caminhos relativos sempre a partir da pasta do projeto."""
    caminho = Path(valor).expanduser() if valor else padrao
    if not caminho.is_absolute():
        caminho = AQUI / caminho
    return str(caminho.resolve())


DB_DIR = _resolver_caminho_projeto(
    os.getenv("BIOMIND_DB_DIR"),
    AQUI / "biomind_db",
)
COLLECTION = os.getenv("BIOMIND_COLLECTION", "casos")
EMBED_MODEL = os.getenv("BIOMIND_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("BIOMIND_EMBED_DEVICE", "cpu")

TOP_K = max(1, int(os.getenv("BIOMIND_TOP_K", "4")))
WINDOW = max(0, int(os.getenv("BIOMIND_WINDOW", "0")))
CANDIDATE_MULTIPLIER = max(
    2,
    int(os.getenv("BIOMIND_CANDIDATE_MULTIPLIER", "8")),
)
PISO_RELEVANCIA = float(os.getenv("BIOMIND_PISO_RELEVANCIA", "0.35"))
MIN_RERANK_SCORE = float(os.getenv("BIOMIND_MIN_RERANK_SCORE", "0.50"))
MAX_CONTEXT_CHARS = max(
    1000,
    int(os.getenv("BIOMIND_MAX_CONTEXT_CHARS", "10000")),
)
EMBED_BATCH_SIZE = max(
    1,
    int(os.getenv("BIOMIND_EMBED_BATCH_SIZE", "8")),
)
MAX_QUERY_VARIANTS = max(
    1,
    int(os.getenv("BIOMIND_MAX_QUERY_VARIANTS", "4")),
)
MAX_RESULTS_PER_SOURCE = max(
    1,
    int(os.getenv("BIOMIND_MAX_RESULTS_PER_SOURCE", "1")),
)
MAX_CONTENT_SIMILARITY = float(
    os.getenv("BIOMIND_MAX_CONTENT_SIMILARITY", "0.86")
)
WINDOW_SAME_SECTION_ONLY = (
    os.getenv("BIOMIND_WINDOW_SAME_SECTION_ONLY", "1") == "1"
)

OFFLINE_ONLY = os.getenv("BIOMIND_OFFLINE", "1") == "1"
ENFORCE_LOCAL_OLLAMA = os.getenv("BIOMIND_LOCAL_OLLAMA_ONLY", "1") == "1"
OLLAMA_ALLOWED_HOSTS = {
    host.strip().lower()
    for host in os.getenv(
        "BIOMIND_ALLOWED_OLLAMA_HOSTS",
        "localhost,127.0.0.1,::1,ollama",
    ).split(",")
    if host.strip()
}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_NUM_CTX = max(1024, int(os.getenv("OLLAMA_NUM_CTX", "4096")))
OLLAMA_NUM_PREDICT = max(64, int(os.getenv("OLLAMA_NUM_PREDICT", "450")))
OLLAMA_CONNECT_TIMEOUT = max(
    1,
    int(os.getenv("OLLAMA_CONNECT_TIMEOUT", "10")),
)
OLLAMA_READ_TIMEOUT = max(
    30,
    int(os.getenv("OLLAMA_READ_TIMEOUT", "300")),
)
MAX_GENERATIONS = max(1, int(os.getenv("BIOMIND_MAX_GENERATIONS", "1")))
REPAIR_UNSAFE_RESPONSE = (
    os.getenv("BIOMIND_REPAIR_UNSAFE_RESPONSE", "1") == "1"
)

PROMPT_VERSION = "clinical-rag-v3.3"

AVISO_LEGAL = (
    "⚠️ Observação: As informações apresentadas são baseadas em estudos, evidências "
    "científicas e boas práticas, tendo como finalidade fornecer sugestões e apoiar a "
    "análise do caso. Esta inteligência artificial não realiza diagnósticos, não define "
    "condutas e não substitui a avaliação clínica. A decisão final sobre a conduta a ser "
    "adotada é de responsabilidade exclusiva da biomédica responsável pelo acompanhamento "
    "do paciente."
)

SYSTEM_PROMPT = """Você é o Biomind, um assistente de apoio à análise clínica em tricologia, utilizado por biomédicas e profissionais habilitadas.

OBJETIVO PRINCIPAL

Responda diretamente à pergunta feita pela profissional usando somente os TRECHOS DA BASE apresentados.

Não repita o caso apenas com outras palavras. Use os dados do caso para produzir uma orientação prática, segura e sustentada pelas fontes.

REGRAS DE FONTES E SEGURANÇA

- Use somente informações contidas nos TRECHOS DA BASE fornecidos nesta solicitação.
- Não use conhecimento externo, memória própria ou suposições clínicas.
- Cite cada afirmação clínica com [1], [2] e assim por diante.
- Nunca invente uma citação.
- Quando a base não sustentar uma conclusão, declare claramente essa limitação.
- Não dê diagnóstico definitivo.
- Não prescreva medicamentos, suplementos, doses, pomadas ou procedimentos.
- Não recomende iniciar, suspender ou trocar medicamentos.
- Não transfira diagnóstico, exame ou tratamento de outro paciente descrito nos documentos.
- Termos de consentimento e formulários administrativos não devem fundamentar a resposta clínica.

PRESERVAÇÃO DO CASO

- Preserve exatamente os sintomas informados pela profissional.
- Não substitua palavras por outras de significado diferente.
- Quando houver provável erro de digitação, mantenha o termo original e indique que ele precisa ser confirmado.
- Não altere sexo, idade, duração, localização ou intensidade relatada.

COMO RESPONDER A “COMO PROSSEGUIR”

Quando a pergunta solicitar como prosseguir, o que fazer, qual a próxima etapa ou como conduzir o caso:

- responda primeiro com ações concretas e seguras;
- informe o que precisa ser verificado antes de uma nova conduta;
- informe o que deve ser perguntado diretamente à paciente;
- informe o que deve ser observado e registrado;
- informe os sinais que justificam encaminhamento;
- não transforme a resposta em uma lista de perguntas teóricas para a profissional;
- não responda apenas que é necessário investigar mais.

SINTOMAS NOVOS APÓS PROCEDIMENTO

Quando houver sintomas novos após mesoterapia, microagulhamento, injeção ou outro procedimento:

- trate o relato como possível evento adverso que precisa ser avaliado, sem concluir diagnóstico;
- relacione temporalmente o início dos sintomas com a última sessão;
- verifique se os sinais estão localizados nos pontos de aplicação;
- diferencie reação leve, reação em progressão e sinais de alerta somente conforme a base;
- informe o que registrar sobre o produto, sessão e técnica quando a base sustentar essa investigação;
- não considere automaticamente coceira, dor, edema ou nódulos como reação normal;
- uma nova sessão somente pode ser discutida depois da avaliação do quadro, quando essa orientação estiver sustentada pelos trechos.

PERGUNTAS À PACIENTE

Na seção de perguntas, escreva somente perguntas que possam ser feitas diretamente à paciente.

Não escreva perguntas teóricas como:

- “Quais são as complicações da mesoterapia?”
- “Qual é o intervalo recomendado?”
- “O que é considerado normal?”

Transforme-as em perguntas aplicáveis ao caso, como:

- quando os sintomas começaram;
- onde surgiram;
- se estão aumentando;
- se existe calor, secreção, edema ou outros sinais mencionados pela base.

QUALIDADE

- Seja específica para o caso.
- Não repita a mesma informação em seções diferentes.
- Não crie uma seção apenas para reescrever a pergunta.
- Explique por que cada pergunta ou observação muda a interpretação.
- Não use frases vagas como “avaliar o quadro” sem dizer o que deve ser avaliado.
- Conclua todas as frases.
- Use no máximo quatro itens por seção.
- Priorize a próxima etapa prática.

FORMATO

Use somente as seções relevantes:

### Como prosseguir agora

Obrigatória quando a pergunta solicitar uma próxima etapa ou quando houver sintoma novo após procedimento.

### Perguntas que mudam a interpretação

Inclua somente perguntas dirigidas à paciente.

### O que observar ou registrar

### Hipóteses sustentadas pela base

### Sinais de alerta / quando encaminhar

### Limites da base

Não crie a seção “Leitura inicial do caso”, salvo quando houver uma interpretação importante que não seja mera repetição.
"""

REPAIR_PROMPT = """Você é um revisor de segurança e utilidade clínica. Reescreva a RESPOSTA PROVISÓRIA usando somente o CASO e os TRECHOS fornecidos.

Obrigatório:
- remova prescrições, doses e ordens para iniciar, suspender ou trocar tratamentos;
- não transfira diagnóstico, exames ou conduta de outro paciente para o caso atual;
- preserve exatamente os sintomas, a duração, a localização e a intensidade descritos;
- preserve apenas afirmações sustentadas e cite-as com [1], [2] etc.;
- elimine repetições, perguntas teóricas e frases vagas;
- quando a pergunta pedir como prosseguir, mantenha a seção “### Como prosseguir agora”;
- nessa seção, informe a próxima etapa segura, o que observar, o que registrar e quando encaminhar;
- formule somente perguntas que possam ser feitas diretamente à paciente;
- quando faltar suporte, diga exatamente qual informação ou protocolo não está disponível;
- conclua todas as frases e mantenha o formato em português do Brasil.

Entregue somente a resposta revisada."""

logger = logging.getLogger("biomind")

_model = None
_model_lock = threading.Lock()
_client = None
_client_lock = threading.Lock()
_http = requests.Session()
_http.trust_env = False
_generation_gate = threading.BoundedSemaphore(MAX_GENERATIONS)


# ---------------------------------------------------------------------------
# Embeddings e Chroma
# ---------------------------------------------------------------------------

def _carregar_classe_sentence_transformer():
    """
    Importa sentence-transformers de forma preguiçosa.

    O nome do pacote no pip é ``sentence-transformers`` e o módulo Python
    correspondente é ``sentence_transformers``.
    """
    try:
        module = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "A dependência 'sentence-transformers' não está instalada no Python "
            f"que executa o Biomind: {sys.executable}. Instale com: "
            f'"{sys.executable}" -m pip install -U sentence-transformers'
        ) from exc

    sentence_transformer = getattr(module, "SentenceTransformer", None)
    if sentence_transformer is None:
        raise RuntimeError(
            "O módulo 'sentence_transformers' foi encontrado, mas não expõe "
            "'SentenceTransformer'. Reinstale a dependência no ambiente virtual."
        )
    return sentence_transformer


def get_model():
    """Carrega uma única instância do modelo de embeddings por processo."""
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                sentence_transformer = _carregar_classe_sentence_transformer()

                try:
                    _model = sentence_transformer(
                        EMBED_MODEL,
                        local_files_only=OFFLINE_ONLY,
                        device=EMBED_DEVICE,
                    )
                except Exception as exc:
                    modo = "offline" if OFFLINE_ONLY else "online"
                    raise RuntimeError(
                        f"Não foi possível carregar o modelo de embeddings "
                        f"'{EMBED_MODEL}' em modo {modo}, dispositivo "
                        f"'{EMBED_DEVICE}'. Verifique se o modelo já está no cache "
                        "local e se o dispositivo configurado está disponível."
                    ) from exc

                logger.info(
                    "Modelo de embeddings carregado. Modelo=%s Dispositivo=%s",
                    EMBED_MODEL,
                    EMBED_DEVICE,
                )

    return _model


def embed_queries(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    encoder = model.encode_query if hasattr(model, "encode_query") else model.encode
    vectors = encoder(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(texto: str) -> list[list[float]]:
    """Compatibilidade com scripts anteriores."""
    return embed_queries([texto])


def get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = chromadb.PersistentClient(path=DB_DIR)
    return _client


def get_collection():
    """Obtém a coleção existente; nunca cria uma coleção vazia por engano."""
    try:
        return get_client().get_collection(name=COLLECTION)
    except Exception:
        return None


def validar_modelo_indice(col) -> None:
    metadata = col.metadata or {}
    modelo_indice = metadata.get("embedding_model")
    if modelo_indice and modelo_indice != EMBED_MODEL:
        raise RuntimeError(
            f"O índice foi criado com '{modelo_indice}', mas o servidor usa "
            f"'{EMBED_MODEL}'. Refaça o build ou ajuste BIOMIND_EMBED_MODEL."
        )


# ---------------------------------------------------------------------------
# Recuperação e reranking
# ---------------------------------------------------------------------------

def _janela(col, meta_central: dict[str, Any]) -> list[dict[str, Any]]:
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

    registros: list[dict[str, Any]] = []
    central_section = str(meta_central.get("section") or "general")
    central_content = str(meta_central.get("content_type") or "general")

    for record_id, document_text, metadata in zip(
        resposta.get("ids") or [],
        resposta.get("documents") or [],
        resposta.get("metadatas") or [],
    ):
        if document_text is None or metadata is None:
            continue

        if WINDOW_SAME_SECTION_ONLY and int(metadata.get("doc_pos", -1)) != posicao:
            neighbor_section = str(metadata.get("section") or "general")
            neighbor_content = str(metadata.get("content_type") or "general")

            same_semantic_section = (
                central_section != "general" and neighbor_section == central_section
            ) or (
                central_content != "general" and neighbor_content == central_content
            )
            if not same_semantic_section:
                continue

        registros.append(
            {
                "id": str(record_id),
                "text": str(document_text),
                "metadata": metadata,
            }
        )

    registros.sort(key=lambda item: int(item["metadata"]["doc_pos"]))
    return registros


def _mesclar_overlap(textos: list[str]) -> str:
    if not textos:
        return ""

    palavras = textos[0].split()
    for texto in textos[1:]:
        seguintes = texto.split()
        max_overlap = min(120, len(palavras), len(seguintes))
        overlap = 0
        for tamanho in range(max_overlap, 4, -1):
            if palavras[-tamanho:] == seguintes[:tamanho]:
                overlap = tamanho
                break
        palavras.extend(seguintes[overlap:])
    return " ".join(palavras)


def _formatar_paginas(metadatas: list[dict[str, Any]]) -> str:
    intervalos = sorted(
        (
            int(meta.get("page_start", meta.get("page", 1))),
            int(meta.get("page_end", meta.get("page_start", meta.get("page", 1)))),
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


def _candidate_pool(pergunta: str, col) -> list[dict[str, Any]]:
    variants = build_query_variants(
        pergunta,
        max_variants=MAX_QUERY_VARIANTS,
    )
    quantidade = min(
        col.count(),
        max(TOP_K * CANDIDATE_MULTIPLIER, TOP_K + 12),
    )

    query_result = col.query(
        query_embeddings=embed_queries(variants),
        n_results=quantidade,
        include=["documents", "metadatas", "distances"],
    )

    candidates: dict[str, dict[str, Any]] = {}

    for variant_index, variant in enumerate(variants):
        ids = query_result.get("ids", [[]])[variant_index]
        documents = query_result.get("documents", [[]])[variant_index]
        metadatas = query_result.get("metadatas", [[]])[variant_index]
        distances = query_result.get("distances", [[]])[variant_index]

        for record_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            if metadata is None or text is None or distance is None:
                continue

            semantic_similarity = 1.0 - float(distance)
            if semantic_similarity < PISO_RELEVANCIA:
                continue

            record_key = str(record_id)
            current = candidates.get(record_key)
            if current is None or semantic_similarity > current["semantic_similarity"]:
                candidates[record_key] = {
                    "id": record_key,
                    "text": str(text),
                    "metadata": metadata,
                    "semantic_similarity": semantic_similarity,
                    "matched_variant": variant,
                }

    scored: list[dict[str, Any]] = []
    for candidate in candidates.values():
        metadata = candidate["metadata"]
        source = str(metadata.get("source") or "Fonte não identificada")
        quality = score_candidate(
            pergunta,
            semantic_similarity=candidate["semantic_similarity"],
            source=source,
            text=candidate["text"],
            metadata=metadata,
        )
        candidate["quality"] = quality
        scored.append(candidate)

    scored.sort(
        key=lambda item: (
            item["quality"].adjusted_score,
            item["semantic_similarity"],
        ),
        reverse=True,
    )
    return scored


def _recuperar_com_debug(pergunta: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pergunta = pergunta.strip()
    if not pergunta:
        return [], []

    col = get_collection()
    if col is None or col.count() == 0:
        return [], []

    validar_modelo_indice(col)
    candidates = _candidate_pool(pergunta, col)

    selected: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    per_source: dict[str, int] = {}

    for candidate in candidates:
        metadata = candidate["metadata"]
        quality = candidate["quality"]
        source = str(metadata.get("source") or "Fonte não identificada")
        debug_item = {
            "id": candidate["id"],
            "source": source,
            "semantic_similarity": round(candidate["semantic_similarity"], 3),
            "adjusted_score": round(quality.adjusted_score, 3),
            "document_type": quality.document_type,
            "content_type": quality.content_type,
            "topic": quality.topic,
            "source_quality": quality.source_quality,
            "matched_variant": candidate["matched_variant"],
            "reasons": list(quality.reasons),
            "selected": False,
            "discard_reason": None,
            "preview": candidate["text"][:600],
        }

        if quality.rejected:
            debug_item["discard_reason"] = "rejeitado pela política de finalidade"
            debug.append(debug_item)
            continue

        if quality.adjusted_score < MIN_RERANK_SCORE:
            debug_item["discard_reason"] = "score ajustado abaixo do mínimo"
            debug.append(debug_item)
            continue

        if per_source.get(source, 0) >= MAX_RESULTS_PER_SOURCE:
            debug_item["discard_reason"] = "limite de resultados por fonte"
            debug.append(debug_item)
            continue

        window = _janela(col, metadata)
        if not window:
            debug_item["discard_reason"] = "janela vazia"
            debug.append(debug_item)
            continue

        window_ids = {item["id"] for item in window}
        if window_ids & used_ids:
            debug_item["discard_reason"] = "janela duplicada"
            debug.append(debug_item)
            continue

        window_metadatas = [item["metadata"] for item in window]
        merged_text = _mesclar_overlap([item["text"] for item in window])

        duplicated_source: str | None = None
        duplicated_similarity = 0.0

        for already_selected in selected:
            similarity = content_similarity(
                merged_text,
                already_selected["text"],
            )

            if similarity >= MAX_CONTENT_SIMILARITY:
                duplicated_source = already_selected["source"]
                duplicated_similarity = similarity
                break

        if duplicated_source is not None:
            debug_item["discard_reason"] = (
                "conteúdo muito semelhante a outro trecho já selecionado: "
                f"{duplicated_source} ({duplicated_similarity:.1%})"
            )
            debug.append(debug_item)
            continue

        # Reavalia a janela completa, porque um vizinho pode alterar a finalidade
        # do conteúdo (por exemplo, arrastar tratamento para um trecho de exame).
        merged_quality = score_candidate(
            pergunta,
            semantic_similarity=candidate["semantic_similarity"],
            source=source,
            text=merged_text,
            metadata=metadata,
        )
        if merged_quality.rejected or merged_quality.adjusted_score < MIN_RERANK_SCORE:
            debug_item["discard_reason"] = "janela completa inadequada"
            debug.append(debug_item)
            continue

        legacy_quality_metadata = enrich_chunk_metadata(source, merged_text)
        selected.append(
            {
                "text": merged_text,
                "source": source,
                "pages": _formatar_paginas(window_metadatas),
                "sim": round(candidate["semantic_similarity"], 3),
                "score": round(merged_quality.adjusted_score, 3),
                "chunk_ids": [item["id"] for item in window],
                "document_type": str(
                    metadata.get("document_type")
                    or legacy_quality_metadata["document_type"]
                ),
                "content_type": str(
                    metadata.get("content_type")
                    or legacy_quality_metadata["content_type"]
                ),
                "topic": str(metadata.get("topic") or legacy_quality_metadata["topic"]),
                "source_quality": str(
                    metadata.get("source_quality")
                    or legacy_quality_metadata["source_quality"]
                ),
                "section": str(metadata.get("section") or legacy_quality_metadata["section"]),
                "reasons": list(merged_quality.reasons),
            }
        )
        used_ids.update(window_ids)
        per_source[source] = per_source.get(source, 0) + 1
        debug_item["selected"] = True
        debug.append(debug_item)

        if len(selected) >= TOP_K:
            break

    return selected, debug


def recuperar(pergunta: str) -> list[dict[str, Any]]:
    """Retorna trechos reranqueados e diversificados."""
    selected, _ = _recuperar_com_debug(pergunta)
    return selected


def debug_recuperacao(pergunta: str) -> dict[str, Any]:
    """Expõe diagnóstico da recuperação sem chamar o Ollama."""
    selected, candidates = _recuperar_com_debug(pergunta)
    return {
        "question": pergunta,
        "query_variants": build_query_variants(
            pergunta,
            max_variants=MAX_QUERY_VARIANTS,
        ),
        "selected": selected,
        "candidates": candidates,
        "config": {
            "top_k": TOP_K,
            "window": WINDOW,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "min_semantic_similarity": PISO_RELEVANCIA,
            "min_rerank_score": MIN_RERANK_SCORE,
            "max_results_per_source": MAX_RESULTS_PER_SOURCE,
            "max_content_similarity": MAX_CONTENT_SIMILARITY,
        },
    }


def montar_contexto(trechos: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Monta o contexto com metadados de finalidade e teto de caracteres."""
    blocks: list[str] = []
    used: list[dict[str, Any]] = []
    total = 0

    for index, trecho in enumerate(trechos, start=1):
        source = html.escape(str(trecho["source"]), quote=True)
        pages = html.escape(str(trecho["pages"]), quote=True)
        block = (
            f'<trecho id="{index}" fonte="{source}" paginas="{pages}" '
            f'tipo_documento="{trecho.get("document_type", "desconhecido")}" '
            f'tipo_conteudo="{trecho.get("content_type", "general")}" '
            f'tema="{trecho.get("topic", "general_trichology")}" '
            f'qualidade_fonte="{trecho.get("source_quality", "medium")}">\n'
            f'{trecho["text"]}\n'
            f'</trecho>'
        )

        if blocks and total + len(block) > MAX_CONTEXT_CHARS:
            continue

        if not blocks and len(block) > MAX_CONTEXT_CHARS:
            allowed = max(0, MAX_CONTEXT_CHARS - 120)
            block = block[:allowed] + "\n[trecho truncado pelo limite]\n</trecho>"

        blocks.append(block)
        used.append(trecho)
        total += len(block)

    return "\n\n".join(blocks), used


# ---------------------------------------------------------------------------
# Ollama e validação da resposta
# ---------------------------------------------------------------------------

def _validar_ollama_local() -> None:
    if not ENFORCE_LOCAL_OLLAMA:
        return

    parsed = urlparse(OLLAMA_URL)
    host = (parsed.hostname or "").lower()
    if host not in OLLAMA_ALLOWED_HOSTS:
        permitidos = ", ".join(sorted(OLLAMA_ALLOWED_HOSTS))
        raise RuntimeError(
            f"O host do Ollama '{host}' não está autorizado. "
            f"Hosts permitidos: {permitidos}. Ajuste "
            "BIOMIND_ALLOWED_OLLAMA_HOSTS conscientemente."
        )
    if parsed.path.rstrip("/") != "/api/chat":
        raise RuntimeError(
            "OLLAMA_URL deve terminar em /api/chat. Exemplo: "
            "http://127.0.0.1:11434/api/chat"
        )


def perguntar_ollama(system_prompt: str, user_prompt: str) -> str:
    _validar_ollama_local()

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": 0.1,
        },
    }

    response = _http.post(
        OLLAMA_URL,
        json=payload,
        timeout=(OLLAMA_CONNECT_TIMEOUT, OLLAMA_READ_TIMEOUT),
    )

    if not response.ok:
        logger.error(
            "Ollama retornou HTTP %s: %s",
            response.status_code,
            response.text[:500],
        )

    response.raise_for_status()

    data = response.json()

    content = data.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama retornou resposta sem conteúdo.")

    done_reason = data.get("done_reason")

    if done_reason == "length":
        logger.warning(
            "Resposta do Ollama interrompida por limite de tokens. "
            "Aumente OLLAMA_NUM_PREDICT. Modelo=%s",
            OLLAMA_MODEL,
        )

    return content.strip()


def _citacoes_encontradas(answer: str) -> list[int]:
    return sorted({int(number) for number in re.findall(r"\[(\d+)\]", answer)})


def _normalizar_validacao(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)
    sem_acentos = "".join(
        caractere
        for caractere in normalizado
        if not unicodedata.combining(caractere)
    )
    return re.sub(r"\s+", " ", sem_acentos.lower()).strip()


def _pergunta_exige_proxima_etapa(pergunta: str) -> bool:
    texto = _normalizar_validacao(pergunta)
    expressoes = (
        "como prosseguir",
        "como proceder",
        "como conduzir",
        "o que fazer",
        "qual a proxima etapa",
        "qual a conduta",
    )
    return any(expressao in texto for expressao in expressoes)


def _caso_pos_procedimento(pergunta: str) -> bool:
    texto = _normalizar_validacao(pergunta)

    procedimentos = (
        "mesoterapia",
        "microagulhamento",
        "injecao",
        "injetavel",
        "intralesional",
        "procedimento",
        "aplicacao",
    )
    sintomas = (
        "coceira",
        "prurido",
        "caroco",
        "nodulo",
        "dolorido",
        "dor",
        "edema",
        "inchaco",
        "vermelh",
        "secrecao",
        "pus",
        "reacao",
    )

    return (
        any(termo in texto for termo in procedimentos)
        and any(termo in texto for termo in sintomas)
    )


_UNSAFE_RESPONSE_PATTERNS = (
    r"\bprescrev(?:a|er|eu|emos|ido|ida)\b",
    r"\b(?:inicie|iniciar|suspenda|suspender|interrompa|interromper|troque|trocar)\b",
    r"\b(?:dose|posologia)\s*(?:de|:)",
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ml)\b",
)


def _problemas_resposta(
    answer: str,
    source_count: int,
    question: str = "",
) -> list[str]:
    problems: list[str] = []
    citations = _citacoes_encontradas(answer)

    if source_count > 0 and not citations:
        problems.append("resposta sem citações")

    invalid = [
        number
        for number in citations
        if number < 1 or number > source_count
    ]
    if invalid:
        problems.append(f"citações inexistentes: {invalid}")

    for pattern in _UNSAFE_RESPONSE_PATTERNS:
        if re.search(pattern, answer, flags=re.I):
            problems.append("linguagem prescritiva ou dose")
            break

    resposta_normalizada = _normalizar_validacao(answer)
    exige_plano = (
        _pergunta_exige_proxima_etapa(question)
        or _caso_pos_procedimento(question)
    )

    if exige_plano:
        if "como prosseguir agora" not in resposta_normalizada:
            problems.append("faltou responder diretamente como prosseguir")

        indicadores_de_acao = (
            "registrar",
            "observar",
            "verificar",
            "examinar",
            "encaminhar",
            "avaliacao",
            "antes de nova sessao",
        )
        if not any(
            indicador in resposta_normalizada
            for indicador in indicadores_de_acao
        ):
            problems.append("resposta sem próxima etapa operacional")

    if resposta_normalizada.endswith(
        (
            "a base",
            "e necessario",
            "nao permite concluir",
            "porque",
            "portanto",
        )
    ):
        problems.append("resposta possivelmente interrompida")

    return problems


def _repair_answer(
    *,
    question: str,
    context: str,
    provisional_answer: str,
    problems: list[str],
) -> str:
    repair_user_prompt = (
        "<caso>\n"
        f"{question}\n"
        "</caso>\n\n"
        "<trechos_da_base>\n"
        f"{context}\n"
        "</trechos_da_base>\n\n"
        "<resposta_provisoria>\n"
        f"{provisional_answer}\n"
        "</resposta_provisoria>\n\n"
        f"Problemas detectados: {', '.join(problems)}."
    )
    return perguntar_ollama(REPAIR_PROMPT, repair_user_prompt)


# ---------------------------------------------------------------------------
# Resposta principal
# ---------------------------------------------------------------------------

def responder(pergunta: str) -> dict[str, Any]:
    pergunta = pergunta.strip()
    if not pergunta:
        return {
            "status": "vazio",
            "message": "Descreva o caso primeiro.",
            "sources": [],
            "model_name": OLLAMA_MODEL,
            "embedding_model": EMBED_MODEL,
            "prompt_version": PROMPT_VERSION,
        }

    try:
        retrieval_started = time.perf_counter()
        trechos = recuperar(pergunta)
        retrieval_seconds = time.perf_counter() - retrieval_started
        logger.info(
            "Recuperação concluída em %.2f segundos. Trechos=%s",
            retrieval_seconds,
            len(trechos),
        )
    except Exception:
        logger.exception("Falha na recuperação do índice")
        return {
            "status": "erro_indice",
            "message": "Não consegui consultar a base local. Verifique o índice e a configuração.",
            "sources": [],
            "model_name": OLLAMA_MODEL,
            "embedding_model": EMBED_MODEL,
            "prompt_version": PROMPT_VERSION,
        }

    if not trechos:
        return {
            "status": "sem_material",
            "best_sim": None,
            "message": (
                "A base não retornou material clínico adequado para orientar este caso "
                "com segurança. Acrescente ou revise referências de avaliação e diagnóstico diferencial."
            ),
            "sources": [],
            "model_name": OLLAMA_MODEL,
            "embedding_model": EMBED_MODEL,
            "prompt_version": PROMPT_VERSION,
        }

    context, used_chunks = montar_contexto(trechos)
    if not used_chunks:
        return {
            "status": "sem_material",
            "best_sim": trechos[0]["sim"],
            "message": "A busca encontrou material, mas não foi possível montar contexto seguro.",
            "sources": [],
            "model_name": OLLAMA_MODEL,
            "embedding_model": EMBED_MODEL,
            "prompt_version": PROMPT_VERSION,
        }

    user_prompt = (
        "<caso_descrito_pela_profissional>\n"
        f"{pergunta}\n"
        "</caso_descrito_pela_profissional>\n\n"

        "<trechos_da_base>\n"
        f"{context}\n"
        "</trechos_da_base>\n\n"

        "Responda diretamente à pergunta da profissional. "

        "Não comece repetindo ou parafraseando o caso. "

        "Preserve exatamente os sintomas, a duração, a localização e os "
        "demais elementos relatados. Caso identifique um provável erro de "
        "digitação, indique que o termo precisa ser confirmado em vez de "
        "substituí-lo por outra palavra. "

        "Quando a pergunta solicitar como prosseguir ou relatar um sintoma "
        "novo após procedimento, inclua obrigatoriamente a seção "
        "'### Como prosseguir agora'. "

        "Nessa seção, apresente a próxima etapa segura, o que precisa ser "
        "avaliado antes de nova conduta, o que registrar e quando encaminhar, "
        "sempre conforme os trechos disponíveis. "

        "Na seção de perguntas, escreva somente perguntas que possam ser "
        "feitas diretamente à paciente. Não inclua perguntas teóricas para "
        "a profissional. "

        "Não prescreva medicamentos, doses ou tratamentos. "
        "Não transfira diagnósticos ou condutas de pacientes descritos nos documentos. "

        "Associe cada afirmação clínica aos números [1], [2] e assim por diante. "

        "Quando a base não permitir uma orientação, indique exatamente qual "
        "informação ou documento está faltando."
    )

    try:
        with _generation_gate:
            generation_started = time.perf_counter()
            answer = perguntar_ollama(SYSTEM_PROMPT, user_prompt)
            generation_seconds = time.perf_counter() - generation_started
            logger.info(
                "Geração principal concluída em %.2f segundos.",
                generation_seconds,
            )

            problems = _problemas_resposta(
                answer,
                len(used_chunks),
                pergunta,
            )
            if problems and REPAIR_UNSAFE_RESPONSE:
                logger.warning("Resposta enviada à revisão automática: %s", problems)
                answer = _repair_answer(
                    question=pergunta,
                    context=context,
                    provisional_answer=answer,
                    problems=problems,
                )
                problems = _problemas_resposta(
                    answer,
                    len(used_chunks),
                    pergunta,
                )
    except requests.exceptions.Timeout:
        logger.exception("Ollama excedeu o tempo limite")
        return {
            "status": "erro_modelo",
            "message": (
                "O modelo local excedeu o tempo limite. Reduza o contexto, use um modelo menor "
                "ou verifique se o Ollama está usando a GPU."
            ),
            "sources": [],
            "model_name": OLLAMA_MODEL,
            "embedding_model": EMBED_MODEL,
            "prompt_version": PROMPT_VERSION,
        }
    except requests.exceptions.RequestException:
        logger.exception("Falha ao acessar o Ollama")
        return {
            "status": "erro_modelo",
            "message": (
                "Não consegui acessar o modelo local. Verifique o Ollama, o endpoint /api/chat "
                f"e se o modelo '{OLLAMA_MODEL}' está instalado."
            ),
            "sources": [],
            "model_name": OLLAMA_MODEL,
            "embedding_model": EMBED_MODEL,
            "prompt_version": PROMPT_VERSION,
        }
    except Exception:
        logger.exception("Falha ao gerar resposta com o Ollama")
        return {
            "status": "erro_modelo",
            "message": "O modelo local não conseguiu gerar uma resposta válida.",
            "sources": [],
            "model_name": OLLAMA_MODEL,
            "embedding_model": EMBED_MODEL,
            "prompt_version": PROMPT_VERSION,
        }

    if problems:
        logger.error("Resposta bloqueada após revisão: %s", problems)
        return {
            "status": "resposta_bloqueada",
            "message": (
                "A resposta gerada não passou pelos controles de segurança e foi bloqueada. "
                "Revise os trechos recuperados ou reformule o caso."
            ),
            "sources": [],
            "model_name": OLLAMA_MODEL,
            "embedding_model": EMBED_MODEL,
            "prompt_version": PROMPT_VERSION,
        }

    citations = _citacoes_encontradas(answer)
    sources = [
        {
            "n": index + 1,
            "source": trecho["source"],
            "pages": trecho["pages"],
            "sim": trecho["sim"],
            "score": trecho.get("score"),
            "chunk_ids": trecho["chunk_ids"],
            "document_type": trecho.get("document_type"),
            "content_type": trecho.get("content_type"),
        }
        for index, trecho in enumerate(used_chunks)
    ]

    return {
        "status": "ok",
        "answer": answer,
        "disclaimer": AVISO_LEGAL,
        "best_sim": used_chunks[0]["sim"],
        "citation_warning": len(citations) == 0,
        "sources": sources,
        "model_name": OLLAMA_MODEL,
        "embedding_model": EMBED_MODEL,
        "prompt_version": PROMPT_VERSION,
    }


def health() -> dict[str, Any]:
    col = get_collection()
    metadata = col.metadata or {} if col is not None else {}
    return {
        "status": "ok" if col is not None and col.count() > 0 else "indice_indisponivel",
        "collection": COLLECTION,
        "chunks": col.count() if col is not None else 0,
        "index_schema_version": metadata.get("schema_version"),
        "offline_only": OFFLINE_ONLY,
        "ollama_model": OLLAMA_MODEL,
        "embedding_model": EMBED_MODEL,
        "prompt_version": PROMPT_VERSION,
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "sentence_transformers_available": (
                importlib.util.find_spec("sentence_transformers") is not None
            ),
            "embed_device": EMBED_DEVICE,
            "max_generations": MAX_GENERATIONS,
        },
        "ollama_runtime": {
            "url": OLLAMA_URL,
            "allowed_hosts": sorted(OLLAMA_ALLOWED_HOSTS),
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "connect_timeout": OLLAMA_CONNECT_TIMEOUT,
            "read_timeout": OLLAMA_READ_TIMEOUT,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        },
        "retrieval": {
            "top_k": TOP_K,
            "window": WINDOW,
            "min_similarity": PISO_RELEVANCIA,
            "min_rerank_score": MIN_RERANK_SCORE,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "max_query_variants": MAX_QUERY_VARIANTS,
            "max_context_chars": MAX_CONTEXT_CHARS,
            "max_results_per_source": MAX_RESULTS_PER_SOURCE,
            "max_content_similarity": MAX_CONTENT_SIMILARITY,
        },
    }
