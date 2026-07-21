"""
Biomind — servidor web local (interface gráfica da demo), v3.

Liga o index.html ao biomind_core e mantém todo o processamento local.

Pré-requisitos:
  1. Índice criado: python 02_embed.py build
  2. Ollama rodando com o modelo configurado

Rodar:
  uvicorn app:app --host 0.0.0.0 --port 8000

Depois abra:
  http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import biomind_core as core

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

AQUI = Path(__file__).resolve().parent
INDEX_FILE = AQUI / "index.html"

ENABLE_DOCS = os.getenv("BIOMIND_ENABLE_DOCS", "0") == "1"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv(
        "BIOMIND_ALLOWED_HOSTS",
        "127.0.0.1,localhost,testserver",
    ).split(",")
    if host.strip()
]

logging.basicConfig(
    level=os.getenv("BIOMIND_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("biomind.app")

app = FastAPI(
    title="Biomind",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

# Protege contra requisições com cabeçalho Host inesperado.
# Para uso em rede local, inclua o IP ou domínio em BIOMIND_ALLOWED_HOSTS.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=ALLOWED_HOSTS,
    www_redirect=False,
)


class Pergunta(BaseModel):
    """Corpo aceito por POST /ask."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    # Não usamos min_length=1 para manter a resposta amigável "status: vazio"
    # esperada pela interface, em vez de devolver erro 422 para texto vazio.
    texto: str = Field(default="", max_length=6000)


# ---------------------------------------------------------------------------
# Cabeçalhos básicos de segurança e privacidade
# ---------------------------------------------------------------------------

@app.middleware("http")
async def adicionar_cabecalhos_seguranca(request, call_next):
    resposta = await call_next(request)
    resposta.headers["X-Content-Type-Options"] = "nosniff"
    resposta.headers["X-Frame-Options"] = "DENY"
    resposta.headers["Referrer-Policy"] = "no-referrer"
    resposta.headers["Cache-Control"] = "no-store"
    return resposta


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False, response_class=FileResponse)
def home() -> FileResponse:
    """Entrega a interface local."""
    if not INDEX_FILE.is_file():
        logger.error("index.html não encontrado em %s", INDEX_FILE)
        raise HTTPException(
            status_code=500,
            detail="A interface não foi encontrada. Verifique se index.html está na pasta do app.py.",
        )

    return FileResponse(
        path=INDEX_FILE,
        media_type="text/html",
    )


@app.get("/health")
def health() -> JSONResponse:
    """Informa se o índice local está disponível."""
    try:
        resultado = core.health()
    except Exception:
        logger.exception("Falha ao consultar o estado do Biomind")
        return JSONResponse(
            status_code=503,
            content={
                "status": "erro",
                "message": "Não foi possível verificar o estado da base local.",
            },
        )

    codigo = 200 if resultado.get("status") == "ok" else 503
    return JSONResponse(status_code=codigo, content=resultado)


@app.post("/ask", response_model=None)
def ask(p: Pergunta) -> Any:
    """Recebe o caso e devolve a orientação estruturada do biomind_core."""
    if not p.texto:
        return {
            "status": "vazio",
            "message": "Descreva o caso primeiro.",
            "sources": [],
        }

    try:
        return core.responder(p.texto)
    except Exception:
        # O core já trata erros esperados. Este bloco cobre falhas inesperadas
        # sem expor detalhes internos, caminhos ou conteúdo clínico ao navegador.
        logger.exception("Erro inesperado ao processar POST /ask")
        return JSONResponse(
            status_code=500,
            content={
                "status": "erro_servidor",
                "message": "Ocorreu um erro interno ao processar a solicitação.",
                "sources": [],
            },
        )