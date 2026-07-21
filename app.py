"""
Biomind — servidor web local com autenticação e histórico de chats.

Responsabilidades deste arquivo:
- iniciar a aplicação FastAPI;
- registrar os routers;
- servir as páginas HTML;
- disponibilizar o health check;
- adicionar proteções HTTP básicas.

Pré-requisitos:
  1. Índice criado:
     python embed.py build

  2. Ollama rodando com o modelo configurado.

  3. Banco atualizado:
     alembic upgrade head

Rodar:
  uvicorn app:app --host 0.0.0.0 --port 8000

Abrir localmente:
  http://127.0.0.1:8000

Na rede:
  http://IP-DO-SERVIDOR:8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import biomind_core as core

from routers import admin, auth, chats


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = BASE_DIR / "index.html"
LOGIN_FILE = BASE_DIR / "login.html"
STATIC_DIR = BASE_DIR / "static"


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

ENABLE_DOCS = os.getenv(
    "BIOMIND_ENABLE_DOCS",
    "0",
) == "1"


ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv(
        "BIOMIND_ALLOWED_HOSTS",
        "127.0.0.1,localhost,testserver",
    ).split(",")
    if host.strip()
]


logging.basicConfig(
    level=os.getenv(
        "BIOMIND_LOG_LEVEL",
        "INFO",
    ).upper(),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


logger = logging.getLogger("biomind.app")


# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Biomind",
    version="4.0.0",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url=None,
    openapi_url=(
        "/openapi.json"
        if ENABLE_DOCS
        else None
    ),
)

if not STATIC_DIR.is_dir():
    raise RuntimeError(
        f"Pasta static não encontrada: {STATIC_DIR}"
    )

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

# ---------------------------------------------------------------------------
# Middlewares
# ---------------------------------------------------------------------------

# Impede requisições usando cabeçalhos Host inesperados.
#
# Para acesso pela rede local, inclua o IP do servidor:
#
# BIOMIND_ALLOWED_HOSTS=127.0.0.1,localhost,192.168.1.50
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=ALLOWED_HOSTS,
    www_redirect=False,
)


@app.middleware("http")
async def adicionar_cabecalhos_seguranca(
    request,
    call_next,
):
    """
    Adiciona cabeçalhos básicos de segurança e privacidade.
    """

    response = await call_next(request)

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = "no-referrer"

    response.headers[
        "Cache-Control"
    ] = "no-store"

    response.headers[
        "Permissions-Policy"
    ] = (
        "camera=(), "
        "microphone=(), "
        "geolocation=()"
    )

    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(
    auth.router,
    prefix="/auth",
    tags=["Autenticação"],
)


app.include_router(
    admin.router,
    prefix="/admin",
    tags=["Administração"],
)


app.include_router(
    chats.router,
    prefix="/chats",
    tags=["Chats"],
)

if not STATIC_DIR.is_dir():
    raise RuntimeError(
        f"Pasta de arquivos estáticos não encontrada: {STATIC_DIR}"
    )

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

# ---------------------------------------------------------------------------
# Páginas HTML
# ---------------------------------------------------------------------------

@app.get(
    "/login",
    include_in_schema=False,
    response_class=FileResponse,
)
def login_page() -> FileResponse:
    """
    Entrega a página de autenticação.
    """

    if not LOGIN_FILE.is_file():
        logger.error(
            "login.html não encontrado em %s",
            LOGIN_FILE,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "A página de login não foi encontrada. "
                "Verifique se login.html está na mesma "
                "pasta do app.py."
            ),
        )

    return FileResponse(
        path=LOGIN_FILE,
        media_type="text/html",
    )


@app.get(
    "/",
    include_in_schema=False,
    response_class=FileResponse,
)
def home() -> FileResponse:
    """
    Entrega a interface principal.

    A página deve chamar GET /auth/me ao carregar.
    Caso receba 401, deve redirecionar para /login.
    """

    if not INDEX_FILE.is_file():
        logger.error(
            "index.html não encontrado em %s",
            INDEX_FILE,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "A interface não foi encontrada. "
                "Verifique se index.html está na mesma "
                "pasta do app.py."
            ),
        )

    return FileResponse(
        path=INDEX_FILE,
        media_type="text/html",
    )


# ---------------------------------------------------------------------------
# Saúde da aplicação
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Sistema"],
)
def health() -> JSONResponse:
    """
    Verifica se o núcleo do Biomind e o índice estão disponíveis.

    Essa rota permanece pública porque pode ser utilizada por ferramentas
    de monitoramento.
    """

    try:
        result = core.health()

    except Exception:
        logger.exception(
            "Falha ao consultar o estado do Biomind"
        )

        return JSONResponse(
            status_code=503,
            content={
                "status": "erro",
                "message": (
                    "Não foi possível verificar "
                    "o estado da base local."
                ),
            },
        )

    status_code = (
        200
        if result.get("status") == "ok"
        else 503
    )

    return JSONResponse(
        status_code=status_code,
        content=result,
    )