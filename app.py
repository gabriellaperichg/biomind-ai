"""
Biomind — servidor web local (interface gráfica da demo).

Sobe uma página de chat e liga ela ao pipeline offline (biomind_core).
Roda tudo na sua máquina, nada sai para a internet.

Pré-requisitos: índice criado (python 02_embed.py build) e Ollama rodando.

Rodar:
  uvicorn app:app --host 127.0.0.1 --port 8000
Depois abra http://127.0.0.1:8000 no navegador.
"""

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import biomind_core as core

app = FastAPI(title="Biomind")
AQUI = os.path.dirname(os.path.abspath(__file__))


class Pergunta(BaseModel):
    texto: str


@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(AQUI, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
def ask(p: Pergunta):
    texto = (p.texto or "").strip()
    if not texto:
        return {"status": "vazio", "message": "Descreva o caso primeiro."}
    return core.responder(texto)