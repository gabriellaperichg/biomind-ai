"""
Biomind — servidor web (com login) para o piloto na RunPod.

- Login por usuária: cada biomédica entra com seu usuário/senha.
- Registra cada pergunta/resposta em logs/biomind_log.jsonl, incluindo QUEM perguntou.
- Usuários e segredo de sessão vêm de variáveis de ambiente (nunca no código).

Configurar antes de subir (no terminal do pod):
  export BIOMIND_USERS="ana:senhaAna,bia:senhaBia,carol:senhaCarol"
  export BIOMIND_SECRET="uma-frase-longa-e-aleatoria-qualquer"

Rodar:
  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import json
import time
import secrets
import threading
from datetime import datetime

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

import biomind_core as core

AQUI = os.path.dirname(os.path.abspath(__file__))

# ------------------- usuários e sessão -------------------
def _carregar_usuarios():
    bruto = os.environ.get("BIOMIND_USERS", "").strip()
    usuarios = {}
    for par in bruto.split(","):
        if ":" in par:
            u, s = par.split(":", 1)
            usuarios[u.strip()] = s.strip()
    if not usuarios:
        print("AVISO: BIOMIND_USERS não definido. Usando usuário padrão 'biomedica' / 'trocar-senha'.")
        usuarios = {"biomedica": "trocar-senha"}
    return usuarios

USUARIOS = _carregar_usuarios()
SECRET = os.environ.get("BIOMIND_SECRET") or secrets.token_hex(16)

app = FastAPI(title="Biomind")
app.add_middleware(SessionMiddleware, secret_key=SECRET)

# ------------------- logs -------------------
LOG_DIR = os.environ.get("BIOMIND_LOG_DIR", os.path.join(AQUI, "logs"))
LOG_FILE = os.path.join(LOG_DIR, "biomind_log.jsonl")
_log_lock = threading.Lock()


def registrar(entrada: dict):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with _log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception:
        pass


LOGIN_HTML = """<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Biomind — Acesso</title>
<style>
body{font-family:system-ui,sans-serif;background:#11302D;color:#16302E;height:100vh;margin:0;
display:grid;place-items:center}
.box{background:#fff;padding:34px 30px;border-radius:16px;width:300px;box-shadow:0 10px 40px rgba(0,0,0,.25)}
h1{font-size:22px;margin:0 0 4px;color:#1E5B57}
p{font-size:13px;color:#5F716E;margin:0 0 20px}
label{font-size:12px;color:#5F716E;display:block;margin:12px 0 4px}
input{width:100%;padding:10px;border:1px solid #E2DED3;border-radius:9px;font-size:14px;box-sizing:border-box}
button{width:100%;margin-top:18px;padding:11px;border:0;border-radius:10px;background:#1E5B57;color:#fff;
font-size:14px;cursor:pointer}
.erro{color:#9A3B23;font-size:12.5px;margin-top:12px;text-align:center}
</style></head><body>
<form class="box" method="post" action="/login">
<h1>Biomind</h1><p>Apoio à decisão em tricologia</p>
<label>Usuária</label><input name="usuario" autofocus>
<label>Senha</label><input name="senha" type="password">
<button type="submit">Entrar</button>
__ERRO__
</form></body></html>"""


class Pergunta(BaseModel):
    texto: str


@app.get("/login", response_class=HTMLResponse)
def login_form(erro: int = 0):
    msg = '<div class="erro">Usuária ou senha inválida.</div>' if erro else ""
    return LOGIN_HTML.replace("__ERRO__", msg)


@app.post("/login")
def login(request: Request, usuario: str = Form(...), senha: str = Form(...)):
    if USUARIOS.get(usuario) and secrets.compare_digest(USUARIOS[usuario], senha):
        request.session["user"] = usuario
        return RedirectResponse("/", status_code=303)
    return RedirectResponse("/login?erro=1", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login", status_code=303)
    with open(os.path.join(AQUI, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
def ask(p: Pergunta, request: Request):
    usuario = request.session.get("user")
    if not usuario:
        return JSONResponse({"status": "nao_autenticado",
                             "message": "Sessão expirada. Recarregue a página e entre novamente."},
                            status_code=401)

    texto = (p.texto or "").strip()
    if not texto:
        return {"status": "vazio", "message": "Descreva o caso primeiro."}

    inicio = time.time()
    resultado = core.responder(texto)
    duracao_ms = int((time.time() - inicio) * 1000)

    registrar({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "usuario": usuario,
        "origem": request.client.host if request.client else None,
        "pergunta": texto,
        "status": resultado.get("status"),
        "best_sim": resultado.get("best_sim"),
        "resposta": resultado.get("answer") or resultado.get("message"),
        "fontes": resultado.get("sources"),
        "duracao_ms": duracao_ms,
    })
    return resultado