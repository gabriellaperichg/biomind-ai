# Biomind — autenticação, chats e RAG local

Assistente interno de apoio à investigação clínica em tricologia. O RAG, os embeddings e a geração são executados na infraestrutura autorizada.

## Estrutura principal

- `chunk.py`: transforma os documentos em chunks.
- `embed.py`: cria o índice Chroma em `biomind_db/`.
- `biomind_core.py`: recuperação, piso de relevância e integração com Ollama.
- `app.py`: aplicação FastAPI.
- `routers/`: autenticação, usuários e chats.
- `services/`: auditoria e integração com o núcleo RAG.
- `static/`: páginas e arquivos do navegador.
- `data/biomind_app.db`: banco SQLite local, criado pelas migrações e não versionado.

## Instalação

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash no Windows
python -m pip install -r requirements.txt
```

## Preparar o banco de aplicação

```bash
alembic upgrade head
python -m scripts.create_admin
```

## Preparar a base RAG

```bash
python chunk.py
python embed.py build
```

## Executar

Desenvolvimento local:

```bash
export BIOMIND_ENABLE_DOCS=1
export BIOMIND_ALLOWED_HOSTS="127.0.0.1,localhost,testserver"
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Abra:

- `http://127.0.0.1:8000/login`
- `http://127.0.0.1:8000/docs` durante o desenvolvimento

Para acesso pela rede local, adicione o IP ou hostname do servidor a `BIOMIND_ALLOWED_HOSTS` e use HTTPS antes de inserir casos reais.

## Atualizar o banco após mudanças nos modelos

```bash
alembic revision --autogenerate -m "Descrição da alteração"
alembic upgrade head
```

## Testes rápidos

```bash
python -m compileall -q .
alembic check
python -c "import app; print(app.INDEX_FILE.exists(), app.LOGIN_FILE.exists())"
```
