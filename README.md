# Biomind V4 — RAG integrado

Esta pasta é uma versão integrada do repositório Biomind enviado para esta conversa.

## O que foi preservado

A aplicação existente foi mantida como base:

- `app.py`
- autenticação em `routers/auth.py`
- administração em `routers/admin.py`
- chats em `routers/chats.py`
- `database.py` / SQLAlchemy
- `models.py`
- `security.py`
- `services/`
- `static/`
- `migrations/`
- testes existentes
- `data/biomind_app.db`
- `data/trichon_app.db`

**O código novo de RAG não altera o banco de usuários, sessões, chats ou mensagens.**

Antes de qualquer reconstrução do índice, faça:

```powershell
python -m scripts.backup_app_db
```

## Pastas locais reservadas

As seguintes pastas são destinadas aos arquivos locais e podem continuar fora do controle de versão:

```text
pdfs/
biomind_db/
data/
```

No ZIP foram mantidos placeholders quando uma pasta não possuía conteúdo no repositório enviado. A extração sobre uma pasta que já possui arquivos não deve apagar esses arquivos.

## Nova arquitetura

```text
                         DOCUMENTO
                             |
                      Document Router
                             |
               +-------------+-------------+
               |             |             |
              PDF          DOCX          Slides
               |             |             |
               +-------------+-------------+
                             |
                    Structure / Sections
                             |
                      Parent / Child
                             |
                         Metadata
                             |
                         Embedding
                             |
                    +--------+--------+
                    |                 |
                 Chroma             BM25
                    |                 |
                    +--------+--------+
                             |
                           Hybrid
                             |
                          Reranker
                             |
                        Parent Context
                             |
                           Ollama
                             |
                     resposta + fontes
```

### Ingestão

- `ingestion/router.py`: identifica o tipo estrutural do documento.
- `ingestion/parsers.py`: lê PDF e DOCX.
- `ingestion/structure.py`: preserva títulos, seções e páginas.
- `ingestion/metadata.py`: gera metadata de conteúdo sem injetá-la artificialmente no embedding.
- `ingestion/chunker.py`: cria chunks semânticos e relações `parent_id` / `child_index`.

### Retrieval

- `retrieval/dense.py`: embeddings.
- `retrieval/bm25.py`: índice lexical local.
- `retrieval/hybrid.py`: combina BM25 e dense com RRF.
- `retrieval/reranker.py`: reranker neural opcional; se o modelo não estiver disponível em cache, usa fallback sem quebrar a aplicação.

### RAG

- `rag/query.py`: variações determinísticas da pergunta.
- `rag/context.py`: recompõe o contexto do parent.
- `rag/pipeline.py`: fluxo de retrieval → reranking.

## Reconstruir a base RAG

Coloque os documentos em:

```text
pdfs/
```

PDF e DOCX são aceitos. Subpastas também são percorridas.

Depois:

```powershell
python -m scripts.backup_app_db
python -m scripts.build_rag
```

Isso executa:

```text
scripts/ingest_v4.py
        ↓
chunks.jsonl
        ↓
embed.py build
        ↓
Chroma + BM25
```

### Importante sobre o banco

`embed.py` opera em `biomind_db/`, que é o índice RAG.

Ele **não acessa nem recria `data/biomind_app.db`**.

O arquivo `data/biomind_app.db` é o banco da aplicação e contém usuários, sessões, chats, mensagens, fontes e auditoria.

## Usuários existentes

Para confirmar o banco antes de executar qualquer migração:

```powershell
python -m scripts.verify_data
```

A V4 não inclui migração destrutiva das tabelas da aplicação.

## Testar retrieval sem Ollama

```powershell
python test_retrieval.py "Paciente feminina, queda na coroa, couro cabeludo oleoso e escamando."
```

Ou:

```powershell
python evaluate_retrieval.py
```

## Executar a aplicação

```powershell
alembic upgrade head
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Não é necessário executar `alembic` para a nova camada de RAG, pois ela usa Chroma + BM25 e não adiciona tabelas ao banco de usuários nesta versão.

## Reranker

Por padrão:

```dotenv
BIOMIND_RERANKER_ENABLED=1
BIOMIND_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
BIOMIND_OFFLINE=1
```

O modelo do reranker precisa estar disponível no cache local quando `BIOMIND_OFFLINE=1`.

Durante a primeira preparação, pode-se desativar temporariamente o modo offline para baixar os modelos, e depois voltar para modo offline.

## Segurança

Não versionar:

```text
.env
biomind_db/
pdfs/
chunks.jsonl
debug_retrieval_report.json
*.db
```

Mantenha o Ollama local quando possível e não exponha diretamente as portas 8000 ou 11434 à internet.

## Compatibilidade

Os arquivos antigos abaixo foram mantidos para compatibilidade com o projeto atual:

```text
chunk.py
embed.py
retrieval_quality.py
```

A nova ingestão V4 é acionada por:

```text
python -m scripts.ingest_v4
```

e a integração de retrieval é feita pelo `biomind_core.py`.
