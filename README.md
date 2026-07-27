# Biomind — autenticação, chats e RAG clínico local

Assistente interno de apoio à investigação em tricologia. O projeto combina FastAPI, autenticação, histórico de chats, ChromaDB, embeddings BGE-M3 e geração local via Ollama.

## Melhorias desta versão

A recuperação deixou de selecionar apenas os primeiros resultados por similaridade vetorial. O fluxo agora inclui:

- variações determinísticas da consulta para sintomas, padrão de queda e histórico medicamentoso;
- reranking por finalidade clínica, cobertura dos termos e qualidade documental;
- descarte de termos de consentimento em perguntas clínicas gerais;
- penalização de procedimentos e complicações quando não foram perguntados;
- bloqueio de condutas retiradas de outro caso clínico em investigação inicial;
- limite de resultados por fonte para aumentar diversidade;
- metadados de `document_type`, `content_type`, `topic`, `section` e `source_quality`;
- prompt clínico mais conservador;
- revisão automática de respostas sem citações ou com linguagem prescritiva;
- scripts para inspecionar e avaliar a qualidade dos chunks.

## Estrutura principal

- `chunk.py`: limpa os PDFs, preserva fronteiras clínicas e gera `chunks.jsonl`.
- `retrieval_quality.py`: classificação, intenção da consulta, filtros e reranking.
- `embed.py`: cria o índice Chroma em `biomind_db/`.
- `biomind_core.py`: recuperação, contexto, prompt, Ollama e validação da resposta.
- `test_retrieval.py`: mostra selecionados, descartes e motivos sem chamar o Ollama.
- `evaluate_retrieval.py`: executa casos de regressão definidos em `evaluation_cases.json`.
- `app.py`: aplicação FastAPI.
- `routers/`: autenticação, usuários e chats.
- `services/`: auditoria e integração com o núcleo RAG.
- `static/`: interface do navegador.
- `data/biomind_app.db`: banco SQLite da aplicação.

## Instalação no Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Ajuste no `.env` o caminho de `BIOMIND_DB_DIR`, o modelo do Ollama e os hosts permitidos.

## Preparar o banco de aplicação

```powershell
alembic upgrade head
python -m scripts.create_admin
```

As melhorias de RAG não exigem nova migração do banco de chats.

## Reconstruir a base RAG

A reconstrução é recomendada porque a versão nova gera chunks menores, remove identificação de licença e grava metadados de finalidade.

1. Faça backup do índice atual:

```powershell
Rename-Item biomind_db biomind_db_backup
```

2. Coloque os PDFs em `pdfs/` e execute:

```powershell
python chunk.py
python embed.py build
```

O build é transacional: o índice anterior é preservado quando possível até a nova coleção ficar completa.

### Migração rápida sem refazer os chunks

Caso `chunks.jsonl` já exista, você pode executar apenas:

```powershell
python embed.py build
```

O `embed.py` classifica chunks antigos durante o build. A qualidade melhora, mas a separação entre descrição, diagnóstico e tratamento só será completa após rodar também `chunk.py`.

## Testar a recuperação antes do Ollama

```powershell
python test_retrieval.py "Paciente feminina, queda na coroa, couro cabeludo oleoso e escamando, toma anticoncepcional e antidepressivo."
```

O script gera:

```text
debug_retrieval_report.json
```

Revise principalmente:

- `TRECHOS SELECIONADOS`;
- `CANDIDATOS E DESCARTES`;
- `document_type` e `content_type`;
- motivo do descarte;
- se as fontes tratam realmente de avaliação e diagnóstico diferencial.

Para executar o conjunto de regressão:

```powershell
python evaluate_retrieval.py
```

Os casos podem ser ampliados em `evaluation_cases.json` com perguntas já validadas pela equipe.

## Executar os testes de código

```powershell
pytest -q
python -m compileall -q .
```

## Testar o Ollama

```powershell
ollama list
ollama run llama3.1:latest "Responda somente OK"
```

O `.env` deve conter o endpoint completo:

```dotenv
OLLAMA_URL=http://127.0.0.1:11434/api/chat
OLLAMA_MODEL=llama3.1:latest
```

## Executar o Biomind

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --env-file .env
```

Abra:

- `http://127.0.0.1:8000/login`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`, somente quando `BIOMIND_ENABLE_DOCS=1`

Para Tailscale Serve, mantenha o Uvicorn em `127.0.0.1` e publique a porta de forma privada:

```powershell
tailscale serve --bg 8000
tailscale serve status
```

Use Tailscale Serve, não Funnel, para manter o acesso restrito à tailnet.

## Configuração inicial recomendada

```dotenv
BIOMIND_TOP_K=4
BIOMIND_WINDOW=0
BIOMIND_CANDIDATE_MULTIPLIER=8
BIOMIND_MAX_QUERY_VARIANTS=4
BIOMIND_MAX_RESULTS_PER_SOURCE=1
BIOMIND_PISO_RELEVANCIA=0.35
BIOMIND_MIN_RERANK_SCORE=0.50
BIOMIND_MAX_CONTEXT_CHARS=10000

OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=450
OLLAMA_KEEP_ALIVE=30m
```

Não aumente `TOP_K` e o contexto apenas para obter respostas mais longas. Primeiro confirme que os trechos selecionados são clinicamente adequados.

## Privacidade

- Não versione `.env`, `biomind_db/`, `chunks.jsonl`, `pdfs/` ou bancos SQLite.
- Ative `BIOMIND_OFFLINE=1` somente depois que o BGE-M3 estiver completo no cache local.
- Mantenha `BIOMIND_LOCAL_OLLAMA_ONLY=1` quando o Ollama estiver na mesma máquina.
- Não exponha as portas `8000` ou `11434` diretamente à internet.
- Faça backup periódico de `data/biomind_app.db` e `biomind_db/`.
