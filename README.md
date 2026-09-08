# Trichon AI --- Plataforma de IA com RAG

## Visão geral

O **Trichon AI** é uma aplicação web de inteligência artificial construída
para oferecer uma experiência de consulta conversacional apoiada por uma
base de conhecimento própria.

A aplicação combina:

-   autenticação e controle de acesso;
-   gerenciamento administrativo;
-   conversas e histórico de chats;
-   pipeline de **RAG (Retrieval-Augmented Generation)**;
-   busca semântica e lexical;
-   reranking neural;
-   reconstrução de contexto a partir da estrutura dos documentos;
-   geração de respostas com **Ollama**;
-   processamento de documentos PDF e DOCX;
-   banco de dados separado para os dados da aplicação;
-   índice RAG separado dos dados de usuários e conversas.

A arquitetura foi organizada para que a evolução da camada de
conhecimento não altere o banco responsável pela aplicação.

------------------------------------------------------------------------

## Principais funcionalidades

### Interface de chat

O Trichon AI possui uma interface web para interação conversacional com o
modelo de IA. O usuário envia perguntas e recebe respostas geradas a
partir do conhecimento recuperado pela camada RAG.

Fluxo:

``` text
Usuário
   ↓
Interface de Chat
   ↓
FastAPI / Aplicação
   ↓
Pipeline RAG
   ↓
Recuperação de conhecimento
   ↓
Contexto relevante
   ↓
Ollama / LLM
   ↓
Resposta
```

### Autenticação

A aplicação possui sistema de autenticação próprio, organizado
principalmente em:

``` text
routers/auth.py
database.py
models.py
security.py
```

A autenticação é separada da base de conhecimento. A reconstrução do
índice RAG não precisa recriar os dados de usuários.

### Administração

A aplicação possui rotas administrativas em:

``` text
routers/admin.py
```

### Chats e histórico

A camada de chats está em:

``` text
routers/chats.py
```

Os dados de usuários, sessões, chats, mensagens, fontes e auditoria
pertencem ao banco da aplicação.

Banco principal:

``` text
data/Trichon AI_app.db
```

Também existe:

``` text
data/trichon_app.db
```

Esses arquivos são dados locais e não devem ser versionados.

------------------------------------------------------------------------

# RAG --- Retrieval-Augmented Generation

O principal diferencial técnico do Trichon AI é o uso de RAG.

Em vez de enviar apenas a pergunta diretamente ao modelo, a aplicação
primeiro procura informações relevantes na base de documentos.

``` text
Pergunta
   ↓
Variações determinísticas da consulta
   ↓
Dense + BM25
   ↓
Hybrid Retrieval
   ↓
Reranker
   ↓
Contexto dos documentos
   ↓
Ollama
   ↓
Resposta + fontes
```

------------------------------------------------------------------------

## Ingestão de documentos

Os documentos ficam em:

``` text
pdfs/
```

Formatos aceitos:

-   PDF;
-   DOCX.

Subpastas também são percorridas.

Arquitetura:

``` text
Documento
   ↓
Document Router
   ↓
Parser
   ↓
Structure / Sections
   ↓
Parent / Child
   ↓
Metadata
   ↓
Chunking
   ↓
Embedding
   ↓
Índice RAG
```

### Document Router

Arquivo:

``` text
ingestion/router.py
```

Identifica o tipo estrutural do documento e direciona o processamento.

### Parsers

Arquivo:

``` text
ingestion/parsers.py
```

Responsável pela leitura dos documentos PDF e DOCX.

### Estrutura

Arquivo:

``` text
ingestion/structure.py
```

Preserva informações como títulos, seções e páginas, permitindo manter o
contexto original do conteúdo.

### Parent / Child

Arquivo:

``` text
ingestion/chunker.py
```

Cria chunks semânticos e mantém relações como:

-   `parent_id`;
-   `child_index`.

A recuperação pode utilizar unidades menores, enquanto o contexto final
pode ser recomposto a partir do conteúdo pai.

### Metadata

Arquivo:

``` text
ingestion/metadata.py
```

Gera metadata relacionada ao conteúdo sem injetá-la artificialmente no
texto utilizado para embeddings.

------------------------------------------------------------------------

# Retrieval

A recuperação combina diferentes estratégias:

``` text
                 Pergunta
                    |
          +---------+---------+
          |                   |
       Dense                BM25
          |                   |
          +---------+---------+
                    |
                  RRF
                    |
                 Hybrid
                    |
                Reranker
```

## Dense Retrieval

Arquivo:

``` text
retrieval/dense.py
```

Utiliza embeddings com:

``` text
BAAI/bge-m3
```

O modelo representa documentos e perguntas como vetores, permitindo
encontrar conteúdo semanticamente relacionado mesmo quando as palavras
utilizadas são diferentes.

## BM25

Arquivo:

``` text
retrieval/bm25.py
```

Realiza busca lexical e é especialmente útil para termos específicos,
nomes, expressões e palavras-chave.

## Hybrid Retrieval

Arquivo:

``` text
retrieval/hybrid.py
```

Combina Dense Retrieval e BM25 usando **RRF (Reciprocal Rank Fusion)**.

Assim, a recuperação aproveita tanto similaridade semântica quanto
correspondência lexical.

## Reranker

Arquivo:

``` text
retrieval/reranker.py
```

Os candidatos recuperados podem ser reordenados por um reranker neural.

Modelo:

``` text
BAAI/bge-reranker-v2-m3
```

O reranker é opcional. Caso o modelo não esteja disponível ou ocorra uma
falha, existe fallback para os candidatos recuperados.

------------------------------------------------------------------------

# Pipeline RAG

## Variações de consulta

Arquivo:

``` text
rag/query.py
```

Pode gerar variações determinísticas da pergunta para ampliar a
cobertura da recuperação.

## Reconstrução de contexto

Arquivo:

``` text
rag/context.py
```

Recompõe contexto relacionado ao documento pai a partir dos chunks
recuperados.

## Pipeline principal

Arquivo:

``` text
rag/pipeline.py
```

Coordena:

``` text
query
  ↓
retrieval
  ↓
reranking
  ↓
context
```

A integração com a aplicação ocorre através de:

``` text
Trichon AI_core.py
```

------------------------------------------------------------------------

# Geração da resposta

O modelo gerador é executado através do **Ollama**.

Modelos utilizados no ambiente de desenvolvimento incluem:

``` text
qwen2.5:14b
llama3.1:latest
```

Os papéis são diferentes:

  Componente     Função
  -------------- ---------------------
  BGE-M3         Embeddings
  BM25           Busca lexical
  BGE Reranker   Reordenação
  Qwen / Llama   Geração da resposta

------------------------------------------------------------------------

# Armazenamento

## Banco da aplicação

``` text
data/
├── Trichon AI_app.db
└── trichon_app.db
```

Armazena os dados da aplicação.

## Índice RAG

``` text
Trichon AI_db/
```

Contém o armazenamento usado pela recuperação, incluindo Chroma e os
índices relacionados ao BM25.

A camada RAG não recria o banco de usuários.

### Separação

``` text
APLICAÇÃO
data/Trichon AI_app.db
        |
        +-- usuários
        +-- sessões
        +-- chats
        +-- mensagens
        +-- fontes
        +-- auditoria

RAG
Trichon AI_db/
        |
        +-- Chroma
        +-- embeddings
        +-- BM25
        +-- documentos indexados
```

------------------------------------------------------------------------

# Scripts

## Backup

Antes de reconstruir a base:

``` powershell
python -m scripts.backup_app_db
```

## Construção do RAG

Coloque os documentos em `pdfs/` e execute:

``` powershell
python -m scripts.build_rag
```

O fluxo é:

``` text
scripts/ingest_v4.py
        ↓
chunks.jsonl
        ↓
embed.py build
        ↓
Chroma + BM25
```

## Verificação dos dados

``` powershell
python -m scripts.verify_data
```

A V4 não inclui migração destrutiva das tabelas da aplicação.

## Teste de retrieval

Sem executar o Ollama:

``` powershell
python test_retrieval.py "Paciente feminina, queda na coroa, couro cabeludo oleoso e escamando."
```

Avaliação:

``` powershell
python evaluate_retrieval.py
```

------------------------------------------------------------------------

# Configuração

Variáveis sensíveis devem ficar em:

``` text
.env
```

O arquivo não deve ser versionado. Um modelo pode ser mantido em:

``` text
.env.example
```

Configuração do reranker:

``` dotenv
Trichon AI_RERANKER_ENABLED=1
Trichon AI_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
Trichon AI_OFFLINE=1
```

Com `Trichon AI_OFFLINE=1`, os modelos precisam estar disponíveis no cache
local.

Durante a preparação inicial, o modo offline pode ser desativado
temporariamente para permitir o download dos modelos e depois reativado.

------------------------------------------------------------------------

# Execução local

Atualize as migrações da aplicação:

``` powershell
alembic upgrade head
```

Inicie a aplicação:

``` powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

O RAG não exige Alembic para sua própria camada, pois utiliza Chroma +
BM25 e não adiciona tabelas ao banco de usuários nesta versão.

------------------------------------------------------------------------

# Estrutura do projeto

``` text
Trichon AI-ai/
│
├── app.py
├── Trichon AI_core.py
├── config.py
├── database.py
├── models.py
├── security.py
├── alembic.ini
├── requirements.txt
│
├── routers/
│   ├── auth.py
│   ├── admin.py
│   └── chats.py
│
├── services/
├── static/
├── migrations/
│
├── ingestion/
│   ├── router.py
│   ├── parsers.py
│   ├── structure.py
│   ├── metadata.py
│   └── chunker.py
│
├── retrieval/
│   ├── dense.py
│   ├── bm25.py
│   ├── hybrid.py
│   └── reranker.py
│
├── rag/
│   ├── query.py
│   ├── context.py
│   └── pipeline.py
│
├── evaluation/
│   └── retrieval.py
│
├── scripts/
│   ├── backup_app_db.py
│   ├── build_rag.py
│   ├── ingest_v4.py
│   └── verify_data.py
│
├── pdfs/
├── Trichon AI_db/
├── data/
│
├── chunks.jsonl
├── embed.py
├── chunk.py
└── retrieval_quality.py
```

------------------------------------------------------------------------

# Compatibilidade

Arquivos antigos foram mantidos para compatibilidade:

``` text
chunk.py
embed.py
retrieval_quality.py
```

A nova ingestão V4 é acionada por:

``` powershell
python -m scripts.ingest_v4
```

A integração do retrieval com a aplicação ocorre através de:

``` text
Trichon AI_core.py
```

------------------------------------------------------------------------

# Segurança e dados

Não versionar:

``` text
.env
Trichon AI_db/
pdfs/
chunks.jsonl
debug_retrieval_report.json
*.db
```

Também devem permanecer fora do Git:

``` text
.venv/
__pycache__/
*.log
```

O Ollama deve permanecer protegido sempre que possível.

Não exponha diretamente as portas:

``` text
8000
11434
```

à internet sem uma camada adequada de segurança, autenticação e controle
de acesso.

------------------------------------------------------------------------

# Fluxo completo

``` text
                         ┌───────────────┐
                         │    Usuário    │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   FastAPI     │
                         │   + Chat UI   │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Trichon AI_core  │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   RAG Query   │
                         └───────┬───────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             ┌─────────────┐          ┌─────────────┐
             │    Dense    │          │    BM25     │
             │  BGE-M3     │          │   Lexical   │
             └──────┬──────┘          └──────┬──────┘
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                         ┌───────────────┐
                         │    Hybrid     │
                         │      RRF      │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   Reranker    │
                         │ BGE Reranker  │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    Context    │
                         │ Parent/Child  │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │    Ollama     │
                         │ Qwen / Llama  │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │ Resposta +    │
                         │ fontes        │
                         └───────────────┘
```

Preparação da base:

``` text
PDF / DOCX
    ↓
Document Router
    ↓
Parser
    ↓
Structure
    ↓
Metadata
    ↓
Parent / Child Chunking
    ↓
BGE-M3
    ↓
Chroma + BM25
    ↓
Base de conhecimento pronta
```

------------------------------------------------------------------------

# Manutenção segura

Antes de qualquer reconstrução do índice ou alteração importante:

``` powershell
python -m scripts.backup_app_db
```

Depois:

``` powershell
python -m scripts.build_rag
```

A reconstrução trabalha sobre:

``` text
Trichon AI_db/
```

e não deve recriar:

``` text
data/Trichon AI_app.db
```

O princípio fundamental é:

> **O índice RAG pode ser reconstruído; os dados da aplicação devem ser
> preservados.**

------------------------------------------------------------------------

# Status da arquitetura V4

## Aplicação

-   FastAPI
-   autenticação
-   administração
-   chats
-   histórico
-   SQLAlchemy
-   Alembic
-   interface web

## RAG

-   ingestão estruturada;
-   PDF e DOCX;
-   metadata;
-   chunking semântico;
-   parent/child;
-   embeddings BGE-M3;
-   Chroma;
-   BM25;
-   Hybrid Retrieval;
-   RRF;
-   reranking com BGE Reranker;
-   reconstrução de contexto;
-   integração com Ollama;
-   testes e avaliação de retrieval.

## Princípio arquitetural

``` text
Dados da aplicação ≠ Índice RAG
```

A separação permite evoluir, reconstruir e avaliar a base de
conhecimento sem colocar em risco os usuários, sessões, chats e
mensagens da aplicação.
