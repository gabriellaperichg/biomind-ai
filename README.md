# Biomind — apoio à decisão (offline)

Assistente que orienta a investigação de casos a partir da base de PDFs da clínica.
Roda 100% na máquina local — nenhum dado sai para a internet.

## Arquivos
- `01_chunk.py` — lê os PDFs (pasta `pdfs/`) e gera os chunks (`chunks.jsonl`).
- `02_embed.py` — gera embeddings e monta o índice (Chroma) em `biomind_db/`.
- `biomind_core.py` — lógica central (busca + piso de relevância + Ollama).
- `03_answer.py` — usar pela linha de comando.
- `app.py` + `index.html` — interface gráfica (web) para a demo.

## Instalação (uma vez)
```bash
pip install -r requirements.txt
```
Instale o Ollama (https://ollama.com) e baixe um modelo:
```bash
ollama pull llama3.1
```

## Preparar a base (sempre que adicionar/atualizar PDFs)
```bash
# coloque os PDFs em ./pdfs/
python 01_chunk.py
python 02_embed.py build
```

## Rodar a interface (a demo)
```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```
Abra **http://127.0.0.1:8000** no navegador.

Ou, sem interface, pela linha de comando:
```bash
python 03_answer.py "mulher de 29 anos, rarefacao no topo, deseja engravidar"
```

## Botões para calibrar (em `biomind_core.py`)
- `OLLAMA_MODEL` — o modelo que você baixou no Ollama.
- `PISO_RELEVANCIA` — similaridade mínima para responder. Calibre: rode casos que
  a base cobre bem e casos que ela não cobre; o piso fica no meio dos dois grupos.

## Dicas para a demo
- **Latência**: um modelo local pode levar de alguns segundos a mais de um minuto por
  resposta, dependendo da máquina. Sem GPU, prefira um modelo menor (ex.: `llama3.2:3b`).
- **Aquecimento**: a primeira pergunta carrega o modelo na memória e é a mais lenta.
  Faça uma pergunta de teste antes da demo para "aquecer".
- **Ensaio**: rode hoje, de ponta a ponta, com os PDFs reais e exatamente os casos que
  você vai mostrar. Deixe o Ollama rodando antes de começar.