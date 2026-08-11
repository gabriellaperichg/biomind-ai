#!/usr/bin/env bash
# Biomind — sobe o serviço no pod da RunPod com um comando.
# Rode de dentro de /workspace:  bash start.sh
set -e
cd "$(dirname "$0")"

# 1) Ollama: sobe em segundo plano se ainda não estiver rodando
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo ">> iniciando Ollama..."
  ollama serve > ollama.log 2>&1 &
  sleep 5
fi

# 2) garante o modelo baixado (troque pelo que você escolheu)
MODELO="${OLLAMA_MODEL:-qwen2.5:14b}"
if ! ollama list | grep -q "$MODELO"; then
  echo ">> baixando modelo $MODELO (uma vez)..."
  ollama pull "$MODELO"
fi

# 3) sobe a interface web
echo ">> subindo Biomind em :8000"
exec uvicorn app:app --host 0.0.0.0 --port 8000
