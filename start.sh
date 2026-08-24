#!/bin/bash

set -e

echo "======================================"
echo "        INICIANDO BIOMIND"
echo "======================================"

PROJECT_DIR="/workspace/biomind-ai"
OLLAMA_DIR="/workspace/ollama"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
OLLAMA_MODELS="$OLLAMA_DIR/models"

cd "$PROJECT_DIR"

# --------------------------------------------------
# 1. Ollama
# --------------------------------------------------

export PATH="$OLLAMA_DIR/bin:$PATH"
export OLLAMA_MODELS="$OLLAMA_MODELS"
export OLLAMA_HOST="0.0.0.0:11434"

echo ""
echo "[1/4] Verificando Ollama..."

if [ ! -x "$OLLAMA_BIN" ]; then
    echo "ERRO: Ollama não encontrado em:"
    echo "$OLLAMA_BIN"
    exit 1
fi

echo "Ollama: $("$OLLAMA_BIN" --version)"

# --------------------------------------------------
# 2. Ambiente Python
# --------------------------------------------------

echo ""
echo "[2/4] Ativando ambiente Python..."

if [ ! -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    echo "ERRO: .venv não encontrado."
    exit 1
fi

source "$PROJECT_DIR/.venv/bin/activate"

echo "Python: $(which python)"
python --version

# --------------------------------------------------
# 3. Iniciar Ollama
# --------------------------------------------------

echo ""
echo "[3/4] Iniciando Ollama..."

# Verifica se já existe um Ollama rodando
if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama já está rodando."
else
    echo "Iniciando Ollama..."

    nohup "$OLLAMA_BIN" serve \
        > "$PROJECT_DIR/ollama.log" 2>&1 &

    OLLAMA_PID=$!

    echo "Ollama PID: $OLLAMA_PID"

    echo "Aguardando Ollama..."

    for i in {1..30}; do
        if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            echo "Ollama está pronto."
            break
        fi

        sleep 1
    done

    if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "ERRO: Ollama não iniciou."
        echo ""
        echo "Últimas linhas do log:"
        tail -n 30 "$PROJECT_DIR/ollama.log"
        exit 1
    fi
fi

# --------------------------------------------------
# 4. Verificar modelo
# --------------------------------------------------

echo ""
echo "Modelo instalado:"

"$OLLAMA_BIN" list

if ! "$OLLAMA_BIN" list | grep -q "llama3.1:latest"; then
    echo ""
    echo "ERRO: llama3.1:latest não está instalado."
    echo "Instale manualmente com:"
    echo ""
    echo "ollama pull llama3.1:latest"
    exit 1
fi

# --------------------------------------------------
# 5. Iniciar FastAPI
# --------------------------------------------------

echo ""
echo "[4/4] Iniciando Biomind..."
echo ""

echo "======================================"
echo " API: http://0.0.0.0:8000"
echo " Ollama: http://0.0.0.0:11434"
echo "======================================"
echo ""

exec python -m uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000