#!/bin/bash

# Avvia il wrapper FastAPI del progetto, che espone l'API OpenAI-compatible
# e applica la validazione JSON lato server.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

cd "$SCRIPT_DIR"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"
mkdir -p "$HF_HOME/hub"

# Default coerenti con il wrapper Python; tutto resta sovrascrivibile via env.
export VLLM_MODEL_ID="${VLLM_MODEL_ID:-${MODEL_ID:-Qwen/Qwen3-14B}}"
export MODEL_ID="${MODEL_ID:-$VLLM_MODEL_ID}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

PYTHON_BIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYENV_ENV_NAME="${PYENV_ENV_NAME:-llm-host-py312}"
    if command -v pyenv >/dev/null 2>&1; then
        PYENV_PREFIX="$(pyenv prefix "${PYENV_ENV_NAME}" 2>/dev/null || true)"
        if [[ -n "${PYENV_PREFIX:-}" && -x "$PYENV_PREFIX/bin/python" ]]; then
            PYTHON_BIN="$PYENV_PREFIX/bin/python"
        fi
    fi
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="python3"
fi

if ! "$PYTHON_BIN" -c 'import fastapi, uvicorn, dotenv' >/dev/null 2>&1; then
    echo "Missing wrapper dependencies. Install them with: pip install -r requirements.txt"
    exit 1
fi

if [[ "${LOAD_IN_4BIT:-false}" == "true" ]] && ! "$PYTHON_BIN" -c 'import bitsandbytes' >/dev/null 2>&1; then
    echo "Missing bitsandbytes for LOAD_IN_4BIT=true. Install it with: pip install -r requirements.txt"
    exit 1
fi

echo "Avvio del wrapper FastAPI con modello: $MODEL_ID"
echo "Endpoint: http://$HOST:$PORT/v1/chat/completions"

exec "$PYTHON_BIN" -m uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info
