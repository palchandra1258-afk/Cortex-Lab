#!/bin/bash
#
# Launch script for Cortex Lab — Fine-Tuned DeepSeek-R1-7B Agentic RAG
# Starts BOTH the Python backend (model + RAG engine) and the Next.js frontend
#

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║     Cortex Lab · Fine-Tuned DeepSeek-R1-7B · Agentic RAG ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ──────────────────────────────────────────
MERGED_MODEL="$ROOT/fine_tuned/stage15_spin/merged/config.json"
if [ -f "$MERGED_MODEL" ]; then
    echo "  ✓ Fine-tuned model found (stage15_spin/merged)"
else
    echo "  ⚠ Fine-tuned model not found — will fall back to HuggingFace"
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  ✓ GPU: $GPU_NAME ($GPU_MEM)"
fi

# ── 1. Activate Virtual Environment ────────────────────────────
if [ -f "$ROOT/venv/bin/activate" ]; then
    source "$ROOT/venv/bin/activate"
    echo "  ✓ Virtual environment activated"
fi

# ── 2. Kill stale processes ─────────────────────────────────────
for PORT in 8000 3000; do
    if lsof -ti:$PORT &>/dev/null; then
        echo "  ⚠ Port $PORT in use — killing stale process"
        kill $(lsof -ti:$PORT) 2>/dev/null || true
        sleep 1
    fi
done

# ── 3. Backend ──────────────────────────────────────────────────
echo ""
echo "  [1/2] Starting Python backend (FastAPI + Fine-Tuned 7B + RAG Engine) …"
echo "        → http://localhost:8000"
echo ""

cd "$ROOT/backend"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python server.py &
BACKEND_PID=$!

# Wait for backend
echo "  ⏳ Waiting for backend …"
for i in {1..60}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "  ✓ Backend responding"
        break
    fi
    sleep 2
done

# ── 4. Frontend ─────────────────────────────────────────────────
echo ""
echo "  [2/2] Starting Next.js frontend → http://localhost:3000"
echo ""

cd "$ROOT/frontend"
npx next dev --port 3000 &
FRONTEND_PID=$!

# ── Cleanup ─────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "  Shutting down …"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "  ✓ Stopped."
    exit 0
}
trap cleanup INT TERM

echo ""
echo "  ✓ Both servers starting. Open http://localhost:3000"
echo "  ✓ Press Ctrl+C to stop."
echo ""

wait
