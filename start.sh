#!/bin/bash
#
# Launch script for DeepSeek-R1 Chat Interface
# Starts BOTH the Python backend and the Next.js frontend
#

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║       DeepSeek R1 · Full-Stack Chat Interface        ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Backend ──────────────────────────────────────────────────
echo "  [1/2] Starting Python backend (FastAPI + Model) …"
echo "        → http://localhost:8000"
echo ""

cd "$ROOT/backend"

# Activate venv if it exists one level up
if [ -f "$ROOT/venv/bin/activate" ]; then
    source "$ROOT/venv/bin/activate"
fi

python server.py &
BACKEND_PID=$!

# ── 2. Frontend ─────────────────────────────────────────────────
echo "  [2/2] Starting Next.js frontend …"
echo "        → http://localhost:3000"
echo ""

cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

# ── Cleanup on Ctrl-C ──────────────────────────────────────────
trap "echo ''; echo '  Shutting down …'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

echo ""
echo "  ✓ Both servers starting."
echo "  ✓ Open http://localhost:3000 in your browser."
echo "  ✓ Press Ctrl+C to stop both servers."
echo ""

wait
