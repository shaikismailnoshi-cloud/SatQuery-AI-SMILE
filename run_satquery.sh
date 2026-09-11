#!/bin/bash
# SatQuery AI — One-Command Production Launcher
# SIH26167 | Agentic Multimodal Remote-Sensing Vision-Language AI

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  🛰️ SatQuery AI: Agentic Remote-Sensing AI System (SIH26167)"
echo "============================================================"

# Activate Virtual Environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r backend/requirements.txt python-multipart
fi

echo "1. Checking Python & PyTorch GPU / MPS Acceleration..."
python3 -c "import torch; print('PyTorch Version:', torch.__version__); print('MPS Device Available:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()); print('CUDA Available:', torch.cuda.is_available())"

echo "2. Running Unit & Validation Tests..."
python3 -m unittest discover -s tests -p "test_*.py"

echo "3. Starting FastAPI Backend API Server on http://127.0.0.1:8000..."
python3 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "4. Serving Research Web Frontend on http://127.0.0.1:3000..."
python3 -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

sleep 2

echo "============================================================"
echo "  🚀 SatQuery AI System is LIVE and ready for queries!"
echo "  👉 Web Interface: http://localhost:3000"
echo "  👉 API Server & Docs: http://localhost:8000/docs"
echo "============================================================"

# Trap CTRL+C to cleanly terminate both servers
trap "echo 'Stopping SatQuery AI servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

wait
