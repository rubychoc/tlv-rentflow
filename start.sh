#!/usr/bin/env bash
# First-time setup and launch.
# Usage: ./start.sh [port]

set -e

PORT=${1:-8000}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -e . -q

if [ ! -f ".env" ]; then
  echo "No .env file found."
  read -rp "Paste your OpenAI API key: " key
  echo "OPENAI_API_KEY=$key" > .env
  echo ".env created."
fi

(sleep 1.5 && open "http://localhost:$PORT") &

echo "Starting RentFlow on http://localhost:$PORT  (Ctrl+C to stop)"
uvicorn rentflow.ingestion.app:app --port "$PORT"
