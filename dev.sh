#!/usr/bin/env bash
# Start the RentFlow server and open the UI in the browser.
# Usage: ./dev.sh [port]

set -e

PORT=${1:-8000}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"
source .venv/bin/activate

# Open the browser a moment after uvicorn binds the port.
(sleep 1.5 && open "http://localhost:$PORT") &

echo "Starting RentFlow on http://localhost:$PORT  (Ctrl+C to stop)"
uvicorn rentflow.ingestion.app:app --port "$PORT"
