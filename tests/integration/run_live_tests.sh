#!/usr/bin/env bash
# Run live integration tests that hit the real OpenAI API.
# Usage: ./tests/integration/run_live_tests.sh [extra pytest args]
# Example: ./tests/integration/run_live_tests.sh -v --tb=short

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"
source .venv/bin/activate

# OPENAI_API_KEY is loaded from .env by the test file itself via load_dotenv.
# No shell-level key handling needed here.

echo "Running live integration tests..."
python -m pytest tests/integration/ -m live -v "$@"
