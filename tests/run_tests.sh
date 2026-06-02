#!/usr/bin/env bash
# Run the test suite.
#
# Usage:
#   ./tests/run_tests.sh           # all fast deterministic tests (default)
#   ./tests/run_tests.sh --live    # include live OpenAI tests (requires OPENAI_API_KEY)
#   ./tests/run_tests.sh --slow    # include slow stress/concurrency tests
#   ./tests/run_tests.sh --all     # everything
#   ./tests/run_tests.sh [pytest args]   # pass any extra args directly to pytest

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"
source .venv/bin/activate

LIVE=false
SLOW=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --live) LIVE=true ;;
        --slow) SLOW=true ;;
        --all)  LIVE=true; SLOW=true ;;
        *)      EXTRA_ARGS+=("$arg") ;;
    esac
done

if [ "$LIVE" = true ]; then
    if [ -z "$OPENAI_API_KEY" ] && [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: --live requires OPENAI_API_KEY. Add it to .env or export it first."
        exit 1
    fi
fi

echo "Running tests (live=$LIVE, slow=$SLOW)..."

if [ "$LIVE" = true ] && [ "$SLOW" = true ]; then
    python -m pytest tests/ -p no:warnings "${EXTRA_ARGS[@]}"
elif [ "$LIVE" = true ]; then
    python -m pytest tests/ -m "not slow" -p no:warnings "${EXTRA_ARGS[@]}"
elif [ "$SLOW" = true ]; then
    python -m pytest tests/ -m "not live" -p no:warnings "${EXTRA_ARGS[@]}"
else
    python -m pytest tests/ -m "not live and not slow" -p no:warnings --no-header -q "${EXTRA_ARGS[@]}" \
        | grep -E "^[0-9]+ passed|^FAILED|error"
fi
