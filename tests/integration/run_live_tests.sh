#!/usr/bin/env bash
# Run live integration tests that hit the real OpenAI API.
# OPENAI_API_KEY is loaded from .env by the test files via load_dotenv.
#
# Usage:
#   ./tests/integration/run_live_tests.sh              # extraction tests only (fast)
#   ./tests/integration/run_live_tests.sh --probes     # hallucination/consistency probes only
#   ./tests/integration/run_live_tests.sh --cost       # cost/token analysis only
#   ./tests/integration/run_live_tests.sh --all        # everything
#   ./tests/integration/run_live_tests.sh [pytest args] # extra args passed through
#
# Examples:
#   ./tests/integration/run_live_tests.sh -v
#   ./tests/integration/run_live_tests.sh --probes -s   # -s shows probe distributions
#   ./tests/integration/run_live_tests.sh --cost -v

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"
source .venv/bin/activate

PROBES=false
COST=false
ALL=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --probes) PROBES=true ;;
        --cost)   COST=true ;;
        --all)    ALL=true ;;
        *)        EXTRA_ARGS+=("$arg") ;;
    esac
done

COST_FILE="tests/integration/test_cost_analysis.py"

if [ "$ALL" = true ]; then
    TARGET="tests/integration/"
    echo "Running all live tests (extraction + probes + cost)..."
elif [ "$PROBES" = true ]; then
    TARGET="tests/integration/test_llm_probes.py"
    echo "Running hallucination/consistency probes..."
elif [ "$COST" = true ]; then
    if [ ! -f "$COST_FILE" ]; then
        echo "Cost analysis file not found: $COST_FILE"
        exit 1
    fi
    TARGET="$COST_FILE"
    echo "Running cost/token analysis..."
else
    TARGET="tests/integration/test_extraction_live.py"
    echo "Running live extraction tests..."
fi

python -m pytest "$TARGET" -m live -v "${EXTRA_ARGS[@]}"
