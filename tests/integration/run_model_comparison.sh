#!/usr/bin/env bash
# Run all live tests across multiple models and produce a combined comparison report.
#
# For each model it runs:
#   1. test_cost_analysis.py      → cost per offer, token stats
#   2. test_extraction_live.py    → extraction accuracy on clear fixtures
#   3. test_llm_probes.py         → hallucination / consistency accuracy
#
# Then merges all results into tests/integration/model_comparison.json.
#
# Usage:
#   ./tests/integration/run_model_comparison.sh
#   ./tests/integration/run_model_comparison.sh --models "gpt-4.1-mini gpt-4.1-nano"
#
# OPENAI_API_KEY is loaded from .env by the test files.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

# Default models to compare
MODELS=("gpt-4.1-mini" "gpt-4.1-nano" "gpt-4.1")

# Parse --models override
for arg in "$@"; do
    case "$arg" in
        --models)
            shift
            IFS=' ' read -ra MODELS <<< "$1"
            ;;
    esac
done

RESULTS_DIR="$SCRIPT_DIR/model_results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "TLV-RentFlow — Multi-Model Comparison Run"
echo "Models: ${MODELS[*]}"
echo "Results dir: $RESULTS_DIR"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  Model: $MODEL"
    echo "------------------------------------------------------------"

    export EXTRACTION_MODEL="$MODEL"
    MODEL_DIR="$RESULTS_DIR/$MODEL"
    mkdir -p "$MODEL_DIR"

    # 1. Cost analysis (marked live+slow, use -s to see the printed table)
    echo "  [1/3] Cost analysis..."
    python -m pytest tests/integration/test_cost_analysis.py \
        -m "live and slow" -s -q --no-header 2>&1 | tail -5 || true
    # Cost report is now written directly to the integration folder
    [ -f "$SCRIPT_DIR/cost_analysis_results.json" ] && cp "$SCRIPT_DIR/cost_analysis_results.json" "$MODEL_DIR/cost.json"

    # 2. Extraction live tests — use pytest-json-report if installed, else plain
    echo "  [2/3] Extraction live tests..."
    if python -c "import pytest_json_report" 2>/dev/null; then
        python -m pytest tests/integration/test_extraction_live.py \
            -m live -q --no-header \
            --json-report --json-report-file="$MODEL_DIR/extraction.json" \
            2>&1 | tail -3 || true
    else
        # Fallback: capture counts from pytest output
        OUTPUT=$(python -m pytest tests/integration/test_extraction_live.py \
            -m live -q --no-header 2>&1 || true)
        echo "$OUTPUT" | tail -3
        PASSED=$(echo "$OUTPUT" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
        FAILED=$(echo "$OUTPUT" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)
        echo "{\"passed\": $PASSED, \"failed\": $FAILED, \"source\": \"stdout_parse\"}" \
            > "$MODEL_DIR/extraction.json"
    fi

    # 3. LLM probes (writes probe_report.json automatically)
    echo "  [3/3] LLM probes..."
    python -m pytest tests/integration/test_llm_probes.py \
        -m live -q --no-header 2>&1 | tail -3 || true
    [ -f "$SCRIPT_DIR/probe_report.json" ] && cp "$SCRIPT_DIR/probe_report.json" "$MODEL_DIR/probes.json"

    echo "  Done → $MODEL_DIR/"
done

unset EXTRACTION_MODEL

# Merge into one comparison file
echo ""
echo "------------------------------------------------------------"
echo "  Merging results across models..."
echo "------------------------------------------------------------"
python "$SCRIPT_DIR/merge_model_results.py"

echo ""
echo "============================================================"
echo "  Combined report: $SCRIPT_DIR/model_comparison.json"
echo "============================================================"
