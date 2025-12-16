#!/bin/bash

# VNPT Track 2 - Inference Pipeline Runner
# Usage:
#   ./bin/inference.sh test          # Quick test on 5 questions
#   ./bin/inference.sh eval          # Full evaluation on test.json (JSON output)
#   ./bin/inference.sh eval val      # Full evaluation on val.json (JSON output)
#   ./bin/inference.sh eval test csv # Full evaluation on test.json (CSV output)
#   ./bin/inference.sh eval test csv vnpt  # Full evaluation with VNPT provider
#   ./bin/inference.sh eval test csv azure # Full evaluation with Azure provider
#   ./bin/inference.sh inference     # Inference only (no metrics)

MODE="${1:-test}"
DATASET="${2:-test}"
OUTPUT_FORMAT="${3:-json}"  # json or csv
PROVIDER="${4:-vnpt}"       # ollama, vnpt, or azure
MODEL="${5:-}"              # optional model override
USE_AGENT="${6:-}"          # optional: --use-agent flag

INPUT_FILE="data/${DATASET}.json"
OUTPUT_DIR="results"
OUTPUT_EXT="${OUTPUT_FORMAT}"

# Add suffix for agent mode in filename
if [ "$USE_AGENT" = "--use-agent" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/predictions_${DATASET}_agent.${OUTPUT_EXT}"
else
    OUTPUT_FILE="${OUTPUT_DIR}/predictions_${DATASET}.${OUTPUT_EXT}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

case "$MODE" in
    test)
        MODE_DISPLAY="Simple"
        if [ "$USE_AGENT" = "--use-agent" ]; then
            MODE_DISPLAY="Agent"
        fi
        echo "🧪 Running quick test on 5 questions with $PROVIDER ($MODE_DISPLAY mode)..."
        
        CMD="uv run python predict.py --mode test --input \"$INPUT_FILE\" --provider \"$PROVIDER\" --n 5"
        if [ -n "$MODEL" ]; then
            CMD="$CMD --model \"$MODEL\""
        fi
        if [ "$USE_AGENT" = "--use-agent" ]; then
            CMD="$CMD --use-agent"
        fi
        eval $CMD
        ;;
    eval)
        MODE_DISPLAY="Simple"
        if [ "$USE_AGENT" = "--use-agent" ]; then
            MODE_DISPLAY="Agent"
        fi
        echo "📊 Running full evaluation on $DATASET.json (output: $OUTPUT_FORMAT) with $PROVIDER ($MODE_DISPLAY mode)..."
        
        CMD="uv run python predict.py --mode eval --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" --provider \"$PROVIDER\""
        if [ -n "$MODEL" ]; then
            CMD="$CMD --model \"$MODEL\""
        fi
        if [ "$USE_AGENT" = "--use-agent" ]; then
            CMD="$CMD --use-agent"
        fi
        eval $CMD
        ;;
    inference)
        MODE_DISPLAY="Simple"
        if [ "$USE_AGENT" = "--use-agent" ]; then
            MODE_DISPLAY="Agent"
        fi
        echo "🚀 Running inference on $DATASET.json (output: $OUTPUT_FORMAT) with $PROVIDER ($MODE_DISPLAY mode)..."
        
        CMD="uv run python predict.py --mode inference --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" --provider \"$PROVIDER\""
        if [ -n "$MODEL" ]; then
            CMD="$CMD --model \"$MODEL\""
        fi
        if [ "$USE_AGENT" = "--use-agent" ]; then
            CMD="$CMD --use-agent"
        fi
        eval $CMD
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "VNPT Track 2 - Inference Pipeline Runner"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        echo "Usage:"
        echo "  $0 MODE [DATASET] [FORMAT] [PROVIDER] [MODEL] [--use-agent]"
        echo ""
        echo "Modes:"
        echo "  test        Quick test on first 5 questions"
        echo "  eval        Full evaluation with metrics"
        echo "  inference   Inference only (no metrics)"
        echo ""
        echo "Arguments:"
        echo "  DATASET     Dataset name (default: test)"
        echo "              Available: test, val"
        echo "  FORMAT      Output format (default: json)"
        echo "              Available: json, csv"
        echo "  PROVIDER    LLM provider (default: vnpt)"
        echo "              Available: vnpt, azure, ollama"
        echo "  MODEL       Model name override (optional)"
        echo "  --use-agent Enable agent pipeline with task routing (optional)"
        echo "              Default: simple prompting mode"
        echo ""
        echo "Examples:"
        echo ""
        echo "  Basic Usage:"
        echo "  $0 test                                      # Quick test (simple mode)"
        echo "  $0 eval val                                 # Full eval on val.json"
        echo "  $0 eval val csv azure                       # Val, CSV, Azure"
        echo ""
        echo "  With Agent Mode:"
        echo "  $0 test test json azure \"\" --use-agent      # Test with agent"
        echo "  $0 eval val csv azure \"\" --use-agent        # Eval with agent"
        echo "  $0 eval val csv azure gpt-4o --use-agent    # With model & agent"
        echo ""
        echo "  Output files:"
        echo "  • Simple mode: predictions_val.csv"
        echo "  • Agent mode:  predictions_val_agent.csv"
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "TIP: For more flexibility (--qids, --n, --no-agent option),"
        echo "     use the eval.sh script:"
        echo "     ./bin/eval.sh --help"
        echo "═══════════════════════════════════════════════════════════════"
        exit 1
        ;;
esac

