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

INPUT_FILE="data/${DATASET}.json"
OUTPUT_DIR="results"
OUTPUT_EXT="${OUTPUT_FORMAT}"
OUTPUT_FILE="${OUTPUT_DIR}/predictions_${DATASET}.${OUTPUT_EXT}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

case "$MODE" in
    test)
        echo "🧪 Running quick test on 5 questions with $PROVIDER..."
        if [ -n "$MODEL" ]; then
            uv run python predict.py --mode test --input "$INPUT_FILE" --provider "$PROVIDER" --model "$MODEL" --n 5
        else
            uv run python predict.py --mode test --input "$INPUT_FILE" --provider "$PROVIDER" --n 5
        fi
        ;;
    eval)
        echo "📊 Running full evaluation on $DATASET.json (output: $OUTPUT_FORMAT) with $PROVIDER..."
        if [ -n "$MODEL" ]; then
            uv run python predict.py --mode eval --input "$INPUT_FILE" --output "$OUTPUT_FILE" --provider "$PROVIDER" --model "$MODEL"
        else
            uv run python predict.py --mode eval --input "$INPUT_FILE" --output "$OUTPUT_FILE" --provider "$PROVIDER"
        fi
        ;;
    inference)
        echo "🚀 Running inference on $DATASET.json (output: $OUTPUT_FORMAT) with $PROVIDER..."
        if [ -n "$MODEL" ]; then
            uv run python predict.py --mode inference --input "$INPUT_FILE" --output "$OUTPUT_FILE" --provider "$PROVIDER" --model "$MODEL"
        else
            uv run python predict.py --mode inference --input "$INPUT_FILE" --output "$OUTPUT_FILE" --provider "$PROVIDER"
        fi
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo ""
        echo "Usage:"
        echo "  $0 test [dataset] [format] [provider] [model]      # Quick test"
        echo "  $0 eval [dataset] [format] [provider] [model]      # Full evaluation"
        echo "  $0 inference [dataset] [format] [provider] [model] # Inference only"
        echo ""
        echo "Supported formats: json (default), csv"
        echo "Supported providers: vnpt (default), azure, ollama"
        echo ""
        echo "Examples:"
        echo "  $0 test                                # Test on data/test.json (VNPT)"
        echo "  $0 eval val                           # Eval on data/val.json (VNPT, JSON)"
        echo "  $0 eval val csv azure                 # Eval on val.json with Azure (CSV)"
        echo "  $0 eval test json azure gpt-4.1       # Eval with Azure GPT-4.1"
        echo "  $0 inference test csv ollama qwen3:1.7b # Inference with Ollama"
        exit 1
        ;;
esac

