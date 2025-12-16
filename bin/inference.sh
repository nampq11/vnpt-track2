#!/bin/bash

# VNPT Track 2 - Inference Pipeline Runner
# Usage:
#   ./bin/inference.sh test          # Quick test on 5 questions
#   ./bin/inference.sh eval          # Full evaluation on test.json (JSON output)
#   ./bin/inference.sh eval val      # Full evaluation on val.json (JSON output)
#   ./bin/inference.sh eval test csv # Full evaluation on test.json (CSV output)
#   ./bin/inference.sh inference     # Inference only (no metrics)

MODE="${1:-test}"
DATASET="${2:-test}"
MODEL="${3:-qwen3:1.7b}"
OUTPUT_FORMAT="${4:-json}"  # json or csv

INPUT_FILE="data/${DATASET}.json"
OUTPUT_DIR="results"
OUTPUT_EXT="${OUTPUT_FORMAT}"
OUTPUT_FILE="${OUTPUT_DIR}/predictions_${DATASET}.${OUTPUT_EXT}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

case "$MODE" in
    test)
        echo "🧪 Running quick test on 5 questions..."
        uv run python predict.py --mode test --input "$INPUT_FILE" --model "$MODEL" --n 5
        ;;
    eval)
        echo "📊 Running full evaluation on $DATASET.json (output: $OUTPUT_FORMAT)..."
        uv run python predict.py --mode eval --input "$INPUT_FILE" --output "$OUTPUT_FILE" --model "$MODEL"
        ;;
    inference)
        echo "🚀 Running inference on $DATASET.json (output: $OUTPUT_FORMAT)..."
        uv run python predict.py --mode inference --input "$INPUT_FILE" --output "$OUTPUT_FILE" --model "$MODEL"
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo ""
        echo "Usage:"
        echo "  $0 test [dataset] [model] [format]      # Quick test"
        echo "  $0 eval [dataset] [model] [format]      # Full evaluation"
        echo "  $0 inference [dataset] [model] [format]  # Inference only"
        echo ""
        echo "Supported formats: json (default), csv"
        echo ""
        echo "Examples:"
        echo "  $0 test                           # Test on data/test.json"
        echo "  $0 eval val                      # Eval on data/val.json (JSON)"
        echo "  $0 eval test qwen3:1.7b csv     # Eval with CSV output"
        echo "  $0 inference test qwen3:1.7b csv # Inference with CSV output"
        exit 1
        ;;
esac

