#!/bin/bash

# VNPT Track 2 - Inference Pipeline Runner
# Usage:
#   ./bin/inference.sh test          # Quick test on 5 questions
#   ./bin/inference.sh eval          # Full evaluation on test.json
#   ./bin/inference.sh eval val      # Full evaluation on val.json
#   ./bin/inference.sh inference     # Inference only (no metrics)

MODE="${1:-test}"
DATASET="${2:-test}"
MODEL="${3:-qwen3:1.7b}"

INPUT_FILE="data/${DATASET}.json"
OUTPUT_DIR="results"
OUTPUT_FILE="${OUTPUT_DIR}/predictions_${DATASET}.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

case "$MODE" in
    test)
        echo "🧪 Running quick test on 5 questions..."
        uv run python predict.py --mode test --input "$INPUT_FILE" --model "$MODEL" --n 5
        ;;
    eval)
        echo "📊 Running full evaluation on $DATASET.json..."
        uv run python predict.py --mode eval --input "$INPUT_FILE" --output "$OUTPUT_FILE" --model "$MODEL"
        ;;
    inference)
        echo "🚀 Running inference (no evaluation)..."
        uv run python predict.py --mode inference --input "$INPUT_FILE" --output "$OUTPUT_FILE" --model "$MODEL"
        ;;
    *)
        echo "❌ Unknown mode: $MODE"
        echo ""
        echo "Usage:"
        echo "  $0 test [dataset] [model]     # Quick test"
        echo "  $0 eval [dataset] [model]     # Full evaluation"
        echo "  $0 inference [dataset] [model] # Inference only"
        echo ""
        echo "Examples:"
        echo "  $0 test                # Test on data/test.json"
        echo "  $0 eval val           # Eval on data/val.json"
        echo "  $0 eval test qwen3:1.7b  # Eval with specific model"
        exit 1
        ;;
esac

