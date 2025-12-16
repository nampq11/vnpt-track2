#!/bin/bash

# VNPT Track 2 - Evaluation Script
# Usage:
#   ./bin/eval.sh                              # Default: 5 questions, vnpt, agent mode
#   ./bin/eval.sh --n 10                       # First 10 questions
#   ./bin/eval.sh --provider azure --n 20      # Azure provider, 20 questions
#   ./bin/eval.sh --dataset val --no-agent     # val.json, simple prompting (no agent)
#   ./bin/eval.sh --all                        # Full dataset (all questions)

set -e

# Default values
DATASET="val"
PROVIDER="azure"
N_QUESTIONS="5"
QIDS=""
USE_AGENT="--use-agent"
MODEL=""
OUTPUT_FORMAT="csv"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --n)
            N_QUESTIONS="$2"
            shift 2
            ;;
        --qids)
            QIDS="$2"
            N_QUESTIONS=""  # Clear N_QUESTIONS when using QIDs
            shift 2
            ;;
        --all)
            N_QUESTIONS=""
            shift
            ;;
        --agent)
            USE_AGENT="--use-agent"
            shift
            ;;
        --no-agent)
            USE_AGENT=""
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --json)
            OUTPUT_FORMAT="json"
            shift
            ;;
        --csv)
            OUTPUT_FORMAT="csv"
            shift
            ;;
        -h|--help)
            echo "VNPT Track 2 - Evaluation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET    Dataset to use (default: val)"
            echo "                       Available: val, test"
            echo "  --provider PROVIDER  LLM provider (default: azure)"
            echo "                       Available: vnpt, azure, ollama"
            echo "  --n NUMBER           Number of questions to test (default: 5)"
            echo "  --qids QID1,QID2,... Test specific questions by ID"
            echo "                       Example: val_0002,val_0005,val_0010"
            echo "  --all                Test all questions in dataset"
            echo "  --agent              Use agent pipeline (default)"
            echo "  --no-agent           Use simple prompting (no agent)"
            echo "  --model MODEL        Model name (optional)"
            echo "  --csv                Output as CSV (default)"
            echo "  --json               Output as JSON"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                      # 5 questions, azure, agent"
            echo "  $0 --n 10                              # First 10 questions"
            echo "  $0 --qids val_0002                     # Single specific question"
            echo "  $0 --qids val_0002,val_0005,val_0010   # Multiple specific questions"
            echo "  $0 --provider azure --n 20             # Azure, 20 questions"
            echo "  $0 --dataset val --no-agent            # Simple prompting mode"
            echo "  $0 --all --provider vnpt               # Full dataset with VNPT"
            echo "  $0 --provider azure --model gpt-4o     # Specific model"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Construct input/output paths
INPUT_FILE="data/${DATASET}.json"
OUTPUT_DIR="results"
mkdir -p "$OUTPUT_DIR"

# Generate output filename
AGENT_SUFFIX=""
if [ -n "$USE_AGENT" ]; then
    AGENT_SUFFIX="_agent"
fi

N_SUFFIX=""
if [ -n "$QIDS" ]; then
    # Count number of QIDs
    IFS=',' read -ra QID_ARRAY <<< "$QIDS"
    QID_COUNT=${#QID_ARRAY[@]}
    N_SUFFIX="_qids${QID_COUNT}"
elif [ -n "$N_QUESTIONS" ]; then
    N_SUFFIX="_n${N_QUESTIONS}"
fi

OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}_${PROVIDER}${AGENT_SUFFIX}${N_SUFFIX}.${OUTPUT_FORMAT}"

# Construct command
CMD="uv run python predict.py --mode eval --input \"$INPUT_FILE\" --output \"$OUTPUT_FILE\" --provider \"$PROVIDER\""

if [ -n "$USE_AGENT" ]; then
    CMD="$CMD $USE_AGENT"
fi

if [ -n "$QIDS" ]; then
    CMD="$CMD --qids \"$QIDS\""
elif [ -n "$N_QUESTIONS" ]; then
    CMD="$CMD --n $N_QUESTIONS"
fi

if [ -n "$MODEL" ]; then
    CMD="$CMD --model \"$MODEL\""
fi

# Print configuration
echo "================================"
echo "🧪 VNPT Track 2 - Evaluation"
echo "================================"
echo "Dataset:       $DATASET.json"
echo "Provider:      $PROVIDER"
if [ -n "$MODEL" ]; then
    echo "Model:         $MODEL"
fi
if [ -n "$QIDS" ]; then
    echo "Questions:     Specific QIDs: $QIDS"
elif [ -n "$N_QUESTIONS" ]; then
    echo "Questions:     First $N_QUESTIONS"
else
    echo "Questions:     All"
fi
if [ -n "$USE_AGENT" ]; then
    echo "Mode:          Agent Pipeline"
else
    echo "Mode:          Simple Prompting"
fi
echo "Output:        $OUTPUT_FILE"
echo "================================"
echo ""

# Run evaluation
eval $CMD

echo ""
echo "✅ Evaluation complete!"
echo "📄 Results saved to: $OUTPUT_FILE"
if [ -f "${OUTPUT_FILE%.csv}_metrics.json" ] || [ -f "${OUTPUT_FILE%.json}_metrics.json" ]; then
    METRICS_FILE="${OUTPUT_FILE%.*}_metrics.json"
    echo "📊 Metrics saved to: $METRICS_FILE"
fi

