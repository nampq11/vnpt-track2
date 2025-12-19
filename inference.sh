#!/bin/bash
# VNPT Track 2 - Submission Inference Script
# This script runs the complete pipeline for submission:
# - Reads from /code/private_test.json (mounted by BTC)
# - Outputs submission.csv (required format)

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════"
echo "VNPT Track 2 - Submission Inference Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if input file exists
INPUT_FILE="/code/private_test.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not found: $INPUT_FILE"
    echo "   Please ensure the test file is mounted at /code/private_test.json"
    exit 1
fi

echo "✓ Input file found: $INPUT_FILE"

# Check if knowledge base exists (optional, but recommended)
KB_PATH="data/embeddings/knowledge"
if [ ! -d "$KB_PATH" ]; then
    echo "⚠️  Warning: Knowledge base not found at $KB_PATH"
    echo "   Pipeline will run without RAG capabilities"
else
    echo "✓ Knowledge base found at $KB_PATH"
fi

# Run inference pipeline
# - Mode: inference (no evaluation)
# - Input: /code/private_test.json
# - Output: submission.csv (root level)
# - Provider: vnpt (default)
# - Use agent: True (enabled)
# - Batch size: 6 (optimal for 4-8 threads as recommended by BTC)
echo ""
echo "🚀 Starting inference pipeline..."
echo "   Input:  $INPUT_FILE"
echo "   Output: submission.csv"
echo "   Provider: vnpt"
echo "   Agent mode: enabled"
echo "   Batch size: 6 threads (optimal for 4-8 range)"
echo ""

python predict.py \
    --mode inference \
    --input "$INPUT_FILE" \
    --output submission.csv \
    --provider vnpt \
    --use-agent \
    --batch-size 6

# Verify output file exists
if [ ! -f "submission.csv" ]; then
    echo ""
    echo "❌ Error: submission.csv was not generated!"
    exit 1
fi

# Verify CSV format (check header)
if ! head -1 submission.csv | grep -q "qid,answer"; then
    echo ""
    echo "⚠️  Warning: CSV header may be incorrect. Expected: qid,answer"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Inference complete! Output saved to: submission.csv"
echo "═══════════════════════════════════════════════════════════════"

