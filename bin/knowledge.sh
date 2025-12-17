#!/bin/bash
# Knowledge Base Manager - CLI for LanceDB knowledge base management
#
# Usage:
#   ./bin/knowledge.sh <command> [options]
#
# Commands:
#   info                         Show knowledge base statistics
#   build                        Build index from scratch (~10 min)
#   upsert                       Add/update documents incrementally (~30 sec per 100)
#   delete                       Delete documents by file or category
#
# Quick Examples:
#   ./bin/knowledge.sh info
#   ./bin/knowledge.sh build --data-dir data/data --provider azure
#   ./bin/knowledge.sh upsert --data-dir data/new --provider azure
#   ./bin/knowledge.sh delete --file "data/data/Bac_Ho/Hồ_Chí_Minh.txt"
#   ./bin/knowledge.sh delete --category "test_category"
#
# Common Options:
#   --provider {azure,vnpt}      Embedding provider (default: azure)
#   --index-dir PATH             Index directory (default: data/embeddings/knowledge)
#   --data-dir PATH              Source data directory
#   --chunk-size N               Chunk size in chars (default: 512)
#   --overlap N                  Chunk overlap in chars (default: 50)
#
# Full Help:
#   ./bin/knowledge.sh --help
#   ./bin/knowledge.sh build --help
#   ./bin/knowledge.sh upsert --help
#   ./bin/knowledge.sh delete --help
#
# Current Status:
#   Location: data/embeddings/knowledge/knowledge.lance/
#   Vectors: 26,513 chunks
#   Dimension: 1536 (Azure embeddings)
#   Categories: 26 (History, Culture, Law, etc.)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
uv run python src/utils/knowledge_manager.py "$@"

