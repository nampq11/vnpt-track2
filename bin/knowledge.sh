#!/bin/bash
# Simple wrapper for knowledge_manager.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
uv run python src/utils/knowledge_manager.py "$@"

