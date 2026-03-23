#!/bin/bash
set -e

echo "=============================================="
echo "Autoresearch with Local LLM"
echo "Started: $(date)"
echo "=============================================="

# Step 1: Prepare data (downloads shards + trains tokenizer)
echo ""
echo "=== Step 1: Preparing data ==="
echo "Downloading data shards and training tokenizer..."
echo ""
uv run prepare.py --num-shards 10

echo ""
echo "Data preparation complete."
echo ""

# Step 2: Create experiment branch
echo "=== Step 2: Creating experiment branch ==="
BRANCH="autoresearch/local-llm-$(date +%Y%m%d)"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
echo "On branch: $BRANCH"
echo ""

# Step 3: Start autonomous training loop
echo "=== Step 3: Starting autonomous training agent ==="
echo "Model: ${AUTORESEARCH_MODEL:-Qwen3.5:latest} via ollama"
echo "Press Ctrl+C to stop."
echo ""
PYTHONUNBUFFERED=1 uv run python3 agent.py
