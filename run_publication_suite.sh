#!/usr/bin/env bash
# run_publication_suite.sh — One-command reproduction of publication artifacts.

set -euo pipefail
cd "$(dirname "$0")"

MODE=""
if [[ "$1" == "--fast" ]]; then
    MODE="--fast"
    echo "=== RDT Publication Suite (FAST mode) ==="
else
    echo "=== RDT Publication Suite (FULL mode) ==="
fi

echo ""
echo "Step 1/4: Correctness tests..."
python3 tests/test_pub_correctness.py

echo ""
echo "Step 2/4: Benchmarks..."
python3 benchmarks/pub_benchmark.py $MODE

echo ""
echo "Step 3/4: Original benchmark suite (for comparison with prior results)..."
python3 benchmarks/compare_indexes.py || true

echo ""
echo "Step 4/4: Generate figures and tables..."
python3 benchmarks/generate_figures.py

echo ""
echo "=== Complete! ==="
echo "Results in: publication/RAW_RESULTS/"
echo "Figures in: publication/PAPER_FIGURES/"
echo "Tables in:  publication/PAPER_TABLES/"
