#!/bin/bash
#
# Quick Test Script for PDBbind Benchmarking
#
# This script runs a quick test with 5 complexes and 2 algorithms
# to verify that everything is set up correctly.
#
# Expected runtime: 5-10 minutes
#

set -e  # Exit on error

echo "======================================================================"
echo "PandaDock PDBbind Benchmarking - Quick Test"
echo "======================================================================"
echo ""
echo "This will test the benchmarking pipeline with:"
echo "  - 5 complexes from PDBbind v2020"
echo "  - 2 algorithms (hierarchical_cpu, cuda_genetic_algorithm)"
echo "  - Expected runtime: 5-10 minutes"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

OUTPUT_DIR="benchmarking/quick_test_results_$(date +%Y%m%d_%H%M%S)"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run benchmark
echo "======================================================================"
echo "Step 1: Running Docking Benchmark"
echo "======================================================================"
python benchmarking/run_pdbbind_benchmark.py \
  --max-complexes 5 \
  --algorithms hierarchical_cpu cuda_genetic_algorithm \
  --output-dir "$OUTPUT_DIR" || {
    echo ""
    echo "ERROR: Benchmarking failed!"
    echo "Check if CUDA/GPU is available for cuda_genetic_algorithm"
    echo ""
    echo "Try CPU-only test:"
    echo "  python benchmarking/run_pdbbind_benchmark.py --max-complexes 5 --algorithms hierarchical_cpu"
    exit 1
}

echo ""
echo "======================================================================"
echo "Step 2: Analyzing Results"
echo "======================================================================"
python benchmarking/analyze_pdbbind_results.py \
  --results "$OUTPUT_DIR/benchmark_results.csv" \
  --output "$OUTPUT_DIR/analysis" || {
    echo ""
    echo "ERROR: Analysis failed!"
    echo "Check the benchmark results file"
    exit 1
}

echo ""
echo "======================================================================"
echo "Quick Test SUCCESSFUL!"
echo "======================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check the following:"
echo "  1. Benchmark results: $OUTPUT_DIR/benchmark_results.csv"
echo "  2. Algorithm summary: $OUTPUT_DIR/algorithm_summary.csv"
echo "  3. Figures: $OUTPUT_DIR/analysis/figures/"
echo "  4. Tables: $OUTPUT_DIR/analysis/tables/"
echo ""
echo "Next steps:"
echo "  - Review the results in $OUTPUT_DIR"
echo "  - If successful, run larger benchmark:"
echo "    python benchmarking/run_pdbbind_benchmark.py --max-complexes 100 \\"
echo "      --algorithms hierarchical_cpu enhanced_hierarchical_cpu cuda_genetic_algorithm"
echo ""
echo "======================================================================"
