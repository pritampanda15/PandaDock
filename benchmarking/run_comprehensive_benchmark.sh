#!/bin/bash
################################################################################
# Comprehensive PandaDock Benchmarking Workflow for Publication
#
# This script runs the complete benchmarking pipeline:
# 1. Ligand Preparation (fix kekulization, add hydrogens, minimize)
# 2. Comprehensive Docking (150 complexes × 8 algorithms = 1,200 runs)
# 3. Statistical Analysis & Publication Figures
#
# Usage:
#   # Quick test (10 complexes)
#   ./run_comprehensive_benchmark.sh --test
#
#   # Full benchmark (150 complexes)
#   ./run_comprehensive_benchmark.sh --full
#
#   # Resume from previous run
#   ./run_comprehensive_benchmark.sh --resume
#
# Author: PandaDock Benchmarking Suite
# Date: December 2025
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PANDADOCK_ROOT="/mnt/cce9630f-8b3b-4312-932d-ff04311ba514/SSD/PandaDock"
BENCHMARKING_DIR="${PANDADOCK_ROOT}/benchmarking"
PDBBIND_DIR="${BENCHMARKING_DIR}/full_benchmark/PDBbind_v2020"
GRID_CENTERS="${BENCHMARKING_DIR}/full_benchmark/grid_centers.csv"

# Parse arguments
MODE="full"
MAX_COMPLEXES=150

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MODE="test"
            MAX_COMPLEXES=10
            shift
            ;;
        --full)
            MODE="full"
            MAX_COMPLEXES=150
            shift
            ;;
        --resume)
            MODE="resume"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test|--full|--resume]"
            exit 1
            ;;
    esac
done

# Set output directory based on mode
if [ "$MODE" == "test" ]; then
    OUTPUT_DIR="${BENCHMARKING_DIR}/test_benchmark_${MAX_COMPLEXES}"
    PREPARED_DIR="${BENCHMARKING_DIR}/ligands_prepared_${MAX_COMPLEXES}"
else
    OUTPUT_DIR="${BENCHMARKING_DIR}/complete_benchmark_comprehensive_150"
    PREPARED_DIR="${BENCHMARKING_DIR}/ligands_prepared_150"
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PandaDock Comprehensive Benchmarking Pipeline - Publication Ready ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Mode:${NC} $MODE"
echo -e "${GREEN}Complexes:${NC} $MAX_COMPLEXES"
echo -e "${GREEN}Output directory:${NC} $OUTPUT_DIR"
echo ""

# Change to PandaDock root
cd "$PANDADOCK_ROOT"

################################################################################
# STEP 1: LIGAND PREPARATION
################################################################################

if [ "$MODE" != "resume" ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}STEP 1/3: Ligand Preparation${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Preparing ligands with proper sanitization, hydrogens, and minimization..."
    echo "This fixes kekulization errors and ensures all ligands are docking-ready."
    echo ""

    python3 benchmarking/prepare_ligands_comprehensive.py \
        --pdbbind-dir "$PDBBIND_DIR" \
        --output-dir "$PREPARED_DIR" \
        --grid-centers "$GRID_CENTERS" \
        --max-ligands "$MAX_COMPLEXES"

    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Ligand preparation failed!${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Ligand preparation completed!${NC}"
    echo ""
fi

################################################################################
# STEP 2: COMPREHENSIVE DOCKING BENCHMARK
################################################################################

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 2/3: Comprehensive Docking Benchmark${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Running $MAX_COMPLEXES complexes × 8 algorithms = $((MAX_COMPLEXES * 8)) docking runs"
echo "Timeout: 600 seconds per docking"
echo ""

# Define algorithms to test
ALGORITHMS="monte_carlo_cpu genetic_algorithm_cpu hierarchical_cpu enhanced_hierarchical_cpu crystal_guided_cpu cuda_monte_carlo cuda_genetic_algorithm enhanced_hierarchical_gpu"

nohup python3 benchmarking/run_pdbbind_benchmark.py \
    --pdbbind-dir "$PREPARED_DIR" \
    --grid-centers "$GRID_CENTERS" \
    --output-dir "$OUTPUT_DIR" \
    --max-complexes "$MAX_COMPLEXES" \
    --timeout 600 \
    --algorithms $ALGORITHMS \
    > "${OUTPUT_DIR}_benchmark.log" 2>&1 &

BENCH_PID=$!
echo -e "${GREEN}Benchmark started with PID: $BENCH_PID${NC}"
echo "Log file: ${OUTPUT_DIR}_benchmark.log"
echo ""
echo "Monitoring progress (Ctrl+C to stop monitoring, benchmark will continue)..."
echo ""

# Monitor progress
tail -f "${OUTPUT_DIR}_benchmark.log" &
TAIL_PID=$!

# Wait for benchmark to complete
wait $BENCH_PID
BENCH_EXIT_CODE=$?

# Stop tail
kill $TAIL_PID 2>/dev/null || true

if [ $BENCH_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}✗ Benchmark failed with exit code $BENCH_EXIT_CODE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docking benchmark completed!${NC}"
echo ""

################################################################################
# STEP 3: STATISTICAL ANALYSIS & PUBLICATION FIGURES
################################################################################

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}STEP 3/3: Statistical Analysis & Publication Figures${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Generating publication-quality figures and statistical analysis..."
echo ""

python3 benchmarking/analyze_pdbbind_results.py \
    --results "${OUTPUT_DIR}/benchmark_results.csv" \
    --output "${OUTPUT_DIR}/analysis"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Analysis failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Analysis completed!${NC}"
echo ""

################################################################################
# SUMMARY
################################################################################

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    BENCHMARKING COMPLETE! 🎉                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}All results are ready for publication:${NC}"
echo ""
echo "📁 Results directory: $OUTPUT_DIR"
echo ""
echo "📊 Data files:"
echo "   - benchmark_results.csv (all 1,200 runs)"
echo "   - algorithm_summary.csv (summary statistics)"
echo ""
echo "📈 Publication figures:"
echo "   - analysis/figures/fig1_success_rates.png/pdf"
echo "   - analysis/figures/fig2_rmsd_distribution.png/pdf"
echo "   - analysis/figures/fig3_rmsd_boxplot.png/pdf"
echo "   - analysis/figures/fig4_runtime_comparison.png/pdf"
echo "   - analysis/figures/fig5_success_vs_runtime.png/pdf"
echo ""
echo "📝 Documentation:"
echo "   - PUBLICATION_SUMMARY.md (comprehensive analysis)"
echo "   - RESULTS_FOR_PUBLICATION.md (ready-to-use text)"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Review figures in: ${OUTPUT_DIR}/analysis/figures/"
echo "2. Check statistics in: ${OUTPUT_DIR}/analysis/tables/"
echo "3. Use RESULTS_FOR_PUBLICATION.md for your manuscript"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
