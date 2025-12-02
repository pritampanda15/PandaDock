#!/bin/bash
# Quick test of benchmarking pipeline
# Tests 1 complex with all 5 CPU algorithms (~5 minutes)

set -e  # Exit on error

echo "==========================================="
echo "PandaDock Benchmarking Quick Test"
echo "==========================================="
echo ""

# Change to project directory
cd "$(dirname "$0")/.."

echo "Step 1: Verify benchmark dataset exists..."
if [ ! -d "benchmarking/simple_benchmark_set" ]; then
    echo "ERROR: Benchmark dataset not found!"
    echo "Run: python benchmarking/prepare_benchmark_simple.py"
    exit 1
fi

COMPLEXES=$(wc -l < benchmarking/simple_benchmark_set/benchmark_metadata.csv)
echo "✓ Found $(($COMPLEXES - 1)) complexes"
echo ""

echo "Step 2: List available PandaDock algorithms..."
PYTHONPATH=. python3 -c "
from pandadock.docking_cli import engine
print('CPU Algorithms:')
for name in sorted(engine._algorithms.keys()):
    print(f'  - {name}')
" 2>/dev/null
echo ""

echo "Step 3: Create test with 1 complex (faster)..."
mkdir -p benchmarking/test_quick_set/receptors
mkdir -p benchmarking/test_quick_set/ligands

# Copy just first complex
head -2 benchmarking/simple_benchmark_set/benchmark_metadata.csv > benchmarking/test_quick_set/benchmark_metadata.csv

PDB_ID=$(tail -1 benchmarking/test_quick_set/benchmark_metadata.csv | cut -d',' -f1)
echo "Using complex: $PDB_ID"

cp benchmarking/simple_benchmark_set/receptors/${PDB_ID}_receptor.pdb benchmarking/test_quick_set/receptors/
cp benchmarking/simple_benchmark_set/ligands/${PDB_ID}_ligand.sdf benchmarking/test_quick_set/ligands/
echo ""

echo "Step 4: Run benchmark with ALL 5 CPU algorithms..."
echo "This will test:"
echo "  1. monte_carlo_cpu"
echo "  2. genetic_algorithm_cpu"
echo "  3. hierarchical_cpu"
echo "  4. enhanced_hierarchical_cpu"
echo "  5. crystal_guided_cpu"
echo ""
echo "Estimated time: 5-10 minutes for 1 complex × 5 algorithms"
echo ""

python benchmarking/run_benchmark_comparison.py \
  --benchmark-dir benchmarking/test_quick_set \
  --output-dir benchmarking/test_quick_results

echo ""
echo "Step 5: Check results..."
if [ -f "benchmarking/test_quick_results/benchmark_results.csv" ]; then
    echo "✓ Results file created"
    echo ""
    echo "Results summary:"
    python3 -c "
import pandas as pd
df = pd.read_csv('benchmarking/test_quick_results/benchmark_results.csv')
print(df[['algorithm', 'success', 'runtime']].to_string(index=False))
"
    echo ""
else
    echo "✗ Results file not found!"
    exit 1
fi

echo "==========================================="
echo "✓ Quick test completed successfully!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Run full test (10 complexes):"
echo "     python benchmarking/run_benchmark_comparison.py \\"
echo "       --benchmark-dir benchmarking/simple_benchmark_set \\"
echo "       --output-dir benchmarking/results_simple"
echo ""
echo "  2. Generate figures:"
echo "     python benchmarking/analyze_results.py \\"
echo "       --results benchmarking/results_simple/benchmark_results.csv \\"
echo "       --metadata benchmarking/simple_benchmark_set/benchmark_metadata.csv \\"
echo "       --output-dir benchmarking/analysis_simple"
echo ""
echo "  3. Register for PDBbind (290 complexes for publication):"
echo "     http://www.pdbbind.org.cn/"
echo ""
