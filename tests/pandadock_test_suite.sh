#!/bin/bash

# === PandaDock Automated Testing Script ===
# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Input files (Change if needed)
PROTEIN="beta-2_alpha-1.pdb"
LIGAND="ligand.sdf"
REFERENCE="reference.sdf"  # Only needed for tethered docking

# Basic command
PANDADOCK="pandadock"

# Helper function
run_test() {
    echo -e "${YELLOW}\n=== Running Test: $1 ===${NC}\n"
    echo -e "${BLUE}Command:${NC} $2"
    eval $2
    echo -e "${GREEN}=== Finished: $1 ===${NC}\n"
    sleep 2
}

# ======== TEST CASES ========

# 1. Minimal quick test
run_test "Minimal Docking" "$PANDADOCK -p $PROTEIN -l $LIGAND"

# 2. Define Active Site
run_test "Active Site Defined" "$PANDADOCK -p $PROTEIN -l $LIGAND -s 10 10 10 -r 10"

# 3. Fast Mode
run_test "Fast Mode" "$PANDADOCK -p $PROTEIN -l $LIGAND --fast-mode"

# 4. Physics-Based Docking
run_test "Physics-Based Docking" "$PANDADOCK -p $PROTEIN -l $LIGAND --physics-based --prepare-molecules"

# 5. Genetic Algorithm + Local Opt + Enhanced Scoring
run_test "Genetic + Local Optimization + Enhanced" "$PANDADOCK -p $PROTEIN -l $LIGAND -a genetic --local-opt --enhanced-scoring"

# 6. Random Search
run_test "Random Search" "$PANDADOCK -p $PROTEIN -l $LIGAND -a random"

# 7. Monte Carlo Sampling
run_test "Monte Carlo Sampling" "$PANDADOCK -p $PROTEIN -l $LIGAND --monte-carlo --mc-steps 2000 --temperature 500"

# 8. Pandadock Algorithm
run_test "PANDADOCK Algorithm" "$PANDADOCK -p $PROTEIN -l $LIGAND -a pandadock --high-temp 800 --target-temp 300 --md-steps 500 --minimize-steps 100"

# 9. Pocket Detection
run_test "Pocket Detection" "$PANDADOCK -p $PROTEIN -l $LIGAND --detect-pockets"

# 10. GPU Physics-Based Docking
run_test "Physics-Based Docking with GPU" "$PANDADOCK -p $PROTEIN -l $LIGAND --physics-based --use-gpu --gpu-precision float32"

# 11. Auto Flexible Residues
run_test "Auto-Flex Residues" "$PANDADOCK -p $PROTEIN -l $LIGAND --auto-flex"

# 12. Advanced Search - Gradient
run_test "Advanced Gradient Search" "$PANDADOCK -p $PROTEIN -l $LIGAND --advanced-search gradient --gradient-step 0.1 --convergence-threshold 0.01"

# 13. Advanced Search - Replica Exchange
run_test "Advanced Replica Exchange Search" "$PANDADOCK -p $PROTEIN -l $LIGAND --advanced-search replica-exchange --n-replicas 4 --exchange-steps 10"

# 14. ML-Guided Docking
run_test "Advanced ML-Guided Search" "$PANDADOCK -p $PROTEIN -l $LIGAND --advanced-search ml-guided --surrogate-model rf --exploitation-factor 0.7"

# 15. Fragment-Based Docking
run_test "Advanced Fragment-Based Docking" "$PANDADOCK -p $PROTEIN -l $LIGAND --advanced-search fragment-based --fragment-min-size 5 --growth-steps 3"

# 16. Hybrid GA + Local Search
run_test "Hybrid Search (GA + L-BFGS)" "$PANDADOCK -p $PROTEIN -l $LIGAND --advanced-search hybrid --ga-iterations 50 --lbfgs-iterations 50 --top-n-for-local 5"

# 17. Full Reporting - HTML, CSV, JSON
run_test "Generate Full Reports" "$PANDADOCK -p $PROTEIN -l $LIGAND --report-format all --detailed-energy --generate-analysis-report --analysis-report-format html"

# 18. Custom CPU/GPU Workload Balance
run_test "Custom CPU/GPU Workload Balance" "$PANDADOCK -p $PROTEIN -l $LIGAND --use-gpu --workload-balance 0.7"

# 19. Tethered Docking with Reference
run_test "Tethered Docking (Reference Guided)" "$PANDADOCK -p $PROTEIN -l $LIGAND --tethered-docking --reference $REFERENCE --tether-weight 10.0"

# ======== DONE ========
echo -e "${CYAN}\n✅ All PandaDock tests completed!\n${NC}"
