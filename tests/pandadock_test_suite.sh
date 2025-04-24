#!/bin/bash

# Create a log directory
mkdir -p docking_logs

# Define common parameters
PROTEIN="receptor.pdb"
LIGAND="ligand.sdf"
SITE_X=-15.7
SITE_Y=-17.7
SITE_Z=8.1
OUTPUT_DIR="docking_results"
LOG_DIR="docking_logs"

# Function to run docking with logging
run_docking() {
  DESC=$1
  shift
  echo ">>> Running: $DESC"
  pandadock "$@" > "$LOG_DIR/${DESC// /_}.log" 2>&1
}

# 1. Physics-Based Docking (GPU)
run_docking "physics_gpu" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 10 -a genetic --use-gpu --physics-based --gpu-id 0 --detailed-energy

# 2. Physics-Based Docking (CPU)
run_docking "physics_cpu" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 10 -a genetic --physics-based --detailed-energy

# 3. Enhanced Scoring (GPU)
run_docking "enhanced_gpu" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 10 -a genetic --use-gpu --enhanced-scoring --gpu-id 0 --detailed-energy

# 4. Enhanced Scoring (CPU)
run_docking "enhanced_cpu" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 10 -a genetic --enhanced-scoring --detailed-energy

# 5. Fast Mode Docking
run_docking "fast_mode" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 5 -a random --fast-mode

# 6. Monte Carlo Sampling (GPU)
run_docking "montecarlo_gpu" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z --monte-carlo --mc-steps 1500 --temperature 400 --use-gpu --gpu-id 0 --physics-based --detailed-energy

# 7. Auto Algorithm Selection (GPU)
run_docking "auto_algorithm" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z --auto-algorithm --use-gpu --gpu-id 0 --physics-based --detailed-energy

# 8. Full Report + Per-Residue Energy
run_docking "full_analysis" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 10 -a genetic --use-gpu --physics-based --gpu-id 0 \
  --energy-decomposition --per-residue-energy --generate-analysis-report --analysis-report-format html --detailed-energy

# ✅ Test 9: PandaDock algorithm (CHARMm + Simulated Annealing)
run_docking "Pandadock_algorithm" -p receptor.pdb -l ligand.sdf -s -15.7 -17.7 8.1 -a pandadock --use-gpu --physics-based \
  --num-conformers 5 --num-orientations 5 --md-steps 1000 --minimize-steps 200 --detailed-energy \
  --report-format all 