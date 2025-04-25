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

run_docking "genetic_gpu" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 1 -a genetic --use-gpu --gpu-id 0 --detailed-energy
run_docking "genetic_cpu" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 1 -a genetic  --detailed-energy
run_docking "physics_gpu" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 1 -a genetic --use-gpu --physics-based --gpu-id 0 --detailed-energy
run_docking "physics_cpu" -p $PROTEIN -l $LIGAND -s $SITE_X $SITE_Y $SITE_Z -i 1 -a genetic  --physics-based --detailed-energy
