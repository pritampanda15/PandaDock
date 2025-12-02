#!/bin/bash

# PandaDock Full Accuracy Algorithm Testing Script
# Tests all CPU algorithms with all scoring functions (without --fast mode)
# This will take longer but provide more accurate results
# Author: Claude Code Assistant

echo "========================================================================"
echo "PandaDock FULL ACCURACY Algorithm & Scoring Function Testing"
echo "========================================================================"
echo "Testing all combinations without --fast mode (higher accuracy, longer runtime)"
echo "Base parameters: etomidate ligand with similarity-based grid"
echo "========================================================================"

# Define arrays of algorithms and scoring functions
CPU_ALGORITHMS=("monte_carlo_cpu" "genetic_algorithm_cpu" "hierarchical_cpu" "enhanced_hierarchical_cpu")
SCORING_FUNCTIONS=("physics_based" "empirical" "precision_score" "hybrid")

# Base command parameters
RECEPTOR="preprocessing/intermediate/receptor_processed.pdb"
LIGAND="preprocessing/intermediate/ligand_processed.sdf"
GRID_CONFIG="gridbox_output/gridbox_similarity.json"

# Create results directory
RESULTS_DIR="algorithm_comparison_full"
mkdir -p "$RESULTS_DIR"

# Create summary log
SUMMARY_LOG="$RESULTS_DIR/summary_full.log"
echo "PandaDock FULL ACCURACY Algorithm Comparison Results" > "$SUMMARY_LOG"
echo "====================================================" >> "$SUMMARY_LOG"
echo "Started at: $(date)" >> "$SUMMARY_LOG"
echo "Mode: Full accuracy (no --fast flag)" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

# Function to run docking and collect results
run_docking_full() {
    local algorithm=$1
    local scoring=$2
    local output_dir="$RESULTS_DIR/${algorithm}_${scoring}_full"

    echo ""
    echo "========================================================================"
    echo "FULL MODE Testing: $algorithm with $scoring scoring"
    echo "Output directory: $output_dir"
    echo "========================================================================"

    # Record start time
    start_time=$(date +%s)

    # Run docking (NO --fast flag for full accuracy)
    pandadock dock \
        -r "$RECEPTOR" \
        -l "$LIGAND" \
        --grid-config "$GRID_CONFIG" \
        --algorithm "$algorithm" \
        --scoring "$scoring" \
        --num-poses 20 \
        -o "$output_dir" > "$output_dir.log" 2>&1

    # Record end time and calculate duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Extract key results
    if [ -f "$output_dir.log" ]; then
        # Get the best energy (from Top 5 poses section)
        best_energy=$(grep -A 5 "Top.*poses:" "$output_dir.log" | grep "Energy:" | head -1 | sed 's/.*Energy: \([^,]*\).*/\1/')
        # Get number of poses generated
        num_poses=$(grep "Generated.*poses" "$output_dir.log" | tail -1 | sed 's/.*Generated \([0-9]*\) poses.*/\1/')
        # Get ensemble binding energy
        ensemble_energy=$(grep "Ensemble binding energy:" "$output_dir.log" | sed 's/.*Ensemble binding energy: \([^ ]*\).*/\1/')
        # Get runtime
        runtime=$(grep "Docking completed in" "$output_dir.log" | sed 's/.*completed in \([^ ]*\) seconds.*/\1/')

        # Check if interaction analysis worked
        if [ -f "$output_dir/interaction_analysis.json" ]; then
            interactions=$(grep "total_interactions" "$output_dir/interaction_analysis.json" | awk '{print $2}' | tr -d ',')
        else
            interactions="N/A"
        fi

        # Write to summary
        printf "%-25s %-15s %8s %6s %12s %8s %12s %8s\n" \
            "$algorithm" "$scoring" "$best_energy" "$num_poses" "$ensemble_energy" "${runtime}s" "$interactions" "${duration}s" >> "$SUMMARY_LOG"

        echo "✓ Completed: $algorithm + $scoring (${duration}s)"
        echo "  Best Energy: $best_energy kcal/mol"
        echo "  Poses Generated: $num_poses"
        echo "  Ensemble ΔG: $ensemble_energy kcal/mol"
        echo "  Interactions: $interactions"
        echo "  Runtime: ${runtime}s"
    else
        printf "%-25s %-15s %8s %6s %12s %8s %12s %8s\n" \
            "$algorithm" "$scoring" "FAILED" "0" "N/A" "N/A" "N/A" "${duration}s" >> "$SUMMARY_LOG"
        echo "✗ Failed: $algorithm + $scoring"
    fi
}

# Write summary header
printf "\n%-25s %-15s %8s %6s %12s %8s %12s %8s\n" \
    "Algorithm" "Scoring" "BestE" "Poses" "EnsembleΔG" "Runtime" "Contacts" "WallTime" >> "$SUMMARY_LOG"
echo "=========================================================================================" >> "$SUMMARY_LOG"

# Test all combinations
total_tests=$((${#CPU_ALGORITHMS[@]} * ${#SCORING_FUNCTIONS[@]}))
current_test=0

echo "⚠️  WARNING: Full mode testing will take significantly longer!"
echo "   Each test may take 1-10 minutes depending on the algorithm"
echo "   Total estimated time: 15-60 minutes for all combinations"
echo ""
read -p "Continue with full accuracy testing? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing cancelled. Use test_all_algorithms.sh for fast testing."
    exit 1
fi

for algorithm in "${CPU_ALGORITHMS[@]}"; do
    for scoring in "${SCORING_FUNCTIONS[@]}"; do
        current_test=$((current_test + 1))
        echo "Progress: $current_test/$total_tests"
        run_docking_full "$algorithm" "$scoring"

        # Add a small delay to prevent system overload
        sleep 5
    done
done

# Generate analysis summary
echo "" >> "$SUMMARY_LOG"
echo "Analysis Summary (Full Mode):" >> "$SUMMARY_LOG"
echo "=============================" >> "$SUMMARY_LOG"
echo "Completed at: $(date)" >> "$SUMMARY_LOG"

echo ""
echo "========================================================================"
echo "FULL ACCURACY TESTING COMPLETE"
echo "========================================================================"

echo "Results saved in: $RESULTS_DIR/"
echo "Summary log: $SUMMARY_LOG"

# Show summary table
echo ""
echo "Summary of all FULL MODE runs:"
cat "$SUMMARY_LOG" | tail -n +9

echo ""
echo "Full mode provides:"
echo "- More conformer generation"
echo "- Extensive pose sampling"
echo "- Complete refinement cycles"
echo "- Better energy optimization"
echo "- Higher accuracy results"

echo ""
echo "Compare with fast mode results in algorithm_comparison_results/"
echo "========================================================================"