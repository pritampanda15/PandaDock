#!/bin/bash

# PandaDock Comprehensive Algorithm Testing Script
# Tests all CPU algorithms with all scoring functions
# Author: Claude Code Assistant
# Date: $(date)

echo "========================================================================"
echo "PandaDock Comprehensive Algorithm & Scoring Function Testing"
echo "========================================================================"
echo "Testing all combinations of CPU algorithms and scoring functions"
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
RESULTS_DIR="algorithm_comparison_results"
mkdir -p "$RESULTS_DIR"

# Create summary log
SUMMARY_LOG="$RESULTS_DIR/summary.log"
echo "PandaDock Algorithm Comparison Results" > "$SUMMARY_LOG"
echo "=======================================" >> "$SUMMARY_LOG"
echo "Started at: $(date)" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

# Function to run docking and collect results
run_docking() {
    local algorithm=$1
    local scoring=$2
    local output_dir="$RESULTS_DIR/${algorithm}_${scoring}"

    echo ""
    echo "========================================================================"
    echo "Testing: $algorithm with $scoring scoring"
    echo "Output directory: $output_dir"
    echo "========================================================================"

    # Record start time
    start_time=$(date +%s)

    # Run docking
    pandadock-dock dock \
        -r "$RECEPTOR" \
        -l "$LIGAND" \
        --grid-config "$GRID_CONFIG" \
        --algorithm "$algorithm" \
        --scoring "$scoring" \
        --num-poses 10 \
        --fast \
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

for algorithm in "${CPU_ALGORITHMS[@]}"; do
    for scoring in "${SCORING_FUNCTIONS[@]}"; do
        current_test=$((current_test + 1))
        echo "Progress: $current_test/$total_tests"
        run_docking "$algorithm" "$scoring"

        # Add a small delay to prevent system overload
        sleep 2
    done
done

# Generate analysis summary
echo "" >> "$SUMMARY_LOG"
echo "Analysis Summary:" >> "$SUMMARY_LOG"
echo "=================" >> "$SUMMARY_LOG"
echo "Completed at: $(date)" >> "$SUMMARY_LOG"

# Create a simple results analysis
echo ""
echo "========================================================================"
echo "TESTING COMPLETE - ANALYSIS"
echo "========================================================================"

echo "Results saved in: $RESULTS_DIR/"
echo "Summary log: $SUMMARY_LOG"
echo ""
echo "Quick Analysis:"
echo "---------------"

# Show summary table
echo "Summary of all runs:"
cat "$SUMMARY_LOG" | tail -n +8

# Find best performing combinations
echo ""
echo "Analysis Notes:"
echo "- Lower energies are generally better for binding"
echo "- More interactions indicate better binding site engagement"
echo "- Check individual result directories for detailed analysis"
echo "- Look at pose PDB files to visually inspect binding modes"

echo ""
echo "Individual result directories contain:"
echo "- Pose PDB files (pose1.pdb, pose2.pdb, etc.)"
echo "- Complex PDB files (complex1.pdb, complex2.pdb, etc.)"
echo "- Interaction analysis JSON"
echo "- Binding affinity plots"
echo "- Detailed logs"

echo ""
echo "Recommended next steps:"
echo "1. Check '$SUMMARY_LOG' for overview"
echo "2. Visual inspection of poses in PyMOL/ChimeraX"
echo "3. Compare with crystal structure"
echo "4. Select best algorithm/scoring combination"

echo ""
echo "========================================================================"
echo "Testing completed successfully!"
echo "========================================================================"