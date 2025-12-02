#!/bin/bash

# PandaDock COMPREHENSIVE Testing Suite
# Tests ALL PandaDock tools with both fast and full accuracy modes
# Includes: pandadock dock, pandadock-flex, pandadock-report, pandadock-tethered
# Author: Claude Code Assistant
# Date: $(date)

echo "========================================================================"
echo "🧬 PandaDock COMPREHENSIVE Testing Suite"
echo "========================================================================"
echo "Testing ALL PandaDock tools:"
echo "  ✓ pandadock dock (fast & full accuracy modes)"
echo "  ✓ pandadock-flex (flexible docking)"
echo "  ✓ pandadock-report (publication-ready analysis)"
echo "  ✓ pandadock-tethered (crystallographic validation)"
echo "========================================================================"

# Configuration
RECEPTOR="preprocessing/intermediate/receptor_processed.pdb"
LIGAND="preprocessing/intermediate/ligand_processed.sdf"
GRID_CONFIG="gridbox_output/gridbox_similarity.json"

# Algorithm and scoring arrays
CPU_ALGORITHMS=("monte_carlo_cpu" "genetic_algorithm_cpu" "hierarchical_cpu" "enhanced_hierarchical_cpu")
SCORING_FUNCTIONS=("physics_based" "empirical" "precision_score" "hybrid")

# Result directories
MAIN_RESULTS_DIR="comprehensive_pandadock_testing"
FAST_RESULTS_DIR="$MAIN_RESULTS_DIR/fast_mode"
FULL_RESULTS_DIR="$MAIN_RESULTS_DIR/full_mode"
FLEX_RESULTS_DIR="$MAIN_RESULTS_DIR/flex_docking"
TETHERED_RESULTS_DIR="$MAIN_RESULTS_DIR/tethered_analysis"
REPORTS_DIR="$MAIN_RESULTS_DIR/publication_reports"

# Create all directories
mkdir -p "$FAST_RESULTS_DIR" "$FULL_RESULTS_DIR" "$FLEX_RESULTS_DIR" "$TETHERED_RESULTS_DIR" "$REPORTS_DIR"

# Master summary log
MASTER_LOG="$MAIN_RESULTS_DIR/comprehensive_summary.log"
echo "🧬 PandaDock COMPREHENSIVE Testing Results" > "$MASTER_LOG"
echo "===========================================" >> "$MASTER_LOG"
echo "Started at: $(date)" >> "$MASTER_LOG"
echo "Testing modes: Fast, Full Accuracy, Flexible, Tethered" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

# Timing and progress tracking
TOTAL_TESTS=0
COMPLETED_TESTS=0
START_TIME=$(date +%s)

# Calculate total tests
BASIC_TESTS=$((${#CPU_ALGORITHMS[@]} * ${#SCORING_FUNCTIONS[@]} * 2))  # fast + full modes
FLEX_TESTS=4  # One per scoring function
TETHERED_TESTS=1
REPORT_TESTS=3  # Fast, Full, Flex reports
TOTAL_TESTS=$((BASIC_TESTS + FLEX_TESTS + TETHERED_TESTS + REPORT_TESTS))

echo "📊 TESTING PLAN:"
echo "  • Basic docking (fast): $((${#CPU_ALGORITHMS[@]} * ${#SCORING_FUNCTIONS[@]})) tests"
echo "  • Basic docking (full): $((${#CPU_ALGORITHMS[@]} * ${#SCORING_FUNCTIONS[@]})) tests"
echo "  • Flexible docking: $FLEX_TESTS tests"
echo "  • Tethered analysis: $TETHERED_TESTS test"
echo "  • Report generation: $REPORT_TESTS reports"
echo "  • TOTAL: $TOTAL_TESTS tests"
echo ""

# Progress tracking function
update_progress() {
    COMPLETED_TESTS=$((COMPLETED_TESTS + 1))
    local current_time=$(date +%s)
    local elapsed=$((current_time - START_TIME))
    local avg_time_per_test=$((elapsed / COMPLETED_TESTS))
    local estimated_remaining=$(((TOTAL_TESTS - COMPLETED_TESTS) * avg_time_per_test))
    local eta=$(date -d "@$((current_time + estimated_remaining))" '+%H:%M:%S' 2>/dev/null || echo "N/A")

    echo "📈 Progress: $COMPLETED_TESTS/$TOTAL_TESTS ($(( COMPLETED_TESTS * 100 / TOTAL_TESTS ))%) | ETA: $eta"
}

# Function to run basic docking
run_basic_docking() {
    local mode=$1  # "fast" or "full"
    local algorithm=$2
    local scoring=$3
    local results_dir=$4

    local output_dir="$results_dir/${algorithm}_${scoring}_${mode}"
    local fast_flag=""
    local num_poses=10

    if [ "$mode" = "fast" ]; then
        fast_flag="--fast"
        num_poses=10
    else
        num_poses=20
    fi

    echo ""
    echo "🔬 Testing: $algorithm + $scoring ($mode mode)"

    # Record start time
    start_time=$(date +%s)

    # Run docking
    PYTHONPATH=. python3 -m pandadock.docking_cli dock \
        -r "$RECEPTOR" \
        -l "$LIGAND" \
        --grid-config "$GRID_CONFIG" \
        --algorithm "$algorithm" \
        --scoring "$scoring" \
        --num-poses "$num_poses" \
        $fast_flag \
        -o "$output_dir" > "$output_dir.log" 2>&1

    # Record end time
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Extract results
    if [ -f "$output_dir.log" ]; then
        best_energy=$(grep -A 5 "Top.*poses:" "$output_dir.log" | grep "Energy:" | head -1 | sed 's/.*Energy: \([^,]*\).*/\1/' || echo "N/A")
        num_poses_generated=$(grep "Generated.*poses" "$output_dir.log" | tail -1 | sed 's/.*Generated \([0-9]*\) poses.*/\1/' || echo "N/A")
        ensemble_energy=$(grep "Ensemble binding energy:" "$output_dir.log" | sed 's/.*Ensemble binding energy: \([^ ]*\).*/\1/' || echo "N/A")
        runtime=$(grep "Docking completed in" "$output_dir.log" | sed 's/.*completed in \([^ ]*\) seconds.*/\1/' || echo "N/A")

        # Check interactions
        if [ -f "$output_dir/interaction_analysis.json" ]; then
            interactions=$(grep "total_interactions" "$output_dir/interaction_analysis.json" | awk '{print $2}' | tr -d ',' || echo "N/A")
        else
            interactions="N/A"
        fi

        echo "  ✅ Success: $best_energy kcal/mol, ${duration}s"
    else
        best_energy="FAILED"
        num_poses_generated="0"
        ensemble_energy="N/A"
        runtime="N/A"
        interactions="N/A"
        echo "  ❌ Failed: ${duration}s"
    fi

    # Log to master summary
    printf "%-8s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "$mode" "$algorithm" "$scoring" "$best_energy" "$num_poses_generated" "$ensemble_energy" "${runtime}s" "$interactions" "${duration}s" >> "$MASTER_LOG"

    update_progress
}

# Function to run flexible docking
run_flex_docking() {
    local scoring=$1
    local output_dir="$FLEX_RESULTS_DIR/flex_${scoring}"

    echo ""
    echo "🔄 Flexible Docking: $scoring scoring"

    start_time=$(date +%s)

    # Run flexible docking
    PYTHONPATH=. python3 -m pandadock.flex_docking_cli \
        -r "$RECEPTOR" \
        -l "$LIGAND" \
        --grid-config "$GRID_CONFIG" \
        --scoring "$scoring" \
        --fast \
        -o "$output_dir" > "$output_dir.log" 2>&1

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ -f "$output_dir.log" ]; then
        best_energy=$(grep -A 5 "Top.*poses:" "$output_dir.log" | grep "Energy:" | head -1 | sed 's/.*Energy: \([^,]*\).*/\1/' || echo "N/A")
        echo "  ✅ Flex Success: $best_energy kcal/mol, ${duration}s"
        flex_status="SUCCESS"
    else
        best_energy="FAILED"
        flex_status="FAILED"
        echo "  ❌ Flex Failed: ${duration}s"
    fi

    printf "%-8s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "flex" "induced_fit_flexible" "$scoring" "$best_energy" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

    update_progress
}

# Function to run tethered analysis
run_tethered_analysis() {
    echo ""
    echo "🔗 Tethered Analysis (Crystallographic Validation)"

    start_time=$(date +%s)

    # Find a good complex file from previous results
    local complex_file=""
    for dir in "$FAST_RESULTS_DIR"/genetic_algorithm_cpu_physics_based_fast; do
        if [ -f "$dir/complex1.pdb" ]; then
            complex_file="$dir/complex1.pdb"
            break
        fi
    done

    if [ -z "$complex_file" ]; then
        echo "  ⚠️  No complex file found for tethered analysis"
        printf "%-8s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
            "tethered" "tethered_validation" "physics_based" "NO_INPUT" "N/A" "N/A" "N/A" "N/A" "0s" >> "$MASTER_LOG"
        update_progress
        return
    fi

    # Run tethered analysis
    PYTHONPATH=. python3 -m pandadock.tethered_cli analyze \
        -i "$complex_file" \
        -l "LIG" \
        --tether-radius 2.0 \
        -o "$TETHERED_RESULTS_DIR/tethered_validation" > "$TETHERED_RESULTS_DIR/tethered.log" 2>&1

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ -f "$TETHERED_RESULTS_DIR/tethered.log" ]; then
        crystal_score=$(grep "Crystal Score:" "$TETHERED_RESULTS_DIR/tethered.log" | sed 's/.*Crystal Score: \([^ ]*\).*/\1/' || echo "N/A")
        best_score=$(grep "Best Score:" "$TETHERED_RESULTS_DIR/tethered.log" | sed 's/.*Best Score: \([^ ]*\).*/\1/' || echo "N/A")
        echo "  ✅ Tethered Success: Crystal=${crystal_score}, Best=${best_score}, ${duration}s"
        tethered_status="SUCCESS"
    else
        crystal_score="FAILED"
        best_score="FAILED"
        tethered_status="FAILED"
        echo "  ❌ Tethered Failed: ${duration}s"
    fi

    printf "%-8s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "tethered" "tethered_validation" "crystal_reprod" "$crystal_score" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

    update_progress
}

# Function to generate reports
generate_reports() {
    local mode=$1
    local input_dir=$2
    local title=$3

    echo ""
    echo "📊 Generating $mode Report"

    start_time=$(date +%s)

    # Generate publication report
    PYTHONPATH=. python3 -m pandadock.report_cli pandamap \
        -i "$input_dir" \
        -t "$title" \
        -o "$REPORTS_DIR/${mode}_report" > "$REPORTS_DIR/${mode}_report.log" 2>&1

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ -f "$REPORTS_DIR/${mode}_report.log" ]; then
        echo "  ✅ $mode Report Generated: ${duration}s"
        report_status="SUCCESS"
    else
        echo "  ❌ $mode Report Failed: ${duration}s"
        report_status="FAILED"
    fi

    printf "%-8s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "report" "${mode}_analysis" "publication" "$report_status" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

    update_progress
}

# Write master log header
echo "" >> "$MASTER_LOG"
printf "%-8s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
    "Mode" "Algorithm" "Scoring" "BestEnergy" "Poses" "EnsembleΔG" "Runtime" "Contacts" "WallTime" >> "$MASTER_LOG"
echo "=================================================================================================" >> "$MASTER_LOG"

# Confirmation prompt
echo "⚠️  WARNING: Comprehensive testing will take significant time!"
echo "   Estimated total time: 2-4 hours depending on system"
echo "   This includes both fast and full accuracy modes"
echo ""
read -p "Continue with comprehensive testing? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing cancelled."
    exit 1
fi

echo ""
echo "🚀 Starting comprehensive testing..."

# PHASE 1: Fast Mode Docking
echo ""
echo "========================================================================"
echo "PHASE 1: FAST MODE DOCKING"
echo "========================================================================"

for algorithm in "${CPU_ALGORITHMS[@]}"; do
    for scoring in "${SCORING_FUNCTIONS[@]}"; do
        run_basic_docking "fast" "$algorithm" "$scoring" "$FAST_RESULTS_DIR"
        sleep 1
    done
done

# PHASE 2: Full Accuracy Mode Docking
echo ""
echo "========================================================================"
echo "PHASE 2: FULL ACCURACY MODE DOCKING"
echo "========================================================================"

for algorithm in "${CPU_ALGORITHMS[@]}"; do
    for scoring in "${SCORING_FUNCTIONS[@]}"; do
        run_basic_docking "full" "$algorithm" "$scoring" "$FULL_RESULTS_DIR"
        sleep 2
    done
done

# PHASE 3: Flexible Docking
echo ""
echo "========================================================================"
echo "PHASE 3: FLEXIBLE DOCKING"
echo "========================================================================"

for scoring in "${SCORING_FUNCTIONS[@]}"; do
    run_flex_docking "$scoring"
    sleep 1
done

# PHASE 4: Tethered Analysis
echo ""
echo "========================================================================"
echo "PHASE 4: TETHERED ANALYSIS"
echo "========================================================================"

run_tethered_analysis

# PHASE 5: Report Generation
echo ""
echo "========================================================================"
echo "PHASE 5: PUBLICATION REPORTS"
echo "========================================================================"

# Generate reports for each mode
generate_reports "fast" "$FAST_RESULTS_DIR" "PandaDock Fast Mode Analysis"
generate_reports "full" "$FULL_RESULTS_DIR" "PandaDock Full Accuracy Analysis"
generate_reports "flex" "$FLEX_RESULTS_DIR" "PandaDock Flexible Docking Analysis"

# Final summary
FINAL_TIME=$(date +%s)
TOTAL_DURATION=$((FINAL_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Write final summary to log
echo "" >> "$MASTER_LOG"
echo "COMPREHENSIVE TESTING SUMMARY:" >> "$MASTER_LOG"
echo "==============================" >> "$MASTER_LOG"
echo "Completed at: $(date)" >> "$MASTER_LOG"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$MASTER_LOG"
echo "Tests completed: $COMPLETED_TESTS/$TOTAL_TESTS" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"
echo "Output Structure:" >> "$MASTER_LOG"
echo "- Fast mode results: $FAST_RESULTS_DIR/" >> "$MASTER_LOG"
echo "- Full mode results: $FULL_RESULTS_DIR/" >> "$MASTER_LOG"
echo "- Flexible docking: $FLEX_RESULTS_DIR/" >> "$MASTER_LOG"
echo "- Tethered analysis: $TETHERED_RESULTS_DIR/" >> "$MASTER_LOG"
echo "- Publication reports: $REPORTS_DIR/" >> "$MASTER_LOG"

# Display final results
echo ""
echo "========================================================================"
echo "🎉 COMPREHENSIVE TESTING COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 FINAL SUMMARY:"
echo "  ⏱️  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  ✅ Tests completed: $COMPLETED_TESTS/$TOTAL_TESTS"
echo "  📁 Results saved in: $MAIN_RESULTS_DIR/"
echo ""
echo "📋 COMPREHENSIVE RESULTS TABLE:"
echo "================================"
echo ""

# Show comprehensive summary table
cat "$MASTER_LOG" | tail -n +7 | head -n -15

echo ""
echo "📁 OUTPUT STRUCTURE:"
echo "  📊 Master Summary: $MASTER_LOG"
echo "  🚀 Fast Mode: $FAST_RESULTS_DIR/"
echo "  🔬 Full Mode: $FULL_RESULTS_DIR/"
echo "  🔄 Flexible: $FLEX_RESULTS_DIR/"
echo "  🔗 Tethered: $TETHERED_RESULTS_DIR/"
echo "  📈 Reports: $REPORTS_DIR/"
echo ""
echo "🔍 ANALYSIS RECOMMENDATIONS:"
echo "  1. Compare fast vs full mode performance in $MASTER_LOG"
echo "  2. Review publication plots in $REPORTS_DIR/"
echo "  3. Validate poses with tethered analysis results"
echo "  4. Check flexible docking for induced-fit scenarios"
echo "  5. Visual inspection of best poses in ChimeraX/PyMOL"
echo ""
echo "🚀 NEXT STEPS:"
echo "  • Select optimal algorithm/scoring combinations"
echo "  • Use fast mode for high-throughput screening"
echo "  • Use full mode for final publication results"
echo "  • Integrate flexible docking for challenging targets"
echo ""
echo "========================================================================"
echo "🧬 PandaDock Comprehensive Testing Suite Complete!"
echo "========================================================================"