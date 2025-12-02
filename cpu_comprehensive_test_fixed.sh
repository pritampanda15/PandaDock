#!/bin/bash

# PandaDock CPU-ONLY Comprehensive Testing Suite (FIXED VERSION)
# Fixed: Added timeouts to prevent hanging on slow algorithms
# Fixed: Reduced sampling for hierarchical algorithms
# Fixed: Better error handling and progress monitoring
# Author: PandaDock Testing Suite
# Date: $(date)

echo "========================================================================"
echo "🖥️  PandaDock CPU-ONLY Comprehensive Testing Suite (FIXED)"
echo "========================================================================"
echo "Improvements:"
echo "  ✓ Timeouts added (10min fast, 30min full) to prevent hanging"
echo "  ✓ Reduced sampling for slow hierarchical algorithms"
echo "  ✓ Better error handling and process monitoring"
echo "========================================================================"

# Configuration
RECEPTOR="preprocessing/intermediate/receptor_processed.pdb"
LIGAND="preprocessing/intermediate/ligand_processed.sdf"
GRID_CONFIG="gridbox_output/gridbox_similarity.json"

# Auto-detect CPU cores (leave 2 for system)
TOTAL_CORES=$(nproc)
CPU_WORKERS=$((TOTAL_CORES - 2))
if [ "$CPU_WORKERS" -lt 1 ]; then
    CPU_WORKERS=1
fi

echo "Detected $TOTAL_CORES CPU cores, using $CPU_WORKERS for docking"
echo ""

# Timeouts (in seconds)
FAST_MODE_TIMEOUT=600    # 10 minutes for fast mode
FULL_MODE_TIMEOUT=1800   # 30 minutes for full mode
FLEX_TIMEOUT=2400        # 40 minutes for flexible docking

# CPU-specific algorithms and scoring
CPU_ALGORITHMS=("monte_carlo_cpu" "genetic_algorithm_cpu" "hierarchical_cpu" "enhanced_hierarchical_cpu" "crystal_guided_cpu")
CPU_SCORING=("physics_based" "empirical" "precision_score" "hybrid")

# Result directories
MAIN_RESULTS_DIR="cpu_comprehensive_testing_fixed"
CPU_FAST_DIR="$MAIN_RESULTS_DIR/cpu_fast_${CPU_WORKERS}cores"
CPU_FULL_DIR="$MAIN_RESULTS_DIR/cpu_full_${CPU_WORKERS}cores"
FLEX_CPU_DIR="$MAIN_RESULTS_DIR/flex_cpu_${CPU_WORKERS}cores"
REPORTS_DIR="$MAIN_RESULTS_DIR/cpu_publication_reports"

# Create all directories
mkdir -p "$CPU_FAST_DIR" "$CPU_FULL_DIR" "$FLEX_CPU_DIR" "$REPORTS_DIR"

# Master summary log
MASTER_LOG="$MAIN_RESULTS_DIR/cpu_comprehensive_summary.log"
DETAILED_CSV="$MAIN_RESULTS_DIR/cpu_detailed_results.csv"

echo "🖥️  PandaDock CPU-ONLY Comprehensive Testing Results (FIXED)" > "$MASTER_LOG"
echo "==========================================================" >> "$MASTER_LOG"
echo "Started at: $(date)" >> "$MASTER_LOG"
echo "Hardware: $CPU_WORKERS CPU cores" >> "$MASTER_LOG"
echo "Timeouts: Fast=${FAST_MODE_TIMEOUT}s, Full=${FULL_MODE_TIMEOUT}s" >> "$MASTER_LOG"
echo "Testing modes: CPU Fast, CPU Full, Flexible CPU" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

# Create CSV header
echo "Mode,Algorithm,Scoring,BestEnergy,NumPoses,EnsembleΔG,Runtime,Contacts,WallTime,Status" > "$DETAILED_CSV"

# Timing and progress tracking
TOTAL_TESTS=0
COMPLETED_TESTS=0
START_TIME=$(date +%s)

# Calculate total tests
CPU_DOCKING_TESTS=$((${#CPU_ALGORITHMS[@]} * ${#CPU_SCORING[@]} * 2))  # fast + full modes
FLEX_CPU_TESTS=${#CPU_SCORING[@]}
REPORT_TESTS=3  # CPU Fast, CPU Full, Flex CPU
TOTAL_TESTS=$((CPU_DOCKING_TESTS + FLEX_CPU_TESTS + REPORT_TESTS))

echo "📊 CPU TESTING PLAN:"
echo "  • CPU docking ($CPU_WORKERS cores, fast): $((${#CPU_ALGORITHMS[@]} * ${#CPU_SCORING[@]})) tests"
echo "  • CPU docking ($CPU_WORKERS cores, full): $((${#CPU_ALGORITHMS[@]} * ${#CPU_SCORING[@]})) tests"
echo "  • Flexible CPU docking: $FLEX_CPU_TESTS tests"
echo "  • CPU report generation: $REPORT_TESTS reports"
echo "  • TOTAL: $TOTAL_TESTS CPU tests"
echo "  • Timeouts: Fast=${FAST_MODE_TIMEOUT}s ($(($FAST_MODE_TIMEOUT/60))min), Full=${FULL_MODE_TIMEOUT}s ($(($FULL_MODE_TIMEOUT/60))min)"
echo ""

# Progress tracking function
update_progress() {
    COMPLETED_TESTS=$((COMPLETED_TESTS + 1))
    local current_time=$(date +%s)
    local elapsed=$((current_time - START_TIME))
    local avg_time_per_test=$((elapsed / COMPLETED_TESTS))
    local estimated_remaining=$(((TOTAL_TESTS - COMPLETED_TESTS) * avg_time_per_test))
    local hours=$((estimated_remaining / 3600))
    local minutes=$(((estimated_remaining % 3600) / 60))
    local seconds=$((estimated_remaining % 60))

    echo "📈 Progress: $COMPLETED_TESTS/$TOTAL_TESTS ($(( COMPLETED_TESTS * 100 / TOTAL_TESTS ))%) | Est. remaining: ${hours}h ${minutes}m ${seconds}s"
}

# Function to run CPU-optimized docking WITH TIMEOUT
run_cpu_docking() {
    local mode=$1  # "fast" or "full"
    local algorithm=$2
    local scoring=$3
    local results_dir=$4

    local output_dir="$results_dir/${algorithm}_${scoring}_${mode}"
    local fast_flag=""
    local num_poses=10
    local timeout_seconds=$FAST_MODE_TIMEOUT

    if [ "$mode" = "fast" ]; then
        fast_flag="--fast"
        num_poses=20
        timeout_seconds=$FAST_MODE_TIMEOUT

        # Reduce sampling for slow hierarchical algorithms in fast mode
        if [[ "$algorithm" == "hierarchical_cpu" ]]; then
            num_poses=10  # Reduced from 20
        fi
    else
        num_poses=50  # More poses for full mode
        timeout_seconds=$FULL_MODE_TIMEOUT

        # Reduce sampling for slow hierarchical algorithms in full mode
        if [[ "$algorithm" == "hierarchical_cpu" ]]; then
            num_poses=20  # Reduced from 50
        fi
    fi

    echo ""
    echo "🖥️  CPU Testing: $algorithm + $scoring ($mode mode, $CPU_WORKERS workers, timeout: $((timeout_seconds/60))min)"

    # Record start time
    start_time=$(date +%s)

    # Run CPU-optimized docking WITH TIMEOUT
    timeout $timeout_seconds bash -c "
        PYTHONPATH=. python3 -m pandadock.docking_cli dock \
            -r '$RECEPTOR' \
            -l '$LIGAND' \
            --grid-config '$GRID_CONFIG' \
            --algorithm '$algorithm' \
            --scoring '$scoring' \
            --num-poses '$num_poses' \
            --cpuworkers '$CPU_WORKERS' \
            --ensemble \
            --visualize \
            $fast_flag \
            -o '$output_dir'
    " > "$output_dir.log" 2>&1

    local exit_code=$?

    # Record end time
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Extract results
    local status="SUCCESS"
    if [ $exit_code -eq 124 ]; then
        # Timeout occurred
        best_energy="TIMEOUT"
        num_poses_generated="0"
        ensemble_energy="N/A"
        runtime="N/A"
        interactions="N/A"
        status="TIMEOUT"
        echo "  ⏱️  CPU TIMEOUT: Exceeded ${timeout_seconds}s limit (${duration}s elapsed)"
    elif [ -f "$output_dir.log" ]; then
        # Check for errors
        if grep -q "Error:" "$output_dir.log" || grep -q "Traceback" "$output_dir.log" || grep -q "No poses generated" "$output_dir.log"; then
            best_energy="FAILED"
            num_poses_generated="0"
            ensemble_energy="N/A"
            runtime="N/A"
            interactions="N/A"
            status="FAILED"
            error_msg=$(grep -A 2 "Error:" "$output_dir.log" | head -1 || echo "Unknown error")
            echo "  ❌ CPU Failed: $error_msg"
        else
            best_energy=$(grep -A 5 "Top.*poses:" "$output_dir.log" | grep "Energy:" | head -1 | sed 's/.*Energy: \([^,]*\).*/\1/' || echo "N/A")
            num_poses_generated=$(grep "Generated.*poses" "$output_dir.log" | tail -1 | sed 's/.*Generated \([0-9]*\) poses.*/\1/' || echo "0")
            ensemble_energy=$(grep "Ensemble binding energy:" "$output_dir.log" | sed 's/.*Ensemble binding energy: \([^ ]*\).*/\1/' || echo "N/A")
            runtime=$(grep "Docking completed in" "$output_dir.log" | sed 's/.*completed in \([^ ]*\) seconds.*/\1/' || echo "N/A")

            # Check interactions
            if [ -f "$output_dir/interaction_analysis.json" ]; then
                interactions=$(python3 -c "import json; data=json.load(open('$output_dir/interaction_analysis.json')); print(data.get('total_interactions', 'N/A'))" 2>/dev/null || echo "N/A")
            else
                interactions="N/A"
            fi

            # Check if no poses generated
            if [ "$num_poses_generated" = "0" ]; then
                status="NO_POSES"
                echo "  ⚠️  CPU Warning: No poses generated, ${duration}s ($CPU_WORKERS cores)"
            else
                echo "  ✅ CPU Success: $best_energy kcal/mol, ${duration}s ($CPU_WORKERS cores)"
            fi
        fi
    else
        best_energy="FAILED"
        num_poses_generated="0"
        ensemble_energy="N/A"
        runtime="N/A"
        interactions="N/A"
        status="NO_LOG"
        echo "  ❌ CPU Failed: No log file, ${duration}s"
    fi

    # Log to master summary
    printf "%-10s %-25s %-15s %15s %6s %15s %10s %12s %8s\n" \
        "CPU-$mode" "$algorithm" "$scoring" "$best_energy" "$num_poses_generated" "$ensemble_energy" "${runtime}s" "$interactions" "${duration}s" >> "$MASTER_LOG"

    # Log to CSV
    echo "CPU-$mode,$algorithm,$scoring,$best_energy,$num_poses_generated,$ensemble_energy,$runtime,$interactions,$duration,$status" >> "$DETAILED_CSV"

    update_progress
}

# Function to run flexible CPU docking WITH TIMEOUT
run_flex_cpu_docking() {
    local scoring=$1
    local output_dir="$FLEX_CPU_DIR/flex_cpu_${scoring}_results"

    echo ""
    echo "🔄 Flexible CPU Docking: $scoring scoring ($CPU_WORKERS workers, timeout: $((FLEX_TIMEOUT/60))min)"

    start_time=$(date +%s)

    # Run flexible docking with CPU optimization WITH TIMEOUT
    timeout $FLEX_TIMEOUT bash -c "
        PYTHONPATH=. python3 -m pandadock.flex_docking_cli \
            -r '$RECEPTOR' \
            -l '$LIGAND' \
            --center 129.249 120.21 145.249 \
            --radius 15.0 \
            --refine-distance 5.0 \
            --refine-loops \
            --refine-ligand \
            --initial-poses-to-retain 20 \
            --final-poses-to-retain 10 \
            --scoring-function ifd_composite \
            --cpu-workers '$CPU_WORKERS' \
            --max-memory-gb 32.0 \
            -o '$output_dir'
    " > "$output_dir.log" 2>&1

    local exit_code=$?

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    local status="SUCCESS"
    if [ $exit_code -eq 124 ]; then
        best_energy="TIMEOUT"
        status="TIMEOUT"
        echo "  ⏱️  Flex CPU TIMEOUT: Exceeded ${FLEX_TIMEOUT}s limit"
    elif [ -f "$output_dir.log" ]; then
        if grep -q "Error:" "$output_dir.log" || grep -q "Traceback" "$output_dir.log"; then
            best_energy="FAILED"
            status="FAILED"
            echo "  ❌ Flex CPU Failed: ${duration}s"
        else
            best_energy=$(grep "Best IFD Score:" "$output_dir.log" | sed 's/.*Best IFD Score: \([^ ]*\).*/\1/' || echo "N/A")
            echo "  ✅ Flex CPU Success: $best_energy score, ${duration}s ($CPU_WORKERS workers)"
        fi
    else
        best_energy="FAILED"
        status="NO_LOG"
        echo "  ❌ Flex CPU Failed: No log, ${duration}s"
    fi

    printf "%-10s %-25s %-15s %15s %6s %15s %10s %12s %8s\n" \
        "FLEX-CPU" "induced_fit_flexible" "$scoring" "$best_energy" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

    echo "FLEX-CPU,induced_fit_flexible,$scoring,$best_energy,N/A,N/A,N/A,N/A,$duration,$status" >> "$DETAILED_CSV"

    update_progress
}

# Function to generate CPU reports
generate_cpu_reports() {
    local mode=$1
    local input_dir=$2
    local title=$3

    echo ""
    echo "📊 Generating CPU $mode Report"

    start_time=$(date +%s)

    # Create report directory
    local report_dir="$REPORTS_DIR/${mode}_report"
    mkdir -p "$report_dir"

    # Generate publication plots WITH TIMEOUT (5 minutes)
    timeout 300 bash -c "
        PYTHONPATH=. python3 << 'EOF' > '$report_dir/report.log' 2>&1
from pandadock.report_cli import main as report_main
import sys
sys.argv = ['report_cli', 'plots', '-i', '$input_dir', '-t', '$title', '-o', '$report_dir']
try:
    report_main()
    print('Report generation completed successfully')
except Exception as e:
    print(f'Report generation error: {e}')
    import traceback
    traceback.print_exc()
EOF
    "
    local exit_code=$?

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    local status="SUCCESS"
    if [ $exit_code -eq 124 ]; then
        echo "  ⏱️  $mode Report TIMEOUT: Exceeded 300s limit"
        status="TIMEOUT"
    elif [ -f "$report_dir/report.log" ]; then
        if grep -q "error:" "$report_dir/report.log" || grep -q "Traceback" "$report_dir/report.log"; then
            echo "  ⚠️  $mode Report: Partial success, ${duration}s (check log)"
            status="PARTIAL"
        elif grep -q "completed successfully" "$report_dir/report.log"; then
            echo "  ✅ $mode Report Generated: ${duration}s"
            status="SUCCESS"
        else
            echo "  ⚠️  $mode Report: Unknown status, ${duration}s"
            status="UNKNOWN"
        fi
    else
        echo "  ❌ $mode Report Failed: No log, ${duration}s"
        status="FAILED"
    fi

    printf "%-10s %-25s %-15s %15s %6s %15s %10s %12s %8s\n" \
        "REPORT" "${mode}_analysis" "publication" "$status" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

    echo "REPORT,${mode}_analysis,publication,$status,N/A,N/A,N/A,N/A,$duration,$status" >> "$DETAILED_CSV"

    update_progress
}

# Write master log header
echo "" >> "$MASTER_LOG"
printf "%-10s %-25s %-15s %15s %6s %15s %10s %12s %8s\n" \
    "Mode" "Algorithm" "Scoring" "BestEnergy" "Poses" "EnsembleΔG" "Runtime" "Contacts" "WallTime" >> "$MASTER_LOG"
echo "=========================================================================================================================" >> "$MASTER_LOG"

# Confirmation prompt
echo "⚠️  WARNING: CPU-only comprehensive testing with TIMEOUTS!"
echo "   Hardware: $CPU_WORKERS CPU cores"
echo "   Timeouts: Fast=$((FAST_MODE_TIMEOUT/60))min, Full=$((FULL_MODE_TIMEOUT/60))min"
echo "   Estimated total time: 1-3 hours (with timeouts preventing hangs)"
echo ""
read -p "Continue with CPU testing? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing cancelled."
    exit 1
fi

echo ""
echo "🖥️  Starting CPU-only comprehensive testing with timeout protection..."

# PHASE 1: CPU Fast Mode Docking
echo ""
echo "========================================================================"
echo "PHASE 1: CPU FAST MODE DOCKING ($CPU_WORKERS cores, timeout: $((FAST_MODE_TIMEOUT/60))min)"
echo "========================================================================"

for algorithm in "${CPU_ALGORITHMS[@]}"; do
    for scoring in "${CPU_SCORING[@]}"; do
        run_cpu_docking "fast" "$algorithm" "$scoring" "$CPU_FAST_DIR"
        sleep 1
    done
done

# PHASE 2: CPU Full Accuracy Mode Docking
echo ""
echo "========================================================================"
echo "PHASE 2: CPU FULL ACCURACY MODE DOCKING ($CPU_WORKERS cores, timeout: $((FULL_MODE_TIMEOUT/60))min)"
echo "========================================================================"

for algorithm in "${CPU_ALGORITHMS[@]}"; do
    for scoring in "${CPU_SCORING[@]}"; do
        run_cpu_docking "full" "$algorithm" "$scoring" "$CPU_FULL_DIR"
        sleep 2
    done
done

# PHASE 3: Flexible CPU Docking
echo ""
echo "========================================================================"
echo "PHASE 3: FLEXIBLE CPU DOCKING ($CPU_WORKERS cores, timeout: $((FLEX_TIMEOUT/60))min)"
echo "========================================================================"

for scoring in "${CPU_SCORING[@]}"; do
    run_flex_cpu_docking "$scoring"
    sleep 1
done

# PHASE 4: CPU Report Generation
echo ""
echo "========================================================================"
echo "PHASE 4: CPU PUBLICATION REPORTS"
echo "========================================================================"

# Generate reports for each CPU mode
generate_cpu_reports "cpu_fast" "$CPU_FAST_DIR" "PandaDock CPU Fast Mode ($CPU_WORKERS cores)"
generate_cpu_reports "cpu_full" "$CPU_FULL_DIR" "PandaDock CPU Full Accuracy ($CPU_WORKERS cores)"
generate_cpu_reports "flex_cpu" "$FLEX_CPU_DIR" "PandaDock Flexible CPU ($CPU_WORKERS cores)"

# Final summary
FINAL_TIME=$(date +%s)
TOTAL_DURATION=$((FINAL_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Write final summary to log
echo "" >> "$MASTER_LOG"
echo "CPU COMPREHENSIVE TESTING SUMMARY:" >> "$MASTER_LOG"
echo "==================================" >> "$MASTER_LOG"
echo "Completed at: $(date)" >> "$MASTER_LOG"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$MASTER_LOG"
echo "Tests completed: $COMPLETED_TESTS/$TOTAL_TESTS" >> "$MASTER_LOG"
echo "Hardware utilized: $CPU_WORKERS CPU cores" >> "$MASTER_LOG"
echo "Timeouts: Fast=${FAST_MODE_TIMEOUT}s, Full=${FULL_MODE_TIMEOUT}s, Flex=${FLEX_TIMEOUT}s" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"
echo "Output Structure:" >> "$MASTER_LOG"
echo "- CPU fast results: $CPU_FAST_DIR/" >> "$MASTER_LOG"
echo "- CPU full results: $CPU_FULL_DIR/" >> "$MASTER_LOG"
echo "- Flexible CPU: $FLEX_CPU_DIR/" >> "$MASTER_LOG"
echo "- CPU publication reports: $REPORTS_DIR/" >> "$MASTER_LOG"
echo "- Detailed CSV: $DETAILED_CSV" >> "$MASTER_LOG"

# Display final results
echo ""
echo "========================================================================"
echo "🎉 CPU-ONLY COMPREHENSIVE TESTING COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 FINAL SUMMARY:"
echo "  🖥️  Hardware: $CPU_WORKERS CPU cores"
echo "  ⏱️  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  ✅ Tests completed: $COMPLETED_TESTS/$TOTAL_TESTS"
echo "  📁 Results saved in: $MAIN_RESULTS_DIR/"
echo ""
echo "📋 COMPREHENSIVE RESULTS TABLE:"
echo "================================"
echo ""

# Show comprehensive summary table
tail -n +7 "$MASTER_LOG" | head -n -10

echo ""
echo "📁 OUTPUT STRUCTURE:"
echo "  📊 Master Summary: $MASTER_LOG"
echo "  📊 Detailed CSV: $DETAILED_CSV"
echo "  🖥️  CPU Fast: $CPU_FAST_DIR/"
echo "  🖥️  CPU Full: $CPU_FULL_DIR/"
echo "  🔄 Flex CPU: $FLEX_CPU_DIR/"
echo "  📈 CPU Reports: $REPORTS_DIR/"
echo ""

# Count timeouts and failures
TIMEOUT_TESTS=$(grep -c "TIMEOUT" "$MASTER_LOG" || echo "0")
FAILED_TESTS=$(grep -c "FAILED\|NO_POSES" "$MASTER_LOG" || echo "0")

if [ "$TIMEOUT_TESTS" -gt "0" ]; then
    echo "⏱️  TIMEOUTS: $TIMEOUT_TESTS tests exceeded time limits"
    echo "  These tests were terminated to prevent hanging"
    echo "  Consider running these individually with longer timeouts if needed:"
    grep "TIMEOUT" "$DETAILED_CSV" | cut -d',' -f1-3 | while IFS=',' read mode alg score; do
        echo "    - $mode: $alg + $score"
    done
    echo ""
fi

if [ "$FAILED_TESTS" -gt "0" ]; then
    echo "⚠️  FAILURES: $FAILED_TESTS tests failed or generated no poses"
    echo "  Check individual log files for error details"
    echo ""
fi

echo "🖥️  CPU PERFORMANCE ANALYSIS:"
echo "  1. Compare algorithm performance across $CPU_WORKERS cores"
echo "  2. Identify algorithms that hit timeout limits"
echo "  3. Review scoring function differences"
echo "  4. Evaluate fast vs full mode timing and accuracy"
echo "  5. Check detailed CSV for complete statistics"
echo ""
echo "💡 RECOMMENDATIONS:"
echo "  • Timed-out tests indicate slow algorithms (hierarchical_cpu)"
echo "  • Consider using faster algorithms (monte_carlo, genetic_algorithm)"
echo "  • Or run slow algorithms individually with extended timeouts"
echo "  • Fast mode recommended for screening, Full mode for final analysis"
echo ""
echo "========================================================================"
echo "🖥️  PandaDock CPU-Only Testing Suite Complete (With Timeout Protection)!"
echo "========================================================================"
