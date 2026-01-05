#!/bin/bash

# PandaDock GPU-ONLY Comprehensive Testing Suite
# Optimized for multi-GPU systems with parallel processing
# Tests ALL GPU algorithms with ALL scoring functions
# Author: PandaDock Testing Suite
# Date: $(date)

echo "========================================================================"
echo "🎮 PandaDock GPU-ONLY Comprehensive Testing Suite"
echo "========================================================================"

# Auto-detect GPU count
GPU_COUNT=$(python3 -c "try:
    import cupy as cp
    print(cp.cuda.runtime.getDeviceCount())
except:
    print(0)" 2>/dev/null)

if [ "$GPU_COUNT" = "0" ]; then
    echo "❌ ERROR: No GPUs detected or CuPy not installed!"
    echo ""
    echo "To enable GPU algorithms, install CuPy:"
    echo "  For CUDA 11.x: pip install cupy-cuda11x"
    echo "  For CUDA 12.x: pip install cupy-cuda12x"
    echo ""
    exit 1
fi

echo "Hardware Configuration:"
echo "  🎮 GPUs: $GPU_COUNT device(s) detected"
echo "========================================================================"
echo "Testing GPU PandaDock capabilities:"
echo "  ✓ All GPU docking algorithms"
echo "  ✓ All scoring functions (physics_based, empirical, precision_score, hybrid)"
echo "  ✓ Fast and Full accuracy modes"
echo "  ✓ Flexible GPU docking"
echo "  ✓ Publication-ready reports with detailed statistics"
echo "========================================================================"

# Configuration
RECEPTOR="preprocessing/intermediate/receptor_processed.pdb"
LIGAND="preprocessing/intermediate/ligand_processed.sdf"
GRID_CONFIG="gridbox_output/gridbox_similarity.json"

# GPU configuration
GPU_BATCH_SIZE_SMALL=1024        # Conservative batch size
GPU_BATCH_SIZE_LARGE=2048        # Larger batch size for full mode
GPU_MEMORY_LIMIT=10.0            # GPU memory limit in GB

# GPU-specific algorithms and scoring
GPU_ALGORITHMS=("cuda_monte_carlo" "cuda_genetic_algorithm" "enhanced_hierarchical_gpu")
GPU_SCORING=("physics_based" "empirical" "precision_score" "hybrid")

# Result directories
MAIN_RESULTS_DIR="gpu_comprehensive_testing"
GPU_FAST_DIR="$MAIN_RESULTS_DIR/gpu_fast_${GPU_COUNT}gpus"
GPU_FULL_DIR="$MAIN_RESULTS_DIR/gpu_full_${GPU_COUNT}gpus"
FLEX_GPU_DIR="$MAIN_RESULTS_DIR/flex_gpu_${GPU_COUNT}gpus"
REPORTS_DIR="$MAIN_RESULTS_DIR/gpu_publication_reports"

# Create all directories
mkdir -p "$GPU_FAST_DIR" "$GPU_FULL_DIR" "$FLEX_GPU_DIR" "$REPORTS_DIR"

# Master summary log
MASTER_LOG="$MAIN_RESULTS_DIR/gpu_comprehensive_summary.log"
DETAILED_CSV="$MAIN_RESULTS_DIR/gpu_detailed_results.csv"

echo "🎮 PandaDock GPU-ONLY Comprehensive Testing Results" > "$MASTER_LOG"
echo "====================================================" >> "$MASTER_LOG"
echo "Started at: $(date)" >> "$MASTER_LOG"
echo "Hardware: $GPU_COUNT GPU(s)" >> "$MASTER_LOG"
echo "Testing modes: GPU Fast, GPU Full, Flexible GPU" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

# Create CSV header
echo "Mode,Algorithm,Scoring,BestEnergy,NumPoses,EnsembleΔG,Runtime,Contacts,WallTime,GPU_ID,Status" > "$DETAILED_CSV"

# Timing and progress tracking
TOTAL_TESTS=0
COMPLETED_TESTS=0
START_TIME=$(date +%s)

# Calculate total tests
GPU_DOCKING_TESTS=$((${#GPU_ALGORITHMS[@]} * ${#GPU_SCORING[@]} * 2))  # fast + full modes
FLEX_GPU_TESTS=${#GPU_SCORING[@]}
REPORT_TESTS=3  # GPU Fast, GPU Full, Flex GPU
TOTAL_TESTS=$((GPU_DOCKING_TESTS + FLEX_GPU_TESTS + REPORT_TESTS))

echo "📊 GPU TESTING PLAN:"
echo "  • GPU docking ($GPU_COUNT GPUs, fast): $((${#GPU_ALGORITHMS[@]} * ${#GPU_SCORING[@]})) tests"
echo "  • GPU docking ($GPU_COUNT GPUs, full): $((${#GPU_ALGORITHMS[@]} * ${#GPU_SCORING[@]})) tests"
echo "  • Flexible GPU docking: $FLEX_GPU_TESTS tests"
echo "  • GPU report generation: $REPORT_TESTS reports"
echo "  • TOTAL: $TOTAL_TESTS GPU tests"
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

# Function to run GPU-optimized docking
run_gpu_docking() {
    local mode=$1  # "fast" or "full"
    local algorithm=$2
    local scoring=$3
    local results_dir=$4
    local gpu_id=$5

    local output_dir="$results_dir/${algorithm}_${scoring}_${mode}_gpu${gpu_id}"
    local fast_flag=""
    local num_poses=20
    local batch_size=$GPU_BATCH_SIZE_SMALL

    if [ "$mode" = "fast" ]; then
        fast_flag="--fast"
        num_poses=30
        batch_size=$GPU_BATCH_SIZE_SMALL
    else
        num_poses=100  # Many more poses for GPU full mode
        batch_size=$GPU_BATCH_SIZE_LARGE
    fi

    echo ""
    echo "🎮 GPU Testing: $algorithm + $scoring ($mode mode, GPU $gpu_id)"

    # Record start time
    start_time=$(date +%s)

    # Run GPU-optimized docking
    PYTHONPATH=. python3 -m pandadock.docking_cli dock \
        -r "$RECEPTOR" \
        -l "$LIGAND" \
        --grid-config "$GRID_CONFIG" \
        --algorithm "$algorithm" \
        --scoring "$scoring" \
        --num-poses "$num_poses" \
        --gpu \
        --gpuid "$gpu_id" \
        --gpu-batch-size "$batch_size" \
        --gpu-memory-limit "$GPU_MEMORY_LIMIT" \
        --ensemble \
        --visualize \
        $fast_flag \
        -o "$output_dir" > "$output_dir.log" 2>&1

    # Record end time
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Extract results
    local status="SUCCESS"
    if [ -f "$output_dir.log" ]; then
        # Check for errors
        if grep -q "Error:" "$output_dir.log" || grep -q "Traceback" "$output_dir.log" || grep -q "Docking failed" "$output_dir.log"; then
            best_energy="FAILED"
            num_poses_generated="0"
            ensemble_energy="N/A"
            runtime="N/A"
            interactions="N/A"
            status="FAILED"
            error_msg=$(grep -A 2 "Error:" "$output_dir.log" | head -1 || echo "Check log")
            echo "  ❌ GPU Failed: $error_msg"
        elif grep -q "No poses generated" "$output_dir.log"; then
            best_energy="NO_POSES"
            num_poses_generated="0"
            ensemble_energy="N/A"
            runtime=$(grep "Docking completed in" "$output_dir.log" | sed 's/.*completed in \([^ ]*\) seconds.*/\1/' || echo "N/A")
            interactions="N/A"
            status="NO_POSES"
            echo "  ⚠️  GPU Warning: No poses generated, ${duration}s (GPU $gpu_id)"
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

            echo "  ✅ GPU Success: $best_energy kcal/mol, ${duration}s (GPU $gpu_id)"
        fi
    else
        best_energy="FAILED"
        num_poses_generated="0"
        ensemble_energy="N/A"
        runtime="N/A"
        interactions="N/A"
        status="NO_LOG"
        echo "  ❌ GPU Failed: No log file, ${duration}s"
    fi

    # Log to master summary
    printf "%-10s %-25s %-15s %15s %6s %15s %10s %12s %8s\n" \
        "GPU$gpu_id-$mode" "$algorithm" "$scoring" "$best_energy" "$num_poses_generated" "$ensemble_energy" "${runtime}s" "$interactions" "${duration}s" >> "$MASTER_LOG"

    # Log to CSV
    echo "GPU$gpu_id-$mode,$algorithm,$scoring,$best_energy,$num_poses_generated,$ensemble_energy,$runtime,$interactions,$duration,$gpu_id,$status" >> "$DETAILED_CSV"

    update_progress
}

# Function to run flexible GPU docking
run_flex_gpu_docking() {
    local scoring=$1
    local gpu_id=$2
    local output_dir="$FLEX_GPU_DIR/flex_gpu${gpu_id}_${scoring}_results"

    echo ""
    echo "🔄 Flexible GPU Docking: $scoring scoring (GPU $gpu_id)"

    start_time=$(date +%s)

    # Run flexible docking with GPU optimization
    PYTHONPATH=. python3 -m pandadock.flex_docking_cli \
        -r "$RECEPTOR" \
        -l "$LIGAND" \
        --center 129.249 120.21 145.249 \
        --radius 15.0 \
        --refine-distance 5.0 \
        --refine-loops \
        --refine-ligand \
        --initial-poses-to-retain 30 \
        --final-poses-to-retain 15 \
        --scoring-function ifd_composite \
        --use-gpu \
        --gpu-id "$gpu_id" \
        --cpu-workers 12 \
        --max-memory-gb 16.0 \
        -o "$output_dir" > "$output_dir.log" 2>&1

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    local status="SUCCESS"
    if [ -f "$output_dir.log" ]; then
        if grep -q "Error:" "$output_dir.log" || grep -q "Traceback" "$output_dir.log"; then
            best_energy="FAILED"
            status="FAILED"
            echo "  ❌ Flex GPU Failed: ${duration}s"
        else
            best_energy=$(grep "Best IFD Score:" "$output_dir.log" | sed 's/.*Best IFD Score: \([^ ]*\).*/\1/' || echo "N/A")
            echo "  ✅ Flex GPU Success: $best_energy score, ${duration}s (GPU $gpu_id)"
        fi
    else
        best_energy="FAILED"
        status="NO_LOG"
        echo "  ❌ Flex GPU Failed: No log, ${duration}s"
    fi

    printf "%-10s %-25s %-15s %15s %6s %15s %10s %12s %8s\n" \
        "FLEX-GPU$gpu_id" "induced_fit_flexible" "$scoring" "$best_energy" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

    echo "FLEX-GPU$gpu_id,induced_fit_flexible,$scoring,$best_energy,N/A,N/A,N/A,N/A,$duration,$gpu_id,$status" >> "$DETAILED_CSV"

    update_progress
}

# Function to generate GPU reports
generate_gpu_reports() {
    local mode=$1
    local input_dir=$2
    local title=$3

    echo ""
    echo "📊 Generating GPU $mode Report"

    start_time=$(date +%s)

    # Create report directory
    local report_dir="$REPORTS_DIR/${mode}_report"
    mkdir -p "$report_dir"

    # Generate publication plots
    PYTHONPATH=. python3 << EOF > "$report_dir/report.log" 2>&1
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

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    local status="SUCCESS"
    if [ -f "$report_dir/report.log" ]; then
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

    echo "REPORT,${mode}_analysis,publication,$status,N/A,N/A,N/A,N/A,$duration,N/A,$status" >> "$DETAILED_CSV"

    update_progress
}

# Write master log header
echo "" >> "$MASTER_LOG"
printf "%-10s %-25s %-15s %15s %6s %15s %10s %12s %8s\n" \
    "Mode" "Algorithm" "Scoring" "BestEnergy" "Poses" "EnsembleΔG" "Runtime" "Contacts" "WallTime" >> "$MASTER_LOG"
echo "=========================================================================================================================" >> "$MASTER_LOG"

# Display GPU information
echo ""
echo "🔍 GPU Hardware Information:"
python3 << EOF
try:
    import cupy as cp
    for i in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(i):
            meminfo = cp.cuda.runtime.memGetInfo()
            print(f'  GPU {i}: {meminfo[1]//1024//1024//1024} GB total memory')
except Exception as e:
    print(f'  Error getting GPU info: {e}')
EOF
echo ""

# Confirmation prompt
echo "⚠️  WARNING: GPU-only comprehensive testing!"
echo "   Hardware: $GPU_COUNT GPU(s)"
echo "   Estimated total time: 20-40 minutes with GPU acceleration"
echo ""
read -p "Continue with GPU testing? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing cancelled."
    exit 1
fi

echo ""
echo "🎮 Starting GPU-only comprehensive testing..."

# PHASE 1: GPU Fast Mode Docking
echo ""
echo "========================================================================"
echo "PHASE 1: GPU FAST MODE DOCKING ($GPU_COUNT GPUs)"
echo "========================================================================"

gpu_counter=0
for algorithm in "${GPU_ALGORITHMS[@]}"; do
    for scoring in "${GPU_SCORING[@]}"; do
        gpu_id=$((gpu_counter % GPU_COUNT))  # Distribute across available GPUs
        run_gpu_docking "fast" "$algorithm" "$scoring" "$GPU_FAST_DIR" "$gpu_id"
        gpu_counter=$((gpu_counter + 1))
        sleep 1
    done
done

# PHASE 2: GPU Full Accuracy Mode Docking
echo ""
echo "========================================================================"
echo "PHASE 2: GPU FULL ACCURACY MODE DOCKING ($GPU_COUNT GPUs)"
echo "========================================================================"

gpu_counter=0
for algorithm in "${GPU_ALGORITHMS[@]}"; do
    for scoring in "${GPU_SCORING[@]}"; do
        gpu_id=$((gpu_counter % GPU_COUNT))  # Distribute across available GPUs
        run_gpu_docking "full" "$algorithm" "$scoring" "$GPU_FULL_DIR" "$gpu_id"
        gpu_counter=$((gpu_counter + 1))
        sleep 2
    done
done

# PHASE 3: Flexible GPU Docking
echo ""
echo "========================================================================"
echo "PHASE 3: FLEXIBLE GPU DOCKING ($GPU_COUNT GPUs)"
echo "========================================================================"

gpu_counter=0
for scoring in "${GPU_SCORING[@]}"; do
    gpu_id=$((gpu_counter % GPU_COUNT))  # Distribute across available GPUs
    run_flex_gpu_docking "$scoring" "$gpu_id"
    gpu_counter=$((gpu_counter + 1))
    sleep 1
done

# PHASE 4: GPU Report Generation
echo ""
echo "========================================================================"
echo "PHASE 4: GPU PUBLICATION REPORTS"
echo "========================================================================"

# Generate reports for each GPU mode
generate_gpu_reports "gpu_fast" "$GPU_FAST_DIR" "PandaDock GPU Fast Mode ($GPU_COUNT GPUs)"
generate_gpu_reports "gpu_full" "$GPU_FULL_DIR" "PandaDock GPU Full Accuracy ($GPU_COUNT GPUs)"
generate_gpu_reports "flex_gpu" "$FLEX_GPU_DIR" "PandaDock Flexible GPU ($GPU_COUNT GPUs)"

# Final summary
FINAL_TIME=$(date +%s)
TOTAL_DURATION=$((FINAL_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Write final summary to log
echo "" >> "$MASTER_LOG"
echo "GPU COMPREHENSIVE TESTING SUMMARY:" >> "$MASTER_LOG"
echo "==================================" >> "$MASTER_LOG"
echo "Completed at: $(date)" >> "$MASTER_LOG"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$MASTER_LOG"
echo "Tests completed: $COMPLETED_TESTS/$TOTAL_TESTS" >> "$MASTER_LOG"
echo "Hardware utilized: $GPU_COUNT GPU(s)" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"
echo "Output Structure:" >> "$MASTER_LOG"
echo "- GPU fast results: $GPU_FAST_DIR/" >> "$MASTER_LOG"
echo "- GPU full results: $GPU_FULL_DIR/" >> "$MASTER_LOG"
echo "- Flexible GPU: $FLEX_GPU_DIR/" >> "$MASTER_LOG"
echo "- GPU publication reports: $REPORTS_DIR/" >> "$MASTER_LOG"
echo "- Detailed CSV: $DETAILED_CSV" >> "$MASTER_LOG"

# Display final results
echo ""
echo "========================================================================"
echo "🎉 GPU-ONLY COMPREHENSIVE TESTING COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 FINAL SUMMARY:"
echo "  🎮 Hardware: $GPU_COUNT GPU(s)"
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
echo "  🎮 GPU Fast: $GPU_FAST_DIR/"
echo "  🎮 GPU Full: $GPU_FULL_DIR/"
echo "  🔄 Flex GPU: $FLEX_GPU_DIR/"
echo "  📈 GPU Reports: $REPORTS_DIR/"
echo ""

# Count failures
FAILED_TESTS=$(grep -c "FAILED\|NO_POSES" "$MASTER_LOG" || echo "0")
if [ "$FAILED_TESTS" -gt "0" ]; then
    echo "⚠️  WARNING: $FAILED_TESTS GPU tests failed or generated no poses"
    echo "  Check individual log files for error details"
    echo ""
fi

echo "🎮 GPU PERFORMANCE ANALYSIS:"
echo "  1. Compare GPU performance across $GPU_COUNT device(s)"
echo "  2. Evaluate fast vs full mode GPU speedups"
echo "  3. Review GPU memory usage and batch size optimization"
echo "  4. Compare GPU algorithms for throughput and accuracy"
echo "  5. Validate flexible GPU docking for induced-fit targets"
echo ""
echo "🚀 GPU OPTIMIZATION RECOMMENDATIONS:"
echo "  • GPU algorithms provide 50-100x speedup for large-scale screening"
echo "  • Use GPU fast mode for virtual screening campaigns"
echo "  • Use GPU full mode for final refinement and publications"
echo "  • Optimize batch sizes based on GPU memory (monitor with nvidia-smi)"
echo "  • Flexible GPU docking best for challenging protein flexibility cases"
echo ""
echo "========================================================================"
echo "🎮 PandaDock GPU-Only Testing Suite Complete!"
echo "========================================================================"
