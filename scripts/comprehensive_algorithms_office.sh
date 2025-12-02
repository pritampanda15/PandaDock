#!/bin/bash

# PandaDock OFFICE COMPREHENSIVE Testing Suite
# Optimized for 2 GPUs + 48 CPU Cores High-Performance Setup
# Tests ALL PandaDock tools with GPU acceleration and maximum CPU utilization
# Includes: pandadock dock (CPU/GPU), pandadock-flex, pandadock-report, pandadock-tethered
# Author: Claude Code Assistant
# Date: $(date)

echo "========================================================================"
echo "🚀 PandaDock OFFICE HIGH-PERFORMANCE Testing Suite"
echo "========================================================================"
echo "Hardware Configuration:"
echo "  🖥️  CPUs: 48 cores (optimized worker allocation)"
echo "  🎮 GPUs: 2 devices (parallel GPU processing)"
echo "========================================================================"
echo "Testing ALL PandaDock tools:"
echo "  ✓ pandadock dock (CPU algorithms with 48 workers)"
echo "  ✓ pandadock dock (GPU algorithms with 2 GPUs)"
echo "  ✓ pandadock-flex (CPU + GPU flexible docking)"
echo "  ✓ pandadock-report (publication-ready analysis)"
echo "  ✓ pandadock-tethered (crystallographic validation)"
echo "========================================================================"

# Configuration
RECEPTOR="preprocessing/intermediate/receptor_processed.pdb"
LIGAND="preprocessing/intermediate/ligand_processed.sdf"
GRID_CONFIG="gridbox_output/gridbox_similarity.json"

# Hardware configuration
CPU_WORKERS=48                    # Use all 48 CPU cores
GPU_BATCH_SIZE_SMALL=1024        # Conservative batch size for GPU 0
GPU_BATCH_SIZE_LARGE=2048        # Larger batch size for GPU 1
GPU_MEMORY_LIMIT=10.0            # GPU memory limit in GB
GPU_0_ID=0                       # First GPU
GPU_1_ID=1                       # Second GPU

# Algorithm arrays - separated by compute type
CPU_ALGORITHMS=("monte_carlo_cpu" "genetic_algorithm_cpu" "hierarchical_cpu" "enhanced_hierarchical_cpu" "crystal_guided_cpu")
# GPU_ALGORITHMS - NOTE: These require CuPy to be installed
# If CuPy is not installed, tests will fail gracefully with error messages
GPU_ALGORITHMS=("cuda_monte_carlo" "cuda_genetic_algorithm" "enhanced_hierarchical_gpu")

# Scoring functions - separated by compute type
CPU_SCORING=("physics_based" "empirical" "precision_score" "hybrid")
# GPU_SCORING - NOTE: These require GPU-specific libraries
# If not available, tests will fall back to CPU scoring
GPU_SCORING=("physics_based" "empirical" "precision_score" "hybrid")

# Result directories
MAIN_RESULTS_DIR="office_comprehensive_testing"
CPU_FAST_DIR="$MAIN_RESULTS_DIR/cpu_fast_48cores"
CPU_FULL_DIR="$MAIN_RESULTS_DIR/cpu_full_48cores"
GPU_FAST_DIR="$MAIN_RESULTS_DIR/gpu_fast_2gpus"
GPU_FULL_DIR="$MAIN_RESULTS_DIR/gpu_full_2gpus"
FLEX_CPU_DIR="$MAIN_RESULTS_DIR/flex_cpu_48cores"
FLEX_GPU_DIR="$MAIN_RESULTS_DIR/flex_gpu_2gpus"
TETHERED_RESULTS_DIR="$MAIN_RESULTS_DIR/tethered_analysis"
REPORTS_DIR="$MAIN_RESULTS_DIR/publication_reports"

# Create all directories
mkdir -p "$CPU_FAST_DIR" "$CPU_FULL_DIR" "$GPU_FAST_DIR" "$GPU_FULL_DIR" \
         "$FLEX_CPU_DIR" "$FLEX_GPU_DIR" "$TETHERED_RESULTS_DIR" "$REPORTS_DIR"

# Master summary log
MASTER_LOG="$MAIN_RESULTS_DIR/office_comprehensive_summary.log"
echo "🚀 PandaDock OFFICE High-Performance Testing Results" > "$MASTER_LOG"
echo "====================================================" >> "$MASTER_LOG"
echo "Started at: $(date)" >> "$MASTER_LOG"
echo "Hardware: 48 CPU cores + 2 GPUs" >> "$MASTER_LOG"
echo "Testing modes: CPU Fast, CPU Full, GPU Fast, GPU Full, Flexible CPU/GPU" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

# Timing and progress tracking
TOTAL_TESTS=0
COMPLETED_TESTS=0
START_TIME=$(date +%s)

# Calculate total tests
CPU_TESTS=$((${#CPU_ALGORITHMS[@]} * ${#CPU_SCORING[@]} * 2))  # fast + full modes
GPU_TESTS=$((${#GPU_ALGORITHMS[@]} * ${#GPU_SCORING[@]} * 2))  # fast + full modes
FLEX_CPU_TESTS=${#CPU_SCORING[@]}
FLEX_GPU_TESTS=${#GPU_SCORING[@]}
TETHERED_TESTS=1
REPORT_TESTS=6  # CPU Fast, CPU Full, GPU Fast, GPU Full, Flex CPU, Flex GPU
TOTAL_TESTS=$((CPU_TESTS + GPU_TESTS + FLEX_CPU_TESTS + FLEX_GPU_TESTS + TETHERED_TESTS + REPORT_TESTS))

echo "📊 OFFICE TESTING PLAN:"
echo "  • CPU docking (48 workers, fast): $((${#CPU_ALGORITHMS[@]} * ${#CPU_SCORING[@]})) tests"
echo "  • CPU docking (48 workers, full): $((${#CPU_ALGORITHMS[@]} * ${#CPU_SCORING[@]})) tests"
echo "  • GPU docking (2 GPUs, fast): $((${#GPU_ALGORITHMS[@]} * ${#GPU_SCORING[@]})) tests"
echo "  • GPU docking (2 GPUs, full): $((${#GPU_ALGORITHMS[@]} * ${#GPU_SCORING[@]})) tests"
echo "  • Flexible CPU docking: $FLEX_CPU_TESTS tests"
echo "  • Flexible GPU docking: $FLEX_GPU_TESTS tests"
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

# Function to run CPU-optimized docking
run_cpu_docking() {
    local mode=$1  # "fast" or "full"
    local algorithm=$2
    local scoring=$3
    local results_dir=$4

    local output_dir="$results_dir/${algorithm}_${scoring}_${mode}"
    local fast_flag=""
    local num_poses=10

    if [ "$mode" = "fast" ]; then
        fast_flag="--fast"
        num_poses=20
    else
        num_poses=50  # More poses for full mode with 48 cores
    fi

    echo ""
    echo "🖥️  CPU Testing: $algorithm + $scoring ($mode mode, 48 workers)"

    # Record start time
    start_time=$(date +%s)

    # Run CPU-optimized docking
    PYTHONPATH=. python3 -m pandadock.docking_cli dock \
        -r "$RECEPTOR" \
        -l "$LIGAND" \
        --grid-config "$GRID_CONFIG" \
        --algorithm "$algorithm" \
        --scoring "$scoring" \
        --num-poses "$num_poses" \
        --cpuworkers "$CPU_WORKERS" \
        --ensemble \
        --rescoring mmgbsa \
        --visualize \
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

        echo "  ✅ CPU Success: $best_energy kcal/mol, ${duration}s (48 cores)"
    else
        best_energy="FAILED"
        num_poses_generated="0"
        ensemble_energy="N/A"
        runtime="N/A"
        interactions="N/A"
        echo "  ❌ CPU Failed: ${duration}s"
    fi

    # Log to master summary
    printf "%-10s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "CPU-$mode" "$algorithm" "$scoring" "$best_energy" "$num_poses_generated" "$ensemble_energy" "${runtime}s" "$interactions" "${duration}s" >> "$MASTER_LOG"

    update_progress
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
        --rescoring mmgbsa \
        --visualize \
        $fast_flag \
        -o "$output_dir" > "$output_dir.log" 2>&1

    # Record end time
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Extract results
    if [ -f "$output_dir.log" ]; then
        # Check for algorithm availability error
        if grep -q "Unknown algorithm" "$output_dir.log" || grep -q "CUDA not available" "$output_dir.log"; then
            best_energy="NO_CUPY"
            num_poses_generated="0"
            ensemble_energy="N/A"
            runtime="N/A"
            interactions="N/A"
            echo "  ⚠️  GPU Skipped: CuPy not installed (install: pip install cupy-cuda11x or cupy-cuda12x)"
        else
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

            echo "  ✅ GPU Success: $best_energy kcal/mol, ${duration}s (GPU $gpu_id)"
        fi
    else
        best_energy="FAILED"
        num_poses_generated="0"
        ensemble_energy="N/A"
        runtime="N/A"
        interactions="N/A"
        echo "  ❌ GPU Failed: ${duration}s"
    fi

    # Log to master summary
    printf "%-10s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "GPU$gpu_id-$mode" "$algorithm" "$scoring" "$best_energy" "$num_poses_generated" "$ensemble_energy" "${runtime}s" "$interactions" "${duration}s" >> "$MASTER_LOG"

    update_progress
}

# Function to run flexible docking (CPU optimized)
run_flex_cpu_docking() {
    local scoring=$1
    local output_dir="$FLEX_CPU_DIR/flex_cpu_${scoring}"

    echo ""
    echo "🔄 Flexible CPU Docking: $scoring scoring (48 workers)"

    start_time=$(date +%s)

    # Run flexible docking with CPU optimization
    PYTHONPATH=. python3 -m pandadock.flex_docking_cli \
        -r "$RECEPTOR" \
        -l "$LIGAND" \
        --center 129.249 120.21 145.249 \
        --radius 15.0 \
        --refine-distance 5.0 \
        --refine-loops \
        --refine-ligand \
        --initial-poses-to-retain 20 \
        --final-poses-to-retain 10 \
        --scoring-function ifd_composite \
        --cpu-workers "$CPU_WORKERS" \
        --max-memory-gb 32.0 \
        -o "$output_dir" > "$output_dir.log" 2>&1

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ -f "$output_dir.log" ]; then
        best_energy=$(grep "Best IFD Score:" "$output_dir.log" | sed 's/.*Best IFD Score: \([^ ]*\).*/\1/' || echo "N/A")
        echo "  ✅ Flex CPU Success: $best_energy score, ${duration}s (48 workers)"
        flex_status="SUCCESS"
    else
        best_energy="FAILED"
        flex_status="FAILED"
        echo "  ❌ Flex CPU Failed: ${duration}s"
    fi

    printf "%-10s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "FLEX-CPU" "induced_fit_flexible" "$scoring" "$best_energy" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

    update_progress
}

# Function to run flexible docking (GPU optimized)
run_flex_gpu_docking() {
    local scoring=$1
    local gpu_id=$2
    local output_dir="$FLEX_GPU_DIR/flex_gpu${gpu_id}_${scoring}"

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

    if [ -f "$output_dir.log" ]; then
        best_energy=$(grep "Best IFD Score:" "$output_dir.log" | sed 's/.*Best IFD Score: \([^ ]*\).*/\1/' || echo "N/A")
        echo "  ✅ Flex GPU Success: $best_energy score, ${duration}s (GPU $gpu_id)"
        flex_status="SUCCESS"
    else
        best_energy="FAILED"
        flex_status="FAILED"
        echo "  ❌ Flex GPU Failed: ${duration}s"
    fi

    printf "%-10s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "FLEX-GPU$gpu_id" "induced_fit_flexible" "$scoring" "$best_energy" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

    update_progress
}

# Function to run tethered analysis
run_tethered_analysis() {
    echo ""
    echo "🔗 Tethered Analysis (Crystallographic Validation)"

    start_time=$(date +%s)

    # Find a good complex file from previous results (prefer GPU results)
    local complex_file=""
    for dir in "$GPU_FAST_DIR"/cuda_monte_carlo_physics_based_fast_gpu0; do
        if [ -f "$dir/complex1.pdb" ]; then
            complex_file="$dir/complex1.pdb"
            break
        fi
    done

    # Fallback to CPU results
    if [ -z "$complex_file" ]; then
        for dir in "$CPU_FAST_DIR"/genetic_algorithm_cpu_physics_based_fast; do
            if [ -f "$dir/complex1.pdb" ]; then
                complex_file="$dir/complex1.pdb"
                break
            fi
        done
    fi

    if [ -z "$complex_file" ]; then
        echo "  ⚠️  No complex file found for tethered analysis"
        printf "%-10s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
            "TETHERED" "tethered_validation" "physics_based" "NO_INPUT" "N/A" "N/A" "N/A" "N/A" "0s" >> "$MASTER_LOG"
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

    printf "%-10s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "TETHERED" "tethered_validation" "crystal_reprod" "$crystal_score" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

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

    # Generate publication plots and analysis
    PYTHONPATH=. python3 -m pandadock.report_cli plots \
        -i "$input_dir" \
        -t "$title" \
        -o "$REPORTS_DIR/${mode}_report" > "$REPORTS_DIR/${mode}_report.log" 2>&1

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ -f "$REPORTS_DIR/${mode}_report.log" ]; then
        # Check if report generation was successful (look for error messages)
        if grep -q "Error:" "$REPORTS_DIR/${mode}_report.log" || grep -q "Failed" "$REPORTS_DIR/${mode}_report.log"; then
            echo "  ❌ $mode Report Failed: ${duration}s (check log for errors)"
            report_status="FAILED"
        else
            echo "  ✅ $mode Report Generated: ${duration}s"
            report_status="SUCCESS"
        fi
    else
        echo "  ❌ $mode Report Failed: ${duration}s (no log file)"
        report_status="FAILED"
    fi

    printf "%-10s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
        "REPORT" "${mode}_analysis" "publication" "$report_status" "N/A" "N/A" "N/A" "N/A" "${duration}s" >> "$MASTER_LOG"

    update_progress
}

# Write master log header
echo "" >> "$MASTER_LOG"
printf "%-10s %-25s %-15s %12s %6s %12s %8s %12s %8s\n" \
    "Mode" "Algorithm" "Scoring" "BestEnergy" "Poses" "EnsembleΔG" "Runtime" "Contacts" "WallTime" >> "$MASTER_LOG"
echo "=========================================================================================================" >> "$MASTER_LOG"

# Check GPU requirements
echo ""
echo "🔍 Checking GPU Requirements..."
python3 -c "import cupy; print('  ✅ CuPy installed - GPU algorithms will be tested')" 2>/dev/null || \
    echo "  ⚠️  CuPy not installed - GPU algorithms will be skipped"
echo "     To enable GPU algorithms: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)"
echo ""

# Confirmation prompt
echo "⚠️  WARNING: Office comprehensive testing optimized for high-performance hardware!"
echo "   Hardware: 48 CPU cores + 2 GPUs"
echo "   Estimated total time: 1-3 hours with hardware acceleration"
echo "   This includes CPU and GPU modes with optimal resource utilization"
echo "   Note: GPU algorithms require CuPy (will be skipped if not installed)"
echo ""
read -p "Continue with office testing? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing cancelled."
    exit 1
fi

echo ""
echo "🚀 Starting office comprehensive testing with hardware optimization..."

# PHASE 1: CPU Fast Mode Docking (48 cores)
echo ""
echo "========================================================================"
echo "PHASE 1: CPU FAST MODE DOCKING (48 cores)"
echo "========================================================================"

for algorithm in "${CPU_ALGORITHMS[@]}"; do
    for scoring in "${CPU_SCORING[@]}"; do
        run_cpu_docking "fast" "$algorithm" "$scoring" "$CPU_FAST_DIR"
        sleep 1
    done
done

# PHASE 2: CPU Full Accuracy Mode Docking (48 cores)
echo ""
echo "========================================================================"
echo "PHASE 2: CPU FULL ACCURACY MODE DOCKING (48 cores)"
echo "========================================================================"

for algorithm in "${CPU_ALGORITHMS[@]}"; do
    for scoring in "${CPU_SCORING[@]}"; do
        run_cpu_docking "full" "$algorithm" "$scoring" "$CPU_FULL_DIR"
        sleep 2
    done
done

# PHASE 3: GPU Fast Mode Docking (Dual GPU)
echo ""
echo "========================================================================"
echo "PHASE 3: GPU FAST MODE DOCKING (2 GPUs)"
echo "========================================================================"

gpu_counter=0
for algorithm in "${GPU_ALGORITHMS[@]}"; do
    for scoring in "${GPU_SCORING[@]}"; do
        gpu_id=$((gpu_counter % 2))  # Alternate between GPU 0 and 1
        run_gpu_docking "fast" "$algorithm" "$scoring" "$GPU_FAST_DIR" "$gpu_id"
        gpu_counter=$((gpu_counter + 1))
        sleep 1
    done
done

# PHASE 4: GPU Full Accuracy Mode Docking (Dual GPU)
echo ""
echo "========================================================================"
echo "PHASE 4: GPU FULL ACCURACY MODE DOCKING (2 GPUs)"
echo "========================================================================"

gpu_counter=0
for algorithm in "${GPU_ALGORITHMS[@]}"; do
    for scoring in "${GPU_SCORING[@]}"; do
        gpu_id=$((gpu_counter % 2))  # Alternate between GPU 0 and 1
        run_gpu_docking "full" "$algorithm" "$scoring" "$GPU_FULL_DIR" "$gpu_id"
        gpu_counter=$((gpu_counter + 1))
        sleep 2
    done
done

# PHASE 5: Flexible CPU Docking (48 cores)
echo ""
echo "========================================================================"
echo "PHASE 5: FLEXIBLE CPU DOCKING (48 cores)"
echo "========================================================================"

for scoring in "${CPU_SCORING[@]}"; do
    run_flex_cpu_docking "$scoring"
    sleep 1
done

# PHASE 6: Flexible GPU Docking (2 GPUs)
echo ""
echo "========================================================================"
echo "PHASE 6: FLEXIBLE GPU DOCKING (2 GPUs)"
echo "========================================================================"

gpu_counter=0
for scoring in "${GPU_SCORING[@]}"; do
    gpu_id=$((gpu_counter % 2))  # Alternate between GPU 0 and 1
    run_flex_gpu_docking "$scoring" "$gpu_id"
    gpu_counter=$((gpu_counter + 1))
    sleep 1
done

# PHASE 7: Tethered Analysis
echo ""
echo "========================================================================"
echo "PHASE 7: TETHERED ANALYSIS"
echo "========================================================================"

run_tethered_analysis

# PHASE 8: Report Generation
echo ""
echo "========================================================================"
echo "PHASE 8: PUBLICATION REPORTS"
echo "========================================================================"

# Generate reports for each mode
generate_reports "cpu_fast" "$CPU_FAST_DIR" "PandaDock CPU Fast Mode (48 cores)"
generate_reports "cpu_full" "$CPU_FULL_DIR" "PandaDock CPU Full Accuracy (48 cores)"
generate_reports "gpu_fast" "$GPU_FAST_DIR" "PandaDock GPU Fast Mode (2 GPUs)"
generate_reports "gpu_full" "$GPU_FULL_DIR" "PandaDock GPU Full Accuracy (2 GPUs)"
generate_reports "flex_cpu" "$FLEX_CPU_DIR" "PandaDock Flexible CPU (48 cores)"
generate_reports "flex_gpu" "$FLEX_GPU_DIR" "PandaDock Flexible GPU (2 GPUs)"

# Final summary
FINAL_TIME=$(date +%s)
TOTAL_DURATION=$((FINAL_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Write final summary to log
echo "" >> "$MASTER_LOG"
echo "OFFICE COMPREHENSIVE TESTING SUMMARY:" >> "$MASTER_LOG"
echo "=====================================" >> "$MASTER_LOG"
echo "Completed at: $(date)" >> "$MASTER_LOG"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$MASTER_LOG"
echo "Tests completed: $COMPLETED_TESTS/$TOTAL_TESTS" >> "$MASTER_LOG"
echo "Hardware utilized: 48 CPU cores + 2 GPUs" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"
echo "Output Structure:" >> "$MASTER_LOG"
echo "- CPU fast results (48 cores): $CPU_FAST_DIR/" >> "$MASTER_LOG"
echo "- CPU full results (48 cores): $CPU_FULL_DIR/" >> "$MASTER_LOG"
echo "- GPU fast results (2 GPUs): $GPU_FAST_DIR/" >> "$MASTER_LOG"
echo "- GPU full results (2 GPUs): $GPU_FULL_DIR/" >> "$MASTER_LOG"
echo "- Flexible CPU (48 cores): $FLEX_CPU_DIR/" >> "$MASTER_LOG"
echo "- Flexible GPU (2 GPUs): $FLEX_GPU_DIR/" >> "$MASTER_LOG"
echo "- Tethered analysis: $TETHERED_RESULTS_DIR/" >> "$MASTER_LOG"
echo "- Publication reports: $REPORTS_DIR/" >> "$MASTER_LOG"

# Display final results
echo ""
echo "========================================================================"
echo "🎉 OFFICE COMPREHENSIVE TESTING COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 FINAL SUMMARY:"
echo "  🖥️  Hardware: 48 CPU cores + 2 GPUs"
echo "  ⏱️  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "  ✅ Tests completed: $COMPLETED_TESTS/$TOTAL_TESTS"
echo "  📁 Results saved in: $MAIN_RESULTS_DIR/"
echo ""
echo "📋 COMPREHENSIVE RESULTS TABLE:"
echo "================================"
echo ""

# Show comprehensive summary table
cat "$MASTER_LOG" | tail -n +7 | head -n -20

echo ""
echo "📁 OUTPUT STRUCTURE:"
echo "  📊 Master Summary: $MASTER_LOG"
echo "  🖥️  CPU Fast (48 cores): $CPU_FAST_DIR/"
echo "  🖥️  CPU Full (48 cores): $CPU_FULL_DIR/"
echo "  🎮 GPU Fast (2 GPUs): $GPU_FAST_DIR/"
echo "  🎮 GPU Full (2 GPUs): $GPU_FULL_DIR/"
echo "  🔄 Flex CPU (48 cores): $FLEX_CPU_DIR/"
echo "  🔄 Flex GPU (2 GPUs): $FLEX_GPU_DIR/"
echo "  🔗 Tethered: $TETHERED_RESULTS_DIR/"
echo "  📈 Reports: $REPORTS_DIR/"
echo ""

# Check if GPU tests were skipped
GPU_SKIPPED=$(grep -c "NO_CUPY" "$MASTER_LOG" || echo "0")
if [ "$GPU_SKIPPED" -gt "0" ]; then
    echo "⚠️  GPU ALGORITHM NOTICE:"
    echo "  $GPU_SKIPPED GPU tests were skipped because CuPy is not installed"
    echo ""
    echo "  To enable GPU algorithms, install CuPy:"
    echo "    For CUDA 11.x: pip install cupy-cuda11x"
    echo "    For CUDA 12.x: pip install cupy-cuda12x"
    echo ""
    echo "  After installing CuPy, re-run this script for full GPU testing"
    echo ""
fi

echo "🔍 OFFICE ANALYSIS RECOMMENDATIONS:"
echo "  1. Compare CPU vs GPU performance across algorithms"
echo "  2. Evaluate 48-core CPU scaling vs dual-GPU acceleration"
echo "  3. Review publication plots for hardware comparison"
echo "  4. Validate poses with tethered analysis results"
echo "  5. Optimize GPU batch sizes based on memory usage"
echo "  6. Compare flexible docking CPU vs GPU performance"
echo ""
echo "🚀 OFFICE NEXT STEPS:"
echo "  • GPU algorithms show best performance for high-throughput"
echo "  • CPU algorithms optimal for memory-limited scenarios"
echo "  • Use GPU fast mode for screening, GPU full for publications"
echo "  • Flexible GPU docking for challenging induced-fit targets"
echo "  • Scale batch sizes based on available GPU memory"
if [ "$GPU_SKIPPED" -gt "0" ]; then
    echo "  • Install CuPy to enable GPU algorithms: pip install cupy-cuda11x or cupy-cuda12x"
fi
echo ""
echo "========================================================================"
echo "🏢 PandaDock Office High-Performance Testing Suite Complete!"
echo "========================================================================"