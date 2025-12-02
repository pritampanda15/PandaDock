#!/usr/bin/env python3
"""
PandaDock GPU Diagnostic Script for Office Laptop
Run this on your office laptop to diagnose GPU algorithm registration issues
"""

import sys

print("=" * 80)
print("🔍 PandaDock GPU Diagnostic Tool - Office Laptop")
print("=" * 80)
print()

# Test 1: Check CuPy Installation
print("1️⃣  Testing CuPy Installation...")
try:
    import cupy as cp
    print(f"   ✅ CuPy version: {cp.__version__}")
    print(f"   ✅ CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
    cupy_installed = True
except ImportError as e:
    print(f"   ❌ CuPy not installed: {e}")
    cupy_installed = False
except Exception as e:
    print(f"   ❌ CuPy import error: {e}")
    cupy_installed = False
print()

# Test 2: Check GPU Device Count
if cupy_installed:
    print("2️⃣  Checking GPU Devices...")
    try:
        num_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"   ✅ Number of GPUs detected: {num_gpus}")

        for i in range(num_gpus):
            device = cp.cuda.Device(i)
            with device:
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"   ✅ GPU {i}: Compute Capability {device.compute_capability}")
                print(f"      - Total Memory: {props['totalGlobalMem'] / 1024**3:.2f} GB")
    except Exception as e:
        print(f"   ❌ Failed to query GPU devices: {e}")
    print()

# Test 3: Test CuPy Array Operations
if cupy_installed:
    print("3️⃣  Testing CuPy Array Operations...")
    try:
        # Create test array
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        print(f"   ✅ Array creation: {test_array}")
        print(f"   ✅ Array sum: {result}")

        # Test GPU memory
        test_large = cp.random.rand(1000, 1000)
        mem_info = cp.cuda.Device().mem_info
        print(f"   ✅ GPU memory free: {mem_info[0] / 1024**3:.2f} GB")
        print(f"   ✅ GPU memory total: {mem_info[1] / 1024**3:.2f} GB")

        del test_array, result, test_large
        print("   ✅ CuPy operations working correctly")
    except Exception as e:
        print(f"   ❌ CuPy operation failed: {e}")
        import traceback
        traceback.print_exc()
    print()

# Test 4: Check PandaDock GPU Module Availability
print("4️⃣  Checking PandaDock GPU Module Flags...")
try:
    from pandadock.docking.algorithms import GPU_AVAILABLE
    print(f"   GPU_AVAILABLE flag: {GPU_AVAILABLE}")

    from pandadock.docking.algorithms.monte_carlo_gpu import CUPY_AVAILABLE, NUMBA_CUDA_AVAILABLE, PYCUDA_AVAILABLE
    print(f"   CUPY_AVAILABLE: {CUPY_AVAILABLE}")
    print(f"   NUMBA_CUDA_AVAILABLE: {NUMBA_CUDA_AVAILABLE}")
    print(f"   PYCUDA_AVAILABLE: {PYCUDA_AVAILABLE}")
except Exception as e:
    print(f"   ❌ Failed to import PandaDock modules: {e}")
print()

# Test 5: Test GPU Algorithm Import
print("5️⃣  Testing GPU Algorithm Imports...")
try:
    from pandadock.docking.algorithms.monte_carlo_gpu import CUDAMonteCarloDocker
    print("   ✅ CUDAMonteCarloDocker import successful")
except ImportError as e:
    print(f"   ❌ CUDAMonteCarloDocker import failed: {e}")

try:
    from pandadock.docking.algorithms.genetic_algorithm_gpu import CUDAGeneticAlgorithmDocker
    print("   ✅ CUDAGeneticAlgorithmDocker import successful")
except ImportError as e:
    print(f"   ❌ CUDAGeneticAlgorithmDocker import failed: {e}")

try:
    from pandadock.docking.algorithms.enhanced_hierarchical_gpu import EnhancedHierarchicalGPUDocker
    print("   ✅ EnhancedHierarchicalGPUDocker import successful")
except ImportError as e:
    print(f"   ❌ EnhancedHierarchicalGPUDocker import failed: {e}")
print()

# Test 6: Test GPU Algorithm Instantiation
print("6️⃣  Testing GPU Algorithm Instantiation...")
instantiation_errors = []

try:
    from pandadock.docking.algorithms.monte_carlo_gpu import CUDAMonteCarloDocker
    docker = CUDAMonteCarloDocker()
    print(f"   ✅ CUDAMonteCarloDocker instantiated: {docker.name}")
except Exception as e:
    print(f"   ❌ CUDAMonteCarloDocker instantiation failed:")
    print(f"      Error: {e}")
    instantiation_errors.append(("CUDAMonteCarloDocker", e))

try:
    from pandadock.docking.algorithms.genetic_algorithm_gpu import CUDAGeneticAlgorithmDocker
    docker = CUDAGeneticAlgorithmDocker()
    print(f"   ✅ CUDAGeneticAlgorithmDocker instantiated: {docker.name}")
except Exception as e:
    print(f"   ❌ CUDAGeneticAlgorithmDocker instantiation failed:")
    print(f"      Error: {e}")
    instantiation_errors.append(("CUDAGeneticAlgorithmDocker", e))

try:
    from pandadock.docking.algorithms.enhanced_hierarchical_gpu import EnhancedHierarchicalGPUDocker
    docker = EnhancedHierarchicalGPUDocker()
    print(f"   ✅ EnhancedHierarchicalGPUDocker instantiated: {docker.name}")
except Exception as e:
    print(f"   ❌ EnhancedHierarchicalGPUDocker instantiation failed:")
    print(f"      Error: {e}")
    instantiation_errors.append(("EnhancedHierarchicalGPUDocker", e))
print()

# Test 7: Check Algorithm Registration
print("7️⃣  Checking PandaDock Algorithm Registration...")
try:
    from pandadock.docking_cli import engine, GPU_ALGORITHMS_REGISTERED

    print(f"   GPU_ALGORITHMS_REGISTERED: {GPU_ALGORITHMS_REGISTERED}")
    print(f"   Registered algorithms: {list(engine._algorithms.keys())}")
    print()

    if 'cuda_monte_carlo' in engine._algorithms:
        print("   ✅ cuda_monte_carlo is registered")
    else:
        print("   ❌ cuda_monte_carlo is NOT registered")

    if 'cuda_genetic_algorithm' in engine._algorithms:
        print("   ✅ cuda_genetic_algorithm is registered")
    else:
        print("   ❌ cuda_genetic_algorithm is NOT registered")

    if 'enhanced_hierarchical_gpu' in engine._algorithms:
        print("   ✅ enhanced_hierarchical_gpu is registered")
    else:
        print("   ❌ enhanced_hierarchical_gpu is NOT registered")

except Exception as e:
    print(f"   ❌ Failed to check registration: {e}")
    import traceback
    traceback.print_exc()
print()

# Summary and Recommendations
print("=" * 80)
print("📋 DIAGNOSIS SUMMARY")
print("=" * 80)
print()

if cupy_installed and len(instantiation_errors) == 0:
    print("✅ ALL CHECKS PASSED")
    print("   GPU algorithms should be working correctly")
    print()
elif cupy_installed and len(instantiation_errors) > 0:
    print("⚠️  CUPY INSTALLED BUT ALGORITHMS FAILING TO INSTANTIATE")
    print()
    print("   Common causes:")
    print("   1. GPU memory issue - try closing other GPU applications")
    print("   2. CUDA driver mismatch - ensure CuPy CUDA version matches installed CUDA")
    print("   3. Permissions issue - ensure user has GPU access")
    print()
    print("   Detailed errors:")
    for algo, error in instantiation_errors:
        print(f"   • {algo}: {error}")
    print()
    print("   Try:")
    print("   - Check nvidia-smi to see GPU status")
    print("   - Reinstall CuPy: pip uninstall cupy cupy-cuda12x && pip install cupy-cuda12x")
    print("   - Check CUDA version: nvcc --version")
    print()
else:
    print("❌ CUPY NOT INSTALLED OR NOT WORKING")
    print()
    print("   Install CuPy:")
    print("   For CUDA 12.x: pip install cupy-cuda12x")
    print("   For CUDA 11.x: pip install cupy-cuda11x")
    print()

print("=" * 80)