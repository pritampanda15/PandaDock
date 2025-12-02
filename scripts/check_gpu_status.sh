#!/bin/bash

# PandaDock GPU Status Checker
# Diagnoses GPU availability and provides installation guidance

echo "========================================================================"
echo "🔍 PandaDock GPU Status Diagnostic Tool"
echo "========================================================================"
echo ""

# Check 1: CUDA Runtime
echo "1️⃣  Checking CUDA Runtime..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ nvidia-smi found"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "   ⚠️  nvidia-smi exists but failed to query GPU"
else
    echo "   ❌ nvidia-smi not found - CUDA drivers not installed or not in PATH"
fi
echo ""

# Check 2: CUDA Compiler
echo "2️⃣  Checking CUDA Compiler..."
if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
    echo "   ✅ nvcc found - CUDA version: $nvcc_version"
else
    echo "   ❌ nvcc not found - CUDA toolkit not installed or not in PATH"
fi
echo ""

# Check 3: Python CUDA Libraries
echo "3️⃣  Checking Python CUDA Libraries..."

# Check CuPy
echo "   📦 CuPy:"
python3 -c "
import cupy as cp
print(f'      ✅ CuPy version: {cp.__version__}')
print(f'      ✅ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}')
print(f'      ✅ GPU count: {cp.cuda.runtime.getDeviceCount()}')
for i in range(cp.cuda.runtime.getDeviceCount()):
    device = cp.cuda.Device(i)
    print(f'      ✅ GPU {i}: {device.compute_capability}')
" 2>/dev/null || echo "      ❌ CuPy not installed"

# Check PyTorch
echo "   📦 PyTorch:"
python3 -c "
import torch
print(f'      ✅ PyTorch version: {torch.__version__}')
print(f'      ✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'      ✅ CUDA version: {torch.version.cuda}')
    print(f'      ✅ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'      ✅ GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "      ❌ PyTorch not installed"

echo ""

# Check 4: PandaDock GPU Algorithm Registration
echo "4️⃣  Checking PandaDock GPU Algorithm Registration..."
PYTHONPATH=. python3 -c "
from pandadock.docking.algorithms import GPU_AVAILABLE
from pandadock.docking_cli import GPU_ALGORITHMS_REGISTERED, engine

print(f'   GPU_AVAILABLE flag: {GPU_AVAILABLE}')
print(f'   GPU_ALGORITHMS_REGISTERED: {GPU_ALGORITHMS_REGISTERED}')
print(f'')
print(f'   Available algorithms: {list(engine._algorithms.keys())}')
print(f'   Available scoring: {list(engine._scoring_functions.keys())}')
" 2>/dev/null

echo ""

# Check 5: Test GPU Algorithm Instantiation
echo "5️⃣  Testing GPU Algorithm Instantiation..."
PYTHONPATH=. python3 -c "
try:
    from pandadock.docking.algorithms.monte_carlo_gpu import CUDAMonteCarloDocker
    print('   ✅ CUDAMonteCarloDocker import successful')
    try:
        docker = CUDAMonteCarloDocker()
        print(f'   ✅ CUDAMonteCarloDocker instantiated: {docker.name}')
    except Exception as e:
        print(f'   ❌ Instantiation failed: {e}')
except ImportError as e:
    print(f'   ❌ Import failed: {e}')
" 2>/dev/null

echo ""
echo "========================================================================"
echo "📋 DIAGNOSIS & RECOMMENDATIONS"
echo "========================================================================"
echo ""

# Provide recommendations based on checks
python3 << 'EOF'
import subprocess
import sys

def check_command(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return True
    except:
        return False

def check_module(module):
    try:
        __import__(module)
        return True
    except:
        return False

has_nvidia_smi = check_command("nvidia-smi > /dev/null 2>&1")
has_nvcc = check_command("nvcc --version > /dev/null 2>&1")
has_cupy = check_module("cupy")
has_torch = check_module("torch")

if has_torch:
    import torch
    torch_cuda = torch.cuda.is_available()
else:
    torch_cuda = False

print("Current Status:")
print(f"  • CUDA Drivers: {'✅ Installed' if has_nvidia_smi else '❌ Not found'}")
print(f"  • CUDA Toolkit: {'✅ Installed' if has_nvcc else '❌ Not found'}")
print(f"  • CuPy: {'✅ Installed' if has_cupy else '❌ Not installed'}")
print(f"  • PyTorch CUDA: {'✅ Available' if torch_cuda else '❌ Not available'}")
print()

if not has_nvidia_smi:
    print("⚠️  CRITICAL: CUDA drivers not detected")
    print("   This system may not have an NVIDIA GPU or drivers are not installed")
    print()
    print("   Solutions:")
    print("   1. Install NVIDIA drivers for your GPU")
    print("   2. Run this script on a system with NVIDIA GPU")
    print()

if has_nvidia_smi and not has_cupy:
    print("⚠️  GPU DOCKING ALGORITHMS DISABLED")
    print("   CuPy is required for GPU docking algorithms:")
    print("   - cuda_monte_carlo")
    print("   - cuda_genetic_algorithm")
    print("   - enhanced_hierarchical_gpu")
    print()
    print("   Installation:")

    if has_nvcc:
        # Try to detect CUDA version
        try:
            result = subprocess.run("nvcc --version | grep release | sed 's/.*release \\([0-9]*\\)\\.\\([0-9]*\\).*/\\1\\2/'",
                                  shell=True, capture_output=True, text=True)
            cuda_ver = result.stdout.strip()[:2]
            print(f"   Detected CUDA {cuda_ver}.x - install with:")
            print(f"     pip install cupy-cuda{cuda_ver}x")
        except:
            print("   For CUDA 11.x: pip install cupy-cuda11x")
            print("   For CUDA 12.x: pip install cupy-cuda12x")
    else:
        print("   First install CUDA toolkit, then:")
        print("   For CUDA 11.x: pip install cupy-cuda11x")
        print("   For CUDA 12.x: pip install cupy-cuda12x")
    print()

if has_cupy:
    try:
        import cupy as cp
        from pandadock.docking_cli import GPU_ALGORITHMS_REGISTERED

        if not GPU_ALGORITHMS_REGISTERED:
            print("⚠️  UNEXPECTED: CuPy installed but GPU algorithms not registered")
            print("   This may indicate a registration error in docking_cli.py")
            print()
            print("   Try running:")
            print("   PYTHONPATH=. python3 -m pandadock.docking_cli list-algorithms")
            print()
        else:
            print("✅ GPU DOCKING ALGORITHMS READY")
            print("   All GPU algorithms are registered and available")
            print()
    except Exception as e:
        print(f"⚠️  Error checking GPU algorithms: {e}")
        print()

if torch_cuda:
    print("✅ FLEXIBLE GPU DOCKING READY")
    print("   PyTorch CUDA is available for flexible docking")
    print()

if not has_cupy and not torch_cuda:
    print("🔴 NO GPU ACCELERATION AVAILABLE")
    print("   Only CPU algorithms can be used")
    print()
    print("   To enable GPU features:")
    print("   1. Install CuPy for GPU docking: pip install cupy-cuda12x")
    print("   2. Install PyTorch with CUDA for Flex GPU:")
    print("      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print()

print("For more help: cat GPU_SETUP.md")
EOF

echo ""
echo "========================================================================"