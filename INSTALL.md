# PandaDock Installation Guide

This guide provides detailed instructions for installing PandaDock on various platforms.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Install (CPU Only)](#quick-install-cpu-only)
3. [GPU-Accelerated Installation](#gpu-accelerated-installation)
4. [Platform-Specific Instructions](#platform-specific-instructions)
5. [Optional Dependencies](#optional-dependencies)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements (CPU Only)

- **Operating System**: Linux, macOS, or Windows (WSL2)
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk Space**: 5 GB for installation, additional space for data
- **CPU**: Multi-core processor recommended (4+ cores)

### Recommended Requirements (GPU Acceleration)

- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal or newer)
- **CUDA**: Version 11.0 or higher
- **GPU Memory**: 4 GB minimum, 8+ GB recommended
- **NVIDIA Driver**: Latest stable version
- **RAM**: 16 GB or more

### Supported Platforms

| Platform | CPU Support | GPU Support | Status |
|----------|-------------|-------------|--------|
| Ubuntu 18.04+ | ✅ | ✅ | Fully tested |
| CentOS 7+ | ✅ | ✅ | Fully tested |
| macOS 10.14+ | ✅ | ❌ | CPU only |
| Windows 10+ (WSL2) | ✅ | ✅ | Tested |
| Red Hat 8+ | ✅ | ✅ | Tested |

---

## Quick Install (CPU Only)

The simplest way to install PandaDock for CPU-only usage:

### Step 1: Clone Repository

```bash
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv pandadock_env
source pandadock_env/bin/activate  # On Windows: pandadock_env\Scripts\activate

# Or using conda
conda create -n pandadock python=3.9
conda activate pandadock
```

### Step 3: Install PandaDock

```bash
pip install -e .
```

### Step 4: Verify Installation

```bash
pandadock --version
pandadock list-algorithms
```

You should see:
```
PandaDock v1.0.0 - Pritam Kumar Panda @ Stanford University
```

---

## GPU-Accelerated Installation

For maximum performance with GPU acceleration:

### Prerequisites

1. **Install NVIDIA Driver**

Check if NVIDIA driver is installed:
```bash
nvidia-smi
```

If not installed, follow [NVIDIA Driver Installation](https://www.nvidia.com/Download/index.aspx).

2. **Install CUDA Toolkit**

Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

For Ubuntu:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda
```

Verify CUDA:
```bash
nvcc --version
```

### Install PandaDock with GPU Support

#### Method 1: Using pip (Recommended)

```bash
# Clone repository
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock

# Create environment
conda create -n pandadock python=3.9
conda activate pandadock

# Install PandaDock
pip install -e .

# Install CuPy for GPU acceleration
# For CUDA 11.x:
pip install cupy-cuda11x

# For CUDA 12.x:
pip install cupy-cuda12x
```

#### Method 2: Using conda (Alternative)

```bash
conda create -n pandadock python=3.9
conda activate pandadock

# Install CUDA dependencies via conda
conda install -c conda-forge cudatoolkit=11.8

# Install PandaDock
cd PandaDock
pip install -e .
pip install cupy-cuda11x
```

### Verify GPU Installation

```bash
pandadock list-algorithms
```

You should see GPU algorithms listed:
```
GPU Algorithms:
  - cuda_monte_carlo
  - cuda_genetic_algorithm
  - enhanced_hierarchical_gpu
```

Test GPU functionality:
```bash
pandadock dock -r examples/protein.pdb -l examples/ligand.sdf \
               --center 0 0 0 --box 20 20 20 \
               --algorithm cuda_monte_carlo \
               --gpu
```

---

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    build-essential \
    git \
    wget

# Install PandaDock
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock
pip install -e .
```

### CentOS/RHEL

```bash
# Install system dependencies
sudo yum install -y \
    python39 \
    python39-devel \
    gcc \
    gcc-c++ \
    git \
    wget

# Install PandaDock
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock
pip3 install -e .
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.9 git

# Install PandaDock
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock
pip3 install -e .
```

**Note**: macOS does not support GPU acceleration (CUDA is not available for macOS).

### Windows (WSL2)

1. **Install WSL2**:
   - Follow [Microsoft's WSL2 Installation Guide](https://docs.microsoft.com/en-us/windows/wsl/install)
   - Install Ubuntu 20.04 or 22.04 from Microsoft Store

2. **Install NVIDIA CUDA on WSL2** (for GPU support):
   - Follow [NVIDIA CUDA on WSL2 Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

3. **Install PandaDock**:
```bash
# Inside WSL2 terminal
sudo apt-get update
sudo apt-get install python3-pip git
git clone https://github.com/pritampanda15/PandaDock.git
cd PandaDock
pip install -e .
```

---

## Optional Dependencies

### For ML-Enhanced Docking

```bash
pip install -e ".[ml]"
```

This installs:
- PyTorch (for machine learning models)
- h5py (for data storage)
- Additional ML dependencies

### For Flexible Docking (with OpenMM)

```bash
conda install -c conda-forge openmm pdbfixer
pip install -e ".[conda]"
```

This enables:
- Receptor flexibility
- Energy minimization
- Molecular dynamics refinement

### For Advanced Visualization

```bash
pip install pymol-open-source  # Optional: PyMOL integration
pip install py3Dmol            # Optional: 3D visualization in notebooks
```

---

## Verification

### Test CPU Algorithms

```bash
# Run a quick test
cd PandaDock
pandadock dock -r 6x3v.pdb -l etomidate.sdf \
               --center 0 0 0 --box 20 20 20 \
               --algorithm monte_carlo_cpu \
               --fast \
               -o test_output/
```

Expected output:
```
PandaDock Enhanced Molecular Docking
Receptor: 6x3v.pdb
Ligand: etomidate.sdf
Algorithm: monte_carlo_cpu
Scoring: physics_based
...
Docking completed in XX.XX seconds
Generated N poses
```

### Test GPU Algorithms (if GPU installed)

```bash
pandadock dock -r 6x3v.pdb -l etomidate.sdf \
               --center 0 0 0 --box 20 20 20 \
               --algorithm enhanced_hierarchical_gpu \
               --gpu \
               -o gpu_test_output/
```

### Run Test Suite

```bash
# CPU tests
cd cpu_comprehensive_testing_fixed
bash run_tests.sh

# GPU tests (if GPU available)
cd ../gpu_comprehensive_testing
bash run_tests.sh
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error: No module named 'pandadock'

**Solution**: Ensure you're in the correct environment and PandaDock is installed:
```bash
pip install -e .
```

#### 2. CUDA/GPU Errors

**Problem**: `Error: GPU acceleration requested but CUDA not available`

**Solution**:
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall CuPy with correct CUDA version
pip uninstall cupy-cuda11x cupy-cuda12x
pip install cupy-cuda11x  # Match your CUDA version
```

**Problem**: `CuPy initialization failed`

**Solution**:
```bash
# Check CUDA compatibility
python -c "import cupy as cp; print(cp.cuda.runtime.runtimeGetVersion())"

# If mismatch, reinstall matching version
pip install cupy-cuda12x  # for CUDA 12
```

#### 3. RDKit Import Errors

**Problem**: `ImportError: cannot import name 'Chem' from 'rdkit'`

**Solution**:
```bash
pip uninstall rdkit
pip install rdkit>=2022.9.5
```

#### 4. OpenMM/Flexible Docking Issues

**Problem**: `ModuleNotFoundError: No module named 'openmm'`

**Solution**: OpenMM must be installed via conda:
```bash
conda install -c conda-forge openmm pdbfixer
```

#### 5. Permission Denied Errors

**Problem**: Permission errors when installing

**Solution**: Use virtual environment or `--user` flag:
```bash
pip install --user -e .
```

#### 6. Memory Errors During Docking

**Problem**: `MemoryError` or system runs out of RAM

**Solution**: Reduce batch size or number of poses:
```bash
pandadock dock ... --num-poses 10 --gpu-batch-size 500
```

#### 7. Slow Performance on CPU

**Problem**: Docking takes very long on CPU

**Solution**: Use parallel processing:
```bash
pandadock dock ... --cpuworkers 8  # Use 8 CPU cores
```

Or consider fast mode:
```bash
pandadock dock ... --fast
```

---

## Advanced Configuration

### Environment Variables

```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Set OpenMP threads
export OMP_NUM_THREADS=16

# Set temporary directory
export TMPDIR=/path/to/large/tmp
```

### Configuration File

Create `~/.pandadock/config.json`:
```json
{
  "default_algorithm": "enhanced_hierarchical_cpu",
  "default_scoring": "physics_based",
  "gpu_memory_limit": 4.0,
  "cpu_workers": 8,
  "temp_dir": "/tmp/pandadock"
}
```

---

## Updating PandaDock

### Update from GitHub

```bash
cd PandaDock
git pull origin main
pip install -e .  # Reinstall with updates
```

### Check for Updates

```bash
pandadock --version
```

---

## Uninstallation

### Complete Removal

```bash
# Uninstall PandaDock
pip uninstall pandadock

# Remove environment
conda deactivate
conda env remove -n pandadock

# Remove repository
rm -rf PandaDock
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check Documentation**: [Full Documentation](https://pandadock.readthedocs.io)
2. **Search Issues**: [GitHub Issues](https://github.com/pritampanda15/PandaDock/issues)
3. **Ask Questions**: [GitHub Discussions](https://github.com/pritampanda15/PandaDock/discussions)
4. **Email Support**: pritampanda@stanford.edu

When reporting issues, please include:
- Operating system and version
- Python version (`python --version`)
- PandaDock version (`pandadock --version`)
- Full error message
- Steps to reproduce

---

## Next Steps

After successful installation:
1. Read the [Quick Start Guide](README.md#quick-start)
2. Try the [Tutorial](docs/tutorial.md)
3. Explore [Examples](examples/)
4. Review [Algorithm Documentation](ALGORITHMS.md)

---

## System Testing

### Comprehensive System Test

Run this script to test all components:

```bash
#!/bin/bash
echo "Testing PandaDock Installation"

echo "1. Testing CPU algorithms..."
pandadock dock -r 6x3v.pdb -l etomidate.sdf \
               --center 0 0 0 --box 20 20 20 \
               --algorithm monte_carlo_cpu --fast \
               -o test_cpu/

echo "2. Testing GPU algorithms..."
if pandadock list-algorithms | grep -q "cuda_monte_carlo"; then
    pandadock dock -r 6x3v.pdb -l etomidate.sdf \
                   --center 0 0 0 --box 20 20 20 \
                   --algorithm cuda_monte_carlo --gpu \
                   -o test_gpu/
    echo "GPU test PASSED"
else
    echo "GPU algorithms not available (CPU-only installation)"
fi

echo "3. Testing specialized modes..."
pandadock-flex --help
pandadock-metal --help
pandadock-tethered --help

echo "Installation test completed!"
```

Save as `test_installation.sh`, make executable, and run:
```bash
chmod +x test_installation.sh
./test_installation.sh
```

---

## Performance Optimization

### CPU Optimization

```bash
# Use all available cores
export OMP_NUM_THREADS=$(nproc)

# Run with parallel workers
pandadock dock ... --cpuworkers $(nproc)
```

### GPU Optimization

```bash
# Set optimal batch size based on GPU memory
# For 8GB GPU:
pandadock dock ... --gpu-batch-size 2000 --gpu-memory-limit 6.0

# For 4GB GPU:
pandadock dock ... --gpu-batch-size 1000 --gpu-memory-limit 3.0
```

### Storage Optimization

```bash
# Use fast SSD for temporary files
export TMPDIR=/path/to/fast/ssd

# Limit output files
pandadock dock ... --num-poses 10  # Generate fewer pose files
```

---

## Docker Installation (Coming Soon)

PandaDock Docker containers will be available soon for easy deployment:

```bash
# Pull CPU image
docker pull pritampanda15/pandadock:latest

# Pull GPU image
docker pull pritampanda15/pandadock:gpu

# Run container
docker run -v $(pwd):/data pritampanda15/pandadock:latest \
    pandadock dock -r /data/protein.pdb -l /data/ligand.sdf \
    --center 0 0 0 --box 20 20 20 -o /data/results/
```

---

## Citation

If you use PandaDock in your research, please cite:

```bibtex
@software{panda2024pandadock,
  author = {Panda, Pritam Kumar},
  title = {PandaDock: Next-Generation Molecular Docking Suite},
  year = {2024},
  url = {https://github.com/pritampanda15/PandaDock}
}
```

---

## License

PandaDock is released under the MIT License. See [LICENSE](LICENSE) for details.
