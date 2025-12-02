# PandaDock GPU Algorithm Setup Guide

## Current Status

Your system has:
- ✅ CUDA installed
- ✅ PyTorch with CUDA support
- ❌ CuPy **not installed** (required for GPU algorithms)

## GPU Algorithms in PandaDock

PandaDock includes three GPU-accelerated algorithms:
1. **cuda_monte_carlo** - CUDA-accelerated Monte Carlo docking
2. **cuda_genetic_algorithm** - GPU-parallel genetic algorithm
3. **enhanced_hierarchical_gpu** - GPU-accelerated hierarchical search

These provide **50-100x speedup** over CPU algorithms for large-scale screening.

## Installation

### Step 1: Check Your CUDA Version

```bash
nvcc --version
# or
nvidia-smi
```

### Step 2: Install CuPy

**For CUDA 11.x:**
```bash
pip install cupy-cuda11x
```

**For CUDA 12.x:**
```bash
pip install cupy-cuda12x
```

**Alternative (build from source if pip fails):**
```bash
pip install cupy
```

### Step 3: Verify Installation

```bash
python3 -c "import cupy; print(f'CuPy {cupy.__version__} installed successfully')"
```

### Step 4: Test GPU Algorithms

```bash
PYTHONPATH=. python3 -m pandadock.docking_cli list-algorithms
```

You should now see:
```
GPU Algorithms:
  - cuda_monte_carlo
  - cuda_genetic_algorithm
  - enhanced_hierarchical_gpu
```

### Step 5: Run GPU Docking Test

```bash
PYTHONPATH=. python3 -m pandadock.docking_cli dock \
  -r preprocessing/intermediate/receptor_processed.pdb \
  -l preprocessing/intermediate/ligand_processed.sdf \
  -a cuda_monte_carlo \
  --gpu \
  --center 129.2 120.2 145.2 \
  --box 20 20 20 \
  -o test_gpu_docking
```

## Troubleshooting

### ImportError: libcuda.so.1: cannot open shared object file

Your CUDA drivers are not properly installed or not in the library path:
```bash
# Add CUDA to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CuPy version mismatch

If you get version mismatch errors, reinstall matching your CUDA version:
```bash
pip uninstall cupy cupy-cuda11x cupy-cuda12x
# Then install the correct version for your CUDA
pip install cupy-cuda11x  # or cupy-cuda12x
```

### Out of GPU memory

Reduce batch size or number of poses:
```bash
--gpu-batch-size 512 \
--num-poses 10
```

## Performance Tips

1. **Use GPU for screening**: Fast mode with many ligands
2. **Use CPU for accuracy**: Full mode with single high-value targets
3. **Monitor GPU memory**: Use `nvidia-smi` to check usage
4. **Batch size tuning**: Larger batches = better GPU utilization

## Re-run Comprehensive Testing

After installing CuPy, re-run the office testing script:
```bash
bash scripts/comprehensive_algorithms_office.sh
```

GPU tests will now be included instead of being skipped.

## Need Help?

- Check CUDA installation: `nvcc --version`
- Check GPU availability: `nvidia-smi`
- Test PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Test CuPy: `python3 -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"`