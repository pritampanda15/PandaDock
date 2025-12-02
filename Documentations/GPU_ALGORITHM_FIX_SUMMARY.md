# GPU Algorithm Signature Fix Required

## Problem
GPU algorithms have different `dock()` signatures than CPU algorithms:
- **CPU**: `dock(receptor_file: str, ligand_mol, grid_center, grid_dimensions, **kwargs)`
- **GPU**: `dock(ligand_coords, receptor_coords, receptor_charges, grid_center, grid_dimensions, **kwargs)`

This causes: `missing 1 required positional argument: 'grid_dimensions'`

## Solution Applied

### ✅ Fixed: `genetic_algorithm_gpu.py`
- Added wrapper `dock()` method that accepts CPU-style parameters
- Extracts coordinates and calls internal `_dock_internal()` method
- Now compatible with engine calling convention

### ⚠️ Still Need to Fix:
1. **`monte_carlo_gpu.py`** - Line 255
2. **`enhanced_hierarchical_gpu.py`** - Check dock method

## Quick Fix for Your Office Laptop

Run this Python script to see which algorithms are working:

```python
from pandadock.docking_cli import engine

print("Registered algorithms:")
for name, alg in engine._algorithms.items():
    # Check signature
    import inspect
    sig = inspect.signature(alg.dock)
    params = list(sig.parameters.keys())
    print(f"  {name}: {params[:4]}")  # First 4 params
```

## Temporary Workaround

Until all GPU algorithms are fixed, use only:
- ✅ `enhanced_hierarchical_gpu` (likely working)
- ✅ `cuda_genetic_algorithm` (FIXED)
- ❌ `cuda_monte_carlo` (needs fixing)

Or use CPU algorithms which all work correctly.