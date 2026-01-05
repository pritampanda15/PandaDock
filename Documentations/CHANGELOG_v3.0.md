# PandaDock v3.0 - Intelligent Algorithm Selection

## 🎯 Major Updates

### 1. Intelligent Algorithm Auto-Selection ⭐ NEW

PandaDock now automatically selects the optimal docking algorithm based on your workload characteristics. No more guessing which algorithm to use!

**How it works:**
```bash
# Simple command - PandaDock handles the rest
pandadock dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20
```

**Auto-selection logic:**

| Scenario | Selected Algorithm | Rationale |
|----------|-------------------|-----------|
| Single ligand, standard use | `enhanced_hierarchical_cpu` | 0.30s, 0.014Å RMSD, 100% success |
| Batch processing (1000+ ligands) + GPU | `enhanced_hierarchical_gpu` | Optimal batch throughput |
| Large conformer libraries (100+) + GPU | `cuda_genetic_algorithm` | GPU parallelizes conformer evaluation |
| Multiple ligands, no GPU | `enhanced_hierarchical_cpu` | Best CPU performance |

**Features:**
- ✅ **Automatic workload detection**: Analyzes ligand input (single file vs directory vs multi-mol SDF)
- ✅ **GPU-aware**: Automatically uses GPU algorithms only when beneficial
- ✅ **Runtime warnings**: Alerts users about problematic algorithms (e.g., cuda_monte_carlo 48.7% success rate)
- ✅ **Performance recommendations**: Suggests optimizations based on workload
- ✅ **Backward compatible**: Manual algorithm selection still works with `--algorithm` flag

**Example output:**
```
🤖 Auto-selected because:
   Best accuracy and speed for single/small-scale docking
   0.30s per ligand, 0.014Å RMSD, 100% success rate

🔧 Algorithm: enhanced_hierarchical_cpu
   Expected time: ~0.30s per ligand
   Expected accuracy: 0.014Å RMSD
   Success rate: 100.0%
```

---

### 2. Python 3.9 Compatibility Fixed ✅

**Issue:** GitHub Actions CI was failing for Python 3.9 with:
```
ERROR: No matching distribution found for rdkit>=2022.9.5
```

**Solution:** Updated `setup.py` to use `rdkit>=2022.3.1` which is compatible with Python 3.9+

**Impact:**
- ✅ CI/CD now passes for Python 3.9, 3.10, 3.11, 3.12
- ✅ Broader compatibility for users on older Python versions
- ✅ Maintains all RDKit functionality needed by PandaDock

---

### 3. Enhanced Documentation 📚

**Updated README.md with:**

1. **Transparent GPU Performance Claims**
   - Changed "100x speedup" to "up to 100x speedup for batch processing"
   - Added clear notes: CPU is faster for single ligands (0.30s vs 0.82s)
   - Specified GPU sweet spots: batch processing, large conformer libraries

2. **Algorithm Selection Guide**
   - New section explaining intelligent auto-selection
   - Clear examples of auto vs manual selection
   - Performance characteristics table with benchmark data

3. **Runtime Warnings**
   - Warning for `cuda_monte_carlo` (48.7% success rate)
   - Notes about superseded algorithms (e.g., `hierarchical_cpu`)
   - GPU availability checks and recommendations

4. **Improved Quick Start**
   - Default command now uses auto-selection
   - Clear explanation of what happens under the hood
   - Examples for common use cases

---

## 🔧 Technical Implementation

### New Module: `algorithm_selector.py`

**Location:** `pandadock/docking/algorithm_selector.py`

**Key Components:**

1. **`AlgorithmSelector` class**
   - `auto_select()`: Main auto-selection logic
   - `get_algorithm_warning()`: Runtime warnings for problematic algorithms
   - `get_recommendation()`: Performance hints for selected algorithm
   - `explain_selection()`: User-friendly explanation of auto-selection

2. **`detect_batch_mode()` function**
   - Detects single file vs directory vs multi-mol SDF
   - Returns (num_ligands, is_batch_mode)

3. **Algorithm Performance Profiles**
   - Benchmark data embedded in code
   - Speed, accuracy, success rate, GPU requirement
   - Best-use-case classification

### CLI Integration

**Modified:** `pandadock/docking_cli.py`

**Changes:**
1. Added `'auto'` to algorithm choices (now default)
2. Integrated auto-selection logic before docking
3. Added runtime warnings for manually selected algorithms
4. Shows explanation of auto-selection when used

---

## 📊 Algorithm Performance Summary

Based on **150 diverse protein-ligand complexes** from PDBbind v2020:

### Recommended for Production ✅

| Algorithm | Speed | Accuracy | Success Rate | Use Case |
|-----------|-------|----------|--------------|----------|
| **enhanced_hierarchical_cpu** | 0.30s | 0.014Å | 100% | Single ligand, default |
| **enhanced_hierarchical_gpu** | 0.82s | 0.015Å | 91.3% | Batch processing (1000+) |
| **cuda_genetic_algorithm** | 35.24s | 0.014Å | 100% | Large conformer libraries |
| **genetic_algorithm_cpu** | 4.84s | 2.246Å | 89.3% | Complex binding sites |

### Use with Caution ⚠️

| Algorithm | Issue | Recommendation |
|-----------|-------|----------------|
| **cuda_monte_carlo** | 48.7% success rate | Use enhanced_hierarchical_cpu instead |
| **hierarchical_cpu** | Superseded by enhanced version | Use enhanced_hierarchical_cpu (60x faster) |

---

## 🚀 User Impact

### Before v3.0:
```bash
# User had to know which algorithm to use
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               --algorithm enhanced_hierarchical_cpu  # ❓ Which one?
```

### After v3.0:
```bash
# Just run it - PandaDock figures out the rest
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20
# ✅ Auto-selects enhanced_hierarchical_cpu (best for single ligand)
```

### For Power Users:
```bash
# Still supports manual selection
pandadock dock -r protein.pdb -l ligand.sdf \
               --center 10 20 30 --box 20 20 20 \
               --algorithm cuda_genetic_algorithm

# Shows warning if problematic:
# ⚠️  WARNING: cuda_monte_carlo has 48.7% success rate...
```

---

## 🔄 Migration Guide

### No Changes Required!

**v2.x code still works:**
```bash
# Explicitly specifying algorithm (v2.x style)
pandadock dock ... --algorithm enhanced_hierarchical_cpu
```

**New v3.0 style:**
```bash
# Auto-selection (recommended)
pandadock dock ...
# Automatically uses best algorithm
```

**Opt-in to new features:**
- Remove `--algorithm` flag to use auto-selection
- System will explain why it chose each algorithm
- Override any time with `--algorithm <name>`

---

## 📝 Files Changed

### Modified Files:
1. **`setup.py`**
   - Changed `rdkit>=2022.9.5` → `rdkit>=2022.3.1` (Python 3.9 compatibility)

2. **`pandadock/docking_cli.py`**
   - Added `'auto'` algorithm option (default)
   - Integrated `AlgorithmSelector` for intelligent selection
   - Added runtime warnings and recommendations
   - Shows auto-selection explanations

3. **`README.md`**
   - Added "Algorithm Selection" section
   - Updated "Quick Start" examples
   - Added GPU performance notes
   - Clarified 100x speedup claims
   - Added warnings for experimental algorithms

### New Files:
1. **`pandadock/docking/algorithm_selector.py`**
   - Intelligent algorithm selection logic
   - Performance profiling database
   - Runtime warning system
   - Batch mode detection

---

## 🎓 Best Practices (Updated)

### For Most Users:
```bash
# Just use default (auto-selection)
pandadock dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20
```

### For Virtual Screening (1000+ ligands):
```bash
# Auto-selection will use GPU if available
pandadock dock -r protein.pdb -l ligands_directory/ --center 10 20 30 --box 20 20 20
```

### For High Accuracy:
```bash
# Auto-selection defaults to enhanced_hierarchical_cpu (best accuracy)
# No need to specify algorithm!
pandadock dock -r protein.pdb -l ligand.sdf --center 10 20 30 --box 20 20 20
```

### When to Override Auto-Selection:
- Validation studies: Use `--algorithm crystal_guided_cpu`
- Benchmarking: Specify exact algorithm for reproducibility
- Testing: Try different algorithms to compare

---

## 🐛 Bug Fixes

1. **Python 3.9 RDKit installation failure** (GitHub Actions CI)
   - Fixed: Updated RDKit version requirement to 2022.3.1

2. **Misleading GPU speedup claims**
   - Fixed: Updated README with accurate performance data
   - Added context: GPU faster only for batch processing

---

## 🔮 Future Enhancements

Potential additions for future versions:

1. **Machine Learning-based Selection**
   - Train ML model on ligand/protein characteristics
   - Predict optimal algorithm for specific protein family

2. **Conformer Count Detection**
   - Automatically detect conformer library size from input
   - Adjust algorithm selection accordingly

3. **Performance Profiling**
   - Track actual runtime vs predicted
   - Adaptive algorithm selection based on historical performance

4. **Multi-Algorithm Consensus**
   - Run 2-3 fast algorithms in parallel
   - Ensemble results for better accuracy

---

## 📊 Impact Summary

**Lines of Code:**
- Added: ~300 lines (`algorithm_selector.py`)
- Modified: ~50 lines (CLI integration)
- Documentation: ~150 lines (README updates)

**User Experience:**
- ✅ Simplified default usage (no algorithm selection needed)
- ✅ Smarter resource utilization (GPU only when beneficial)
- ✅ Better error prevention (warnings for problematic algorithms)
- ✅ Transparent performance expectations

**Performance:**
- ⚡ No overhead (selection happens once at startup)
- ⚡ Better average performance (users get optimal algorithm automatically)
- ⚡ Prevents common mistakes (e.g., using GPU for single ligand)

---

## 🙏 Acknowledgments

This update addresses user feedback about:
- Difficulty choosing the right algorithm
- Misleading GPU performance claims
- Python 3.9 compatibility issues
- Lack of guidance for beginners

Thank you to the PandaDock community for valuable feedback!

---

**Version:** 3.0.0
**Release Date:** 2026-01-04
**Author:** Pritam Kumar Panda @ Stanford University
**License:** MIT
