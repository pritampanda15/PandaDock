# PandaDock Benchmarking - Quick Start Guide

## Current Status

✅ **Simple benchmark set prepared**: 10 complexes ready for testing
✅ **Benchmarking scripts created**: All tools ready to use
✅ **Analysis pipeline ready**: Automated figure generation

## For Publication: What You Need

### Minimum Requirements (Tier 2 Journal - JCAMD, Molecules)
- **Dataset**: 100-200 diverse complexes
- **Comparisons**: PandaDock vs Vina + 1-2 other tools
- **Metrics**: Success rate (RMSD < 2Å), runtime, basic statistics
- **Timeline**: 2-3 weeks

### Strong Publication (Tier 1 Journal - JCIM, J Med Chem)
- **Dataset**: 290 PDBbind Core Set + 100 category-specific = **~400 complexes**
- **Comparisons**: PandaDock vs Vina, Smina, AutoDock-GPU, (Glide if available)
- **Metrics**: Full suite (RMSD, affinity correlation, enrichment, runtime)
- **Statistical tests**: Wilcoxon, Bonferroni correction
- **Timeline**: 4-6 weeks

## Recommended Strategy

### Week 1: Test with Simple Set (10 complexes) ✅ DONE
```bash
cd /Users/pritam/Desktop/PandaDock_Ultimate_Version

# Already prepared!
ls benchmarking/simple_benchmark_set/benchmark_metadata.csv
```

**What you have now:**
- 10 diverse protein-ligand complexes
- HIV protease, kinases, neuraminidase, acetylcholinesterase, etc.
- pKd range: 7.0 - 10.1 (good diversity)
- Ready for immediate testing

### Week 2: Register for PDBbind (START NOW)

**Action Required:**
1. Go to: http://www.pdbbind.org.cn/
2. Click "Download" → "Register"
3. Use institutional email
4. Request access to "PDBbind v2020 Core Set"
5. Wait 1-2 days for approval

**Why PDBbind is essential:**
- 290 complexes (standard benchmark)
- Experimental binding affinities
- Every docking paper uses it
- Reviewers expect it

### Week 3-4: Run Full Benchmark

Once you have PDBbind:
```bash
# Convert PDBbind to PandaDock format
python benchmarking/convert_pdbbind.py \
  --pdbbind-dir /path/to/PDBbind_v2020 \
  --output benchmarking/pdbbind_core_set

# Run comprehensive benchmark
python benchmarking/run_benchmark_comparison.py \
  --benchmark-dir benchmarking/pdbbind_core_set \
  --output-dir benchmarking/results_full \
  --algorithms \
    pandadock_hierarchical_cpu_physics_based \
    pandadock_cuda_genetic_algorithm \
    pandadock_cuda_monte_carlo \
    vina \
    smina
```

### Week 5: Analysis and Figures
```bash
# Generate all publication figures
python benchmarking/analyze_results.py \
  --results benchmarking/results_full/benchmark_results.csv \
  --metadata benchmarking/pdbbind_core_set/benchmark_metadata.csv \
  --output-dir benchmarking/analysis
```

## What to Do RIGHT NOW

### Option 1: Test PandaDock Works (30 minutes)

Run a quick test on the 10-complex set:

```bash
cd /Users/pritam/Desktop/PandaDock_Ultimate_Version

# Test PandaDock CPU algorithm (1 complex, fast)
PYTHONPATH=. python3 -m pandadock.docking_cli dock \
  --receptor benchmarking/simple_benchmark_set/receptors/1hpx_receptor.pdb \
  --ligand benchmarking/simple_benchmark_set/ligands/1hpx_ligand.sdf \
  --algorithm hierarchical_cpu_physics_based \
  --output-dir test_docking_output

# Test PandaDock GPU algorithm (1 complex, verify GPU works)
PYTHONPATH=. python3 -m pandadock.docking_cli dock \
  --receptor benchmarking/simple_benchmark_set/receptors/1hpx_receptor.pdb \
  --ligand benchmarking/simple_benchmark_set/ligands/1hpx_ligand.sdf \
  --algorithm cuda_genetic_algorithm \
  --output-dir test_docking_output_gpu
```

Expected output:
- Docking completes in 30-120 seconds
- Generates poses with scores
- No errors

### Option 2: Register for PDBbind (5 minutes)

**DO THIS TODAY:**
1. Open http://www.pdbbind.org.cn/
2. Register with institutional email
3. Request "PDBbind v2020 Core Set" access

While waiting for approval (1-2 days), continue with Option 1 testing.

### Option 3: Run Full Simple Set Benchmark (2-3 hours)

Test the complete benchmarking pipeline with all 5 CPU algorithms:

```bash
# Run PandaDock on all 10 complexes with ALL CPU algorithms
python benchmarking/run_benchmark_comparison.py \
  --benchmark-dir benchmarking/simple_benchmark_set \
  --output-dir benchmarking/results_simple

# By default, tests all 5 CPU algorithms:
#   - monte_carlo_cpu
#   - genetic_algorithm_cpu
#   - hierarchical_cpu
#   - enhanced_hierarchical_cpu
#   - crystal_guided_cpu

# Generate figures
python benchmarking/analyze_results.py \
  --results benchmarking/results_simple/benchmark_results.csv \
  --metadata benchmarking/simple_benchmark_set/benchmark_metadata.csv \
  --output-dir benchmarking/analysis_simple
```

This verifies:
- ✓ Benchmarking scripts work
- ✓ PandaDock algorithms work
- ✓ Analysis pipeline works
- ✓ Figure generation works

## Expected Results

### Simple Set (10 complexes) - What to Expect

**Success Rate (RMSD < 2Å):**
- PandaDock Hierarchical: 40-60% (good)
- PandaDock CUDA GA: 35-55% (acceptable)

**Mean Runtime:**
- PandaDock Hierarchical CPU: 60-180 seconds/complex
- PandaDock CUDA GA (GPU): 10-30 seconds/complex
- PandaDock CUDA MC (GPU): 20-40 seconds/complex

**If you see these numbers:** ✅ PandaDock is working correctly!

### PDBbind Core (290 complexes) - Target Metrics

**For publication, you want:**
- Success rate (RMSD < 2Å): **>50%** (competitive)
- Success rate (RMSD < 2Å): **>55%** (strong)
- Success rate (RMSD < 2Å): **>60%** (excellent)

**Comparison to baselines:**
- AutoDock Vina: typically 50-60%
- Smina: typically 52-62%
- Glide SP: typically 55-65%

**Your goal:** Match or exceed Vina on PDBbind Core Set

## Comparison Tools Installation

### AutoDock Vina (Required)
```bash
conda install -c conda-forge vina
```

### Smina (Recommended)
```bash
conda install -c conda-forge smina
# Or download: https://sourceforge.net/projects/smina/
```

### AutoDock-GPU (Optional but impressive)
```bash
git clone https://github.com/ccsb-scripps/AutoDock-GPU.git
cd AutoDock-GPU
make DEVICE=CUDA
```

### Glide (Optional - if you have license)
Requires Schrödinger suite license (commercial)

## Common Issues

### Issue 1: "Only 10 complexes - is this enough?"
**Answer:** NO for publication, YES for testing

- 10 complexes: Test PandaDock works
- 100 complexes: Minimum for publication
- 290 complexes (PDBbind Core): Standard for publication
- 400+ complexes: Strong publication

### Issue 2: "PDBbind download requires registration"
**Answer:** Yes, it's necessary

- Registration takes 1-2 days
- Worth the wait - it's THE standard benchmark
- Alternative: Start with 10-complex test set now

### Issue 3: "Benchmarking takes too long"
**Solution:** Use GPU algorithms

- CPU algorithm: ~120 sec/complex × 290 = 9.7 hours
- GPU algorithm: ~20 sec/complex × 290 = 1.6 hours
- Run overnight or use multiple GPUs

### Issue 4: "Some complexes fail to dock"
**Expected:** 5-10% failure rate is normal

- Ligand parsing errors
- Very large ligands (>100 atoms)
- Missing hydrogens
- Unusual chemistry

**What matters:** Overall success rate on majority of complexes

## Files Created

```
benchmarking/
├── README.md                           # Comprehensive guide
├── download_pdbbind_manual.md          # PDBbind instructions
├── prepare_benchmark_simple.py         # Create 10-complex test set ✅
├── run_benchmark_comparison.py         # Run docking comparisons
├── analyze_results.py                  # Generate figures
│
├── simple_benchmark_set/               # Ready for testing ✅
│   ├── benchmark_metadata.csv          # 10 complexes
│   ├── receptors/                      # Prepared proteins
│   └── ligands/                        # Prepared ligands
│
└── pdbbind_core_set/                   # Create after registration
    ├── benchmark_metadata.csv          # 290 complexes
    ├── receptors/
    └── ligands/
```

## Next Steps Summary

**TODAY (30 min):**
1. ✅ Test PandaDock on 1 complex (verify it works)
2. ✅ Register for PDBbind account
3. ✅ Install AutoDock Vina

**THIS WEEK (2-3 hours):**
4. Run full 10-complex benchmark
5. Verify analysis scripts work
6. Check figure quality

**NEXT WEEK (after PDBbind approval):**
7. Download PDBbind Core Set
8. Convert to PandaDock format
9. Start full 290-complex benchmark

**WEEK 3-4:**
10. Complete benchmarking
11. Generate all figures
12. Statistical analysis

**WEEK 5-6:**
13. Write manuscript
14. Submit to journal

## Target Journals

Based on 400-complex benchmark (PDBbind Core + category):

**Tier 1 (Aim for these):**
- Journal of Chemical Information and Modeling (JCIM) - IF ~5.6
- Journal of Computer-Aided Molecular Design (JCAMD) - IF ~3.0
- Molecules - IF ~4.6 (open access)

**Tier 2 (Backup):**
- Journal of Molecular Graphics and Modelling
- Computational Biology and Chemistry

**Requirements:**
- 200-300 complexes minimum
- Compare to 3+ established tools
- Full statistical analysis
- Clear novelty (GPU acceleration + metal handling)

## Questions?

See `benchmarking/README.md` for detailed documentation.

## Summary

**You have:** 10-complex test set ready ✅
**You need:** 290-complex PDBbind Core Set for publication
**Timeline:** 4-6 weeks from now to submission
**Next action:** Register for PDBbind TODAY

Good luck with your publication! 🎉
