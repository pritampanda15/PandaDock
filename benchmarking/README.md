# PandaDock Benchmarking Suite

Reproducible redocking, affinity correlation, pose-rescoring, and SAIR preprocessing/evaluation scripts for PandaDock.

## Scripts

```
benchmarking/
├── README.md                  # This file
├── prepare_complexes.py       # Split whole PDB entries into receptor/ligand pairs via PDB CCD
├── validate_prepared.py       # Check for ligand leakage into receptors before spending compute
├── redock_benchmark.py        # Flexible-ligand redocking harness (CPU or --device cuda|mps)
├── analyze_redock.py          # Split search failures from scoring/ranking failures
├── make_report.py             # Generate Markdown and LaTeX publication tables
├── gnn_affinity_benchmark.py  # Score complexes with PandaDock-GNN and correlate per family
├── gnn_affinity_check.py      # Evaluate SAIR checkpoint on crystal structures & pose displacement
├── gnn_pose_rescoring.py      # Compare Vina vs GNN pose selection on real docked ensembles
├── ligand_only_baseline.py    # 16-descriptor ligand-only Ridge control (target-disjoint)
├── compare_runs.py            # Paired per-target Wilcoxon + bootstrap comparison of two runs
├── sair_preprocess.py         # Stream SAIR ModelCIFs from S3 into compact gzipped shards
├── sair_label_audit.py        # Audit label distributions and between/within-target variance
└── sair_evaluate.py           # Report pooled, target-ranking, and within-target correlations
```

## Quick Start

### Redocking Benchmark (Pose Accuracy)

```bash
# 1. Split complexes into receptor/ligand pairs
python benchmarking/prepare_complexes.py --input complexes/ --output prepared/

# 2. Check for ligand leakage into receptors
python benchmarking/validate_prepared.py --manifest prepared/manifest.csv

# 3. Redock (CPU or GPU)
python benchmarking/redock_benchmark.py \
    --manifest prepared/manifest.csv \
    --output results_cpu/ --padding 5 --seed 42 -j 8

python benchmarking/redock_benchmark.py \
    --manifest prepared/manifest.csv \
    --device cuda --n-chains 512 \
    --output results_cuda/

# 4. Separate search vs ranking failures and generate tables
python benchmarking/analyze_redock.py results_cpu/redock_results.csv
python benchmarking/make_report.py results_cpu/redock_results.csv --output results_cpu/report
```

### SAIR GNN Training & Evaluation

```bash
# 1. Build compact shard cache from SAIR parquet + S3 CIFs
python benchmarking/sair_preprocess.py --parquet sair.parquet --cif-root s3://... --output sair_cache/

# 2. Audit label spread and between-target variance
python benchmarking/sair_label_audit.py --cache sair_cache/

# 3. Train PandaDock-GNN with target-grouped within-target ranking
pandadock-gnn train-sair --cache sair_cache/ --output models/sair/ --rank-weight 1.0 --target-grouped

# 4. Evaluate pooled vs within-target correlation and compare against ligand-only baseline
python benchmarking/sair_evaluate.py --cache sair_cache/ --model models/sair/best_model.pt --output eval_sair/
python benchmarking/ligand_only_baseline.py --cache sair_cache/ --output eval_baseline/
python benchmarking/compare_runs.py eval_sair/predictions.csv eval_baseline/predictions.csv
```

## Verified Benchmark Summary (v4.1+)

> **Note:** Earlier v4.0 figures (`R = 0.88` on PDBbind, `R = 0.82` on ULVSH) were withdrawn in August 2026 due to small-subset label-informed rescaling and 96.1% censored ULVSH labels. All numbers below are measured on full datasets with fixed pipelines.

### Pose Accuracy (814 Complexes, 14 Target Families)

| Backend | Wall Time | Top-1 ≤ 2 Å | Best-of-N ≤ 2 Å |
|---|---|---|---|
| CPU (auto exhaustiveness) | ~6.5 d (full est.) | 33.7% | 57.0% |
| CUDA (NVIDIA A10G, 512 chains) | 20.2 h (814 complexes) | 30.2% | 50.5% |
| Paired 131-complex CPU vs CUDA | 25.25 h vs 2.96 h (**8.5×**) | 34.4% vs 30.5% | **56.5% vs 56.5%** |

### Affinity Prediction (PandaDock-GNN, Native Crystal / Held-out Splits)

| Model | Evaluation Set | N | Pearson R | Median Within-Target R |
|---|---|---|---|---|
| SAIR (741k, target-disjoint) | SAIR held-out test | 90,219 | 0.407 | 0.24 |
| SAIR (741k, independent) | Crystal structures | 202 | 0.467 | — |
| SAIR (741k, independent) | PDBbind v2020 refined | 4,640 | 0.531 | 0.31 |
| PDBbind-only (target-disjoint) | Own held-out test | 624 | 0.690 | — |

### Pose Rescoring Control (50 Complexes × 20 Poses)

| Selection | Median RMSD | ≤ 2 Å |
|---|---|---|
| Empirical Vina score | 2.09 Å | 48% |
| Affinity GNN rescoring | 5.36 Å | 22% |
| Best pose available | 1.26 Å | 80% |

The affinity GNN ranks ligands against each other and should **not** be used to select poses over the empirical scoring function.
