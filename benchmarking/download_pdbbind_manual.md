# How to Get PDBbind Core Set (290 complexes)

The PDBbind database requires registration and cannot be automatically downloaded. Here's how to get it:

## Option 1: Register and Download (RECOMMENDED for publication)

### Step 1: Register
1. Go to: http://www.pdbbind.org.cn/
2. Click "Download" → "Register"
3. Fill in registration form (institutional email required)
4. Wait for approval email (usually 1-2 days)

### Step 2: Download
After approval:
1. Login to PDBbind website
2. Navigate to "Download" section
3. Download these files:
   - **PDBbind v2020 Core Set** (290 complexes)
   - **INDEX_core_data.2020** (binding affinity data)

### Step 3: Extract
```bash
# Extract downloaded file
tar -xzf PDBbind_v2020_core_set.tar.gz

# Should create structure:
PDBbind_v2020/
├── INDEX_core_data.2020
└── v2020-core/
    ├── 10gs/
    │   ├── 10gs_protein.pdb
    │   ├── 10gs_ligand.mol2
    │   └── 10gs_pocket.pdb
    ├── 11gs/
    └── ... (290 total)
```

### Step 4: Convert to PandaDock Format
```bash
python benchmarking/convert_pdbbind_format.py \
  --pdbbind-dir PDBbind_v2020/v2020-core \
  --index-file PDBbind_v2020/INDEX_core_data.2020 \
  --output benchmarking/pdbbind_core_set
```

## Option 2: Use Simple Benchmark Set (25 complexes)

For quick testing or initial method development:

```bash
# Download 25 curated complexes directly from PDB
python benchmarking/prepare_benchmark_simple.py \
  --output benchmarking/simple_benchmark_set
```

**Pros:**
- ✅ No registration needed
- ✅ Fully automated
- ✅ Good for testing PandaDock

**Cons:**
- ❌ Only 25 complexes (too small for publication)
- ❌ No comprehensive binding affinity data
- ❌ Less diversity than PDBbind

## Option 3: Alternative Public Datasets

### Astex Diverse Set (85 complexes)
- **Source**: https://github.com/PatWalters/OpenEye-Contrib
- **Focus**: Pose prediction accuracy
- **Size**: 85 complexes
- **Status**: Sufficient for small-scale publication

```bash
git clone https://github.com/PatWalters/OpenEye-Contrib.git
python benchmarking/convert_astex_format.py \
  --astex-dir OpenEye-Contrib/astex_diverse_set \
  --output benchmarking/astex_set
```

### DUD-E (Directory of Useful Decoys - Enhanced)
- **Source**: http://dude.docking.org/
- **Focus**: Virtual screening / enrichment studies
- **Size**: 102 targets with actives + decoys
- **Status**: Good for virtual screening papers

```bash
# Download specific targets
wget http://dude.docking.org/targets/aa2ar/actives_final.mol2.gz
wget http://dude.docking.org/targets/aa2ar/decoys_final.mol2.gz
wget http://dude.docking.org/targets/aa2ar/receptor.pdb
```

### KLIFS Kinase Database
- **Source**: https://klifs.net/
- **Focus**: Kinase-specific benchmarking
- **Size**: 1000+ kinase structures
- **Status**: Excellent for category-specific publication

```bash
# Download via KLIFS API
python benchmarking/download_klifs_set.py \
  --n-structures 100 \
  --output benchmarking/kinase_set
```

## Recommendations by Publication Goal

### For Method Development / Testing
**Use**: Simple benchmark set (25 complexes)
```bash
python benchmarking/prepare_benchmark_simple.py
```
**Timeline**: 10 minutes
**Sufficient**: Testing, debugging, initial validation

### For Strong Publication (JCIM, JCAMD)
**Use**: PDBbind Core Set (290 complexes)
**Timeline**: 2-3 days (registration) + 2 hours (download/prep)
**Required**: Comprehensive benchmark, comparison to 3+ tools

### For Top-Tier Publication (Nature Comms, J Med Chem)
**Use**: PDBbind Core (290) + Category-specific (100+)
**Timeline**: 1 week (data collection) + 1-2 weeks (benchmarking)
**Required**: Large-scale validation + prospective studies

## Quick Comparison

| Dataset | Complexes | Registration | Time to Prepare | Publication Tier |
|---------|-----------|--------------|-----------------|------------------|
| Simple Set | 25 | None | 10 min | Not sufficient |
| Astex Diverse | 85 | None | 30 min | Small study |
| PDBbind Core | 290 | Required | 2 days | Strong paper |
| PDBbind General | 5316 | Required | 1 week | Comprehensive |
| Multi-source | 500+ | Some required | 1-2 weeks | Top tier |

## What I Recommend

**For your situation:**

1. **Start now with Simple Set** (25 complexes)
   - Test PandaDock algorithms
   - Verify benchmarking scripts work
   - Generate initial figures

2. **Register for PDBbind today**
   - Takes 1-2 days for approval
   - Download PDBbind Core Set (290 complexes)
   - This is REQUIRED for publication

3. **Optional: Add category-specific** (if you have time)
   - KLIFS kinases (50-100 structures)
   - Metal-binding proteins (50 structures)
   - Shows PandaDock's strengths

**Timeline:**
- Week 1: Simple set (25) - verify everything works
- Week 2-3: PDBbind Core (290) - main benchmark
- Week 4: Analysis and figure generation
- Week 5-6: Manuscript writing

## Troubleshooting

**Q: PDBbind registration not approved?**
A: Use institutional email, mention "academic research", usually approved in 1-2 days

**Q: Download too slow?**
A: PDBbind Core Set is ~2GB, use stable connection

**Q: Can I use old PDBbind version (2019, 2016)?**
A: Yes, but reviewers prefer latest version (2020)

**Q: Is 25 complexes enough?**
A: For testing YES, for publication NO (minimum 100-200)

**Q: Can I skip PDBbind?**
A: Difficult - it's the standard benchmark, reviewers expect it
