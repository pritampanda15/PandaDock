# Tutorial 1: Basic Protein-Ligand Docking

This tutorial introduces the fundamentals of molecular docking using PandaDock. You'll learn to perform a standard protein-ligand docking simulation, analyze the results, and interpret the binding predictions.

## Prerequisites

- PandaDock installed and working
- Basic familiarity with molecular structures
- 15-20 minutes of time

## Learning Objectives

By the end of this tutorial, you will be able to:
1. Prepare input files for docking
2. Set up and run a basic docking simulation
3. Analyze docking results and binding poses
4. Interpret energy scores and interaction analysis

## Tutorial Overview

We'll dock a small molecule inhibitor to a protein kinase active site. This represents a typical drug discovery scenario where we want to predict how a potential drug binds to its target protein.

**Target**: Protein kinase (PDB ID: 1ATP)
**Ligand**: Adenosine analog inhibitor
**Expected binding mode**: ATP binding site

## Step 1: Prepare Input Files

### Download Example Files

First, let's get the tutorial files:

```bash
# Create working directory
mkdir pandadock_tutorial
cd pandadock_tutorial

# Download example files (or use your own)
wget https://files.rcsb.org/download/1ATP.pdb
```

For the ligand, create a simple SDF file or download one:

```bash
# Download example ligand
wget https://pandadock.org/examples/adenosine_analog.sdf
```

### Examine Input Files

Let's look at our input files:

```bash
# Check the protein structure
head -20 1ATP.pdb

# Check ligand information
grep -A 10 -B 5 "MOL" adenosine_analog.sdf
```

**What to look for:**
- **Protein**: Should contain ATOM records for the protein chain
- **Ligand**: Should be in 3D format with coordinates
- **Missing atoms**: Check for incomplete residues or missing hydrogens

## Step 2: Basic Docking Setup

### Automatic Grid Detection

The simplest approach is to let PandaDock automatically detect the binding site:

```bash
pandadock-dock -r 1ATP.pdb -l adenosine_analog.sdf -o basic_docking/
```

This command:
- `-r 1ATP.pdb`: Specifies the receptor (protein) file
- `-l adenosine_analog.sdf`: Specifies the ligand file
- `-o basic_docking/`: Sets the output directory

### Monitor Progress

PandaDock will display progress information:

```
PandaDock Enhanced Molecular Docking
Receptor: 1ATP.pdb
Ligand: adenosine_analog.sdf
Algorithm: enhanced_hierarchical_cpu
Scoring: physics_based
Starting docking...
Detecting binding sites...
Found 3 potential binding sites
Using largest cavity for docking
Grid center: (15.2, 22.1, 8.7) A
Grid dimensions: (20.0, 20.0, 20.0) A
Generating ligand conformers...
Performing docking...
Docking completed in 3.2 seconds
Generated 20 poses
```

## Step 3: Examine Output Files

After docking completes, examine the generated files:

```bash
ls -la basic_docking/
```

You should see:
- `pose1.pdb`, `pose2.pdb`, etc.: Individual ligand poses
- `complex1.pdb`, `complex2.pdb`, etc.: Complete protein-ligand complexes
- `docking_summary.json`: Detailed results and energies
- `interaction_analysis.json`: Binding interaction details
- `binding_affinities.png`: Energy distribution plot

### View Docking Summary

```bash
cat basic_docking/docking_summary.json | head -30
```

This shows:
- **Energy scores** for each pose
- **Confidence scores** (0-1, higher is better)
- **Algorithm parameters** used
- **Runtime statistics**

### View Interaction Analysis

```bash
cat basic_docking/interaction_analysis.json
```

This provides:
- **Total number of interactions**
- **Interaction types** (hydrogen bonds, hydrophobic contacts, etc.)
- **Binding affinity estimate**
- **Key interacting residues**

## Step 4: Analyze Results

### Energy Analysis

Let's examine the top poses:

```bash
# Extract top 5 pose energies
grep -A 5 "poses" basic_docking/docking_summary.json
```

**What to look for:**
- **Energy range**: Typical values are -15 to +5 kcal/mol
- **Energy clustering**: Similar energies suggest consistent binding modes
- **Outliers**: Very high energies may indicate poor poses

### Interaction Analysis

Check the interaction analysis:

```bash
# View interaction summary
grep -A 10 "interaction_types" basic_docking/interaction_analysis.json
```

**Good binding poses typically show:**
- **Multiple interaction types**: Hydrogen bonds, hydrophobic contacts, van der Waals
- **Reasonable total interactions**: 20-100 for typical drug-like molecules
- **Key protein contacts**: Known important residues for your target

### Visual Inspection

Use a molecular viewer to examine the poses:

```bash
# Using PyMOL (if available)
pymol basic_docking/complex1.pdb

# Or use ChimeraX, VMD, or other viewers
```

**What to check:**
1. **Binding site location**: Is the ligand in the expected pocket?
2. **Orientation**: Does the binding mode make chemical sense?
3. **Clashes**: Are there any obvious atomic overlaps?
4. **Interactions**: Can you see hydrogen bonds and hydrophobic contacts?

## Step 5: Detailed Analysis

### Compare Multiple Poses

Look at the top few poses to understand binding diversity:

```bash
# View top 3 poses
for i in {1..3}; do
    echo "Pose $i:"
    grep "Energy:" basic_docking/pose${i}.pdb 2>/dev/null || echo "Check docking_summary.json for energy"
done
```

### Binding Site Analysis

If you know the expected binding site, verify the docking results:

```bash
# For known targets, compare to literature or crystal structures
# Check if key interactions are present
```

## Step 6: Improve Results (Optional)

If the initial results need improvement, try these modifications:

### Use Specific Grid Center

If you know the binding site location:

```bash
pandadock-dock -r 1ATP.pdb -l adenosine_analog.sdf \
  --center 15.2 22.1 8.7 \
  --box 20 20 20 \
  -o manual_grid_docking/
```

### Try Different Algorithm

For higher accuracy at the cost of speed:

```bash
pandadock-dock -r 1ATP.pdb -l adenosine_analog.sdf \
  -a genetic_algorithm_cpu \
  -s precision_score \
  -o high_accuracy_docking/
```

### Increase Sampling

Generate more poses for better sampling:

```bash
pandadock-dock -r 1ATP.pdb -l adenosine_analog.sdf \
  -n 50 \
  --ensemble \
  -o extended_sampling/
```

## Step 7: Generate Report

Create a comprehensive analysis report:

```bash
pandadock-report -i basic_docking/ -t "Kinase Inhibitor Docking"
```

This generates:
- **Binding affinity plots**
- **Interaction diagrams**
- **Pose clustering analysis**
- **Summary statistics**

## Understanding Results

### Energy Scores

**Binding Energy**: The predicted strength of protein-ligand binding
- **More negative = stronger binding**
- **Typical range**: -15 to +5 kcal/mol for drug-like molecules
- **Experimental correlation**: ~65-85% depending on system and scoring function

### Confidence Scores

**Confidence**: Algorithm's certainty in the pose prediction (0-1 scale)
- **>0.8**: High confidence, likely correct binding mode
- **0.5-0.8**: Moderate confidence, reasonable prediction
- **<0.5**: Low confidence, consider alternative poses

### Interaction Types

**Hydrogen Bonds**: Directional, high-specificity interactions
- **Count**: 0-8 typical for drug-like molecules
- **Distance**: Usually 2.5-3.5 Å
- **Importance**: Key for binding specificity

**Hydrophobic Contacts**: Shape complementarity interactions
- **Count**: 10-50 typical for drug-like molecules
- **Importance**: Major contributor to binding affinity

**Van der Waals**: General shape complementarity
- **Count**: Usually the largest category
- **Importance**: Overall geometric fit

## Common Issues and Solutions

### Problem: No poses generated

**Symptoms**: "Generated 0 poses" in output

**Solutions**:
1. Check input file formats
2. Increase energy threshold: `--energy-threshold 100`
3. Expand grid box: `--box 25 25 25`
4. Try different algorithm: `-a monte_carlo_cpu`

### Problem: Poor binding poses

**Symptoms**: Ligand outside expected binding site, very high energies

**Solutions**:
1. Specify manual grid center if known
2. Use crystal-guided docking if reference available
3. Try ensemble averaging: `--ensemble`
4. Increase sampling: `-n 50`

### Problem: Unrealistic energies

**Symptoms**: Energies outside typical range (-15 to +5 kcal/mol)

**Solutions**:
1. Check structure quality (missing atoms, weird geometry)
2. Try empirical scoring: `-s empirical`
3. Use structure preparation tools if available

## Next Steps

Congratulations! You've completed your first PandaDock simulation. To build on this knowledge:

1. **Try Tutorial 2**: [High-Throughput Virtual Screening](virtual-screening.md)
2. **Explore algorithms**: Test different docking algorithms on your system
3. **Learn advanced features**: Flexible docking, metal coordination, GPU acceleration
4. **Validate results**: Compare with experimental data or known crystal structures

## Key Takeaways

- **PandaDock automates** binding site detection and docking setup
- **Multiple poses** are generated to explore binding diversity
- **Energy and interaction analysis** help evaluate binding quality
- **Visual inspection** is crucial for validating predicted binding modes
- **Parameter tuning** can improve results for challenging systems

## Additional Resources

- [Algorithm Documentation](../algorithms/index.md): Detailed algorithm descriptions
- [Scoring Functions](../scoring/index.md): Understanding energy calculations
- [CLI Reference](../cli/index.md): Complete command-line options
- [Troubleshooting Guide](troubleshooting.md): Solutions for common problems

## Exercise

Practice with your own system:
1. Find a protein-ligand complex from the PDB
2. Separate the protein and ligand
3. Dock the ligand back to the protein
4. Compare your result with the crystal structure
5. Calculate RMSD and analyze binding mode accuracy

This exercise helps validate your docking setup and understand method limitations.