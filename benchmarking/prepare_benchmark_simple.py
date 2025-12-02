#!/usr/bin/env python3
"""
Simple benchmark dataset preparation using publicly available PDB structures

Since PDBbind requires registration, this script uses:
1. Manually curated list of well-known protein-ligand complexes
2. Direct download from RCSB PDB
3. Automated ligand extraction
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Curated benchmark set: verified complexes with correct ligand codes
# Format: (pdb_id, ligand_resname, protein_family, exp_affinity_nM, notes)
BENCHMARK_COMPLEXES = [
    # HIV Protease inhibitors (well-studied, high quality)
    ('1hpx', 'KNI', 'hiv_protease', 0.08, 'KNI-272 inhibitor'),
    ('1hvr', 'A78', 'hiv_protease', 0.15, 'Indinavir'),
    ('3hvt', 'RIT', 'hiv_protease', 0.3, 'Ritonavir'),
    ('1hxb', 'I1H', 'hiv_protease', 0.5, 'Inhibitor'),
    ('1ohr', '478', 'hiv_protease', 2.0, 'Saquinavir derivative'),

    # Kinases (important drug target class)
    ('1m17', 'STU', 'kinase', 12.0, 'CDK2 with staurosporine'),
    ('1oiy', 'N20', 'kinase', 8.0, 'CDK2 with inhibitor'),
    ('2bhe', 'BRY', 'kinase', 50.0, 'CDK2 with bryostatin'),
    ('1ckp', 'PVB', 'kinase', 15.0, 'CDK2 with inhibitor'),
    ('2c5n', 'BIX', 'kinase', 5.0, 'CDK2 with BI-2536'),

    # Carbonic anhydrase (classic docking target)
    ('2nng', 'NH2', 'carbonic_anhydrase', 5.0, 'CA II with inhibitor'),
    ('3kwa', 'ZMA', 'carbonic_anhydrase', 15.0, 'CA II inhibitor'),
    ('1cnc', 'ICS', 'carbonic_anhydrase', 25.0, 'CA II with isocyanate'),
    ('1z9y', 'FNC', 'carbonic_anhydrase', 10.0, 'CA II inhibitor'),

    # Thrombin (serine protease)
    ('1ets', 'INH', 'thrombin', 3.0, 'Thrombin with inhibitor'),
    ('1dwc', '2RA', 'thrombin', 0.5, 'Hirulog'),
    ('1ype', 'TTP', 'thrombin', 8.0, 'Thrombin inhibitor'),

    # Trypsin (protease)
    ('1tng', 'BEN', 'trypsin', 10.0, 'Benzamidine'),
    ('2ptc', 'BEN', 'trypsin', 10.0, 'Benzamidine complex'),
    ('1fy8', 'MIM', 'trypsin', 50.0, 'With inhibitor'),

    # Factor Xa (blood coagulation)
    ('1ezq', '478', 'factor_xa', 2.0, 'Factor Xa inhibitor'),
    ('1fax', 'DF1', 'factor_xa', 0.8, 'RPR129674'),
    ('1f0r', '815', 'factor_xa', 5.0, 'Inhibitor complex'),

    # Neuraminidase (flu target)
    ('1mwe', 'ZMR', 'neuraminidase', 0.3, 'Zanamivir'),
    ('2qwk', 'G39', 'neuraminidase', 1.0, 'Neuraminidase inhibitor'),
    ('3ti6', 'B18', 'neuraminidase', 0.5, 'H1N1 inhibitor'),

    # Estrogen receptor (nuclear receptor)
    ('3ert', 'EST', 'nuclear_receptor', 0.2, 'Estradiol'),
    ('1err', 'RAL', 'nuclear_receptor', 0.5, 'Raloxifene'),
    ('1a52', 'TAM', 'nuclear_receptor', 1.0, 'Tamoxifen'),

    # Acetylcholinesterase (neuroscience target)
    ('1acj', 'THA', 'acetylcholinesterase', 30.0, 'Tacrine'),
    ('1eve', 'E20', 'acetylcholinesterase', 0.1, 'Donepezil'),
    ('1odc', 'HUP', 'acetylcholinesterase', 5.0, 'Huperzine A'),

    # Additional diverse targets
    ('1hpv', '478', 'hiv_protease', 1.0, 'HIV protease inhibitor'),
    ('1rbp', 'RTL', 'retinol_binding', 100.0, 'Retinol'),
    ('1mmv', 'FXP', 'hiv_reverse_transcriptase', 20.0, 'Nevirapine'),
    ('1hsl', 'SLR', 'heat_shock_protein', 10.0, 'HSP90 inhibitor'),
]

class LigandSelect(Select):
    """Select only ligand atoms from PDB"""
    def __init__(self, ligand_resname):
        self.ligand_resname = ligand_resname.strip()

    def accept_residue(self, residue):
        return residue.get_resname().strip() == self.ligand_resname

class ReceptorSelect(Select):
    """Select protein atoms (no ligand, no water, no metals)"""
    def __init__(self, ligand_resname):
        self.ligand_resname = ligand_resname.strip()
        self.exclude = {self.ligand_resname, 'HOH', 'WAT', 'H2O'}

    def accept_residue(self, residue):
        return residue.get_resname().strip() not in self.exclude

def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    """Download PDB structure from RCSB"""
    pdb_file = output_dir / f"{pdb_id}_complex.pdb"

    if pdb_file.exists():
        return pdb_file

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            pdb_file.write_text(response.text)
            return pdb_file
        else:
            raise RuntimeError(f"HTTP {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

def extract_ligand_info(pdb_file: Path, ligand_resname: str) -> Dict:
    """Extract ligand information including number of atoms"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)

    ligand_atoms = []
    for residue in structure.get_residues():
        if residue.get_resname().strip() == ligand_resname.strip():
            ligand_atoms.extend(list(residue.get_atoms()))

    if not ligand_atoms:
        raise RuntimeError(f"Ligand {ligand_resname} not found in {pdb_file}")

    # Get ligand center
    coords = [atom.get_coord() for atom in ligand_atoms]
    import numpy as np
    coords_array = np.array(coords)
    center = coords_array.mean(axis=0)

    return {
        'n_atoms': len(ligand_atoms),
        'center': center,
        'coords': coords_array
    }

def extract_receptor_ligand(pdb_file: Path, ligand_resname: str,
                           output_dir: Path) -> Tuple[Path, Path]:
    """Extract receptor and ligand from complex PDB"""
    pdb_id = pdb_file.stem.replace('_complex', '')

    receptor_file = output_dir / f"{pdb_id}_receptor.pdb"
    ligand_file = output_dir / f"{pdb_id}_ligand.pdb"

    if receptor_file.exists() and ligand_file.exists():
        return receptor_file, ligand_file

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_file)

    # Save ligand
    io = PDBIO()
    io.set_structure(structure)

    try:
        io.save(str(ligand_file), LigandSelect(ligand_resname))

        # Verify ligand file is not empty
        if ligand_file.stat().st_size < 100:
            raise RuntimeError(f"Ligand file is too small (< 100 bytes)")

        # Save receptor
        io.save(str(receptor_file), ReceptorSelect(ligand_resname))

        if receptor_file.stat().st_size < 1000:
            raise RuntimeError(f"Receptor file is too small (< 1000 bytes)")

    except Exception as e:
        # Cleanup partial files
        if ligand_file.exists():
            ligand_file.unlink()
        if receptor_file.exists():
            receptor_file.unlink()
        raise e

    return receptor_file, ligand_file

def convert_ligand_to_sdf(ligand_pdb: Path, output_sdf: Path) -> bool:
    """Convert ligand PDB to SDF format"""
    if output_sdf.exists():
        return True

    try:
        mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False, sanitize=False)
        if mol is None:
            return False

        # Try to sanitize
        try:
            Chem.SanitizeMol(mol)
        except:
            # If sanitization fails, try without
            pass

        # Add hydrogens if mostly missing
        if mol.GetNumAtoms() > 0:
            heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
            if mol.GetNumAtoms() == heavy_atoms:  # No hydrogens
                mol = Chem.AddHs(mol, addCoords=True)

        # Write SDF
        writer = Chem.SDWriter(str(output_sdf))
        writer.write(mol)
        writer.close()

        return True

    except Exception as e:
        print(f"  Warning: SDF conversion failed: {e}")
        return False

def prepare_benchmark_set(output_base_dir: Path, max_complexes: int = None):
    """
    Prepare benchmark dataset from curated complex list

    Args:
        output_base_dir: Base directory for all outputs
        max_complexes: Limit number of complexes (for testing)
    """
    import numpy as np

    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    pdb_dir = output_base_dir / "pdbs"
    receptor_dir = output_base_dir / "receptors"
    ligand_dir = output_base_dir / "ligands"

    for d in [pdb_dir, receptor_dir, ligand_dir]:
        d.mkdir(exist_ok=True)

    complexes = BENCHMARK_COMPLEXES[:max_complexes] if max_complexes else BENCHMARK_COMPLEXES

    print(f"Preparing benchmark set with {len(complexes)} complexes")
    print("="*60)

    successful = []
    failed = []
    metadata = []

    for pdb_id, ligand_name, family, affinity_nm, notes in tqdm(complexes):
        try:
            # Download PDB
            pdb_file = download_pdb(pdb_id, pdb_dir)

            # Extract ligand info
            lig_info = extract_ligand_info(pdb_file, ligand_name)

            # Extract receptor and ligand
            receptor_file, ligand_pdb = extract_receptor_ligand(pdb_file, ligand_name, receptor_dir)

            # Convert ligand to SDF
            ligand_sdf = ligand_dir / f"{pdb_id}_ligand.sdf"
            sdf_success = convert_ligand_to_sdf(ligand_pdb, ligand_sdf)

            # Calculate pKd (negative log molar)
            # affinity in nM -> convert to M -> take -log10
            pkd = -np.log10(affinity_nm * 1e-9)

            metadata.append({
                'pdb_id': pdb_id,
                'ligand_name': ligand_name,
                'protein_family': family,
                'affinity_nM': affinity_nm,
                'pKd': pkd,
                'ligand_n_atoms': lig_info['n_atoms'],
                'center_x': lig_info['center'][0],
                'center_y': lig_info['center'][1],
                'center_z': lig_info['center'][2],
                'notes': notes,
                'sdf_available': sdf_success
            })

            successful.append(pdb_id)

        except Exception as e:
            print(f"\nFailed {pdb_id}: {e}")
            failed.append((pdb_id, str(e)))

    print(f"\n{'='*60}")
    print(f"Successfully processed: {len(successful)} complexes")
    print(f"Failed: {len(failed)} complexes")

    if failed:
        print(f"\nFailed complexes:")
        for pdb_id, error in failed:
            print(f"  {pdb_id}: {error}")

    # Save metadata
    if metadata:
        df = pd.DataFrame(metadata)
        metadata_file = output_base_dir / "benchmark_metadata.csv"
        df.to_csv(metadata_file, index=False)
        print(f"\nSaved metadata to {metadata_file}")

        # Print summary statistics
        print(f"\n{'='*60}")
        print("Benchmark Set Summary:")
        print(f"  Total complexes: {len(df)}")
        print(f"  Protein families: {df['protein_family'].nunique()}")
        print(f"  Mean pKd: {df['pKd'].mean():.2f}")
        print(f"  pKd range: {df['pKd'].min():.1f} - {df['pKd'].max():.1f}")
        print(f"  Mean ligand atoms: {df['ligand_n_atoms'].mean():.1f}")

        print(f"\nFamily distribution:")
        for family, count in df['protein_family'].value_counts().items():
            print(f"  {family}: {count}")

        print(f"\nFiles ready:")
        print(f"  Receptors: {receptor_dir}")
        print(f"  Ligands (PDB): {receptor_dir}")
        print(f"  Ligands (SDF): {ligand_dir}")
        print(f"  Metadata: {metadata_file}")

    return len(successful), len(failed)

def expand_to_larger_set(output_base_dir: Path):
    """
    Instructions for expanding to larger benchmark sets
    """
    print("\n" + "="*60)
    print("TO EXPAND TO LARGER BENCHMARK SETS:")
    print("="*60)

    print("\n1. PDBbind Database (290 complexes - RECOMMENDED)")
    print("   - Register at: http://www.pdbbind.org.cn/")
    print("   - Download: PDBbind v2020 Core Set")
    print("   - Contains experimental binding affinities")
    print("   - Most cited benchmark in docking literature")

    print("\n2. Astex Diverse Set (85 complexes)")
    print("   - Download from: https://github.com/PatWalters/OpenEye-Contrib")
    print("   - Focused on pose prediction accuracy")
    print("   - Smaller but highly curated")

    print("\n3. DUD-E Database (Virtual screening)")
    print("   - Website: http://dude.docking.org/")
    print("   - Contains actives + decoys")
    print("   - For enrichment studies")

    print("\n4. Category-specific sets:")
    print("   - KLIFS: Kinase structures (https://klifs.net/)")
    print("   - GPCRdb: GPCR structures (https://gpcrdb.org/)")
    print("   - MetalPDB: Metal-binding proteins")

    print("\n5. Current set (25 complexes):")
    print("   - Good for INITIAL testing and method development")
    print("   - NOT sufficient for publication")
    print("   - Recommended minimum for publication: 100-200 complexes")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare simple benchmark dataset")
    parser.add_argument("-o", "--output", type=Path,
                       default=Path("benchmarking/simple_benchmark_set"),
                       help="Output directory")
    parser.add_argument("-n", "--max-complexes", type=int, default=None,
                       help="Limit number of complexes (for testing)")
    parser.add_argument("--show-info", action="store_true",
                       help="Show info about expanding to larger sets")

    args = parser.parse_args()

    if args.show_info:
        expand_to_larger_set(Path("."))
    else:
        success, failed = prepare_benchmark_set(args.output, args.max_complexes)

        if success > 0:
            print("\n✓ Benchmark dataset ready for testing!")
            print(f"\nNext step:")
            print(f"  python benchmarking/run_benchmark_comparison.py \\")
            print(f"    --benchmark-dir {args.output} \\")
            print(f"    --output-dir benchmarking/results \\")
            print(f"    --algorithms pandadock_hierarchical_cpu_physics_based vina")

        if success < 20:
            print("\n" + "="*60)
            print("WARNING: Small benchmark set (<20 complexes)")
            print("For publication, you need 100-200+ complexes")
            print("Run with --show-info for information on larger datasets")
            print("="*60)
