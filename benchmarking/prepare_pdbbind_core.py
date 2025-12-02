#!/usr/bin/env python3
"""
Prepare PDBbind Core Set for benchmarking

Downloads and prepares the PDBbind Core Set (2020) for docking benchmarks.
Filters by quality criteria and extracts ligand binding data.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
from tqdm import tqdm

class LigandSelect(Select):
    """Select only ligand atoms from PDB"""
    def __init__(self, ligand_resname):
        self.ligand_resname = ligand_resname

    def accept_residue(self, residue):
        return residue.get_resname() == self.ligand_resname

def download_pdbbind_index(output_dir: Path):
    """Download PDBbind index file with binding affinities"""
    index_url = "http://www.pdbbind.org.cn/download/PDBbind_2020_plain_text_index/index/INDEX_core_data.2020"
    index_file = output_dir / "INDEX_core_data.2020"

    if index_file.exists():
        print(f"Index file already exists: {index_file}")
        return index_file

    print(f"Downloading PDBbind Core Set index...")
    response = requests.get(index_url)
    if response.status_code == 200:
        index_file.write_text(response.text)
        print(f"Downloaded to {index_file}")
    else:
        print(f"Failed to download index file. Status: {response.status_code}")
        print("Please manually download from http://www.pdbbind.org.cn/")
        return None

    return index_file

def parse_pdbbind_index(index_file: Path) -> pd.DataFrame:
    """Parse PDBbind index file into DataFrame"""
    data = []

    with open(index_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 8:
                continue

            pdb_id = parts[0]
            resolution = parts[1]
            release_year = parts[2]
            binding_data = parts[3]  # e.g., "Kd=2.3nM"
            ligand_name = parts[4]

            # Parse binding affinity
            try:
                if 'Kd=' in binding_data:
                    value = float(binding_data.split('=')[1].replace('nM', '').replace('uM', '').replace('mM', ''))
                    unit = 'nM' if 'nM' in binding_data else ('uM' if 'uM' in binding_data else 'mM')
                    affinity_type = 'Kd'
                elif 'Ki=' in binding_data:
                    value = float(binding_data.split('=')[1].replace('nM', '').replace('uM', '').replace('mM', ''))
                    unit = 'nM' if 'nM' in binding_data else ('uM' if 'uM' in binding_data else 'mM')
                    affinity_type = 'Ki'
                elif 'IC50=' in binding_data:
                    value = float(binding_data.split('=')[1].replace('nM', '').replace('uM', '').replace('mM', ''))
                    unit = 'nM' if 'nM' in binding_data else ('uM' if 'uM' in binding_data else 'mM')
                    affinity_type = 'IC50'
                else:
                    continue

                # Convert to pKd (negative log molar)
                if unit == 'nM':
                    pkd = -1 * (9 - np.log10(value))
                elif unit == 'uM':
                    pkd = -1 * (6 - np.log10(value))
                elif unit == 'mM':
                    pkd = -1 * (3 - np.log10(value))

                data.append({
                    'pdb_id': pdb_id,
                    'resolution': float(resolution) if resolution != 'NMR' else None,
                    'release_year': int(release_year),
                    'affinity_type': affinity_type,
                    'affinity_value': value,
                    'affinity_unit': unit,
                    'pKd': pkd,
                    'ligand_name': ligand_name
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(data)

def download_pdb_structure(pdb_id: str, output_dir: Path) -> Path:
    """Download PDB structure from RCSB"""
    pdb_file = output_dir / f"{pdb_id}_complex.pdb"

    if pdb_file.exists():
        return pdb_file

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)

    if response.status_code == 200:
        pdb_file.write_text(response.text)
        return pdb_file
    else:
        raise RuntimeError(f"Failed to download {pdb_id}")

def extract_ligand_from_pdb(pdb_file: Path, ligand_name: str, output_dir: Path) -> Tuple[Path, Path]:
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
    io.save(str(ligand_file), LigandSelect(ligand_name))

    # Save receptor (everything except ligand and water)
    class ReceptorSelect(Select):
        def accept_residue(self, residue):
            return residue.get_resname() not in [ligand_name, 'HOH', 'WAT']

    io.save(str(receptor_file), ReceptorSelect())

    return receptor_file, ligand_file

def convert_ligand_to_sdf(ligand_pdb: Path, output_sdf: Path):
    """Convert ligand PDB to SDF format"""
    if output_sdf.exists():
        return output_sdf

    mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False)
    if mol is None:
        raise RuntimeError(f"Failed to read ligand: {ligand_pdb}")

    # Add hydrogens if missing
    if mol.GetNumAtoms() == mol.GetNumHeavyAtoms():
        mol = Chem.AddHs(mol, addCoords=True)

    writer = Chem.SDWriter(str(output_sdf))
    writer.write(mol)
    writer.close()

    return output_sdf

def prepare_benchmark_set(output_base_dir: Path, max_complexes: int = None):
    """
    Complete workflow to prepare PDBbind Core Set benchmark

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

    # Download and parse index
    print("Step 1: Downloading PDBbind index...")
    index_file = download_pdbbind_index(output_base_dir)
    if index_file is None:
        print("ERROR: Could not download index file.")
        print("Please manually download INDEX_core_data.2020 from http://www.pdbbind.org.cn/")
        return

    print("\nStep 2: Parsing index file...")
    df = parse_pdbbind_index(index_file)
    print(f"Found {len(df)} complexes in Core Set")

    # Quality filtering
    print("\nStep 3: Filtering by quality criteria...")
    df_filtered = df[
        (df['resolution'].notna()) &  # X-ray structures only
        (df['resolution'] <= 2.5) &   # Good resolution
        (df['pKd'] >= 4.0) &          # Reasonable affinity
        (df['pKd'] <= 11.0)           # Realistic range
    ].copy()

    print(f"After filtering: {len(df_filtered)} complexes")

    if max_complexes:
        df_filtered = df_filtered.head(max_complexes)
        print(f"Limited to {max_complexes} complexes for testing")

    # Save metadata
    metadata_file = output_base_dir / "benchmark_metadata.csv"
    df_filtered.to_csv(metadata_file, index=False)
    print(f"\nSaved metadata to {metadata_file}")

    # Download and process structures
    print(f"\nStep 4: Downloading and processing {len(df_filtered)} structures...")

    successful = []
    failed = []

    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        pdb_id = row['pdb_id']
        ligand_name = row['ligand_name']

        try:
            # Download PDB
            pdb_file = download_pdb_structure(pdb_id, pdb_dir)

            # Extract receptor and ligand
            receptor_file, ligand_pdb = extract_ligand_from_pdb(pdb_file, ligand_name, receptor_dir)

            # Convert ligand to SDF
            ligand_sdf = ligand_dir / f"{pdb_id}_ligand.sdf"
            convert_ligand_to_sdf(ligand_pdb, ligand_sdf)

            successful.append(pdb_id)

        except Exception as e:
            print(f"\nFailed to process {pdb_id}: {e}")
            failed.append(pdb_id)

    print(f"\n{'='*60}")
    print(f"Successfully processed: {len(successful)} complexes")
    print(f"Failed: {len(failed)} complexes")

    if failed:
        print(f"\nFailed PDB IDs: {', '.join(failed)}")
        failed_file = output_base_dir / "failed_pdbs.txt"
        failed_file.write_text('\n'.join(failed))

    # Create summary
    summary = {
        'total_in_index': len(df),
        'after_filtering': len(df_filtered),
        'successfully_processed': len(successful),
        'failed': len(failed),
        'mean_resolution': df_filtered['resolution'].mean(),
        'mean_pKd': df_filtered['pKd'].mean(),
        'pKd_range': (df_filtered['pKd'].min(), df_filtered['pKd'].max())
    }

    print(f"\n{'='*60}")
    print("Benchmark Set Summary:")
    print(f"  Total complexes: {summary['successfully_processed']}")
    print(f"  Mean resolution: {summary['mean_resolution']:.2f} Å")
    print(f"  Mean pKd: {summary['mean_pKd']:.2f}")
    print(f"  pKd range: {summary['pKd_range'][0]:.1f} - {summary['pKd_range'][1]:.1f}")
    print(f"\nReady for benchmarking!")
    print(f"  Receptors: {receptor_dir}")
    print(f"  Ligands: {ligand_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare PDBbind Core Set for benchmarking")
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmarking/pdbbind_core_set"),
                       help="Output directory")
    parser.add_argument("-n", "--max-complexes", type=int, default=None,
                       help="Limit number of complexes (for testing)")

    args = parser.parse_args()

    prepare_benchmark_set(args.output, args.max_complexes)
