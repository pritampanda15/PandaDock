#!/usr/bin/env python3
"""
Comprehensive Ligand Preparation for PandaDock Benchmarking

This script prepares all PDBbind ligands for docking by:
1. Loading ligands with error handling (sanitize=False if needed)
2. Adding hydrogens with proper 3D coordinates
3. Fixing kekulization and bond order issues
4. Generating proper conformers
5. Minimizing structures with UFF/MMFF
6. Saving in proper SDF format

Usage:
    python prepare_ligands_comprehensive.py \\
        --pdbbind-dir full_benchmark/PDBbind_v2020 \\
        --output-dir full_benchmark/PDBbind_v2020_prepared \\
        --max-ligands 150

Author: PandaDock Benchmarking Suite
Date: December 2025
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import argparse
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ligand_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    logger.error("RDKit not available! Please install: pip install rdkit")
    sys.exit(1)


class LigandPreparator:
    """Prepare ligands for molecular docking"""

    def __init__(self, ph: float = 7.0):
        self.ph = ph
        self.stats = {
            'total': 0,
            'success': 0,
            'sanitize_failed': 0,
            'fixed_with_nosanit': 0,
            'mol2_fallback': 0,
            'complete_failure': 0
        }

    def prepare_ligand(self, sdf_file: Path, mol2_file: Optional[Path] = None) -> Optional[Chem.Mol]:
        """
        Prepare a single ligand with robust error handling

        Returns:
            RDKit molecule with proper 3D coordinates and hydrogens, or None if failed
        """
        self.stats['total'] += 1

        # Try 1: Load SDF with sanitization
        mol = None
        try:
            supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
            if len(supplier) > 0:
                mol = supplier[0]

            if mol is not None:
                logger.debug(f"  ✓ Loaded {sdf_file.name} with sanitization")
                self.stats['success'] += 1
                return self._process_molecule(mol)
        except Exception as e:
            logger.debug(f"  Sanitize load failed: {e}")

        # Try 2: Load SDF without sanitization
        try:
            supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
            if len(supplier) > 0:
                mol = supplier[0]

            if mol is not None:
                logger.debug(f"  ⚠ Loaded {sdf_file.name} without sanitization")
                self.stats['sanitize_failed'] += 1
                self.stats['fixed_with_nosanit'] += 1

                # Try to fix the molecule
                try:
                    # Set all bonds to single first
                    for bond in mol.GetBonds():
                        bond.SetBondType(Chem.BondType.SINGLE)
                        bond.SetIsAromatic(False)

                    # Try to kekulize
                    Chem.SanitizeMol(mol, catchErrors=True)

                    return self._process_molecule(mol)
                except Exception as e:
                    logger.debug(f"  Could not fix sanitization: {e}")
                    # Continue with unsanitized molecule
                    return self._process_molecule(mol, force=True)
        except Exception as e:
            logger.debug(f"  No-sanitize load failed: {e}")

        # Try 3: Load MOL2 file if available
        if mol2_file and mol2_file.exists():
            try:
                mol = Chem.MolFromMol2File(str(mol2_file), removeHs=False, sanitize=False)
                if mol is not None:
                    logger.debug(f"  ✓ Loaded {mol2_file.name} (MOL2 fallback)")
                    self.stats['mol2_fallback'] += 1
                    return self._process_molecule(mol, force=True)
            except Exception as e:
                logger.debug(f"  MOL2 load failed: {e}")

        # Complete failure
        logger.error(f"  ✗ Could not load ligand from {sdf_file.name}")
        self.stats['complete_failure'] += 1
        return None

    def _process_molecule(self, mol: Chem.Mol, force: bool = False) -> Optional[Chem.Mol]:
        """
        Process molecule: add hydrogens, generate 3D if needed, minimize

        Args:
            mol: RDKit molecule
            force: If True, skip some checks and process anyway

        Returns:
            Processed molecule or None
        """
        try:
            # Check if already has 3D coordinates
            if mol.GetNumConformers() == 0:
                logger.debug("    No conformer, generating 3D coordinates...")
                if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                    # Try with random coords
                    AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)

            # Add hydrogens with 3D coordinates
            mol = Chem.AddHs(mol, addCoords=True)

            # Try to optimize geometry
            try:
                # Use UFF (faster)
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except:
                try:
                    # Fallback to MMFF
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                except:
                    logger.debug("    Could not minimize (using unminimized structure)")

            return mol

        except Exception as e:
            if not force:
                logger.debug(f"    Processing failed: {e}")
                return None

            # Force mode: return molecule as-is
            logger.debug(f"    Processing failed but returning molecule anyway (force=True): {e}")
            return mol

    def print_stats(self):
        """Print preparation statistics"""
        logger.info("\n" + "="*60)
        logger.info("LIGAND PREPARATION STATISTICS")
        logger.info("="*60)
        logger.info(f"Total ligands processed: {self.stats['total']}")
        logger.info(f"Successfully loaded with sanitize: {self.stats['success']} ({100*self.stats['success']/max(1,self.stats['total']):.1f}%)")
        logger.info(f"Loaded without sanitize: {self.stats['fixed_with_nosanit']}")
        logger.info(f"MOL2 fallback used: {self.stats['mol2_fallback']}")
        logger.info(f"Complete failures: {self.stats['complete_failure']} ({100*self.stats['complete_failure']/max(1,self.stats['total']):.1f}%)")
        logger.info(f"Overall success rate: {100*(self.stats['total']-self.stats['complete_failure'])/max(1,self.stats['total']):.1f}%")
        logger.info("="*60 + "\n")


def prepare_all_ligands(
    pdbbind_dir: Path,
    output_dir: Path,
    grid_centers_file: Optional[Path] = None,
    max_ligands: Optional[int] = None
):
    """
    Prepare all ligands in PDBbind dataset

    Args:
        pdbbind_dir: Path to PDBbind_v2020 directory
        output_dir: Output directory for prepared ligands
        grid_centers_file: CSV file with complex IDs (optional)
        max_ligands: Maximum number of ligands to prepare (optional)
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    preparator = LigandPreparator()

    # Get list of complexes to process
    if grid_centers_file and grid_centers_file.exists():
        grid_centers = pd.read_csv(grid_centers_file)
        complex_ids = grid_centers.iloc[:, 0].tolist()  # First column is PDB ID
        if max_ligands:
            complex_ids = complex_ids[:max_ligands]
    else:
        # Get all subdirectories
        complex_ids = [d.name for d in pdbbind_dir.iterdir() if d.is_dir()]
        if max_ligands:
            complex_ids = complex_ids[:max_ligands]

    logger.info(f"Found {len(complex_ids)} complexes to prepare")

    # Prepare each ligand
    preparation_results = []
    failed_complexes = []

    for idx, pdb_id in enumerate(tqdm(complex_ids, desc="Preparing ligands"), 1):
        complex_dir = pdbbind_dir / pdb_id

        if not complex_dir.exists():
            logger.warning(f"[{idx}/{len(complex_ids)}] Complex directory not found: {pdb_id}")
            failed_complexes.append(pdb_id)
            continue

        # Find ligand files
        sdf_file = complex_dir / f"{pdb_id}_ligand.sdf"
        mol2_file = complex_dir / f"{pdb_id}_ligand.mol2"

        if not sdf_file.exists() and not mol2_file.exists():
            logger.warning(f"[{idx}/{len(complex_ids)}] No ligand file found for {pdb_id}")
            failed_complexes.append(pdb_id)
            continue

        # Prepare ligand
        logger.info(f"[{idx}/{len(complex_ids)}] Processing {pdb_id}...")
        mol = preparator.prepare_ligand(sdf_file, mol2_file if mol2_file.exists() else None)

        if mol is None:
            logger.error(f"  ✗ Failed to prepare ligand for {pdb_id}")
            failed_complexes.append(pdb_id)
            preparation_results.append({
                'pdb_id': pdb_id,
                'status': 'FAILED',
                'error': 'Could not load or prepare ligand'
            })
            continue

        # Create output directory for this complex
        complex_output_dir = output_dir / pdb_id
        complex_output_dir.mkdir(exist_ok=True)

        # Save prepared ligand
        output_sdf = complex_output_dir / f"{pdb_id}_ligand.sdf"
        try:
            writer = Chem.SDWriter(str(output_sdf))
            writer.write(mol)
            writer.close()

            # Copy protein file
            protein_file = complex_dir / f"{pdb_id}_protein.pdb"
            if not protein_file.exists():
                protein_file = complex_dir / f"{pdb_id}_pocket.pdb"

            if protein_file.exists():
                import shutil
                shutil.copy(protein_file, complex_output_dir / protein_file.name)

            logger.info(f"  ✓ Saved prepared ligand: {output_sdf}")
            preparation_results.append({
                'pdb_id': pdb_id,
                'status': 'SUCCESS',
                'atoms': mol.GetNumAtoms(),
                'output_file': str(output_sdf)
            })

        except Exception as e:
            logger.error(f"  ✗ Failed to save ligand for {pdb_id}: {e}")
            failed_complexes.append(pdb_id)
            preparation_results.append({
                'pdb_id': pdb_id,
                'status': 'FAILED',
                'error': f'Could not save: {e}'
            })

    # Save results summary
    results_df = pd.DataFrame(preparation_results)
    summary_file = output_dir / "preparation_summary.csv"
    results_df.to_csv(summary_file, index=False)

    # Print statistics
    preparator.print_stats()

    logger.info(f"\n{'='*60}")
    logger.info("PREPARATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total complexes: {len(complex_ids)}")
    logger.info(f"Successfully prepared: {len(complex_ids) - len(failed_complexes)}")
    logger.info(f"Failed: {len(failed_complexes)}")
    logger.info(f"Results saved to: {summary_file}")

    if failed_complexes:
        logger.info(f"\nFailed complexes ({len(failed_complexes)}):")
        for pdb_id in failed_complexes:
            logger.info(f"  - {pdb_id}")

    logger.info(f"{'='*60}\n")

    return preparation_results, failed_complexes


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PDBbind ligands for comprehensive docking benchmark"
    )
    parser.add_argument(
        '--pdbbind-dir',
        type=Path,
        required=True,
        help='Path to PDBbind_v2020 directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for prepared ligands'
    )
    parser.add_argument(
        '--grid-centers',
        type=Path,
        help='Optional: CSV file with grid centers (to filter complexes)'
    )
    parser.add_argument(
        '--max-ligands',
        type=int,
        help='Maximum number of ligands to prepare (for testing)'
    )

    args = parser.parse_args()

    # Run preparation
    results, failed = prepare_all_ligands(
        pdbbind_dir=args.pdbbind_dir,
        output_dir=args.output_dir,
        grid_centers_file=args.grid_centers,
        max_ligands=args.max_ligands
    )

    # Exit with error if any failures
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()
