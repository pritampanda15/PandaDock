#!/usr/bin/env python3
"""
Comprehensive PDBbind v2020 Benchmarking Script for PandaDock

This script performs large-scale benchmarking on the PDBbind v2020 dataset
with all available PandaDock algorithms for research publication.

Features:
- Supports all 8 PandaDock algorithms (5 CPU + 3 GPU)
- Processes 5,315 protein-ligand complexes from PDBbind v2020
- Calculates RMSD for pose prediction evaluation
- Tracks runtime for computational performance analysis
- Saves results progressively (resume capability)
- Publication-ready output format

Usage:
    # Quick test (10 complexes, 2 algorithms)
    python run_pdbbind_benchmark.py --max-complexes 10 \\
        --algorithms hierarchical_cpu cuda_genetic_algorithm

    # Full benchmark (all complexes, CPU algorithms only)
    python run_pdbbind_benchmark.py --algorithms \\
        monte_carlo_cpu genetic_algorithm_cpu hierarchical_cpu \\
        enhanced_hierarchical_cpu crystal_guided_cpu

    # Full benchmark with GPU acceleration
    python run_pdbbind_benchmark.py --algorithms \\
        monte_carlo_cpu genetic_algorithm_cpu hierarchical_cpu \\
        enhanced_hierarchical_cpu crystal_guided_cpu \\
        cuda_monte_carlo cuda_genetic_algorithm enhanced_hierarchical_gpu

Author: PandaDock Benchmarking Suite
Date: December 2025
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Suppress CuPy distance calculation warnings
import logging
logging.getLogger('cupy').setLevel(logging.ERROR)

# For timeout functionality
import signal
from contextlib import contextmanager

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pandadock.docking_cli import engine
from pandadock.docking.core import Pose

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    print("WARNING: RDKit not available. RMSD calculation will be disabled.")
    RDKIT_AVAILABLE = False


@contextmanager
def timeout(seconds):
    """Context manager for timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError("Operation timed out after {} seconds".format(seconds))

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class PDBbindBenchmarkRunner:
    """
    Comprehensive benchmarking runner for PDBbind v2020 dataset

    Attributes:
        pdbbind_dir: Path to PDBbind_v2020 directory
        grid_centers_file: Path to grid_centers.csv file
        output_dir: Directory for saving results
        algorithms: List of algorithms to benchmark
        box_padding: Padding around ligand for grid box (default: 8.0 Å)
        timeout_seconds: Maximum time allowed per docking (default: 600s = 10 min)
    """

    def __init__(
        self,
        pdbbind_dir: Path,
        grid_centers_file: Path,
        output_dir: Path,
        algorithms: List[str],
        box_padding: float = 8.0,
        max_complexes: Optional[int] = None,
        timeout_seconds: int = 600
    ):
        self.pdbbind_dir = pdbbind_dir
        self.grid_centers_file = grid_centers_file
        self.output_dir = output_dir
        self.algorithms = algorithms
        self.box_padding = box_padding
        self.max_complexes = max_complexes
        self.timeout_seconds = timeout_seconds

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = output_dir / "docking_results"
        self.results_dir.mkdir(exist_ok=True)
        self.poses_dir = output_dir / "top_poses"
        self.poses_dir.mkdir(exist_ok=True)

        # Load grid centers
        print(f"Loading grid centers from {grid_centers_file}")
        self.grid_centers = pd.read_csv(grid_centers_file)

        if max_complexes:
            self.grid_centers = self.grid_centers.head(max_complexes)

        print(f"Loaded {len(self.grid_centers)} complexes for benchmarking")

        # Validate algorithms
        self._validate_algorithms()

        # Initialize progress tracker
        self.progress_file = output_dir / "progress.json"
        self.completed_tasks = self._load_progress()

    def _validate_algorithms(self):
        """Validate that all requested algorithms are available"""
        available = list(engine._algorithms.keys())
        invalid = [alg for alg in self.algorithms if alg not in available]

        if invalid:
            print(f"\nERROR: Invalid algorithms: {invalid}")
            print(f"Available algorithms: {available}")
            sys.exit(1)

        print(f"\nSelected algorithms ({len(self.algorithms)}):")
        for alg in self.algorithms:
            print(f"  ✓ {alg}")

    def _load_progress(self) -> Dict:
        """Load progress from previous run"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_progress(self):
        """Save progress to allow resume"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.completed_tasks, f, indent=2)

    def get_complex_files(self, pdb_id: str) -> Optional[Dict[str, Path]]:
        """Get file paths for a given PDB complex"""
        complex_dir = self.pdbbind_dir / pdb_id

        if not complex_dir.exists():
            print(f"  WARNING: Directory not found for {pdb_id}")
            return None

        # Check for protein file
        protein_file = complex_dir / f"{pdb_id}_protein.pdb"
        if not protein_file.exists():
            protein_file = complex_dir / f"{pdb_id}_pocket.pdb"

        if not protein_file.exists():
            print(f"  WARNING: No protein file found for {pdb_id}")
            return None

        # Check for ligand file
        ligand_file = complex_dir / f"{pdb_id}_ligand.sdf"
        if not ligand_file.exists():
            ligand_file = complex_dir / f"{pdb_id}_ligand.mol2"

        if not ligand_file.exists():
            print(f"  WARNING: No ligand file found for {pdb_id}")
            return None

        return {
            'protein': protein_file,
            'ligand': ligand_file,
            'complex_dir': complex_dir
        }

    def calculate_binding_box(
        self,
        ligand_file: Path,
        grid_center: Tuple[float, float, float]
    ) -> Dict[str, float]:
        """
        Calculate binding site box from ligand coordinates

        Args:
            ligand_file: Path to ligand SDF/MOL2 file
            grid_center: Pre-computed grid center (X, Y, Z)

        Returns:
            Dictionary with center and size coordinates
        """
        if not RDKIT_AVAILABLE:
            # Use default box size if RDKit not available
            return {
                'center_x': grid_center[0],
                'center_y': grid_center[1],
                'center_z': grid_center[2],
                'size_x': 25.0,
                'size_y': 25.0,
                'size_z': 25.0
            }

        # Load ligand
        if str(ligand_file).endswith('.sdf'):
            mol = Chem.MolFromMolFile(str(ligand_file), removeHs=False)
        elif str(ligand_file).endswith('.mol2'):
            mol = Chem.MolFromMol2File(str(ligand_file), removeHs=False)
        else:
            mol = None

        if mol is None or mol.GetNumConformers() == 0:
            # Fallback to default box
            return {
                'center_x': grid_center[0],
                'center_y': grid_center[1],
                'center_z': grid_center[2],
                'size_x': 25.0,
                'size_y': 25.0,
                'size_z': 25.0
            }

        # Get ligand coordinates
        conf = mol.GetConformer()
        coords = np.array([
            conf.GetAtomPosition(i)
            for i in range(mol.GetNumAtoms())
        ])

        # Calculate dimensions with padding
        dimensions = coords.max(axis=0) - coords.min(axis=0) + 2 * self.box_padding

        # Ensure minimum box size of 20 Å
        dimensions = np.maximum(dimensions, 20.0)

        return {
            'center_x': float(grid_center[0]),
            'center_y': float(grid_center[1]),
            'center_z': float(grid_center[2]),
            'size_x': float(dimensions[0]),
            'size_y': float(dimensions[1]),
            'size_z': float(dimensions[2])
        }

    def _create_molecule_from_pose(
        self,
        original_mol,
        pose: Pose
    ) -> Optional['Chem.Mol']:
        """
        Create an RDKit molecule from a Pose by updating coordinates

        Args:
            original_mol: Original ligand molecule (for atom types, bonds)
            pose: Pose object containing new coordinates

        Returns:
            New molecule with updated coordinates
        """
        if not RDKIT_AVAILABLE:
            return None

        try:
            # Create a copy of the original molecule
            mol_copy = Chem.Mol(original_mol)

            # Get or create a conformer
            if mol_copy.GetNumConformers() == 0:
                mol_copy.AddConformer(Chem.Conformer(mol_copy.GetNumAtoms()))

            # Update conformer coordinates
            conf = mol_copy.GetConformer(0)
            for i in range(min(mol_copy.GetNumAtoms(), len(pose.coordinates))):
                conf.SetAtomPosition(i, pose.coordinates[i].tolist())

            return mol_copy

        except Exception as e:
            print(f"    WARNING: Could not create molecule from pose: {e}")
            return None

    def calculate_rmsd(
        self,
        reference_file: Path,
        docked_mol: Optional['Chem.Mol']
    ) -> Optional[float]:
        """Calculate RMSD between reference and docked ligand"""
        if not RDKIT_AVAILABLE or docked_mol is None:
            return None

        try:
            # Load reference
            if str(reference_file).endswith('.sdf'):
                ref_mol = Chem.MolFromMolFile(str(reference_file), removeHs=False)
            elif str(reference_file).endswith('.mol2'):
                ref_mol = Chem.MolFromMol2File(str(reference_file), removeHs=False)
            else:
                return None

            if ref_mol is None:
                return None

            # Calculate RMSD (best alignment)
            rmsd = AllChem.GetBestRMS(ref_mol, docked_mol)
            return float(rmsd)

        except Exception as e:
            print(f"    WARNING: RMSD calculation failed: {e}")
            return None

    def run_single_docking(
        self,
        pdb_id: str,
        algorithm: str,
        files: Dict[str, Path],
        box: Dict[str, float]
    ) -> Dict:
        """Run docking for a single complex with one algorithm"""

        # Check if already completed
        task_id = f"{pdb_id}_{algorithm}"
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]

        # Prepare output paths
        result_dir = self.results_dir / algorithm / pdb_id
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / "result.json"

        # Skip if result file exists
        if result_file.exists():
            try:
                result_data = json.loads(result_file.read_text())
                self.completed_tasks[task_id] = result_data
                return result_data
            except:
                pass  # Re-run if JSON is corrupted

        try:
            start_time = time.time()

            # Load ligand
            if RDKIT_AVAILABLE:
                if str(files['ligand']).endswith('.sdf'):
                    ligand_mol = Chem.MolFromMolFile(str(files['ligand']), removeHs=False)
                elif str(files['ligand']).endswith('.mol2'):
                    ligand_mol = Chem.MolFromMol2File(str(files['ligand']), removeHs=False)
                else:
                    raise ValueError(f"Unsupported ligand format: {files['ligand']}")

                if ligand_mol is None:
                    raise ValueError(f"Could not load ligand: {files['ligand']}")
            else:
                raise RuntimeError("RDKit is required for docking")

            # Prepare grid
            grid_center = np.array([box['center_x'], box['center_y'], box['center_z']])
            grid_dimensions = np.array([box['size_x'], box['size_y'], box['size_z']])

            # Run docking with timeout protection
            alg_instance = engine._algorithms[algorithm]

            # Use timeout to prevent hanging
            try:
                with timeout(self.timeout_seconds):
                    docking_result = alg_instance.dock(
                        str(files['protein']),
                        ligand_mol,
                        grid_center,
                        grid_dimensions
                    )
            except TimeoutError as e:
                runtime = time.time() - start_time
                raise TimeoutError("Docking timed out after {}s".format(self.timeout_seconds))

            runtime = time.time() - start_time

            # Extract results
            if docking_result.poses and len(docking_result.poses) > 0:
                top_pose = docking_result.poses[0]
                best_score = top_pose.energy  # Use energy, not score

                # Create ligand molecule with pose coordinates
                pose_mol = self._create_molecule_from_pose(ligand_mol, top_pose)

                # Calculate RMSD
                rmsd = self.calculate_rmsd(files['ligand'], pose_mol)

                # Save top pose
                pose_file = result_dir / "top_pose.pdb"
                try:
                    if pose_mol:
                        Chem.MolToPDBFile(pose_mol, str(pose_file))
                except Exception as e:
                    print(f"    WARNING: Could not save pose: {e}")

                result_data = {
                    'pdb_id': pdb_id,
                    'algorithm': algorithm,
                    'success': True,
                    'runtime_seconds': round(runtime, 2),
                    'best_score': float(best_score),
                    'rmsd_angstrom': float(rmsd) if rmsd is not None else None,
                    'num_poses': len(docking_result.poses),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                result_data = {
                    'pdb_id': pdb_id,
                    'algorithm': algorithm,
                    'success': False,
                    'runtime_seconds': round(runtime, 2),
                    'error': 'No poses generated',
                    'timestamp': datetime.now().isoformat()
                }

            # Save result
            result_file.write_text(json.dumps(result_data, indent=2))

            # Update progress
            self.completed_tasks[task_id] = result_data

            return result_data

        except Exception as e:
            runtime = time.time() - start_time if 'start_time' in locals() else 0

            result_data = {
                'pdb_id': pdb_id,
                'algorithm': algorithm,
                'success': False,
                'runtime_seconds': round(runtime, 2),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

            # Save error result
            result_file.write_text(json.dumps(result_data, indent=2))
            self.completed_tasks[task_id] = result_data

            return result_data

    def run_benchmark(self):
        """Run complete benchmark across all complexes and algorithms"""

        print("\n" + "="*70)
        print("PANDADOCK PDBBIND V2020 BENCHMARKING")
        print("="*70)
        print(f"Complexes: {len(self.grid_centers)}")
        print(f"Algorithms: {len(self.algorithms)}")
        print(f"Total docking runs: {len(self.grid_centers) * len(self.algorithms)}")
        print(f"Output directory: {self.output_dir}")
        print("="*70 + "\n")

        all_results = []
        total_runs = len(self.grid_centers) * len(self.algorithms)
        completed_runs = len(self.completed_tasks)

        # Progress bar
        pbar = tqdm(
            total=total_runs,
            initial=completed_runs,
            desc="Overall progress",
            unit="docking"
        )

        start_time_benchmark = time.time()

        for idx, row in self.grid_centers.iterrows():
            pdb_id = row['ProteinID']
            grid_center = (row['X'], row['Y'], row['Z'])

            # Get complex files
            files = self.get_complex_files(pdb_id)
            if files is None:
                # Mark as failed for all algorithms
                for algorithm in self.algorithms:
                    all_results.append({
                        'pdb_id': pdb_id,
                        'algorithm': algorithm,
                        'success': False,
                        'error': 'Files not found'
                    })
                    pbar.update(1)
                continue

            # Calculate binding box
            box = self.calculate_binding_box(files['ligand'], grid_center)

            # Run with each algorithm
            for algorithm in self.algorithms:
                result = self.run_single_docking(pdb_id, algorithm, files, box)
                all_results.append(result)
                pbar.update(1)

                # Save progress every 10 runs
                if len(all_results) % 10 == 0:
                    self._save_progress()
                    self._save_intermediate_results(all_results)

        pbar.close()

        total_time = time.time() - start_time_benchmark

        # Save final results
        self._save_final_results(all_results)
        self._print_summary(all_results, total_time)

    def _save_intermediate_results(self, results: List[Dict]):
        """Save intermediate results"""
        df = pd.DataFrame(results)
        csv_file = self.output_dir / "benchmark_results_intermediate.csv"
        df.to_csv(csv_file, index=False)

    def _save_final_results(self, results: List[Dict]):
        """Save final benchmark results"""
        df = pd.DataFrame(results)

        # Save CSV
        csv_file = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✓ Results saved to: {csv_file}")

        # Save JSON
        json_file = self.output_dir / "benchmark_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {json_file}")

        # Save summary by algorithm
        self._save_algorithm_summary(df)

    def _save_algorithm_summary(self, df: pd.DataFrame):
        """Save per-algorithm summary statistics"""
        summary_data = []

        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]

            # Success metrics
            total = len(alg_data)
            successful = alg_data['success'].sum()
            success_rate = (successful / total * 100) if total > 0 else 0

            # RMSD metrics
            rmsd_data = alg_data[alg_data['rmsd_angstrom'].notna()]
            if len(rmsd_data) > 0:
                success_2a = (rmsd_data['rmsd_angstrom'] < 2.0).sum()
                success_3a = (rmsd_data['rmsd_angstrom'] < 3.0).sum()
                rate_2a = (success_2a / len(rmsd_data) * 100)
                rate_3a = (success_3a / len(rmsd_data) * 100)
                mean_rmsd = rmsd_data['rmsd_angstrom'].mean()
                median_rmsd = rmsd_data['rmsd_angstrom'].median()
            else:
                success_2a = success_3a = 0
                rate_2a = rate_3a = 0
                mean_rmsd = median_rmsd = None

            # Runtime metrics
            runtime_data = alg_data[alg_data['runtime_seconds'].notna()]
            if len(runtime_data) > 0:
                mean_runtime = runtime_data['runtime_seconds'].mean()
                median_runtime = runtime_data['runtime_seconds'].median()
                total_runtime = runtime_data['runtime_seconds'].sum()
            else:
                mean_runtime = median_runtime = total_runtime = None

            summary_data.append({
                'algorithm': algorithm,
                'total_complexes': total,
                'successful_runs': successful,
                'success_rate_%': round(success_rate, 2),
                'rmsd_2A_count': success_2a,
                'rmsd_2A_rate_%': round(rate_2a, 2),
                'rmsd_3A_count': success_3a,
                'rmsd_3A_rate_%': round(rate_3a, 2),
                'mean_rmsd_A': round(mean_rmsd, 3) if mean_rmsd else None,
                'median_rmsd_A': round(median_rmsd, 3) if median_rmsd else None,
                'mean_runtime_sec': round(mean_runtime, 2) if mean_runtime else None,
                'median_runtime_sec': round(median_runtime, 2) if median_runtime else None,
                'total_runtime_hours': round(total_runtime / 3600, 2) if total_runtime else None
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / "algorithm_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Algorithm summary saved to: {summary_file}")

    def _print_summary(self, results: List[Dict], total_time: float):
        """Print benchmark summary"""
        df = pd.DataFrame(results)

        print("\n" + "="*70)
        print("BENCHMARKING COMPLETE")
        print("="*70)
        print(f"Total runtime: {total_time/3600:.2f} hours")
        print(f"Total docking runs: {len(results)}")
        print()

        for algorithm in self.algorithms:
            alg_data = df[df['algorithm'] == algorithm]

            total = len(alg_data)
            successful = alg_data['success'].sum()
            success_rate = (successful / total * 100) if total > 0 else 0

            print(f"{algorithm}:")
            print(f"  Runs: {total}")
            print(f"  Success rate: {success_rate:.1f}%")

            if 'rmsd_angstrom' in alg_data.columns:
                rmsd_data = alg_data[alg_data['rmsd_angstrom'].notna()]
                if len(rmsd_data) > 0:
                    success_2a = (rmsd_data['rmsd_angstrom'] < 2.0).sum()
                    rate_2a = (success_2a / len(rmsd_data) * 100)
                    mean_rmsd = rmsd_data['rmsd_angstrom'].mean()
                    print(f"  RMSD < 2.0Å: {rate_2a:.1f}% ({success_2a}/{len(rmsd_data)})")
                    print(f"  Mean RMSD: {mean_rmsd:.2f} Å")

            if 'runtime_seconds' in alg_data.columns:
                runtime_data = alg_data[alg_data['runtime_seconds'].notna()]
                if len(runtime_data) > 0:
                    mean_runtime = runtime_data['runtime_seconds'].mean()
                    total_runtime = runtime_data['runtime_seconds'].sum()
                    print(f"  Mean runtime: {mean_runtime:.1f} seconds")
                    print(f"  Total runtime: {total_runtime/3600:.2f} hours")

            print()

        print("="*70)
        print("\nNext steps:")
        print(f"  1. Analyze results:")
        print(f"     python benchmarking/analyze_pdbbind_results.py \\")
        print(f"       --results {self.output_dir}/benchmark_results.csv \\")
        print(f"       --output {self.output_dir}/analysis")
        print()
        print(f"  2. Generate publication figures:")
        print(f"     python benchmarking/create_publication_figures.py \\")
        print(f"       --results {self.output_dir}/benchmark_results.csv")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive PDBbind v2020 benchmarking for PandaDock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 complexes, 2 algorithms)
  %(prog)s --max-complexes 10 --algorithms hierarchical_cpu cuda_genetic_algorithm

  # CPU algorithms only (full dataset)
  %(prog)s --algorithms monte_carlo_cpu genetic_algorithm_cpu hierarchical_cpu \\
      enhanced_hierarchical_cpu crystal_guided_cpu

  # All algorithms including GPU (full dataset)
  %(prog)s --algorithms monte_carlo_cpu genetic_algorithm_cpu hierarchical_cpu \\
      enhanced_hierarchical_cpu crystal_guided_cpu cuda_monte_carlo \\
      cuda_genetic_algorithm enhanced_hierarchical_gpu

  # Resume interrupted benchmark
  %(prog)s --algorithms hierarchical_cpu cuda_genetic_algorithm
  (Progress is automatically saved and resumed)

Available algorithms:
  CPU: monte_carlo_cpu, genetic_algorithm_cpu, hierarchical_cpu,
       enhanced_hierarchical_cpu, crystal_guided_cpu
  GPU: cuda_monte_carlo, cuda_genetic_algorithm, enhanced_hierarchical_gpu
        """
    )

    parser.add_argument(
        "--pdbbind-dir",
        type=Path,
        default=Path("benchmarking/full_benchmark/PDBbind_v2020"),
        help="Path to PDBbind_v2020 directory (default: benchmarking/full_benchmark/PDBbind_v2020)"
    )

    parser.add_argument(
        "--grid-centers",
        type=Path,
        default=Path("benchmarking/full_benchmark/grid_centers.csv"),
        help="Path to grid_centers.csv (default: benchmarking/full_benchmark/grid_centers.csv)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("benchmarking/pdbbind_results"),
        help="Output directory for results (default: benchmarking/pdbbind_results)"
    )

    parser.add_argument(
        "-a", "--algorithms",
        nargs='+',
        default=['hierarchical_cpu', 'enhanced_hierarchical_cpu'],
        help="Algorithms to benchmark (default: hierarchical_cpu enhanced_hierarchical_cpu)"
    )

    parser.add_argument(
        "--box-padding",
        type=float,
        default=8.0,
        help="Padding around ligand for grid box in Angstroms (default: 8.0)"
    )

    parser.add_argument(
        "--max-complexes",
        type=int,
        default=None,
        help="Maximum number of complexes to process (for testing)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each docking run (default: 600 = 10 minutes)"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.pdbbind_dir.exists():
        print(f"ERROR: PDBbind directory not found: {args.pdbbind_dir}")
        sys.exit(1)

    if not args.grid_centers.exists():
        print(f"ERROR: Grid centers file not found: {args.grid_centers}")
        sys.exit(1)

    # Create runner and execute
    runner = PDBbindBenchmarkRunner(
        pdbbind_dir=args.pdbbind_dir,
        grid_centers_file=args.grid_centers,
        output_dir=args.output_dir,
        algorithms=args.algorithms,
        box_padding=args.box_padding,
        max_complexes=args.max_complexes,
        timeout_seconds=args.timeout
    )

    runner.run_benchmark()


if __name__ == "__main__":
    main()
