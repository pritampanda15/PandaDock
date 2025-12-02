#!/usr/bin/env python3
"""
Run comprehensive benchmark comparison between PandaDock and other docking tools

Compares:
- PandaDock (multiple algorithms)
- AutoDock Vina
- Smina
- AutoDock-GPU (if available)
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from Bio.PDB import PDBParser
import multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pandadock.docking_cli import engine

class BenchmarkRunner:
    """Run docking benchmarks across multiple tools"""

    def __init__(self, benchmark_dir: Path, output_dir: Path, n_jobs: int = 1):
        self.benchmark_dir = benchmark_dir
        self.output_dir = output_dir
        self.n_jobs = n_jobs

        # Create output subdirectories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = output_dir / "docking_results"
        self.results_dir.mkdir(exist_ok=True)

        # Load metadata
        metadata_file = benchmark_dir / "benchmark_metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")

        self.metadata = pd.read_csv(metadata_file)
        print(f"Loaded {len(self.metadata)} complexes from benchmark set")

    def get_binding_site_box(self, ligand_file: Path, padding: float = 5.0) -> Dict:
        """Calculate binding site box from ligand coordinates"""
        mol = Chem.MolFromMolFile(str(ligand_file), removeHs=False)
        if mol is None:
            mol = Chem.MolFromPDBFile(str(ligand_file), removeHs=False)

        if mol is None:
            raise RuntimeError(f"Could not read ligand: {ligand_file}")

        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

        center = coords.mean(axis=0)
        dimensions = coords.max(axis=0) - coords.min(axis=0) + 2 * padding

        return {
            'center_x': float(center[0]),
            'center_y': float(center[1]),
            'center_z': float(center[2]),
            'size_x': float(dimensions[0]),
            'size_y': float(dimensions[1]),
            'size_z': float(dimensions[2])
        }

    def run_pandadock(self, pdb_id: str, receptor_file: Path, ligand_file: Path,
                     box: Dict, algorithm: str = "hierarchical_cpu_physics_based") -> Dict:
        """Run PandaDock docking"""
        output_subdir = self.results_dir / f"pandadock_{algorithm}" / pdb_id
        output_subdir.mkdir(parents=True, exist_ok=True)

        result_file = output_subdir / "result.json"

        # Skip if already done
        if result_file.exists():
            return json.loads(result_file.read_text())

        try:
            start_time = time.time()

            # Get ligand mol
            ligand_mol = Chem.MolFromMolFile(str(ligand_file), removeHs=False)

            grid_center = np.array([box['center_x'], box['center_y'], box['center_z']])
            grid_dimensions = np.array([box['size_x'], box['size_y'], box['size_z']])

            # Run docking
            if algorithm not in engine._algorithms:
                raise ValueError(f"Algorithm '{algorithm}' not found. Available: {list(engine._algorithms.keys())}")

            alg = engine._algorithms[algorithm]
            result = alg.dock(str(receptor_file), ligand_mol, grid_center, grid_dimensions)

            runtime = time.time() - start_time

            # Calculate RMSD to crystal structure
            ref_mol = Chem.MolFromMolFile(str(ligand_file), removeHs=False)
            if ref_mol and result.poses:
                from rdkit.Chem import AllChem
                rmsd = AllChem.GetBestRMS(ref_mol, result.poses[0].ligand_mol)
            else:
                rmsd = None

            # Save result
            result_data = {
                'pdb_id': pdb_id,
                'algorithm': algorithm,
                'success': True,
                'runtime': runtime,
                'best_score': result.poses[0].score if result.poses else None,
                'rmsd': rmsd,
                'num_poses': len(result.poses)
            }

            result_file.write_text(json.dumps(result_data, indent=2))

            # Save top pose
            if result.poses:
                pose_file = output_subdir / "top_pose.pdb"
                Chem.MolToPDBFile(result.poses[0].ligand_mol, str(pose_file))

            return result_data

        except Exception as e:
            print(f"PandaDock failed for {pdb_id}: {e}")
            result_data = {
                'pdb_id': pdb_id,
                'algorithm': algorithm,
                'success': False,
                'error': str(e)
            }
            result_file.write_text(json.dumps(result_data, indent=2))
            return result_data

    def run_vina(self, pdb_id: str, receptor_file: Path, ligand_file: Path, box: Dict) -> Dict:
        """Run AutoDock Vina"""
        output_subdir = self.results_dir / "vina" / pdb_id
        output_subdir.mkdir(parents=True, exist_ok=True)

        result_file = output_subdir / "result.json"
        if result_file.exists():
            return json.loads(result_file.read_text())

        try:
            # Prepare receptor (PDBQT)
            receptor_pdbqt = output_subdir / "receptor.pdbqt"
            if not receptor_pdbqt.exists():
                cmd = [
                    "prepare_receptor4.py",
                    "-r", str(receptor_file),
                    "-o", str(receptor_pdbqt),
                    "-A", "hydrogens"
                ]
                subprocess.run(cmd, check=True, capture_output=True)

            # Prepare ligand (PDBQT)
            ligand_pdbqt = output_subdir / "ligand.pdbqt"
            if not ligand_pdbqt.exists():
                cmd = [
                    "prepare_ligand4.py",
                    "-l", str(ligand_file),
                    "-o", str(ligand_pdbqt),
                    "-A", "hydrogens"
                ]
                subprocess.run(cmd, check=True, capture_output=True)

            # Run Vina
            output_pdbqt = output_subdir / "vina_out.pdbqt"
            log_file = output_subdir / "vina.log"

            start_time = time.time()

            cmd = [
                "vina",
                "--receptor", str(receptor_pdbqt),
                "--ligand", str(ligand_pdbqt),
                "--center_x", str(box['center_x']),
                "--center_y", str(box['center_y']),
                "--center_z", str(box['center_z']),
                "--size_x", str(box['size_x']),
                "--size_y", str(box['size_y']),
                "--size_z", str(box['size_z']),
                "--out", str(output_pdbqt),
                "--log", str(log_file),
                "--exhaustiveness", "8"
            ]

            subprocess.run(cmd, check=True, capture_output=True, timeout=600)

            runtime = time.time() - start_time

            # Parse results
            best_score = None
            if log_file.exists():
                with open(log_file) as f:
                    for line in f:
                        if line.strip().startswith('1'):
                            parts = line.split()
                            best_score = float(parts[1])
                            break

            # Calculate RMSD
            rmsd = None
            if output_pdbqt.exists():
                # Convert PDBQT to PDB
                output_pdb = output_subdir / "top_pose.pdb"
                with open(output_pdbqt) as f_in, open(output_pdb, 'w') as f_out:
                    for line in f_in:
                        if line.startswith('MODEL 1'):
                            continue
                        if line.startswith('ENDMDL'):
                            break
                        if not line.startswith('REMARK'):
                            f_out.write(line)

                # Calculate RMSD
                ref_mol = Chem.MolFromMolFile(str(ligand_file), removeHs=False)
                dock_mol = Chem.MolFromPDBFile(str(output_pdb), removeHs=False)

                if ref_mol and dock_mol:
                    from rdkit.Chem import AllChem
                    rmsd = AllChem.GetBestRMS(ref_mol, dock_mol)

            result_data = {
                'pdb_id': pdb_id,
                'algorithm': 'vina',
                'success': True,
                'runtime': runtime,
                'best_score': best_score,
                'rmsd': rmsd
            }

            result_file.write_text(json.dumps(result_data, indent=2))
            return result_data

        except Exception as e:
            print(f"Vina failed for {pdb_id}: {e}")
            result_data = {
                'pdb_id': pdb_id,
                'algorithm': 'vina',
                'success': False,
                'error': str(e)
            }
            result_file.write_text(json.dumps(result_data, indent=2))
            return result_data

    def run_smina(self, pdb_id: str, receptor_file: Path, ligand_file: Path, box: Dict) -> Dict:
        """Run Smina (Vina fork with more scoring functions)"""
        output_subdir = self.results_dir / "smina" / pdb_id
        output_subdir.mkdir(parents=True, exist_ok=True)

        result_file = output_subdir / "result.json"
        if result_file.exists():
            return json.loads(result_file.read_text())

        try:
            output_pdb = output_subdir / "smina_out.pdb"
            log_file = output_subdir / "smina.log"

            start_time = time.time()

            cmd = [
                "smina",
                "--receptor", str(receptor_file),
                "--ligand", str(ligand_file),
                "--center_x", str(box['center_x']),
                "--center_y", str(box['center_y']),
                "--center_z", str(box['center_z']),
                "--size_x", str(box['size_x']),
                "--size_y", str(box['size_y']),
                "--size_z", str(box['size_z']),
                "--out", str(output_pdb),
                "--log", str(log_file),
                "--exhaustiveness", "8"
            ]

            subprocess.run(cmd, check=True, capture_output=True, timeout=600)

            runtime = time.time() - start_time

            # Parse score from log
            best_score = None
            if log_file.exists():
                with open(log_file) as f:
                    for line in f:
                        if 'Affinity:' in line:
                            best_score = float(line.split(':')[1].strip().split()[0])
                            break

            # Calculate RMSD
            rmsd = None
            ref_mol = Chem.MolFromMolFile(str(ligand_file), removeHs=False)
            dock_mol = Chem.MolFromPDBFile(str(output_pdb), removeHs=False)

            if ref_mol and dock_mol:
                from rdkit.Chem import AllChem
                rmsd = AllChem.GetBestRMS(ref_mol, dock_mol)

            result_data = {
                'pdb_id': pdb_id,
                'algorithm': 'smina',
                'success': True,
                'runtime': runtime,
                'best_score': best_score,
                'rmsd': rmsd
            }

            result_file.write_text(json.dumps(result_data, indent=2))
            return result_data

        except Exception as e:
            print(f"Smina failed for {pdb_id}: {e}")
            result_data = {
                'pdb_id': pdb_id,
                'algorithm': 'smina',
                'success': False,
                'error': str(e)
            }
            result_file.write_text(json.dumps(result_data, indent=2))
            return result_data

    def _normalize_algorithm_name(self, algorithm: str) -> str:
        """Normalize algorithm name to match registered names"""
        # Map user-friendly names to registered names
        name_map = {
            'pandadock_hierarchical_cpu_physics_based': 'hierarchical_cpu',
            'pandadock_hierarchical_cpu': 'hierarchical_cpu',
            'pandadock_enhanced_hierarchical_cpu': 'enhanced_hierarchical_cpu',
            'pandadock_monte_carlo_cpu': 'monte_carlo_cpu',
            'pandadock_genetic_algorithm_cpu': 'genetic_algorithm_cpu',
            'pandadock_crystal_guided_cpu': 'crystal_guided_cpu',
            'pandadock_cuda_genetic_algorithm': 'cuda_genetic_algorithm',
            'pandadock_cuda_monte_carlo': 'cuda_monte_carlo',
            'pandadock_enhanced_hierarchical_gpu': 'enhanced_hierarchical_gpu',
        }

        # Remove pandadock_ prefix if present
        if algorithm.startswith('pandadock_'):
            base_name = algorithm.replace('pandadock_', '')
            # Check if it's already a valid registered name
            if base_name in engine._algorithms:
                return base_name
            # Check mapping
            if algorithm in name_map:
                return name_map[algorithm]
            return base_name

        return algorithm

    def run_single_complex(self, row: pd.Series, algorithms: List[str]) -> List[Dict]:
        """Run docking for a single complex with all specified algorithms"""
        pdb_id = row['pdb_id']

        receptor_file = self.benchmark_dir / "receptors" / f"{pdb_id}_receptor.pdb"
        ligand_file = self.benchmark_dir / "ligands" / f"{pdb_id}_ligand.sdf"

        if not receptor_file.exists() or not ligand_file.exists():
            print(f"Missing files for {pdb_id}")
            return []

        # Calculate binding site box
        box = self.get_binding_site_box(ligand_file)

        results = []

        for algorithm in algorithms:
            # Normalize algorithm name
            norm_algorithm = self._normalize_algorithm_name(algorithm)

            if norm_algorithm in engine._algorithms:
                # It's a PandaDock algorithm
                result = self.run_pandadock(pdb_id, receptor_file, ligand_file, box, norm_algorithm)
            elif algorithm == 'vina':
                result = self.run_vina(pdb_id, receptor_file, ligand_file, box)
            elif algorithm == 'smina':
                result = self.run_smina(pdb_id, receptor_file, ligand_file, box)
            else:
                print(f"Unknown algorithm: {algorithm}")
                print(f"Available PandaDock algorithms: {list(engine._algorithms.keys())}")
                continue

            results.append(result)

        return results

    def run_benchmark(self, algorithms: List[str]):
        """Run complete benchmark across all complexes and algorithms"""
        print(f"\nRunning benchmark with {len(algorithms)} algorithms on {len(self.metadata)} complexes")
        print(f"Algorithms: {', '.join(algorithms)}")

        all_results = []

        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            results = self.run_single_complex(row, algorithms)
            all_results.extend(results)

        # Save combined results
        results_df = pd.DataFrame(all_results)
        results_csv = self.output_dir / "benchmark_results.csv"
        results_df.to_csv(results_csv, index=False)

        print(f"\nBenchmark complete! Results saved to {results_csv}")

        # Print summary statistics
        self.print_summary(results_df)

    def print_summary(self, results_df: pd.DataFrame):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        for algorithm in results_df['algorithm'].unique():
            alg_data = results_df[results_df['algorithm'] == algorithm]

            success_rate = (alg_data['success'].sum() / len(alg_data)) * 100

            if 'rmsd' in alg_data.columns:
                rmsd_data = alg_data[alg_data['rmsd'].notna()]
                success_2a = (rmsd_data['rmsd'] < 2.0).sum() / len(rmsd_data) * 100 if len(rmsd_data) > 0 else 0
                mean_rmsd = rmsd_data['rmsd'].mean() if len(rmsd_data) > 0 else None
            else:
                success_2a = 0
                mean_rmsd = None

            if 'runtime' in alg_data.columns:
                mean_runtime = alg_data[alg_data['runtime'].notna()]['runtime'].mean()
            else:
                mean_runtime = None

            print(f"\n{algorithm}:")
            print(f"  Completion rate: {success_rate:.1f}%")
            if mean_rmsd is not None:
                print(f"  Success rate (RMSD < 2Å): {success_2a:.1f}%")
                print(f"  Mean RMSD: {mean_rmsd:.2f} Å")
            if mean_runtime is not None:
                print(f"  Mean runtime: {mean_runtime:.1f} seconds")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run docking benchmark comparison")
    parser.add_argument("-b", "--benchmark-dir", type=Path, required=True,
                       help="Directory with benchmark dataset (from prepare_pdbbind_core.py)")
    parser.add_argument("-o", "--output-dir", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("-a", "--algorithms", nargs='+',
                       default=['monte_carlo_cpu',
                               'genetic_algorithm_cpu',
                               'hierarchical_cpu',
                               'enhanced_hierarchical_cpu',
                               'crystal_guided_cpu'],
                       help="Algorithms to test (available: monte_carlo_cpu, genetic_algorithm_cpu, hierarchical_cpu, enhanced_hierarchical_cpu, crystal_guided_cpu, cuda_genetic_algorithm, cuda_monte_carlo, enhanced_hierarchical_gpu, vina, smina)")
    parser.add_argument("-j", "--n-jobs", type=int, default=1,
                       help="Number of parallel jobs")

    args = parser.parse_args()

    runner = BenchmarkRunner(args.benchmark_dir, args.output_dir, args.n_jobs)
    runner.run_benchmark(args.algorithms)

if __name__ == "__main__":
    main()
