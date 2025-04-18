#!/usr/bin/env python3
"""
GPU-accelerated docking example using PandaDock.

This script demonstrates how to perform molecular docking with GPU acceleration
and CPU parallelization using PandaDock's hybrid manager.
"""

import os
import argparse
import time
from pathlib import Path

from pandadock.protein import Protein
from pandadock.ligand import Ligand
from pandadock.hybrid_manager import HybridDockingManager
from pandadock.utils import save_docking_results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GPU/CPU Accelerated Molecular Docking")
    
    # Required arguments
    parser.add_argument('-p', '--protein', required=True, help='Path to protein PDB file')
    parser.add_argument('-l', '--ligand', required=True, help='Path to ligand MOL/SDF file')
    
    # Optional arguments
    parser.add_argument('-o', '--output', default='gpu_docking_results',
                        help='Output directory for docking results')
    parser.add_argument('-s', '--site', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='Active site center coordinates')
    parser.add_argument('-r', '--radius', type=float, default=10.0,
                        help='Active site radius in Angstroms (default: 10.0)')
    
    # Algorithm options
    parser.add_argument('-a', '--algorithm', choices=['genetic', 'random', 'monte-carlo'],
                        default='genetic', help='Docking algorithm to use (default: genetic)')
    parser.add_argument('-i', '--iterations', type=int, default=100,
                        help='Number of iterations/generations (default: 100)')
    parser.add_argument('--population-size', type=int, default=100,
                        help='Population size for genetic algorithm (default: 100)')
    parser.add_argument('--mc-steps', type=int, default=1000,
                        help='Number of Monte Carlo steps (default: 1000)')
    parser.add_argument('--temperature', type=float, default=300.0,
                        help='Temperature for Monte Carlo simulation in Kelvin (default: 300K)')
    
    # GPU options
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    
    # CPU options
    parser.add_argument('--cpu-workers', type=int, default=None,
                        help='Number of CPU workers (default: all cores)')
    
    # Ensemble options
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of independent docking runs (default: 1)')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Start timer
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize hybrid docking manager
    print("Initializing hybrid docking manager...")
    hybrid_manager = HybridDockingManager(
        use_gpu=args.use_gpu,
        n_cpu_workers=args.cpu_workers,
        gpu_device_id=args.gpu_id
    )
    
    # Load molecules
    print(f"Loading protein from {args.protein}...")
    protein = Protein(args.protein)
    
    print(f"Loading ligand from {args.ligand}...")
    ligand = Ligand(args.ligand)
    
    # Define active site if coordinates provided
    if args.site:
        print(f"Using active site at {args.site} with radius {args.radius}Å")
        protein.define_active_site(args.site, args.radius)
    else:
        print("No active site specified, attempting to detect pockets...")
        pockets = protein.detect_pockets()
        if pockets:
            print(f"Found {len(pockets)} potential binding pockets")
            print(f"Using largest pocket as active site")
            protein.define_active_site(pockets[0]['center'], pockets[0]['radius'])
        else:
            print("No pockets detected, using whole protein")
    
    # Set up algorithm parameters based on algorithm type
    algorithm_kwargs = {'max_iterations': args.iterations}
    
    if args.algorithm == 'genetic':
        algorithm_kwargs['population_size'] = args.population_size
    elif args.algorithm == 'monte-carlo':
        algorithm_kwargs['n_steps'] = args.mc_steps
        algorithm_kwargs['temperature'] = args.temperature
    
    # Run docking
    if args.runs > 1:
        print(f"Running ensemble docking with {args.runs} independent runs...")
        print(f"Using {args.algorithm} algorithm")
        
        results = hybrid_manager.run_ensemble_docking(
            protein=protein,
            ligand=ligand,
            n_runs=args.runs,
            algorithm_type=args.algorithm,
            **algorithm_kwargs
        )
    else:
        print(f"Running single docking simulation using {args.algorithm} algorithm...")
        
        results = hybrid_manager.run_docking(
            protein=protein,
            ligand=ligand,
            algorithm_type=args.algorithm,
            **algorithm_kwargs
        )
    
    # Clean up resources
    hybrid_manager.cleanup()
    
    # Save results
    save_docking_results(results, args.output)
    from .utils import save_complex_to_pdb
    for i, (pose, score) in enumerate(results[:10]):
        complex_path = Path(output_dir) / f"complex_pose_{i+1}_score_{score:.2f}.pdb"
        save_complex_to_pdb(protein, pose, complex_path)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\nDocking Summary:")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Best score: {results[0][1]:.4f}")
    print(f"  Results saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
