#!/usr/bin/env python
# vs_run.py - Standalone virtual screening script for PandaDock

import os
import sys
import argparse
from pathlib import Path

from pandadock.batch_screening import run, run_parallel
from pandadock.main_integration import configure_hardware


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PandaDock Virtual Screening Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('-p', '--protein', required=True, 
                       help='Path to protein PDB file')
    parser.add_argument('-l', '--ligand-library', required=True, 
                       help='Path to directory containing ligand files')
    parser.add_argument('-o', '--output', default='screening_results',
                       help='Output directory for screening results')
    
    # Optional arguments
    parser.add_argument('-a', '--algorithm', choices=['genetic', 'random', 'pandadock'], 
                       default='genetic', help='Docking algorithm to use')
    parser.add_argument('-i', '--iterations', type=int, default=500,
                       help='Number of iterations/generations per ligand')
    parser.add_argument('-e', '--exhaustiveness', type=int, default=8,
                       help='Exhaustiveness of search (1-64)')
    parser.add_argument('--population-size', type=int, default=100,
                       help='Population size for genetic algorithm')
    parser.add_argument('--enhanced-scoring', action='store_true',
                       help='Use enhanced scoring function')
    parser.add_argument('--local-opt', action='store_true',
                       help='Enable local optimization on top poses')
    parser.add_argument('--prepare-molecules', action='store_true',
                       help='Prepare protein and ligands before docking')
    parser.add_argument('--ph', type=float, default=7.4,
                       help='pH for protein preparation')
    
    # Binding site options
    parser.add_argument('-s', '--site', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Active site center coordinates')
    parser.add_argument('-r', '--radius', type=float, default=10.0,
                       help='Active site radius in Angstroms')
    parser.add_argument('--detect-pockets', action='store_true',
                       help='Automatically detect binding pockets')
    
    # Hardware options
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration if available')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID to use')
    parser.add_argument('--cpu-workers', type=int, default=None,
                       help='Number of CPU workers for parallel processing')
    
    # Parallel processing options
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing for screening')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes')
                       
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Configure hardware
    hw_config = configure_hardware(args)
    
    # Setup screening configuration
    screening_config = {
        'protein': args.protein,
        'ligand_library': args.ligand_library,
        'output_dir': args.output,
        'screening_params': {
            'algorithm': args.algorithm,
            'iterations': args.iterations,
            'population_size': args.population_size,
            'exhaustiveness': args.exhaustiveness,
            'scoring_function': 'enhanced' if args.enhanced_scoring else 'standard',
            'local_opt': args.local_opt,
            'prepare_molecules': args.prepare_molecules,
            'hardware': hw_config,
            'site': args.site,
            'radius': args.radius,
            'detect_pockets': args.detect_pockets,
            'ph': args.ph,
        },
        'n_processes': args.processes,
    }
    
    # Print configuration
    print(f"======= PandaDock Virtual Screening =======")
    print(f"Protein: {args.protein}")
    print(f"Ligand library: {args.ligand_library}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Exhaustiveness: {args.exhaustiveness}")
    print(f"GPU acceleration: {'Enabled' if args.use_gpu else 'Disabled'}")
    print(f"Parallel processing: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"==========================================")
    
    try:
        # Run virtual screening
        if args.parallel:
            results = run_parallel(screening_config)
        else:
            results = run(screening_config)
            
        print(f"Virtual screening completed successfully!")
        print(f"Results saved to: {args.output}")
        return 0
        
    except Exception as e:
        print(f"Error during virtual screening: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())