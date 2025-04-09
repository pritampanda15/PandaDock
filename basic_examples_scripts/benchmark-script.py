#!/usr/bin/env python3
"""
Hardware benchmark for PandaDock.

This script runs a series of performance tests to measure the speed of
different hardware configurations for molecular docking with support for
flexible residue benchmarking.
"""

import os
import argparse
import time
import json
import datetime
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import multiprocessing

from pandadock.protein import Protein
from pandadock.ligand import Ligand
from pandadock.scoring import EnhancedScoringFunction
from pandadock.hybrid_manager import HybridDockingManager, HardwareInfo
from pandadock.gpu_scoring import GPUAcceleratedScoringFunction


# Helper function for multiprocessing - must be at module level
def _score_pose_mp(args):
    """
    Score a single pose (multiprocessing-compatible helper function).
    """
    protein, scoring_function, pose = args
    return scoring_function.score(protein, pose)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PandaDock Hardware Benchmark")
    
    # Required arguments
    parser.add_argument('-p', '--protein', required=True, help='Path to protein PDB file')
    parser.add_argument('-l', '--ligand', required=True, help='Path to ligand MOL/SDF file')
    
    # Optional arguments
    parser.add_argument('-o', '--output', default='benchmark_results',
                        help='Output directory for benchmark results')
    parser.add_argument('-t', '--test-sizes', type=int, nargs='+', default=[10, 50, 100],
                        help='Test sizes (number of poses) to benchmark (default: 10 50 100)')
    parser.add_argument('--site', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='Active site center coordinates')
    parser.add_argument('--radius', type=float, default=10.0,
                        help='Active site radius in Angstroms (default: 10.0)')
    
    # Hardware options
    parser.add_argument('--skip-gpu', action='store_true',
                        help='Skip GPU benchmarks')
    parser.add_argument('--skip-cpu', action='store_true',
                        help='Skip CPU benchmarks')
    parser.add_argument('--skip-hybrid', action='store_true',
                        help='Skip hybrid benchmarks')
    parser.add_argument('--cpu-workers', type=int, default=None,
                        help='Number of CPU workers (default: all cores)')
    
    # Flexible residue options
    parser.add_argument('--flex-residues', nargs='+', 
                        help='Specify flexible residue IDs (e.g., A_42 B_57)')
    parser.add_argument('--max-flex-bonds', type=int, default=3,
                        help='Maximum rotatable bonds per residue (default: 3)')
    parser.add_argument('--auto-flex', action='store_true',
                        help='Automatically define flexible residues near binding site')
    parser.add_argument('--flex-radius', type=float, default=5.0,
                        help='Radius for automatic flexible residue detection')
    parser.add_argument('--compare-flex', action='store_true',
                        help='Run both rigid and flexible benchmarks for comparison')
    
    return parser.parse_args()


def generate_random_poses(protein, ligand, n_poses):
    """
    Generate random ligand poses within the active site.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    ligand : Ligand
        Ligand object
    n_poses : int
        Number of poses to generate
    
    Returns:
    --------
    list
        List of random ligand poses
    """
    import random
    from scipy.spatial.transform import Rotation
    
    # Determine search space
    if protein.active_site:
        center = protein.active_site['center']
        radius = protein.active_site['radius']
    else:
        # Use protein center of mass
        center = np.mean(protein.xyz, axis=0)
        radius = 15.0  # Arbitrary search radius
    
    # Generate poses
    poses = []
    for _ in range(n_poses):
        # Make a deep copy of the ligand
        pose = copy.deepcopy(ligand)
        
        # Generate random position within sphere
        r = radius * random.random() ** (1.0/3.0)
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        
        x = center[0] + r * np.sin(phi) * np.cos(theta)
        y = center[1] + r * np.sin(phi) * np.sin(theta)
        z = center[2] + r * np.cos(phi)
        
        # Calculate translation vector
        centroid = np.mean(pose.xyz, axis=0)
        translation = np.array([x, y, z]) - centroid
        
        # Apply translation
        pose.translate(translation)
        
        # Generate random rotation
        rotation = Rotation.random()
        rotation_matrix = rotation.as_matrix()
        
        # Apply rotation around the new center
        centroid = np.mean(pose.xyz, axis=0)
        pose.translate(-centroid)
        pose.rotate(rotation_matrix)
        pose.translate(centroid)
        
        poses.append(pose)
    
    return poses


def run_scoring_benchmark(protein, ligand, n_poses, mode, cpu_workers=None, gpu_id=0):
    """
    Run scoring function benchmark.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    ligand : Ligand
        Ligand object
    n_poses : int
        Number of poses to generate and score
    mode : str
        Benchmark mode ('cpu', 'gpu', or 'hybrid')
    cpu_workers : int
        Number of CPU workers to use
    gpu_id : int
        GPU device ID to use
    
    Returns:
    --------
    dict
        Benchmark results
    """
    print(f"Running {mode} scoring benchmark with {n_poses} poses...")
    
    # Check if protein has flexible residues
    has_flexible_residues = hasattr(protein, 'flexible_residues') and protein.flexible_residues
    if has_flexible_residues:
        print(f"  Using {len(protein.flexible_residues)} flexible residues")
    
    # Generate random poses - moved to a separate function
    poses = generate_random_poses(protein, ligand, n_poses)
    
    # Set up scoring function based on mode
    if mode == 'cpu':
        # Standard CPU scoring
        scoring_function = EnhancedScoringFunction()
        
        # Time scoring
        start_time = time.time()
        scores = []
        
        for pose in poses:
            score = scoring_function.score(protein, pose)
            scores.append(score)
        
        elapsed_time = time.time() - start_time
        
    elif mode == 'gpu':
        # GPU scoring
        hybrid_manager = HybridDockingManager(use_gpu=True, gpu_device_id=gpu_id)
        
        if hybrid_manager.has_gpu:
            scoring_function = hybrid_manager.prepare_gpu_scoring_function(
                GPUAcceleratedScoringFunction
            )
            
            # Time scoring
            start_time = time.time()
            scores = []
            
            for pose in poses:
                score = scoring_function.score(protein, pose)
                scores.append(score)
            
            elapsed_time = time.time() - start_time
            
            # Clean up GPU resources
            hybrid_manager.cleanup()
        else:
            # If GPU not available, report failure
            return {
                'mode': mode,
                'n_poses': n_poses,
                'success': False,
                'error': 'GPU not available',
                'has_flexible_residues': has_flexible_residues
            }
    
    elif mode == 'hybrid':
        # Hybrid CPU/GPU scoring
        hybrid_manager = HybridDockingManager(
            use_gpu=True,
            n_cpu_workers=cpu_workers,
            gpu_device_id=gpu_id
        )
        
        if hybrid_manager.has_gpu:
            scoring_function = hybrid_manager.prepare_gpu_scoring_function(
                GPUAcceleratedScoringFunction
            )
        else:
            scoring_function = EnhancedScoringFunction()
        
        # Time scoring using process pool
        start_time = time.time()
        
        if hybrid_manager.cpu_pool:
            # Create a list of argument tuples for the helper function
            args_list = [(protein, scoring_function, pose) for pose in poses]
            scores = hybrid_manager.cpu_pool.map(_score_pose_mp, args_list)
        else:
            scores = [scoring_function.score(protein, pose) for pose in poses]
        
        elapsed_time = time.time() - start_time
        
        # Clean up resources
        hybrid_manager.cleanup()
    
    else:
        return {
            'mode': mode,
            'n_poses': n_poses,
            'success': False,
            'error': f'Unknown mode: {mode}',
            'has_flexible_residues': has_flexible_residues
        }
    
    # Calculate statistics
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    poses_per_second = n_poses / elapsed_time
    
    # Return benchmark results
    return {
        'mode': mode,
        'n_poses': n_poses,
        'elapsed_time': elapsed_time,
        'poses_per_second': poses_per_second,
        'avg_score': avg_score,
        'min_score': min_score,
        'max_score': max_score,
        'success': True,
        'has_flexible_residues': has_flexible_residues,
        'flexible_residue_count': len(protein.flexible_residues) if has_flexible_residues else 0
    }


# Module-level search function for multiprocessing
def _search_algorithm_worker(args):
    """Search algorithm worker function for multiprocessing."""
    protein, ligand, algorithm = args
    return algorithm.search(protein, ligand)


def run_search_benchmark(protein, ligand, n_iterations, mode, cpu_workers=None, gpu_id=0):
    """
    Run search algorithm benchmark.
    
    Parameters:
    -----------
    protein : Protein
        Protein object
    ligand : Ligand
        Ligand object
    n_iterations : int
        Number of iterations/generations
    mode : str
        Benchmark mode ('cpu', 'gpu', or 'hybrid')
    cpu_workers : int
        Number of CPU workers to use
    gpu_id : int
        GPU device ID to use
    
    Returns:
    --------
    dict
        Benchmark results
    """
    from pandadock.search import GeneticAlgorithm
    from pandadock.parallel_search import ParallelGeneticAlgorithm
    
    print(f"Running {mode} search benchmark with {n_iterations} iterations...")
    
    # Check if protein has flexible residues
    has_flexible_residues = hasattr(protein, 'flexible_residues') and protein.flexible_residues
    if has_flexible_residues:
        print(f"  Using {len(protein.flexible_residues)} flexible residues")
    
    # Set up algorithm based on mode
    if mode == 'cpu':
        # Standard CPU genetic algorithm
        scoring_function = EnhancedScoringFunction()
        algorithm = GeneticAlgorithm(
            scoring_function=scoring_function,
            max_iterations=n_iterations,
            population_size=50
        )
        
        # Time search
        start_time = time.time()
        results = algorithm.search(protein, ligand)
        elapsed_time = time.time() - start_time
        
    elif mode == 'gpu':
        # GPU-accelerated genetic algorithm
        hybrid_manager = HybridDockingManager(use_gpu=True, gpu_device_id=gpu_id)
        
        if hybrid_manager.has_gpu:
            scoring_function = hybrid_manager.prepare_gpu_scoring_function(
                GPUAcceleratedScoringFunction
            )
            
            algorithm = GeneticAlgorithm(
                scoring_function=scoring_function,
                max_iterations=n_iterations,
                population_size=50
            )
            
            # Time search
            start_time = time.time()
            results = algorithm.search(protein, ligand)
            elapsed_time = time.time() - start_time
            
            # Clean up GPU resources
            hybrid_manager.cleanup()
        else:
            # If GPU not available, report failure
            return {
                'mode': mode,
                'n_iterations': n_iterations,
                'success': False,
                'error': 'GPU not available',
                'has_flexible_residues': has_flexible_residues
            }
    
    elif mode == 'hybrid':
        # Hybrid CPU/GPU genetic algorithm
        hybrid_manager = HybridDockingManager(
            use_gpu=True,
            n_cpu_workers=cpu_workers,
            gpu_device_id=gpu_id
        )
        
        if hybrid_manager.has_gpu:
            scoring_function = hybrid_manager.prepare_gpu_scoring_function(
                GPUAcceleratedScoringFunction
            )
        else:
            scoring_function = EnhancedScoringFunction()
        
        # Create algorithm with external process pool to avoid pickling issues
        algorithm = ParallelGeneticAlgorithm(
            scoring_function=scoring_function,
            max_iterations=n_iterations,
            population_size=50,
            n_processes=cpu_workers,
            process_pool=hybrid_manager.cpu_pool
        )
        
        # Make sure we use a picklable approach for search
        start_time = time.time()
        
        # Using the algorithm directly, which should handle parallelization internally
        results = algorithm.search(protein, ligand)
        
        elapsed_time = time.time() - start_time
        
        # Clean up resources
        hybrid_manager.cleanup()
    
    else:
        return {
            'mode': mode,
            'n_iterations': n_iterations,
            'success': False,
            'error': f'Unknown mode: {mode}',
            'has_flexible_residues': has_flexible_residues
        }
    
    # Calculate statistics
    best_score = results[0][1]
    
    # Return benchmark results
    return {
        'mode': mode,
        'n_iterations': n_iterations,
        'elapsed_time': elapsed_time,
        'iterations_per_second': n_iterations / elapsed_time,
        'best_score': best_score,
        'success': True,
        'has_flexible_residues': has_flexible_residues,
        'flexible_residue_count': len(protein.flexible_residues) if has_flexible_residues else 0
    }


def plot_benchmark_results(results, output_dir):
    """
    Plot benchmark results.
    
    Parameters:
    -----------
    results : dict
        Benchmark results
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract scoring benchmark data
    scoring_results = results.get('scoring_benchmark', [])
    if scoring_results:
        # Group by mode
        modes = set(r['mode'] for r in scoring_results)
        
        # Setup plot
        plt.figure(figsize=(10, 6))
        
        # Plot for each mode
        for mode in modes:
            # Split by flexible/rigid
            mode_results_rigid = [r for r in scoring_results if r['mode'] == mode 
                               and r['success'] and not r['has_flexible_residues']]
            mode_results_flex = [r for r in scoring_results if r['mode'] == mode 
                              and r['success'] and r['has_flexible_residues']]
            
            # Plot rigid results
            if mode_results_rigid:
                sizes = [r['n_poses'] for r in mode_results_rigid]
                speeds = [r['poses_per_second'] for r in mode_results_rigid]
                plt.plot(sizes, speeds, marker='o', label=f"{mode.upper()} (Rigid)")
            
            # Plot flexible results
            if mode_results_flex:
                sizes = [r['n_poses'] for r in mode_results_flex]
                speeds = [r['poses_per_second'] for r in mode_results_flex]
                plt.plot(sizes, speeds, marker='s', linestyle='--', 
                        label=f"{mode.upper()} (Flexible)")
        
        plt.title('Scoring Performance')
        plt.xlabel('Number of Poses')
        plt.ylabel('Poses per Second')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scoring_benchmark.png'))
        plt.close()
    
    # Extract search benchmark data
    search_results = results.get('search_benchmark', [])
    if search_results:
        # Group by mode
        modes = set(r['mode'] for r in search_results)
        
        # Setup plot
        plt.figure(figsize=(10, 6))
        
        # Plot for each mode
        for mode in modes:
            # Split by flexible/rigid
            mode_results_rigid = [r for r in search_results if r['mode'] == mode 
                               and r['success'] and not r['has_flexible_residues']]
            mode_results_flex = [r for r in search_results if r['mode'] == mode 
                              and r['success'] and r['has_flexible_residues']]
            
            # Plot rigid results
            if mode_results_rigid:
                iterations = [r['n_iterations'] for r in mode_results_rigid]
                times = [r['elapsed_time'] for r in mode_results_rigid]
                plt.plot(iterations, times, marker='o', label=f"{mode.upper()} (Rigid)")
            
            # Plot flexible results
            if mode_results_flex:
                iterations = [r['n_iterations'] for r in mode_results_flex]
                times = [r['elapsed_time'] for r in mode_results_flex]
                plt.plot(iterations, times, marker='s', linestyle='--',
                        label=f"{mode.upper()} (Flexible)")
        
        plt.title('Search Algorithm Performance')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'search_benchmark.png'))
        plt.close()
    
    # Plot flexible vs rigid comparison if both exist
    flex_results = [r for r in scoring_results if r['success'] and r['has_flexible_residues']]
    rigid_results = [r for r in scoring_results if r['success'] and not r['has_flexible_residues']]
    
    if flex_results and rigid_results:
        # Setup flexibility comparison plot
        plt.figure(figsize=(10, 6))
        
        # Group by mode
        modes = set(r['mode'] for r in scoring_results)
        
        # Calculate slowdown factors for each mode and size
        for mode in modes:
            mode_flex = [r for r in flex_results if r['mode'] == mode]
            mode_rigid = [r for r in rigid_results if r['mode'] == mode]
            
            # Find matching sizes
            sizes = sorted(list(set([r['n_poses'] for r in mode_flex]) & 
                               set([r['n_poses'] for r in mode_rigid])))
            
            if not sizes:
                continue
                
            slowdown_factors = []
            for size in sizes:
                flex_speed = next((r['poses_per_second'] for r in mode_flex 
                                 if r['n_poses'] == size), None)
                rigid_speed = next((r['poses_per_second'] for r in mode_rigid 
                                  if r['n_poses'] == size), None)
                
                if flex_speed and rigid_speed:
                    slowdown_factors.append(rigid_speed / flex_speed)
            
            if slowdown_factors:
                plt.plot(sizes, slowdown_factors, marker='o', label=f"{mode.upper()}")
        
        plt.title('Performance Impact of Flexible Residues')
        plt.xlabel('Number of Poses')
        plt.ylabel('Slowdown Factor (Rigid/Flexible)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'flex_impact_benchmark.png'))
        plt.close()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Print hardware info
    HardwareInfo.print_hardware_summary()
    
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
    
    # Initialize results dictionary
    benchmark_results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'protein': args.protein,
        'ligand': args.ligand,
        'scoring_benchmark': [],
        'search_benchmark': []
    }
    
    # Setup for flexible residue benchmarks
    proteins_to_benchmark = []
    
    # Create a deep copy of the protein for rigid benchmarks if compare-flex is used
    if args.compare_flex:
        import copy
        rigid_protein = copy.deepcopy(protein)
        proteins_to_benchmark.append(("rigid", rigid_protein))
    
    # Setup protein with flexible residues if requested
    if args.flex_residues or args.auto_flex:
        # Keep original protein as is if we're comparing
        if not args.compare_flex:
            proteins_to_benchmark.append(("rigid", protein))
            
            # Create a new copy for flexible residues
            import copy
            flex_protein = copy.deepcopy(protein)
        else:
            flex_protein = protein  # Use original protein for flexible residues
        
        # Set up flexible residues
        if args.flex_residues:
            print(f"Setting {len(args.flex_residues)} residues as flexible...")
            flex_protein.set_flexible_residues(args.flex_residues, max_bonds=args.max_flex_bonds)
        elif args.auto_flex and flex_protein.active_site:
            print(f"Auto-detecting flexible residues within {args.flex_radius}Å of active site...")
            flexible_res = flex_protein.find_residues_near_point(
                flex_protein.active_site['center'], 
                args.flex_radius
            )
            if flexible_res:
                # Limit to reasonable number
                if len(flexible_res) > 10:
                    print(f"Limiting to 10 flexible residues from {len(flexible_res)} detected")
                    flexible_res = flexible_res[:10]
                
                flex_protein.set_flexible_residues(flexible_res, max_bonds=args.max_flex_bonds)
                print(f"Set {len(flexible_res)} residues as flexible: {', '.join(flexible_res)}")
        
        proteins_to_benchmark.append(("flexible", flex_protein))
    else:
        # Just benchmark regular rigid protein
        proteins_to_benchmark.append(("rigid", protein))
    
    # Add flexibility info to benchmark results
    benchmark_results['flexibility'] = {
        'enabled': args.flex_residues is not None or args.auto_flex,
        'compare_mode': args.compare_flex,
        'method': 'manual' if args.flex_residues else 'auto' if args.auto_flex else 'none',
        'max_bonds': args.max_flex_bonds,
        'residue_ids': args.flex_residues if args.flex_residues else []
    }
    
    # Benchmark each protein configuration
    for protein_type, protein_obj in proteins_to_benchmark:
        print(f"\nRunning benchmarks for {protein_type} protein configuration...")
        
        # Run scoring benchmarks
        print("\nRunning scoring benchmarks...")
        
        for size in args.test_sizes:
            # CPU benchmark
            if not args.skip_cpu:
                result = run_scoring_benchmark(
                    protein=protein_obj,
                    ligand=ligand,
                    n_poses=size,
                    mode='cpu',
                    cpu_workers=args.cpu_workers
                )
                benchmark_results['scoring_benchmark'].append(result)
            
            # GPU benchmark
            if not args.skip_gpu:
                result = run_scoring_benchmark(
                    protein=protein_obj,
                    ligand=ligand,
                    n_poses=size,
                    mode='gpu'
                )
                benchmark_results['scoring_benchmark'].append(result)
            
            # Hybrid benchmark
            if not args.skip_hybrid:
                result = run_scoring_benchmark(
                    protein=protein_obj,
                    ligand=ligand,
                    n_poses=size,
                    mode='hybrid',
                    cpu_workers=args.cpu_workers
                )
                benchmark_results['scoring_benchmark'].append(result)
        
        # Run search benchmarks
        print("\nRunning search benchmarks...")
        
        for size in args.test_sizes:
            # Use smaller sizes for search benchmarks as they take longer
            search_size = max(10, size // 2)
            
            # CPU benchmark
            if not args.skip_cpu:
                result = run_search_benchmark(
                    protein=protein_obj,
                    ligand=ligand,
                    n_iterations=search_size,
                    mode='cpu',
                    cpu_workers=args.cpu_workers
                )
                benchmark_results['search_benchmark'].append(result)
            
            # GPU benchmark
            if not args.skip_gpu:
                result = run_search_benchmark(
                    protein=protein_obj,
                    ligand=ligand,
                    n_iterations=search_size,
                    mode='gpu'
                )
                benchmark_results['search_benchmark'].append(result)
            
            # Hybrid benchmark
            if not args.skip_hybrid:
                result = run_search_benchmark(
                    protein=protein_obj,
                    ligand=ligand,
                    n_iterations=search_size,
                    mode='hybrid',
                    cpu_workers=args.cpu_workers
                )
                benchmark_results['search_benchmark'].append(result)
    
    # Calculate total elapsed time
    elapsed_time = time.time() - start_time
    benchmark_results['total_time'] = elapsed_time
    
    # Save results
    results_file = os.path.join(args.output, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Plot results
    plot_benchmark_results(benchmark_results, args.output)
    
    # Print summary
    print("\nBenchmark completed successfully!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Results saved to: {args.output}")
    
    # Print recommendations
    print("\nPerformance Recommendations:")
    
    # Check if GPU benchmarks were successful
    gpu_results = [r for r in benchmark_results['scoring_benchmark'] 
                   if r['mode'] == 'gpu' and r.get('success', False)]
    
    if gpu_results:
        print("  - GPU acceleration is available and working")
        
        # Compare GPU vs CPU performance
        gpu_speeds = [r['poses_per_second'] for r in benchmark_results['scoring_benchmark'] 
                      if r['mode'] == 'gpu' and r.get('success', False)]
        cpu_speeds = [r['poses_per_second'] for r in benchmark_results['scoring_benchmark'] 
                      if r['mode'] == 'cpu' and r.get('success', False)]
        
        if gpu_speeds and cpu_speeds:
            avg_gpu = sum(gpu_speeds) / len(gpu_speeds)
            avg_cpu = sum(cpu_speeds) / len(cpu_speeds)
            
            if avg_gpu > avg_cpu:
                print(f"  - GPU is {avg_gpu/avg_cpu:.1f}x faster than CPU for scoring")
                print("  - Recommended configuration: --use-gpu")
            else:
                print(f"  - CPU is {avg_cpu/avg_gpu:.1f}x faster than GPU for scoring")
                print("  - Recommended configuration: Use CPU-only mode")
        
        # Check hybrid performance
        hybrid_speeds = [r['poses_per_second'] for r in benchmark_results['scoring_benchmark'] 
                         if r['mode'] == 'hybrid' and r.get('success', False)]
        
        if hybrid_speeds and gpu_speeds:
            avg_hybrid = sum(hybrid_speeds) / len(hybrid_speeds)
            
            if avg_hybrid > avg_gpu:
                print(f"  - Hybrid mode is {avg_hybrid/avg_gpu:.1f}x faster than GPU-only")
                print("  - Recommended configuration: Hybrid mode with optimized workload balance")
    else:
        print("  - GPU acceleration is not available or failed")
        print("  - Recommended configuration: Use CPU-only mode with parallel processing")
    
    # Check impact of flexible residues if both were benchmarked
    flex_results = [r for r in benchmark_results['scoring_benchmark'] 
                   if r.get('has_flexible_residues', False) and r.get('success', False)]
    rigid_results = [r for r in benchmark_results['scoring_benchmark'] 
                    if not r.get('has_flexible_residues', False) and r.get('success', False)]
    
    if flex_results and rigid_results:
        avg_flex_speed = sum(r['poses_per_second'] for r in flex_results) / len(flex_results)
        avg_rigid_speed = sum(r['poses_per_second'] for r in rigid_results) / len(rigid_results)
        
        slowdown_factor = avg_rigid_speed / avg_flex_speed
        
        print(f"\nFlexible Residue Performance:")
        print(f"  - Flexible docking is {slowdown_factor:.1f}x slower than rigid docking")
        
        if slowdown_factor > 5:
            print("  - Consider limiting the number of flexible residues for better performance")
            print("  - GPU acceleration is highly recommended for flexible docking")
        elif slowdown_factor > 2:
            print("  - Performance impact of flexible residues is moderate")
            print("  - Hybrid mode recommended for optimal performance")
        else:
            print("  - Performance impact of flexible residues is minimal")
    
    return benchmark_results


if __name__ == "__main__":
    # The multiprocessing with Pool might need this on some platforms
    multiprocessing.set_start_method('spawn', force=True)
    main()
