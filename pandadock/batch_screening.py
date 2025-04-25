"""
Batch screening module for PandaDock.
This module provides functionality for high-throughput virtual screening
of multiple ligands against a protein target.
"""

import os
import time
import json
import csv
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bars (you may need to install this: pip install tqdm)

from .protein import Protein
from .ligand import Ligand
from .utils import save_docking_results
from .preparation import prepare_protein, prepare_ligand
from .main_integration import (
    configure_hardware,
    setup_hardware_acceleration,
    create_optimized_scoring_function,
    create_optimized_search_algorithm,
    get_algorithm_kwargs_from_args
)


# Global variable to store shared configuration for multiprocessing
_mp_config = {}


def run(config):
    """
    Run batch screening of a ligand library against a protein target.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with the following keys:
        - protein: Path to protein PDB file
        - ligand_library: Path to directory containing ligand files
        - output_dir: Output directory for results
        - screening_params: Dictionary of screening parameters
    
    Returns:
    --------
    dict
        Dictionary of results with ligand filenames as keys
    """
    # Extract configuration
    protein_file = config.get('protein')
    ligand_library = config.get('ligand_library')
    output_dir = config.get('output_dir', 'screening_results')
    screening_params = config.get('screening_params', {})
    
    # Verify inputs
    if not protein_file or not os.path.exists(protein_file):
        raise ValueError(f"Protein file not found: {protein_file}")
    
    if not ligand_library or not os.path.exists(ligand_library):
        raise ValueError(f"Ligand library not found: {ligand_library}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"{output_dir}_{timestamp}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure hardware
    hw_config = _parse_hardware_config(screening_params.get('hardware', {}))
    hybrid_manager = setup_hardware_acceleration(hw_config)
    
    # Prepare protein (done once for all ligands)
    print(f"Preparing protein: {protein_file}")
    protein = _prepare_protein(protein_file, screening_params, output_path)
    
    # Get list of ligand files
    ligand_files = _get_ligand_files(ligand_library)
    print(f"Found {len(ligand_files)} ligands for screening")
    
    # Create summary file
    summary_file = output_path / "screening_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ligand', 'Score', 'Runtime_s', 'Status'])
    
    # Prepare docking parameters
    docking_params = _prepare_docking_params(screening_params)
    
    # Set up results dictionary
    all_results = {}
    
    # Process each ligand
    print(f"Starting batch screening of {len(ligand_files)} ligands...")
    
    for i, ligand_file in enumerate(tqdm(ligand_files)):
        ligand_name = Path(ligand_file).stem
        ligand_output = output_path / ligand_name
        ligand_output.mkdir(exist_ok=True)
        
        try:
            # Process this ligand
            start_time = time.time()
            results = _dock_single_ligand(
                protein=protein,
                ligand_file=ligand_file,
                output_dir=ligand_output,
                hybrid_manager=hybrid_manager,
                docking_params=docking_params
            )
            runtime = time.time() - start_time
            
            # Store best score
            best_score = results[0][1] if results else float('inf')
            status = "Success"
            
            # Add to results dictionary
            all_results[ligand_name] = {
                'file': ligand_file,
                'score': best_score,
                'runtime': runtime,
                'poses': [(i, score) for i, (pose, score) in enumerate(results)]
            }
            
        except Exception as e:
            print(f"Error processing ligand {ligand_name}: {e}")
            status = f"Error: {str(e)}"
            runtime = time.time() - start_time
            
            # Add error entry
            all_results[ligand_name] = {
                'file': ligand_file,
                'score': float('inf'),
                'runtime': runtime,
                'error': str(e)
            }
        
        # Update summary file
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ligand_name, 
                             best_score if status == "Success" else "N/A", 
                             f"{runtime:.2f}", 
                             status])
    
    # Clean up resources
    hybrid_manager.cleanup()
    
    # Generate summary report and visualizations
    _generate_summary_report(all_results, output_path)
    
    print(f"Batch screening completed. Results saved to: {output_path}")
    return all_results


# Function to process a single ligand for parallelization
def _process_ligand_parallel(ligand_file):
    """Process a single ligand for parallel screening."""
    # Get configuration from global variable
    config = _mp_config
    output_path = config['output_path']
    prepared_protein_file = config['prepared_protein_file']
    screening_params = config['screening_params']
    docking_params = config['docking_params']
    hw_config = config['hw_config']
    
    ligand_name = Path(ligand_file).stem
    ligand_output = output_path / ligand_name
    ligand_output.mkdir(exist_ok=True)
    
    try:
        # Create separate hybrid manager for each process
        process_hybrid_manager = setup_hardware_acceleration(hw_config)
        
        # Load protein
        protein = Protein(prepared_protein_file)
        
        # Configure active site if needed
        if 'site' in screening_params:
            site = screening_params['site']
            radius = screening_params.get('radius', 10.0)
            protein.define_active_site(site, radius)
        elif screening_params.get('detect_pockets', False):
            pockets = protein.detect_pockets()
            if pockets:
                protein.define_active_site(pockets[0]['center'], pockets[0]['radius'])
        
        # Process this ligand
        start_time = time.time()
        results = _dock_single_ligand(
            protein=protein,
            ligand_file=ligand_file,
            output_dir=ligand_output,
            hybrid_manager=process_hybrid_manager,
            docking_params=docking_params
        )
        runtime = time.time() - start_time
        
        # Store best score
        best_score = results[0][1] if results else float('inf')
        status = "Success"
        
        # Clean up hybrid manager
        process_hybrid_manager.cleanup()
        
        # Return results
        return {
            'name': ligand_name,
            'file': ligand_file,
            'score': best_score,
            'runtime': runtime,
            'status': status
        }
        
    except Exception as e:
        print(f"Error processing ligand {ligand_name}: {e}")
        status = f"Error: {str(e)}"
        runtime = time.time() - start_time if 'start_time' in locals() else 0
        
        # Return error information
        return {
            'name': ligand_name,
            'file': ligand_file,
            'score': float('inf'),
            'runtime': runtime,
            'status': status,
            'error': str(e)
        }


def run_parallel(config):
    """
    Run batch screening in parallel using multiple processes.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with the following keys:
        - protein: Path to protein PDB file
        - ligand_library: Path to directory containing ligand files
        - output_dir: Output directory for results
        - screening_params: Dictionary of screening parameters
        - n_processes: Number of parallel processes (default: CPU count)
    
    Returns:
    --------
    dict
        Dictionary of results with ligand filenames as keys
    """
    global _mp_config
    
    # Extract configuration
    protein_file = config.get('protein')
    ligand_library = config.get('ligand_library')
    output_dir = config.get('output_dir', 'screening_results')
    screening_params = config.get('screening_params', {})
    n_processes = config.get('n_processes', mp.cpu_count())
    
    # Verify inputs
    if not protein_file or not os.path.exists(protein_file):
        raise ValueError(f"Protein file not found: {protein_file}")
    
    if not ligand_library or not os.path.exists(ligand_library):
        raise ValueError(f"Ligand library not found: {ligand_library}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"{output_dir}_{timestamp}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure hardware
    hw_config = _parse_hardware_config(screening_params.get('hardware', {}))
    
    # Prepare protein (done once for all ligands)
    print(f"Preparing protein: {protein_file}")
    prepared_protein_file = _prepare_protein_file(protein_file, screening_params, output_path)
    
    # Get list of ligand files
    ligand_files = _get_ligand_files(ligand_library)
    print(f"Found {len(ligand_files)} ligands for screening")
    
    # Create summary file
    summary_file = output_path / "screening_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ligand', 'Score', 'Runtime_s', 'Status'])
    
    # Prepare docking parameters
    docking_params = _prepare_docking_params(screening_params)
    
    # Set up configuration for parallel processing
    _mp_config = {
        'output_path': output_path,
        'prepared_protein_file': prepared_protein_file,
        'screening_params': screening_params,
        'docking_params': docking_params,
        'hw_config': hw_config
    }
    
    # Run parallel ligand processing
    print(f"Starting batch screening of {len(ligand_files)} ligands using {n_processes} processes...")
    
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(_process_ligand_parallel, ligand_files),
            total=len(ligand_files)
        ))
    
    # Process results
    all_results = {}
    for result in results:
        ligand_name = result['name']
        
        # Update summary file
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ligand_name, 
                result['score'] if result['status'] == "Success" else "N/A", 
                f"{result['runtime']:.2f}", 
                result['status']
            ])
        
        # Add to results dictionary
        all_results[ligand_name] = {
            'file': result['file'],
            'score': result['score'],
            'runtime': result['runtime'],
            'status': result['status']
        }
        
        if 'error' in result:
            all_results[ligand_name]['error'] = result['error']
    
    # Generate summary report and visualizations
    _generate_summary_report(all_results, output_path)
    
    print(f"Batch screening completed. Results saved to: {output_path}")
    return all_results


def _parse_hardware_config(hardware_params):
    """Parse hardware configuration from parameters."""
    hw_config = {
        'use_gpu': hardware_params.get('use_gpu', True),
        'gpu_id': hardware_params.get('gpu_id', 0),
        'cpu_workers': hardware_params.get('cpu_workers', None),
        'workload_balance': hardware_params.get('workload_balance', None),
    }
    return hw_config


def _prepare_protein(protein_file, params, output_dir):
    """Prepare protein for screening."""
    # Check if preparation is requested
    prepare_molecules = params.get('prepare_molecules', True)
    
    if prepare_molecules:
        prepared_file = output_dir / f"prepared_protein.pdb"
        prepared_protein = prepare_protein(
            protein_file, 
            output_file=prepared_file,
            ph=params.get('ph', 7.4)
        )
        protein_path = prepared_file
    else:
        protein_path = protein_file
    
    # Load protein
    protein = Protein(protein_path)
    
    # Define active site if provided
    if 'site' in params:
        site = params['site']
        radius = params.get('radius', 10.0)
        protein.define_active_site(site, radius)
    elif params.get('detect_pockets', False):
        print("Detecting binding pockets...")
        pockets = protein.detect_pockets()
        if pockets:
            print(f"Found {len(pockets)} potential binding pockets")
            print(f"Using largest pocket as active site")
            protein.define_active_site(pockets[0]['center'], pockets[0]['radius'])
    
    return protein


def _prepare_protein_file(protein_file, params, output_dir):
    """Prepare protein file for screening."""
    # Check if preparation is requested
    prepare_molecules = params.get('prepare_molecules', True)
    
    if prepare_molecules:
        prepared_file = output_dir / f"prepared_protein.pdb"
        prepared_protein = prepare_protein(
            protein_file, 
            output_file=prepared_file,
            ph=params.get('ph', 7.4)
        )
        return prepared_file
    else:
        return protein_file


def _get_ligand_files(ligand_library):
    """Get list of ligand files from directory."""
    # Get all potential ligand files
    extensions = ['*.mol', '*.mol2', '*.sdf', '*.pdb']
    ligand_files = []
    
    for ext in extensions:
        ligand_files.extend(glob.glob(os.path.join(ligand_library, ext)))
    
    return ligand_files


def _prepare_docking_params(screening_params):
    """Prepare docking parameters from screening parameters."""
    docking_params = {
        'algorithm': screening_params.get('algorithm', 'genetic'),
        'iterations': screening_params.get('iterations', 1000),
        'population_size': screening_params.get('population_size', 100),
        'exhaustiveness': screening_params.get('exhaustiveness', 1),
        'scoring_function': screening_params.get('scoring_function', 'enhanced'),
        'local_opt': screening_params.get('local_opt', False),
        'prepare_molecules': screening_params.get('prepare_molecules', True),
    }
    return docking_params


def _dock_single_ligand(protein, ligand_file, output_dir, hybrid_manager, docking_params):
    """Dock a single ligand against the protein target."""
    # Prepare ligand
    if docking_params.get('prepare_molecules', True):
        prepared_file = output_dir / f"prepared_{Path(ligand_file).name}"
        ligand_path = prepare_ligand(
            ligand_file,
            output_file=prepared_file
        )
    else:
        ligand_path = ligand_file
    
    # Load ligand
    ligand = Ligand(ligand_path)
    
    # Set up scoring function
    scoring_type = docking_params.get('scoring_function', 'enhanced')
    scoring_function = create_optimized_scoring_function(hybrid_manager, scoring_type)
    
    # Set up search algorithm
    algorithm_type = docking_params.get('algorithm', 'genetic')
    
    # Parse algorithm-specific parameters
    algorithm_kwargs = {
        'max_iterations': docking_params.get('iterations', 1000),
    }
    
    if algorithm_type == 'genetic':
        algorithm_kwargs['population_size'] = docking_params.get('population_size', 100)
        algorithm_kwargs['mutation_rate'] = docking_params.get('mutation_rate', 0.2)
    
    # Create search algorithm
    search_algorithm = create_optimized_search_algorithm(
        hybrid_manager, 
        algorithm_type, 
        scoring_function, 
        **algorithm_kwargs
    )
    
    # Run multiple docking cycles if exhaustiveness > 1
    if docking_params.get('exhaustiveness', 1) > 1:
        exhaustiveness = docking_params.get('exhaustiveness')
        
        # Use ensemble docking with hybrid manager
        results = hybrid_manager.run_ensemble_docking(
            protein=protein,
            ligand=ligand,
            n_runs=exhaustiveness,
            algorithm_type=algorithm_type,
            **algorithm_kwargs
        )
    else:
        # Single docking run
        results = search_algorithm.search(protein, ligand)
    
    # Apply local optimization to top poses if requested
    if docking_params.get('local_opt', False):
        optimized_results = []
        
        # Optimize top 5 poses
        for i, (pose, score) in enumerate(sorted(results, key=lambda x: x[1])[:5]):
            if hasattr(search_algorithm, '_local_optimization'):
                # Use built-in local optimization
                opt_pose, opt_score = search_algorithm._local_optimization(pose, protein)
                optimized_results.append((opt_pose, opt_score))
            else:
                optimized_results.append((pose, score))
        
        # Combine with original results
        combined_results = optimized_results + [r for r in results if r not in results[:5]]
        results = combined_results
    
    # Sort all results by score
    results.sort(key=lambda x: x[1])
    
    # Save docking results
    save_docking_results(results[:10], output_dir)
    from .utils import save_complex_to_pdb
    for i, (pose, score) in enumerate(results[:10]):
        complex_path = Path(output_dir) / f"complex_pose_{i+1}_score_{score:.2f}.pdb"
        save_complex_to_pdb(protein, pose, complex_path)

    return results


def _generate_summary_report(all_results, output_path):
    """Generate summary report and visualizations from screening results."""
    # Extract scores and runtimes
    scores = []
    runtimes = []
    ligands = []
    
    for ligand_name, result in all_results.items():
        if 'error' not in result:
            scores.append(result['score'])
            runtimes.append(result['runtime'])
            ligands.append(ligand_name)
    
    # Create a rank-ordered list
    if scores:
        data = list(zip(ligands, scores, runtimes))
        ranked_data = sorted(data, key=lambda x: x[1])
        
        # Write tabular report
        report_file = output_path / "ranked_ligands.csv"
        with open(report_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Ligand', 'Score', 'Runtime (s)'])
            for i, (ligand, score, runtime) in enumerate(ranked_data):
                writer.writerow([i+1, ligand, f"{score:.2f}", f"{runtime:.2f}"])
        
        # Create score distribution visualization
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, alpha=0.7)
        plt.xlabel('Docking Score')
        plt.ylabel('Count')
        plt.title('Distribution of Docking Scores')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / "score_distribution.png")
        plt.close()
        
        # Create top compounds chart
        top_n = min(20, len(ranked_data))
        plt.figure(figsize=(12, 8))
        top_ligands = [x[0] for x in ranked_data[:top_n]]
        top_scores = [x[1] for x in ranked_data[:top_n]]
        
        y_pos = np.arange(len(top_ligands))
        plt.barh(y_pos, top_scores)
        plt.yticks(y_pos, top_ligands)
        plt.xlabel('Docking Score')
        plt.title(f'Top {top_n} Compounds')
        plt.tight_layout()
        plt.savefig(output_path / "top_compounds.png")
        plt.close()
    
    # Create runtime statistics
    if runtimes:
        avg_runtime = sum(runtimes) / len(runtimes)
        total_runtime = sum(runtimes)
        
        summary = {
            "total_ligands": len(all_results),
            "successful_dockings": len(scores),
            "errors": len(all_results) - len(scores),
            "average_runtime_per_ligand": avg_runtime,
            "total_runtime": total_runtime,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Save runtime stats
        with open(output_path / "screening_stats.json", 'w') as f:
            json.dump(summary, f, indent=2)

def batch_screening(config):
        """
        Wrapper function for batch screening. 
        Just calls run() internally for backward compatibility.
        """
        return run(config)