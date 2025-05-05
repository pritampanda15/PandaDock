"""
CLI integration module for PandaDock hardware acceleration.
This module extends the main.py command-line interface with GPU/CPU options.
"""

# Core CLI utilities
import argparse
import os

# Factory and algorithms
from .scoring_factory import create_scoring_function
from .search import GeneticAlgorithm, RandomSearch
from .parallel_search import ParallelGeneticAlgorithm, ParallelRandomSearch
from .pandadock import PANDADOCKAlgorithm

# Physics-based models
from .physics import (
    MMFFMinimization,
    MonteCarloSampling,
    PhysicsBasedScoring,
    GeneralizedBornSolvation
)

# Preparation and validation
from .preparation import prepare_protein, prepare_ligand
from .validation import validate_docking, calculate_ensemble_rmsd

# Batch and utilities
from .utils import save_docking_results, save_complex_to_pdb
from .utils import calculate_rmsd
def add_hardware_options(parser):
    """
    Add hardware acceleration options to the argument parser.
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        Argument parser to modify
    """
    # Create a hardware acceleration group
    hw_group = parser.add_argument_group('Hardware Acceleration')
    
    # GPU options
    hw_group.add_argument('--use-gpu', action='store_true',
                         help='Use GPU acceleration if available')
    hw_group.add_argument('--gpu-id', type=int, default=0,
                         help='GPU device ID to use (default: 0)')
    hw_group.add_argument('--gpu-precision', choices=['float32', 'float64'], default='float32',
                         help='Numerical precision for GPU calculations (default: float32)')
    
    # CPU options
    hw_group.add_argument('--cpu-workers', type=int, default=None,
                         help='Number of CPU workers for parallel processing (default: all cores)')
    hw_group.add_argument('--cpu-affinity', action='store_true',
                         help='Set CPU affinity for better performance')
    
    # Hybrid options
    hw_group.add_argument('--workload-balance', type=float, default=None,
                         help='GPU/CPU workload balance (0.0-1.0, higher values assign more work to GPU)')
    hw_group.add_argument('--auto-tune', action='store_true',
                         help='Automatically tune hardware parameters for best performance')


def configure_hardware(args):
    """
    Configure hardware settings based on command-line arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    
    Returns:
    --------
    dict
        Hardware configuration dictionary
    """
    # Basic hardware configuration
    hw_config = {
        'use_gpu': args.use_gpu,
        'gpu_id': args.gpu_id,
        'gpu_precision': args.gpu_precision,
        'cpu_workers': args.cpu_workers,
    }
    
    # Add optional workload balance if specified
    if args.workload_balance is not None:
        hw_config['workload_balance'] = max(0.0, min(1.0, args.workload_balance))
    
    # Set CPU affinity if requested
    if args.cpu_affinity:
        try:
            import psutil
            process = psutil.Process()
            # Get available CPU cores
            cpu_count = psutil.cpu_count(logical=True)
            # Set affinity to all cores
            process.cpu_affinity(list(range(cpu_count)))
            print(f"CPU affinity set to {cpu_count} cores")
        except ImportError:
            print("Warning: psutil module not available, CPU affinity not set")
        except Exception as e:
            print(f"Warning: Failed to set CPU affinity: {e}")
    
    return hw_config


def setup_hardware_acceleration(hw_config):
    """
    Set up hardware acceleration based on configuration.
    
    Parameters:
    -----------
    hw_config : dict
        Hardware configuration dictionary
    
    Returns:
    --------
    object
        HybridDockingManager instance
    """
    from .hybrid_manager import HybridDockingManager
    
    # Create hybrid manager with specified configuration
    manager = HybridDockingManager(
        use_gpu=hw_config.get('use_gpu', False),
        n_cpu_workers=hw_config.get('cpu_workers', None),
        gpu_device_id=hw_config.get('gpu_id', 0),
        workload_balance=hw_config.get('workload_balance', 0.8)
    )
    
    return manager


def create_optimized_scoring_function(args):
    """
    Create an optimized scoring function based on user arguments.
    Automatically selects CPU/GPU/physics-based implementation.
    """
    try:
        scoring_function = create_scoring_function(
            use_gpu=getattr(args, 'use_gpu', False),
            physics_based=getattr(args, 'physics_based', False),
            enhanced=getattr(args, 'enhanced_scoring', True),
            tethered=getattr(args, 'tethered_scoring', False),
            reference_ligand=getattr(args, 'reference_ligand', None),
            weights=getattr(args, 'scoring_weights', None),
            device=getattr(args, 'gpu_device', 'cuda'),
            precision=getattr(args, 'gpu_precision', 'float32'),
            verbose=getattr(args, 'verbose', False)
        )
        print(f"[DEBUG] Created scoring function: {type(scoring_function).__name__}")
        return scoring_function

    except Exception as e:
        print("[ERROR] Could not initialize physics-based scoring:", str(e))
        print("Warning: Physics-based modules not available. Some features will be disabled.")
        return create_scoring_function()
def create_optimized_search_algorithm(manager, algorithm_type, scoring_function, **kwargs):
    """
    Create an optimized search algorithm based on available hardware.
    """
    # Extract grid-related settings from kwargs
    grid_spacing = kwargs.pop('grid_spacing', 0.375)
    grid_radius = kwargs.pop('grid_radius', 10.0)
    grid_center = kwargs.pop('grid_center', None)

    # Standard algorithm types
    if algorithm_type == 'genetic':
        try:
            from .parallel_search import ParallelGeneticAlgorithm
             # Remove unsupported kwargs for this algorithm
            kwargs.pop('grid_radius', None)
            kwargs.pop('grid_spacing', None)
            kwargs.pop('grid_center', None)
           
            # Remove unsupported 'n_steps' argument if present
            kwargs.pop('n_steps', None)
            kwargs.pop('temperature', None)
            kwargs.pop('cooling_factor', None)
            kwargs.pop('high_temp', None)
            kwargs.pop('target_temp', None)
            kwargs.pop('num_conformers', None)
            kwargs.pop('num_orientations', None)
            kwargs.pop('md_steps', None)
            kwargs.pop('minimize_steps', None)
            kwargs.pop('use_grid', None)
            kwargs.pop('grid_center', None)
            kwargs.pop('grid_radius', None)
            kwargs.pop('grid_spacing', None)
            kwargs.pop('use_monte_carlo', None)
            kwargs.pop('use_monte_carlo', None)
            kwargs.pop('grid_spacing', None)
            kwargs.pop('n_steps', None)
            kwargs.pop('temperature', None)
            kwargs.pop('cooling_factor', None)
            kwargs.pop('high_temp', None)
            kwargs.pop('target_temp', None)
            kwargs.pop('num_conformers', None)
            kwargs.pop('num_orientations', None)
            kwargs.pop('md_steps', None)
            kwargs.pop('minimize_steps', None)
            kwargs.pop('use_grid', None)
            kwargs.pop('output_dir', None)

                
            return ParallelGeneticAlgorithm(
                scoring_function=scoring_function,
                grid_spacing=grid_spacing,  # Pass grid_spacing explicitly    # 💥 PASS properly
                grid_radius=grid_radius,      # 💥 PASS properly
                grid_center=grid_center,
                output_dir=kwargs.pop('output_dir', None),
                **kwargs
            )
        except ImportError:
            from .search import GeneticAlgorithm
            return GeneticAlgorithm(
                scoring_function=scoring_function,
                **kwargs
            )
            
    elif algorithm_type == 'random':
        try:
            from .parallel_search import ParallelRandomSearch
             # Remove unsupported kwargs for this algorithm
            kwargs.pop('grid_radius', None)
            kwargs.pop('grid_spacing', None)
            kwargs.pop('grid_center', None)
            kwargs.pop('output_dir', None)
            return ParallelRandomSearch(
                scoring_function=scoring_function,
                **kwargs
            )
        except ImportError:
            from .search import RandomSearch
            return RandomSearch(
                scoring_function=scoring_function,
                grid_spacing=grid_spacing,  # Pass grid_spacing explicitly
                grid_radius=grid_radius,    # 💥 PASS properl
                grid_center=grid_center,     # 💥 PASS proper
                output_dir=kwargs.pop('output_dir', None),
                **kwargs
            )
            
    elif algorithm_type == 'monte-carlo':
        try:
            from .unified_scoring import MonteCarloSampling
             # Remove unsupported kwargs for this algorithm
            kwargs.pop('grid_radius', None)
            kwargs.pop('grid_spacing', None)
            kwargs.pop('grid_center', None)
            kwargs.pop('max_iterations', None)
            kwargs.pop('workload_balance', None)            
            kwargs.pop('num_processes', None)
            kwargs.pop('output_dir', None)
            kwargs.pop('mc_steps', None)
            kwargs.pop('temperature', None)
            return MonteCarloSampling(
                scoring_function=scoring_function,
                grid_spacing=grid_spacing,  # Pass grid_spacing explicitly
                grid_radius=grid_radius,    # 💥 PASS properl
                grid_center=grid_center,     # 💥 PASS proper
                output_dir=kwargs.pop('output_dir', None),
                **kwargs
            )
        except ImportError:
            print("Monte Carlo sampling not available. Using genetic algorithm instead.")
            return create_optimized_search_algorithm(
                manager,
                'genetic',
                scoring_function,
                grid_spacing=grid_spacing,  # Pass grid_spacing explicitly
                grid_radius=grid_radius,    # 💥 PASS properly
                grid_center=grid_center,     # 💥 PASS properly
                output_dir=kwargs.pop('output_dir', None),
                **kwargs
            )
            
    elif algorithm_type == 'pandadock':
        try:
            from .pandadock import PANDADOCKAlgorithm
            
            # Remove unsupported kwargs for this algorithm
            kwargs.pop('grid_radius', None)
            kwargs.pop('grid_spacing', None)
            kwargs.pop('grid_center', None)
            kwargs.pop('output_dir', None)

            return PANDADOCKAlgorithm(
                scoring_function=scoring_function,
                output_dir=kwargs.pop('output_dir', None),
                **kwargs
            )
        ##############
        # Add exception handling for PANDADOCK
        ##############

        except ImportError:
            print("PANDADOCK algorithm not available. Using genetic algorithm instead.")
            return create_optimized_search_algorithm(
                manager,
                'genetic',
                scoring_function,
                grid_spacing=grid_spacing,  # Pass grid_spacing explicitly
                grid_radius=grid_radius,    # 💥 PASS properly
                grid_center=grid_center,     # 💥 PASS properly
                output_dir=kwargs.pop('output_dir', None),
                **kwargs
            )
    
    # Unknown algorithm type
    else:
        print(f"Unknown algorithm type: {algorithm_type}. Using genetic algorithm.")
        return create_optimized_search_algorithm(
            manager,
            'genetic',
            scoring_function,
            grid_spacing=grid_spacing,  # Pass grid_spacing explicitly
            grid_radius=grid_radius,    # 💥 PASS properl
            grid_center=grid_center,     # 💥 PASS property
            output_dir=kwargs.pop('output_dir', None),
            **kwargs
        )
def get_scoring_type_from_args(args):
    """
    Determine scoring type based on command-line arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    
    Returns:
    --------
    str
        Scoring type ('standard', 'enhanced', 'pandadock' or 'physics')
    """
    if args.physics_based:
        return 'physics'
    elif args.enhanced_scoring:
        return 'enhanced'
    else:
        return 'standard'
def get_algorithm_type_from_args(args):
    """
    Determine algorithm type based on command-line arguments.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    
    Returns:
    --------
    str
        Algorithm type ('genetic', 'random', 'monte-carlo', or 'pandadock')
    """
    if args.monte_carlo:
        return 'monte-carlo'
    else:
        return args.algorithm  # Now can include 'pandadock' as an option
    


def get_algorithm_kwargs_from_args(args):
    """
    Get algorithm keyword arguments based on command-line arguments and algorithm type.
    """
    algorithm_type = get_algorithm_type_from_args(args)
    algorithm_kwargs = {}
    
    # Common parameters for most algorithms
    if hasattr(args, 'iterations'):
        algorithm_kwargs['max_iterations'] = args.iterations
    
    # Algorithm-specific parameters
    if algorithm_type == 'genetic':
        if hasattr(args, 'population_size'):
            algorithm_kwargs['population_size'] = args.population_size
        
        if hasattr(args, 'mutation_rate'):
            algorithm_kwargs['mutation_rate'] = getattr(args, 'mutation_rate', 0.2)

        if hasattr(args, 'local_opt'):
            algorithm_kwargs['perform_local_opt'] = args.local_opt
        
    elif algorithm_type == 'monte-carlo':
        # Monte Carlo specific parameters
        if hasattr(args, 'mc_steps'):
            algorithm_kwargs['n_steps'] = args.mc_steps
        
        if hasattr(args, 'temperature'):
            algorithm_kwargs['temperature'] = args.temperature
            
        if hasattr(args, 'cooling_factor'):
            algorithm_kwargs['cooling_factor'] = args.cooling_factor
    
    elif algorithm_type == 'pandadock':
        # pandadock specific parameters
        if hasattr(args, 'high_temp'):
            algorithm_kwargs['high_temp'] = args.high_temp
            
        if hasattr(args, 'target_temp'):
            algorithm_kwargs['target_temp'] = args.target_temp
            
        if hasattr(args, 'num_conformers'):
            algorithm_kwargs['num_conformers'] = args.num_conformers
            
        if hasattr(args, 'num_orientations'):
            algorithm_kwargs['num_orientations'] = args.num_orientations
            
        if hasattr(args, 'md_steps'):
            algorithm_kwargs['md_steps'] = args.md_steps
            
        if hasattr(args, 'minimize_steps'):
            algorithm_kwargs['minimize_steps'] = args.minimize_steps
            
        if hasattr(args, 'use_grid'):
            algorithm_kwargs['use_grid'] = args.use_grid
    
    return algorithm_kwargs