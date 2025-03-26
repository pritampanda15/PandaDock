"""
CLI integration module for PandaDock hardware acceleration.
This module extends the main.py command-line interface with GPU/CPU options.
"""

import argparse


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


def create_optimized_scoring_function(manager, scoring_type='enhanced'):
    """
    Create an optimized scoring function based on available hardware.
    
    Parameters:
    -----------
    manager : HybridDockingManager
        Hybrid docking manager
    scoring_type : str
        Type of scoring function ('standard', 'enhanced', or 'physics')
    
    Returns:
    --------
    ScoringFunction
        Optimized scoring function
    """
    # Import appropriate scoring functions
    from .scoring import CompositeScoringFunction, EnhancedScoringFunction
    
    # Check if GPU acceleration is available
    if manager.has_gpu:
        # Import GPU scoring function
        try:
            from .gpu_scoring import GPUAcceleratedScoringFunction
            
            # Create GPU-accelerated scoring function
            if scoring_type == 'standard':
                return manager.prepare_gpu_scoring_function(GPUAcceleratedScoringFunction)
            elif scoring_type == 'enhanced':
                return manager.prepare_gpu_scoring_function(GPUAcceleratedScoringFunction)
            elif scoring_type == 'physics':
                try:
                    from .physics import PhysicsBasedScoring
                    # Note: Physics-based scoring might not be GPU-accelerated yet
                    return PhysicsBasedScoring()
                except ImportError:
                    print("Physics-based scoring not available. Using enhanced scoring instead.")
                    return manager.prepare_gpu_scoring_function(GPUAcceleratedScoringFunction)
            else:
                print(f"Unknown scoring type: {scoring_type}. Using enhanced scoring.")
                return manager.prepare_gpu_scoring_function(GPUAcceleratedScoringFunction)
                
        except ImportError:
            print("GPU scoring module not available. Using CPU scoring function.")
    
    # Fall back to CPU scoring functions
    if scoring_type == 'standard':
        return CompositeScoringFunction()
    elif scoring_type == 'enhanced':
        return EnhancedScoringFunction()
    elif scoring_type == 'physics':
        try:
            from .physics import PhysicsBasedScoring
            return PhysicsBasedScoring()
        except ImportError:
            print("Physics-based scoring not available. Using enhanced scoring instead.")
            return EnhancedScoringFunction()
    else:
        print(f"Unknown scoring type: {scoring_type}. Using enhanced scoring.")
        return EnhancedScoringFunction()


def create_optimized_search_algorithm(manager, algorithm_type, scoring_function, **kwargs):
    """
    Create an optimized search algorithm based on available hardware.
    
    Parameters:
    -----------
    manager : HybridDockingManager
        Hybrid docking manager
    algorithm_type : str
        Type of algorithm ('genetic', 'random', or 'monte-carlo')
    scoring_function : ScoringFunction
        Scoring function to use
    **kwargs : dict
        Additional arguments for the search algorithm
    
    Returns:
    --------
    SearchAlgorithm
        Optimized search algorithm
    """
    # Standard algorithm types
    if algorithm_type in ['genetic', 'random']:
        return manager.prepare_search_algorithm(
            algorithm_type=algorithm_type,
            scoring_function=scoring_function,
            **kwargs
        )
    
    # Monte Carlo requires special handling
    elif algorithm_type == 'monte-carlo':
        try:
            from .physics import MonteCarloSampling
            return MonteCarloSampling(
                scoring_function=scoring_function,
                **kwargs
            )
        except ImportError:
            print("Monte Carlo sampling not available. Using genetic algorithm instead.")
            return manager.prepare_search_algorithm(
                algorithm_type='genetic',
                scoring_function=scoring_function,
                **kwargs
            )
    
    # Unknown algorithm type
    else:
        print(f"Unknown algorithm type: {algorithm_type}. Using genetic algorithm.")
        return manager.prepare_search_algorithm(
            algorithm_type='genetic',
            scoring_function=scoring_function,
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
        Scoring type ('standard', 'enhanced', or 'physics')
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
        Algorithm type ('genetic', 'random', or 'monte-carlo')
    """
    if args.monte_carlo:
        return 'monte-carlo'
    else:
        return args.algorithm


def get_algorithm_kwargs_from_args(args):
    """
    Get algorithm keyword arguments based on command-line arguments and algorithm type.
    """
    algorithm_type = get_algorithm_type_from_args(args)
    algorithm_kwargs = {}
    
    # Common parameters for all algorithms
    if hasattr(args, 'iterations'):
        algorithm_kwargs['max_iterations'] = args.iterations
    
    # Algorithm-specific parameters
    if algorithm_type == 'genetic':
        if hasattr(args, 'population_size'):
            algorithm_kwargs['population_size'] = args.population_size
        
        if hasattr(args, 'mutation_rate'):
            algorithm_kwargs['mutation_rate'] = getattr(args, 'mutation_rate', 0.2)
            
    elif algorithm_type == 'monte-carlo':
        if hasattr(args, 'mc_steps'):
            algorithm_kwargs['n_steps'] = args.mc_steps
        
        if hasattr(args, 'temperature'):
            algorithm_kwargs['temperature'] = args.temperature
    
    return algorithm_kwargs
