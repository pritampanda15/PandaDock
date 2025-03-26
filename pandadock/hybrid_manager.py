"""
Hybrid CPU/GPU manager for PandaDock.
This module provides a unified interface for leveraging both CPU and GPU resources
to optimize molecular docking performance.
"""

import os
import time
import multiprocessing as mp
import warnings
import numpy as np
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class HardwareInfo:
    """Class for detecting and reporting available hardware resources."""
    
    @staticmethod
    def get_cpu_info():
        """Get CPU information."""
        n_cpus = mp.cpu_count()
        
        # Try to get more detailed CPU info on Linux
        cpu_info = {"cpu_count": n_cpus, "model": "Unknown"}
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_info["model"] = line.split(':', 1)[1].strip()
                        break
        except:
            pass
        
        return cpu_info
    
    @staticmethod
    def get_gpu_info():
        """Get GPU information."""
        gpu_info = {"available": False, "device": None, "name": "None", "memory": 0}
        
        # Check for CUDA GPU with PyTorch
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["device"] = "cuda"
                gpu_info["count"] = torch.cuda.device_count()
                gpu_info["name"] = torch.cuda.get_device_name(0)
                gpu_info["memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        # If PyTorch didn't find a GPU, try CuPy
        elif CUPY_AVAILABLE:
            try:
                gpu_info["available"] = True
                gpu_info["device"] = "cuda"
                gpu_info["count"] = cp.cuda.runtime.getDeviceCount()
                props = cp.cuda.runtime.getDeviceProperties(0)
                gpu_info["name"] = props['name'].decode()
                gpu_info["memory"] = props['totalGlobalMem'] / (1024**3)  # GB
            except:
                pass
        
        return gpu_info
    
    @staticmethod
    def print_hardware_summary():
        """Print a summary of available hardware resources."""
        cpu_info = HardwareInfo.get_cpu_info()
        gpu_info = HardwareInfo.get_gpu_info()
        
        print("\n" + "=" * 60)
        print("               HARDWARE CONFIGURATION SUMMARY")
        print("=" * 60)
        
        print(f"\nCPU Information:")
        print(f"  Cores/Threads: {cpu_info['cpu_count']}")
        print(f"  Model: {cpu_info['model']}")
        
        print(f"\nGPU Information:")
        if gpu_info["available"]:
            print(f"  CUDA GPU Available: Yes")
            print(f"  Device: {gpu_info['name']}")
            print(f"  Memory: {gpu_info['memory']:.2f} GB")
        else:
            print(f"  CUDA GPU Available: No")
            print(f"  Using CPU-only mode")
        
        print("\n" + "=" * 60)


class GPUManager:
    """
    Manager for GPU resources, providing utilities for GPU memory management
    and coordination between different GPU tasks.
    """
    
    def __init__(self, device_id=0, memory_fraction=0.9):
        """
        Initialize GPU manager.
        
        Parameters:
        -----------
        device_id : int
            GPU device ID to use
        memory_fraction : float
            Fraction of GPU memory to reserve (0.0 to 1.0)
        """
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.available = False
        self.backend = None
        
        self._init_gpu()
    
    def _init_gpu(self):
        """Initialize GPU and set up memory management."""
        # Try PyTorch first
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.available = True
                self.backend = "pytorch"
                
                # Set device
                torch.cuda.set_device(self.device_id)
                
                # Get device info
                self.device_name = torch.cuda.get_device_name(self.device_id)
                self.total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
                
                # Set memory limit if possible
                try:
                    torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                    print(f"PyTorch GPU memory fraction set to {self.memory_fraction}")
                except:
                    print("Warning: Could not set GPU memory fraction with PyTorch")
                
                print(f"Initialized GPU ({self.device_name}) using PyTorch backend")
                return
        
        # Try CuPy if PyTorch is not available
        if CUPY_AVAILABLE:
            try:
                # Set device
                cp.cuda.Device(self.device_id).use()
                
                # Get device info
                props = cp.cuda.runtime.getDeviceProperties(self.device_id)
                self.device_name = props['name'].decode()
                self.total_memory = props['totalGlobalMem']
                
                # Set memory limit
                try:
                    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
                    print(f"CuPy memory pool initialized")
                except:
                    print("Warning: Could not set up memory pool with CuPy")
                
                self.available = True
                self.backend = "cupy"
                print(f"Initialized GPU ({self.device_name}) using CuPy backend")
                return
            except:
                pass
        
        # No GPU available
        print("No GPU available. Using CPU-only mode.")
        self.available = False
        self.backend = None
    
    def get_free_memory(self):
        """Get free GPU memory in GB."""
        if not self.available:
            return 0
        
        if self.backend == "pytorch":
            reserved = torch.cuda.memory_reserved(self.device_id)
            allocated = torch.cuda.memory_allocated(self.device_id)
            free = reserved - allocated
            return free / (1024**3)  # GB
        
        elif self.backend == "cupy":
            mem_info = cp.cuda.runtime.memGetInfo()
            free = mem_info[0]
            return free / (1024**3)  # GB
        
        return 0
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if not self.available:
            return
        
        if self.backend == "pytorch":
            torch.cuda.empty_cache()
            print("PyTorch GPU cache cleared")
        
        elif self.backend == "cupy":
            cp.get_default_memory_pool().free_all_blocks()
            print("CuPy GPU cache cleared")
    
    def synchronize(self):
        """Synchronize GPU."""
        if not self.available:
            return
        
        if self.backend == "pytorch":
            torch.cuda.synchronize()
        
        elif self.backend == "cupy":
            cp.cuda.stream.get_current_stream().synchronize()
    
    def check_performance(self, matrix_size=2000):
        """
        Run a simple performance test to check GPU speed.
        
        Parameters:
        -----------
        matrix_size : int
            Size of matrix for multiplication test
        
        Returns:
        --------
        float
            Time taken in seconds
        """
        if not self.available:
            print("No GPU available for performance check.")
            return None
        
        print(f"Running performance check with {matrix_size}x{matrix_size} matrix multiplication...")
        
        # Warm up
        if self.backend == "pytorch":
            a = torch.rand(1000, 1000, device="cuda")
            b = torch.rand(1000, 1000, device="cuda")
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        
        elif self.backend == "cupy":
            a = cp.random.rand(1000, 1000)
            b = cp.random.rand(1000, 1000)
            c = cp.matmul(a, b)
            cp.cuda.stream.get_current_stream().synchronize()
        
        # Actual test
        start_time = time.time()
        
        if self.backend == "pytorch":
            a = torch.rand(matrix_size, matrix_size, device="cuda")
            b = torch.rand(matrix_size, matrix_size, device="cuda")
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        
        elif self.backend == "cupy":
            a = cp.random.rand(matrix_size, matrix_size)
            b = cp.random.rand(matrix_size, matrix_size)
            c = cp.matmul(a, b)
            cp.cuda.stream.get_current_stream().synchronize()
        
        elapsed = time.time() - start_time
        
        print(f"Performance check completed in {elapsed:.4f} seconds")
        return elapsed


class HybridDockingManager:
    """
    Manager for optimally distributing molecular docking workloads
    between CPU and GPU resources.
    """
    
    def __init__(self, use_gpu=True, n_cpu_workers=None, 
                 gpu_device_id=0, workload_balance=0.8):
        """
        Initialize the hybrid docking manager.
        
        Parameters:
        -----------
        use_gpu : bool
            Whether to use GPU if available
        n_cpu_workers : int
            Number of CPU workers to use. If None, uses all available cores.
        gpu_device_id : int
            GPU device ID to use
        workload_balance : float
            Fraction of workload to assign to GPU (0.0 to 1.0)
            Higher values assign more work to GPU, lower values to CPU
        """
        self.use_gpu = use_gpu
        self.n_cpu_workers = n_cpu_workers if n_cpu_workers else mp.cpu_count()
        self.gpu_device_id = gpu_device_id
        self.workload_balance = workload_balance
        
        # Initialize GPU if requested
        self.gpu_manager = None
        if self.use_gpu:
            self.gpu_manager = GPUManager(device_id=gpu_device_id)
            self.has_gpu = self.gpu_manager.available
        else:
            self.has_gpu = False
        
        # Initialize CPU pool
        self.cpu_pool = None
        self._init_cpu_pool()
        
        # Print hardware configuration
        HardwareInfo.print_hardware_summary()
        
        # Adaptive configs based on hardware
        self._configure_adaptive_settings()
    
    def _init_cpu_pool(self):
        """Initialize CPU process pool."""
        try:
            self.cpu_pool = mp.Pool(processes=self.n_cpu_workers)
            print(f"Initialized CPU pool with {self.n_cpu_workers} workers")
        except Exception as e:
            print(f"Error initializing CPU pool: {e}")
            print("Running without CPU parallelization")
    
    def _configure_adaptive_settings(self):
        """Configure settings based on available hardware."""
        # Adjust workload balance based on GPU vs CPU performance
        if self.has_gpu:
            # Run a simple benchmark to determine optimal workload split
            gpu_time = self.gpu_manager.check_performance(matrix_size=2000)
            
            # Estimate CPU performance (smaller matrix for CPU)
            cpu_time = self._check_cpu_performance(matrix_size=1000)
            
            # Calculate performance ratio (normalized)
            if gpu_time and cpu_time:
                # Scale CPU time to equivalent of 2000x2000 matrix (roughly 8x slower)
                cpu_time_scaled = cpu_time * 8
                
                # Calculate relative performance ratio
                perf_ratio = cpu_time_scaled / gpu_time
                
                # Adjust workload balance based on relative performance
                # Higher ratio means GPU is relatively faster
                self.workload_balance = min(0.95, max(0.5, 
                                                    0.5 + 0.3 * np.log10(perf_ratio)))
                
                print(f"Adaptive workload balance: {self.workload_balance:.2f} "
                      f"(GPU: {self.workload_balance:.0%}, CPU: {1-self.workload_balance:.0%})")
            else:
                print(f"Using default workload balance: {self.workload_balance:.2f}")
        else:
            # CPU only mode
            self.workload_balance = 0.0
            print("CPU-only mode (no GPU available)")
    
    def _check_cpu_performance(self, matrix_size=1000):
        """
        Run a simple performance test to check CPU speed.
        
        Parameters:
        -----------
        matrix_size : int
            Size of matrix for multiplication test
        
        Returns:
        --------
        float
            Time taken in seconds
        """
        print(f"Running CPU performance check with {matrix_size}x{matrix_size} matrix multiplication...")
        
        try:
            import numpy as np
            
            # Warm up
            a = np.random.rand(500, 500)
            b = np.random.rand(500, 500)
            c = np.matmul(a, b)
            
            # Actual test
            start_time = time.time()
            
            a = np.random.rand(matrix_size, matrix_size)
            b = np.random.rand(matrix_size, matrix_size)
            c = np.matmul(a, b)
            
            elapsed = time.time() - start_time
            
            print(f"CPU performance check completed in {elapsed:.4f} seconds")
            return elapsed
            
        except Exception as e:
            print(f"Error during CPU performance check: {e}")
            return None
    
    def split_workload(self, n_total_tasks):
        """
        Split workload between GPU and CPU.
        
        Parameters:
        -----------
        n_total_tasks : int
            Total number of tasks to distribute
        
        Returns:
        --------
        tuple
            (n_gpu_tasks, n_cpu_tasks)
        """
        if not self.has_gpu or self.workload_balance <= 0:
            return 0, n_total_tasks
        
        # Calculate tasks for each device
        n_gpu_tasks = int(n_total_tasks * self.workload_balance)
        n_cpu_tasks = n_total_tasks - n_gpu_tasks
        
        return n_gpu_tasks, n_cpu_tasks
    
    def prepare_gpu_scoring_function(self, scoring_function_class, **kwargs):
        """
        Prepare a GPU-accelerated scoring function.
        
        Parameters:
        -----------
        scoring_function_class : class
            Scoring function class to use (e.g., GPUAcceleratedScoringFunction)
        **kwargs : dict
            Additional arguments to pass to the scoring function
        
        Returns:
        --------
        ScoringFunction
            Initialized scoring function
        """
        if not self.has_gpu:
            print("Warning: No GPU available. Using CPU scoring function.")
            from .scoring import EnhancedScoringFunction
            return EnhancedScoringFunction()
        
        try:
            # Determine best backend
            if self.gpu_manager.backend == "pytorch":
                device = "cuda"
            elif self.gpu_manager.backend == "cupy":
                device = "cuda"
            else:
                device = "cpu"
            
            # Create instance with appropriate device
            return scoring_function_class(device=device, **kwargs)
        
        except Exception as e:
            print(f"Error initializing GPU scoring function: {e}")
            print("Falling back to CPU scoring function")
            from .scoring import EnhancedScoringFunction
            return EnhancedScoringFunction()
    
    def prepare_search_algorithm(self, algorithm_type, scoring_function, **kwargs):
        """
        Prepare an appropriate search algorithm based on available resources.
        
        Parameters:
        -----------
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
        if algorithm_type.lower() == 'genetic':
            # Filter out parameters not used by genetic algorithm
            genetic_kwargs = {k: v for k, v in kwargs.items() 
                             if k in ['max_iterations', 'population_size', 'mutation_rate']}
            
            # Use parallel genetic algorithm
            from .parallel_search import ParallelGeneticAlgorithm
            return ParallelGeneticAlgorithm(
                scoring_function=scoring_function,
                n_processes=self.n_cpu_workers,
                process_pool=self.cpu_pool,
                **genetic_kwargs
            )
        
        elif algorithm_type.lower() == 'random':
            # Filter out parameters not used by random search
            random_kwargs = {k: v for k, v in kwargs.items() 
                            if k in ['max_iterations']}
            
            # Use parallel random search
            from .parallel_search import ParallelRandomSearch
            return ParallelRandomSearch(
                scoring_function=scoring_function,
                n_processes=self.n_cpu_workers,
                process_pool=self.cpu_pool,
                **random_kwargs
            )
        
        elif algorithm_type.lower() == 'monte-carlo':
            # For Monte Carlo, try to import the appropriate class
            try:
                from .physics import MonteCarloSampling
                return MonteCarloSampling(
                    scoring_function=scoring_function,
                    **kwargs
                )
            except ImportError:
                print("Monte Carlo sampling not available. Using genetic algorithm instead.")
                # Recursively call with genetic algorithm
                return self.prepare_search_algorithm('genetic', scoring_function, **kwargs)
        
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    def run_docking(self, protein, ligand, algorithm_type='genetic', **kwargs):
        """
        Run docking using optimal hardware configuration.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        algorithm_type : str
            Type of algorithm to use ('genetic' or 'random')
        **kwargs : dict
            Additional arguments to pass to the algorithm
        
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score
        """
        # Prepare scoring function
        if self.has_gpu:
            from .gpu_scoring import GPUAcceleratedScoringFunction
            scoring_function = self.prepare_gpu_scoring_function(
                GPUAcceleratedScoringFunction
            )
        else:
            from .scoring import EnhancedScoringFunction
            scoring_function = EnhancedScoringFunction()
        
        # Prepare search algorithm
        search_algorithm = self.prepare_search_algorithm(
            algorithm_type=algorithm_type,
            scoring_function=scoring_function,
            **kwargs
        )
        
        # Run search
        print(f"Running {algorithm_type} search...")
        results = search_algorithm.search(protein, ligand)
        
        # Clean up GPU resources
        if self.has_gpu:
            self.gpu_manager.clear_cache()
        
        return results
    
    def run_ensemble_docking(self, protein, ligand, n_runs=10, algorithm_type='genetic', **kwargs):
        """
        Run multiple docking simulations and aggregate results.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        n_runs : int
            Number of docking runs
        algorithm_type : str
            Type of algorithm to use ('genetic' or 'random')
        **kwargs : dict
            Additional arguments to pass to the algorithm
        
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score
        """
        print(f"Running ensemble docking with {n_runs} runs...")
        
        # Determine GPU/CPU split
        n_gpu_runs, n_cpu_runs = self.split_workload(n_runs)
        
        # Run dockings sequentially to avoid multiprocessing issues
        all_results = []
        
        # GPU runs
        if n_gpu_runs > 0 and self.has_gpu:
            print(f"Running {n_gpu_runs} docking jobs on GPU...")
            for i in range(n_gpu_runs):
                print(f"GPU run {i+1}/{n_gpu_runs}")
                
                # Run single docking on GPU
                from .gpu_scoring import GPUAcceleratedScoringFunction
                scoring_function = self.prepare_gpu_scoring_function(
                    GPUAcceleratedScoringFunction
                )
                
                # Create search algorithm
                if algorithm_type.lower() == 'genetic':
                    from .search import GeneticAlgorithm
                    search_algorithm = GeneticAlgorithm(
                        scoring_function=scoring_function,
                        **kwargs
                    )
                else:
                    from .search import RandomSearch
                    search_algorithm = RandomSearch(
                        scoring_function=scoring_function,
                        **kwargs
                    )
                
                # Run search
                results = search_algorithm.search(protein, ligand)
                
                # Add best result
                all_results.append((i, results[0][0], results[0][1]))
                
                # Clear GPU cache between runs
                if self.has_gpu:
                    self.gpu_manager.clear_cache()
        
        # CPU runs
        if n_cpu_runs > 0:
            print(f"Running {n_cpu_runs} docking jobs on CPU...")
            for i in range(n_cpu_runs):
                print(f"CPU run {i+1}/{n_cpu_runs}")
                
                # Run single docking on CPU
                from .scoring import EnhancedScoringFunction
                scoring_function = EnhancedScoringFunction()
                
                # Create search algorithm
                if algorithm_type.lower() == 'genetic':
                    from .search import GeneticAlgorithm
                    search_algorithm = GeneticAlgorithm(
                        scoring_function=scoring_function,
                        **kwargs
                    )
                else:
                    from .search import RandomSearch
                    search_algorithm = RandomSearch(
                        scoring_function=scoring_function,
                        **kwargs
                    )
                
                # Run search
                results = search_algorithm.search(protein, ligand)
                
                # Add best result
                all_results.append((i + n_gpu_runs, results[0][0], results[0][1]))
        
        # Sort by score
        all_results.sort(key=lambda x: x[2])
        
        # Convert to standard format
        final_results = [(pose, score) for _, pose, score in all_results]
        
        return final_results
    
    def cleanup(self):
        """Clean up resources."""
        # Clean up CPU pool
        if self.cpu_pool:
            self.cpu_pool.close()
            self.cpu_pool.join()
            print("CPU pool closed")
        
        # Clean up GPU resources
        if self.has_gpu:
            self.gpu_manager.clear_cache()
            print("GPU resources cleaned up")


# Example usage in main script
def run_optimized_docking(protein_file, ligand_file, output_dir=None, **kwargs):
    """
    Run docking with optimized hardware utilization.
    
    Parameters:
    -----------
    protein_file : str
        Path to protein PDB file
    ligand_file : str
        Path to ligand MOL/SDF file
    output_dir : str
        Output directory
    **kwargs : dict
        Additional arguments for docking
    
    Returns:
    --------
    list
        List of (pose, score) tuples, sorted by score
    """
    from .protein import Protein
    from .ligand import Ligand
    from .utils import save_docking_results
    import time
    import os
    
    # Start timer
    start_time = time.time()
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize hybrid manager
    use_gpu = kwargs.pop('use_gpu', True)
    n_cpu_workers = kwargs.pop('n_cpu_workers', None)
    
    hybrid_manager = HybridDockingManager(
        use_gpu=use_gpu,
        n_cpu_workers=n_cpu_workers
    )
    
    # Load molecules
    print(f"Loading protein from {protein_file}...")
    protein = Protein(protein_file)
    
    print(f"Loading ligand from {ligand_file}...")
    ligand = Ligand(ligand_file)
    
    # Define active site if coordinates provided
    if 'site' in kwargs:
        site = kwargs.pop('site')
        radius = kwargs.pop('radius', 10.0)
        print(f"Using active site at {site} with radius {radius}Å")
        protein.define_active_site(site, radius)
    else:
        # Try to detect binding pockets
        print("Detecting binding pockets...")
        pockets = protein.detect_pockets()
        if pockets:
            print(f"Found {len(pockets)} potential binding pockets")
            print(f"Using largest pocket as active site")
            protein.define_active_site(pockets[0]['center'], pockets[0]['radius'])
        else:
            print("No pockets detected, using whole protein")
    
    # Run docking
    algorithm_type = kwargs.pop('algorithm', 'genetic')
    n_runs = kwargs.pop('n_runs', 1)
    
    if n_runs > 1:
        print(f"Running ensemble docking with {n_runs} runs...")
        results = hybrid_manager.run_ensemble_docking(
            protein=protein,
            ligand=ligand,
            n_runs=n_runs,
            algorithm_type=algorithm_type,
            **kwargs
        )
    else:
        print("Running single docking simulation...")
        results = hybrid_manager.run_docking(
            protein=protein,
            ligand=ligand,
            algorithm_type=algorithm_type,
            **kwargs
        )
    
    # Clean up resources
    hybrid_manager.cleanup()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Save results if output directory specified
    if output_dir:
        save_docking_results(results, output_dir)
    
    print(f"Docking completed in {elapsed_time:.2f} seconds")
    print(f"Best score: {results[0][1]:.4f}")
    
    return results
