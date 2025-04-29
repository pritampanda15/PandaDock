"""
Example usage of the batch screening module for PandaDock
"""

from pandadock import batch_screening

def basic_example():
    """Basic batch screening example"""
    
    # Configuration for batch screening
    config = {
        'protein': 'receptor.pdb',
        'ligand_library': '/mnt/7b616197-a2a7-4736-bd58-c500d1a8c523/Panda-Software/PandaDock/tests/ligands',
        'output_dir': 'screening_results',
        'screening_params': {
            'algorithm': 'genetic',
            'iterations': 500,
            'exhaustiveness': 3,
            'scoring_function': 'enhanced',
            'hardware': {
                'use_gpu': True,
                'cpu_workers': 16
            }
        }
    }

    # Run batch screening
    results = batch_screening.run(config)
    
    # Print top 5 compounds
    print("\nTop 5 compounds:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'])
    for i, (ligand_name, result) in enumerate(sorted_results[:5]):
        if 'error' not in result:
            print(f"{i+1}. {ligand_name}: {result['score']:.2f}")


def parallel_example():
    """Parallel batch screening example"""
    
    # Configuration for parallel batch screening
    config = {
        'protein': 'receptor.pdb',
        'ligand_library': '/mnt/7b616197-a2a7-4736-bd58-c500d1a8c523/Panda-Software/PandaDock/tests/ligands',
        'output_dir': 'parallel_screening_results',
        'n_processes': 4,  # Number of parallel processes
        'screening_params': {
            'algorithm': 'genetic',
            'iterations': 500,
            'exhaustiveness': 1,  # Lower for parallel processing
            'scoring_function': 'enhanced',
            'hardware': {
                'use_gpu': True,
                'cpu_workers': 4  # Reduced for parallel processing
            }
        }
    }

    # Run parallel batch screening
    results = batch_screening.run_parallel(config)
    
    # Print top 5 compounds
    print("\nTop 5 compounds:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['score'])
    for i, (ligand_name, result) in enumerate(sorted_results[:5]):
        if 'error' not in result:
            print(f"{i+1}. {ligand_name}: {result['score']:.2f}")


def advanced_example():
    """Advanced batch screening example with additional parameters"""
    
    # Configuration for advanced batch screening
    config = {
        'protein': 'receptor.pdb',
        'ligand_library': '/mnt/7b616197-a2a7-4736-bd58-c500d1a8c523/Panda-Software/PandaDock/tests/ligands',
        'output_dir': 'advanced_screening_results',
        'screening_params': {
            'algorithm': 'genetic',
            'iterations': 1000,
            'population_size': 150,
            'mutation_rate': 0.3,
            'exhaustiveness': 5,
            'scoring_function': 'physics',  # Use physics-based scoring if available
            'prepare_molecules': True,      # Prepare proteins and ligands
            'detect_pockets': True,         # Auto-detect binding pockets
            'local_opt': True,              # Perform local optimization
            'ph': 7.4,                      # pH for protein preparation
            'hardware': {
                'use_gpu': True,
                'gpu_id': 0,
                'cpu_workers': 16,
                'workload_balance': 0.7     # 70% GPU, 30% CPU
            }
        }
    }

    # Run batch screening
    results = batch_screening.run(config)


if __name__ == "__main__":
    # Run the basic example
    #basic_example()
    
    # Uncomment to run parallel example
     #parallel_example()
    
    # Uncomment to run advanced example
     advanced_example()