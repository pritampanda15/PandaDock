from pandadock import batch_screening

# Batch screening configuration
config = {
    'protein': 'receptor.pdb',
    'ligand_library': '/workspaces/PandaDock/tests/ligands',
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
# Run parallel batch screening
#results = batch_screening.run_parallel(config)