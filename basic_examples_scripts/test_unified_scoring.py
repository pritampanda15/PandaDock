# test_unified_scoring.py
import numpy as np
from pandadock.scoring_factory import create_scoring_function
from pandadock.protein import Protein
from pandadock.ligand import Ligand

def test_unified_scoring():
    # Load a test protein and ligand
    protein = Protein("tests/receptor.pdb")
    ligand = Ligand("tests/ligand.sdf")

    # --- Fix: Define dummy active site ---
    protein.active_site = {
        "center": np.mean(protein.xyz, axis=0),
        "radius": 10.0,
        "atoms": protein.atoms
    }

    # Create different scoring functions
    cpu_scorer = create_scoring_function(use_gpu=False, enhanced=True)
    gpu_scorer = create_scoring_function(use_gpu=True, enhanced=True)
    physics_scorer = create_scoring_function(physics_based=True, enhanced=True)

    # Score with each function
    cpu_score = cpu_scorer.score(protein, ligand)
    gpu_score = gpu_scorer.score(protein, ligand)
    physics_score = physics_scorer.score(protein, ligand)

    assert cpu_score is not None
    assert gpu_score is not None
    assert physics_score is not None
    assert isinstance(cpu_score, float)
    assert isinstance(gpu_score, float)
    assert isinstance(physics_score, float)
    
    # Print results
    print(f"CPU Score: {cpu_score:.4f}")
    print(f"GPU Score: {gpu_score:.4f}")
    print(f"Physics Score: {physics_score:.4f}")
    
    # Check consistency between CPU and GPU scores
    # (Physics scores may differ slightly due to different terms)
    assert abs(cpu_score - gpu_score) < 1.0, "CPU and GPU scores differ significantly"
    
    print("Unified scoring integration test passed!")

if __name__ == "__main__":
    test_unified_scoring()