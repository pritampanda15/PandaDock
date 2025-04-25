from .unified_scoring import (
    CompositeScoringFunction,
    EnhancedScoringFunction,
    GPUScoringFunction,
    EnhancedGPUScoringFunction,
    TetheredScoringFunction,
)
from .physics import (
    PhysicsBasedScoringFunction,PhysicsBasedScoring
)

from .physics import PhysicsBasedScoringFunction
def create_scoring_function(use_gpu=False, physics_based=False, enhanced=True, 
                           tethered=False, reference_ligand=None, weights=None,
                           device='cuda', precision='float32', verbose=False):
    """
    Factory function to create the correct scoring function.
    """
    if physics_based:
        base_function = PhysicsBasedScoringFunction()
        print(f"[DEBUG] Creating scoring function: physics_based={physics_based}, use_gpu={use_gpu}, enhanced={enhanced}")
        if use_gpu:
            print("[DEBUG] GPU support is not available for PhysicsBasedScoringFunction.")
            raise ValueError("GPU support is not available for PhysicsBasedScoringFunction.")
    elif use_gpu:
        base_function = EnhancedGPUScoringFunction(device=device, precision=precision) if enhanced else GPUScoringFunction(device=device, precision=precision)
    else:
        base_function = EnhancedScoringFunction() if enhanced else CompositeScoringFunction()

    if weights:
        for key, value in weights.items():
            if key in base_function.weights:
                base_function.weights[key] = value

    base_function.verbose = verbose
    return base_function  # Ensure the scoring function is returned
