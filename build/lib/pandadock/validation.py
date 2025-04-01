# validation.py

import numpy as np
from pathlib import Path
from .utils import calculate_rmsd  # Import from utils instead of redefining


def validate_docking(docked_ligand, reference_ligand, output_file=None):
    """
    Validate docking results by comparing with reference ligand coordinates.
    
    Parameters:
    -----------
    docked_ligand : str or Ligand
        Docked ligand as a file path or Ligand object
    reference_ligand : str or Ligand
        Reference ligand as a file path or Ligand object
    output_file : str, optional
        Path to output validation report
    
    Returns:
    --------
    dict
        Validation results with keys:
        - rmsd: Overall RMSD
        - atom_deviations: Per-atom deviations
        - max_deviation: Maximum atomic deviation
        - min_deviation: Minimum atomic deviation
        - success: Whether docking is successful (RMSD < 2.0 Å)
    """
    from .ligand import Ligand
    
    # Load ligands if they are file paths
    if isinstance(docked_ligand, str):
        docked_ligand = Ligand(docked_ligand)
    if isinstance(reference_ligand, str):
        reference_ligand = Ligand(reference_ligand)
    
    # Extract coordinates
    docked_coords = docked_ligand.xyz
    reference_coords = reference_ligand.xyz
    
    # Check if coordinate arrays have the same shape
    if docked_coords.shape != reference_coords.shape:
        print(f"Warning: Docked ligand has {len(docked_coords)} atoms, but reference has {len(reference_coords)} atoms")
        # Use the minimum number of atoms for comparison
        min_atoms = min(len(docked_coords), len(reference_coords))
        docked_coords = docked_coords[:min_atoms]
        reference_coords = reference_coords[:min_atoms]
    
    # Calculate overall RMSD
    rmsd = calculate_rmsd(docked_coords, reference_coords)
    
    # Calculate per-atom deviations
    atom_deviations = np.sqrt(np.sum((docked_coords - reference_coords) ** 2, axis=1))
    max_deviation = np.max(atom_deviations)
    min_deviation = np.min(atom_deviations)
    
    # Determine success based on RMSD threshold
    success = rmsd < 2.0  # Standard threshold for successful docking
    
    # Create results dictionary
    results = {
        'rmsd': rmsd,
        'atom_deviations': atom_deviations,
        'max_deviation': max_deviation,
        'min_deviation': min_deviation,
        'success': success
    }
    
    # Write output report if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("=======================================\n")
            f.write("       Docking Validation Report       \n")
            f.write("=======================================\n\n")
            
            f.write(f"RMSD: {rmsd:.4f} Å\n")
            f.write(f"Maximum Atomic Deviation: {max_deviation:.4f} Å\n")
            f.write(f"Minimum Atomic Deviation: {min_deviation:.4f} Å\n")
            f.write(f"Docking Success: {'Yes' if success else 'No'}\n\n")
            
            f.write("Per-Atom Deviations (Top 10):\n")
            f.write("---------------------------\n")
            sorted_indices = np.argsort(atom_deviations)[::-1]
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                f.write(f"Atom {idx + 1}: {atom_deviations[idx]:.4f} Å\n")
    
    return results


def calculate_ensemble_rmsd(docked_poses, reference_ligand):
    """
    Calculate RMSD for an ensemble of docked poses against a reference.
    
    Parameters:
    -----------
    docked_poses : list
        List of docked ligand poses (Ligand objects)
    reference_ligand : Ligand
        Reference ligand structure
    
    Returns:
    --------
    list
        List of dictionaries with RMSD info for each pose, sorted by RMSD
    """
    results = []
    
    # Reference coordinates
    ref_coords = reference_ligand.xyz
    
    for i, pose in enumerate(docked_poses):
        # Extract pose coordinates
        pose_coords = pose.xyz
        
        # Check if coordinate arrays have the same shape
        if pose_coords.shape != ref_coords.shape:
            print(f"Warning: Pose {i+1} has {len(pose_coords)} atoms, but reference has {len(ref_coords)} atoms")
            # Use the minimum number of atoms for comparison
            min_atoms = min(len(pose_coords), len(ref_coords))
            pose_coords = pose_coords[:min_atoms]
            ref_coords_subset = ref_coords[:min_atoms]
            rmsd = calculate_rmsd(pose_coords, ref_coords_subset)
        else:
            rmsd = calculate_rmsd(pose_coords, ref_coords)
        
        results.append({
            'pose_index': i,
            'rmsd': rmsd,
            'success': rmsd < 2.0
        })
    
    # Sort results by RMSD (lowest first)
    results.sort(key=lambda x: x['rmsd'])
    
    return results


def validate_against_reference(args, results, output_dir):
    """
    Validate docking results against a reference structure if provided.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    results : list
        List of (pose, score) tuples from docking
    output_dir : str
        Output directory
    
    Returns:
    --------
    dict
        Validation results
    """
    if not hasattr(args, 'reference') or not args.reference:
        return None
    
    print("\nValidating docking results against reference structure...")
    
    # Load reference ligand
    from .ligand import Ligand
    reference_ligand = Ligand(args.reference)
    
    # Extract poses from results
    poses = [pose for pose, _ in results]
    
    # Calculate RMSDs for all poses
    validation_results = calculate_ensemble_rmsd(poses, reference_ligand)
    
    # Report results
    best_rmsd = validation_results[0]['rmsd']
    best_index = validation_results[0]['pose_index']
    
    print(f"Best RMSD: {best_rmsd:.2f} Å (Pose {best_index + 1})")
    print(f"Docking accuracy: {'Successful' if best_rmsd < 2.0 else 'Unsuccessful'}")
    
    # Write detailed validation report
    validation_file = Path(output_dir) / "validation_report.txt"
    
    with open(validation_file, 'w') as f:
        f.write("==================================================\n")
        f.write("       PandaDock Validation Against Reference        \n")
        f.write("==================================================\n\n")
        
        f.write(f"Reference Ligand: {args.reference}\n\n")
        
        f.write("RMSD Summary:\n")
        f.write("-------------\n")
        f.write(f"Best RMSD: {best_rmsd:.4f} Å (Pose {best_index + 1})\n")
        f.write(f"Docking accuracy: {'Successful' if best_rmsd < 2.0 else 'Unsuccessful'}\n\n")
        
        f.write("All Poses:\n")
        f.write("----------\n")
        f.write("Rank\tPose Index\tRMSD (Å)\tStatus\n")
        
        for i, result in enumerate(validation_results):
            status = "Success" if result['rmsd'] < 2.0 else "Failure"
            f.write(f"{i+1}\t{result['pose_index']+1}\t{result['rmsd']:.4f}\t{status}\n")
    
    print(f"Validation report written to {validation_file}")
    
    return validation_results