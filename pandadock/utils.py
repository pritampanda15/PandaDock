# utils.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def save_docking_results(results, output_dir='docking_results'):
    """
    Save docking results to output directory.
    
    Parameters:
    -----------
    results : list
        List of (pose, score) tuples
    output_dir : str
        Output directory
    """
    # Create output directory
    out_path = Path(output_dir)
    os.makedirs(out_path, exist_ok=True)
    
    # Save top poses
    for i, (pose, score) in enumerate(results[:10]):  # Save top 10 poses
        # Generate PDB file for the pose
        pdb_path = out_path / f"pose_{i+1}_score_{score:.1f}.pdb"
        with open(pdb_path, 'w') as f:
            f.write(f"REMARK   1 Docking score: {score}\n")
            
            for j, atom in enumerate(pose.atoms):
                coords = atom['coords']
                symbol = atom.get('symbol', 'C')
                
                # PDB ATOM format
                f.write(f"HETATM{j+1:5d} {symbol:<4}{'':<1}{'LIG':<3} {'A':1}{1:4d}    "
                        f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
                        f"{1.0:6.2f}{0.0:6.2f}          {symbol:>2}\n")
    
    # Create a simple score plot
    scores = [score for _, score in results]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), scores)
    plt.xlabel('Pose Rank')
    plt.ylabel('Docking Score')
    plt.title('Docking Results - Score Distribution')
    plt.tight_layout()
    
    plot_path = out_path / "score_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved {len(results)} docking results to {output_dir}")
    print(f"Best docking score: {results[0][1]}")


def calculate_rmsd(coords1, coords2):
    """
    Calculate RMSD between two sets of coordinates.
    
    Parameters:
    -----------
    coords1 : array-like
        First set of coordinates
    coords2 : array-like
        Second set of coordinates
    
    Returns:
    --------
    float
        RMSD value
    """
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have the same shape")
    
    n_atoms = coords1.shape[0]
    squared_diff = np.sum((coords1 - coords2) ** 2)
    rmsd = np.sqrt(squared_diff / n_atoms)
    
    return rmsd


